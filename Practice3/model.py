
import os
import glob
import random
import argparse
from typing import List, Tuple

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Path helpers
# ----------------------------
def get_infection_base(data_root: str) -> str:
   
    cand = os.path.join(data_root, "Infection Segmentation Data", "Infection Segmentation Data")
    if os.path.isdir(cand) and all(os.path.isdir(os.path.join(cand, s)) for s in ["Train", "Val", "Test"]):
        return cand
    raise FileNotFoundError(
        f"Cannot find infection segmentation base at:\n  {cand}\n"
        "Check your dataset folder structure."
    )


def list_pairs(inf_base: str, split: str, classes: List[str]) -> List[Tuple[str, str]]:
    """
    Build (image_path, mask_path) pairs for a split.

    """
    pairs = []
    for cls in classes:
        img_dir = os.path.join(inf_base, split, cls, "images")
        msk_dir = os.path.join(inf_base, split, cls, "infection masks")

        if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
            # Try flexible search if folder names differ slightly
            img_alt = glob.glob(os.path.join(inf_base, split, cls, "*images*"))
            msk_alt = glob.glob(os.path.join(inf_base, split, cls, "*infection*mask*"))
            if img_alt:
                img_dir = img_alt[0]
            if msk_alt:
                msk_dir = msk_alt[0]

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        if len(img_paths) == 0:
            continue

        msk_paths = glob.glob(os.path.join(msk_dir, "*"))
        msk_map = {os.path.splitext(os.path.basename(mp))[0]: mp for mp in msk_paths}

        for ip in img_paths:
            key = os.path.splitext(os.path.basename(ip))[0]
            if key in msk_map:
                pairs.append((ip, msk_map[key]))
            else:
                # fallback partial match (rare)
                cand = [mp for k, mp in msk_map.items() if k in key or key in k]
                if cand:
                    pairs.append((ip, cand[0]))
    return pairs


# ----------------------------
# Dataset
# ----------------------------
class InfectionSegDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], img_size: int = 256, augment: bool = False):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, msk_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {msk_path}")

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # Binarize mask
        mask = (mask > 127).astype(np.float32)

        # Light augmentation (safe for CXR)
        if self.augment:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if random.random() < 0.3:
                angle = random.uniform(-5, 5)
                M = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (self.img_size, self.img_size),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (self.img_size, self.img_size),
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        # Normalize
        img = img.astype(np.float32) / 255.0

        # To torch (C,H,W)
        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return img_t, mask_t


# ----------------------------
# Model: Small U-Net
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        m = self.mid(self.pool3(d3))

        u3 = self.up3(m)
        x = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(x)
        x = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(x)
        x = self.conv1(torch.cat([u1, d1], dim=1))
        return self.out(x)  


# ----------------------------
# Loss / Metric
# ----------------------------
bce = nn.BCEWithLogitsLoss()


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


@torch.no_grad()
def dice_coef_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return float(dice.mean().cpu().item())


# ----------------------------
# Save qualitative overlays (no matplotlib needed)
# ----------------------------
@torch.no_grad()
def save_overlay_samples(model, loader, device, out_dir: str, n: int = 4):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    batch = next(iter(loader))
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    probs = torch.sigmoid(logits).cpu().numpy()
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    n = min(n, x_np.shape[0])

    for i in range(n):
        img = (x_np[i, 0] * 255).astype(np.uint8)
        gt = (y_np[i, 0] * 255).astype(np.uint8)
        pr = ((probs[i, 0] > 0.5).astype(np.uint8) * 255)

        # Create a 3-panel image: [X-ray | GT | Pred]
        # Also add overlay on X-ray for Pred (red)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        overlay = img_bgr.copy()
        overlay[:, :, 2] = np.clip(overlay[:, :, 2].astype(np.int32) + (pr > 0) * 120, 0, 255).astype(np.uint8)

        gt_bgr = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
        pr_bgr = cv2.cvtColor(pr, cv2.COLOR_GRAY2BGR)

        panel = np.hstack([overlay, gt_bgr, pr_bgr])
        cv2.imwrite(os.path.join(out_dir, f"sample_{i}.png"), panel)


# ----------------------------
# Train / Eval
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help=r'Path to dataset root, e.g. D:\Documents\...\Covid_dataset')
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train", type=int, default=600)
    parser.add_argument("--max_val", type=int, default=100)
    parser.add_argument("--classes", type=str, default="COVID-19",
                        help='Comma-separated classes, e.g. "COVID-19" or "COVID-19,Non-COVID,Normal"')
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    inf_base = get_infection_base(args.data_root)
    print("Infection Segmentation base:", inf_base)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    print("Classes:", classes)

    train_pairs = list_pairs(inf_base, "Train", classes)
    val_pairs = list_pairs(inf_base, "Val", classes)

    print(f"Pairs: train = {len(train_pairs)} | val = {len(val_pairs)}")

    if len(train_pairs) == 0 or len(val_pairs) == 0:
        raise RuntimeError("No image-mask pairs found. Check folder names and file matching.")

    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    if args.max_train is not None:
        train_pairs = train_pairs[:min(args.max_train, len(train_pairs))]
    if args.max_val is not None:
        val_pairs = val_pairs[:min(args.max_val, len(val_pairs))]

    print(f"Using subset: train = {len(train_pairs)} | val = {len(val_pairs)}")

    train_ds = InfectionSegDataset(train_pairs, img_size=args.img_size, augment=True)
    val_ds = InfectionSegDataset(val_pairs, img_size=args.img_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )

    model = UNetSmall(in_ch=1, out_ch=1, base=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs("outputs", exist_ok=True)
    best_path = os.path.join("outputs", "best_unet_infection.pt")
    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = bce(logits, y) + dice_loss_from_logits(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # Val
        model.eval()
        val_loss = 0.0
        val_dice_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = bce(logits, y) + dice_loss_from_logits(logits, y)
                val_loss += loss.item() * x.size(0)
                val_dice_sum += dice_coef_from_logits(logits, y) * x.size(0)

        val_loss /= len(val_loader.dataset)
        val_dice = val_dice_sum / len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved best: {best_path} (dice={best_dice:.4f})")

    print("Done. Best val dice:", best_dice)

    # Save a few overlay samples from validation set
    model.load_state_dict(torch.load(best_path, map_location=device))
    save_overlay_samples(model, val_loader, device, out_dir="outputs", n=4)
    print("Saved samples to outputs/ (sample_*.png)")


if __name__ == "__main__":
    main()
