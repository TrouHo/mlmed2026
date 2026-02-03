# -*- coding: utf-8 -*-

import os

split = "Train"
base = INF_BASE

class_counts = {}
for cls in ["COVID-19", "Non-COVID", "Normal"]:
    img_dir = os.path.join(base, split, cls, "images")
    class_counts[cls] = len(os.listdir(img_dir))

class_counts

import cv2
import numpy as np
import glob

empty_masks = 0
total = 0

mask_paths = glob.glob(os.path.join(base, "Train", "*", "infection masks", "*.png"))

for mp in mask_paths:
    mask = cv2.imread(mp, 0)
    if np.sum(mask) == 0:
        empty_masks += 1
    total += 1

empty_masks, total

import matplotlib.pyplot as plt

plt.bar(class_counts.keys(), class_counts.values())
plt.ylabel("Number of images")
plt.title("Class distribution in training set")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.show()

# ---------- 0) Install deps ----------
!pip -q install kagglehub opencv-python tqdm

import os, glob, random, math
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- 1) Download dataset via KaggleHub ----------
import kagglehub
HANDLE = "anasmohammedtahir/covidqu"
DATA_ROOT = kagglehub.dataset_download(HANDLE)
print("Dataset path:", DATA_ROOT)

# ---------- 2) Config (lightweight) ----------
SEED = 42
IMG_SIZE =  256
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use subset
MAX_TRAIN = 600        # set None to use all
MAX_VAL   = 100
# Which classes
CLASSES = ["COVID-19"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------- 3) Helpers ----------
def find_base_folder(root):
    candidates = [
        os.path.join(root, "Infection Segmentation Data", "Infection Segmentation Data"),
    ]
    for c in candidates:
        if os.path.isdir(c) and all(os.path.isdir(os.path.join(c, s)) for s in ["Train", "Val", "Test"]):
            return c
    raise FileNotFoundError("Could not find infection base folder.")

INF_BASE = find_base_folder(DATA_ROOT)
print("Infection Segmentation base:", INF_BASE)

def list_pairs(split, classes):
    """
    Build (image_path, mask_path) pairs for a split.
    """
    pairs = []
    for cls in classes:
        img_dir = os.path.join(INF_BASE, split, cls, "images")
        msk_dir = os.path.join(INF_BASE, split, cls, "infection masks")
        if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):

            img_dir_alt = glob.glob(os.path.join(INF_BASE, split, cls, "*images*"))
            msk_dir_alt = glob.glob(os.path.join(INF_BASE, split, cls, "*infection*mask*"))
            img_dir = img_dir_alt[0] if img_dir_alt else img_dir
            msk_dir = msk_dir_alt[0] if msk_dir_alt else msk_dir

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        if len(img_paths) == 0:
            continue

        # Create a dict of mask files by basename (without ext) for robust matching
        msk_paths = glob.glob(os.path.join(msk_dir, "*"))
        msk_map = {}
        for mp in msk_paths:
            key = os.path.splitext(os.path.basename(mp))[0]
            msk_map[key] = mp

        for ip in img_paths:
            key = os.path.splitext(os.path.basename(ip))[0]
            if key in msk_map:
                pairs.append((ip, msk_map[key]))
            else:
                # if name mismatch, try partial match (rare)
                cand = [mp for k, mp in msk_map.items() if k in key or key in k]
                if cand:
                    pairs.append((ip, cand[0])

                    )
    return pairs

# ---------- 4) Dataset ----------
class InfectionSegDataset(Dataset):
    def __init__(self, pairs, img_size=256, augment=False):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]

        # read grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {msk_path}")

        # resize
        img  = cv2.resize(img,  (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # binarize mask
        mask = (mask > 127).astype(np.float32)

        # simple augmentation (safe for CXR)
        if self.augment:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            # small rotation
            if random.random() < 0.3:
                angle = random.uniform(-5, 5)
                M = cv2.getRotationMatrix2D((self.img_size/2, self.img_size/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (self.img_size, self.img_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (self.img_size, self.img_size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        # normalize to [0,1]
        img = img.astype(np.float32) / 255.0

        # to torch: (C,H,W)
        img_t = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return img_t, mask_t

# ---------- 5) Model: Small U-Net ----------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)

        self.mid = DoubleConv(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.conv3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.conv1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        m  = self.mid(self.pool3(d3))

        u3 = self.up3(m)
        x  = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(x)
        x  = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(x)
        x  = self.conv1(torch.cat([u1, d1], dim=1))
        return self.out(x)  # logits

# ---------- 6) Loss & metrics ----------
bce = nn.BCEWithLogitsLoss()

def dice_coef_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()
    inter = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (union + eps)
    return dice.mean().item()

def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    inter = (probs * targets).sum(dim=(1,2,3))
    union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (union + eps)
    return 1 - dice.mean()

# ---------- 7) Build pairs + subset ----------
train_pairs = list_pairs("Train", CLASSES)
val_pairs   = list_pairs("Val", CLASSES)

print("Pairs:", "train =", len(train_pairs), "| val =", len(val_pairs))

if len(train_pairs) == 0 or len(val_pairs) == 0:
    raise RuntimeError("No image-mask pairs found. Check HANDLE and folder names under Infection Segmentation.")

random.shuffle(train_pairs)
random.shuffle(val_pairs)

if MAX_TRAIN is not None:
    train_pairs = train_pairs[:min(MAX_TRAIN, len(train_pairs))]
if MAX_VAL is not None:
    val_pairs = val_pairs[:min(MAX_VAL, len(val_pairs))]

print("Using subset:", "train =", len(train_pairs), "| val =", len(val_pairs))

train_ds = InfectionSegDataset(train_pairs, img_size=IMG_SIZE, augment=True)
val_ds   = InfectionSegDataset(val_pairs,   img_size=IMG_SIZE, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ---------- 8) Train ----------
model = UNetSmall(in_ch=1, out_ch=1, base=32).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

os.makedirs("outputs", exist_ok=True)
best_dice = -1.0
best_path = "outputs/best_unet_infection.pt"

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = bce(logits, y) + dice_loss_from_logits(logits, y)
        loss.backward()
        opt.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = bce(logits, y) + dice_loss_from_logits(logits, y)
            val_loss += loss.item() * x.size(0)
            val_dice += dice_coef_from_logits(logits, y) * x.size(0)
    val_loss /= len(val_loader.dataset)
    val_dice /= len(val_loader.dataset)

    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), best_path)
        print(f"  -> Saved best: {best_path} (dice={best_dice:.4f})")

print("Done. Best val dice:", best_dice)

# ---------- 9) Save a few qualitative predictions ----------
import matplotlib.pyplot as plt

model.load_state_dict(torch.load(best_path, map_location=DEVICE))
model.eval()

# take first batch from val
x, y = next(iter(val_loader))
x = x.to(DEVICE)
with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits).cpu().numpy()

x_np = x.cpu().numpy()
y_np = y.numpy()

N_SHOW = min(4, x_np.shape[0])
for i in range(N_SHOW):
    img = x_np[i,0]
    gt  = y_np[i,0]
    pr  = (probs[i,0] > 0.5).astype(np.float32)

    # overlay (red) on grayscale
    overlay = np.stack([img, img, img], axis=-1)
    overlay[...,0] = np.clip(overlay[...,0] + 0.7*pr, 0, 1)  # add red

    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(1,3,1); ax1.imshow(img, cmap="gray"); ax1.set_title("X-ray"); ax1.axis("off")
    ax2 = fig.add_subplot(1,3,2); ax2.imshow(gt, cmap="gray"); ax2.set_title("GT infection mask"); ax2.axis("off")
    ax3 = fig.add_subplot(1,3,3); ax3.imshow(overlay); ax3.set_title("Pred overlay"); ax3.axis("off")
    out_path = f"outputs/sample_{i}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

print("Saved samples to outputs/:", os.listdir("outputs"))