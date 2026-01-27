import os
import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")


# -----------------------------
# Utilities
# -----------------------------
def find_first_existing_col(df: pd.DataFrame, candidates):
    cols = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in cols:
            return df.columns[cols.index(cand.lower())]
    return None


def read_grayscale_image(path: str, size=(256, 256)) -> np.ndarray:
    """Read image -> grayscale -> resize -> float32 in [0,1]."""
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def otsu_threshold(x: np.ndarray) -> float:
    """Simplethreshold for grayscale in [0,1]."""
    # histogram over 256 bins
    hist, bin_edges = np.histogram(x.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    prob = hist / (hist.sum() + 1e-12)

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    # between-class variance
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    k = np.argmax(sigma_b2)
    # map bin index to threshold in [0,1]
    thr = (k + 0.5) / 256.0
    return float(thr)


def extract_features(img: np.ndarray) -> np.ndarray:
    
    x = img
    flat = x.ravel()

    mean = flat.mean()
    std = flat.std()
    median = np.median(flat)
    p10 = np.percentile(flat, 10)
    p90 = np.percentile(flat, 90)

    # simple thresholds
    thr_ms = float(np.clip(mean + std, 0.0, 1.0))
    frac_ms = float((flat > thr_ms).mean())

    thr_otsu = otsu_threshold(x)
    frac_otsu = float((flat > thr_otsu).mean())

    # gradient (finite differences)
    gx = np.diff(x, axis=1)
    gy = np.diff(x, axis=0)
    g = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)  # align shapes
    g_flat = g.ravel()
    g_mean = g_flat.mean()
    g_std = g_flat.std()
    g_p90 = np.percentile(g_flat, 90)

    # projections
    proj_h = x.mean(axis=1)  # (H,)
    proj_v = x.mean(axis=0)  # (W,)
    proj_h_mean, proj_h_std = proj_h.mean(), proj_h.std()
    proj_v_mean, proj_v_std = proj_v.mean(), proj_v.std()

    feats = np.array([
        mean, std, median, p10, p90,
        thr_ms, frac_ms,
        thr_otsu, frac_otsu,
        g_mean, g_std, g_p90,
        proj_h_mean, proj_h_std,
        proj_v_mean, proj_v_std
    ], dtype=np.float32)
    return feats


@dataclass
class Paths:
    root: str
    train_img_dir: str
    test_img_dir: str
    train_csv: str
    test_csv: str


def build_paths(root: str) -> Paths:
    return Paths(
        root=root,
        train_img_dir=os.path.join(root, "training_set", "training_set"),
        test_img_dir=os.path.join(root, "test_set", "test_set"),
        train_csv=os.path.join(root, "training_set_pixel_size_and_HC.csv"),
        test_csv=os.path.join(root, "test_set_pixel_size.csv"),
    )


def resolve_image_path(img_dir: str, name: str) -> str:
    
    if os.path.exists(name):
        return name
    p = os.path.join(img_dir, name)
    return p


def make_feature_table(df: pd.DataFrame, img_dir: str,
                       name_col: str,
                       px_col: str = None,
                       py_col: str = None,
                       resize=(256, 256),
                       verbose_every: int = 200) -> np.ndarray:
    feats_list = []
    n = len(df)
    for i, row in df.iterrows():
        img_name = str(row[name_col])
        img_path = resolve_image_path(img_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = read_grayscale_image(img_path, size=resize)
        f = extract_features(img)

        # add pixel size as features if available
        extra = []
        if px_col is not None:
            extra.append(float(row[px_col]))
        if py_col is not None:
            extra.append(float(row[py_col]))

        if extra:
            f = np.concatenate([f, np.array(extra, dtype=np.float32)], axis=0)

        feats_list.append(f)

        if (i + 1) % verbose_every == 0:
            print(f"  processed {i+1}/{n} images...")

    X = np.vstack(feats_list)
    return X


def main():
    parser = argparse.ArgumentParser(description="HC18 fetal head circumference regression (simple ML baseline).")
    parser.add_argument("--data_root", type=str, required=True,
                        help=r'Path to Ultrasound_dataset folder, e.g. "D:\Documents\USTH\ML in MED\Ultrasound_dataset"')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int, default=500) #Can be change to 300
    parser.add_argument("--max_depth", type=int, default=None) #Can be change to 20 
    parser.add_argument("--out_pred_csv", type=str, default="test_predictions.csv",
                        help="Output CSV for test predictions.")
    args = parser.parse_args()

    paths = build_paths(args.data_root)

    # -----------------------------
    # 1) Load CSVs
    # -----------------------------
    if not os.path.exists(paths.train_csv):
        raise FileNotFoundError(f"Training CSV not found: {paths.train_csv}")
    if not os.path.exists(paths.test_csv):
        raise FileNotFoundError(f"Test CSV not found: {paths.test_csv}")

    train_df = pd.read_csv(paths.train_csv)
    test_df = pd.read_csv(paths.test_csv)

    # Try to infer column names robustly
    name_col_train = find_first_existing_col(train_df, ["image_name", "filename", "img", "image"])
    name_col_test = find_first_existing_col(test_df, ["image_name", "filename", "img", "image"])

    hc_col = find_first_existing_col(
    train_df,
    ["HC", "hc", "head_circumference", "Head Circumference (mm)"]
)


    px_col_train = find_first_existing_col(train_df, ["pixel_size_x", "px", "spacing_x"])
    py_col_train = find_first_existing_col(train_df, ["pixel_size_y", "py", "spacing_y"])

    px_col_test = find_first_existing_col(test_df, ["pixel_size_x", "px", "spacing_x"])
    py_col_test = find_first_existing_col(test_df, ["pixel_size_y", "py", "spacing_y"])

    if name_col_train is None or name_col_test is None:
        raise ValueError("Cannot find image filename column in CSV. Expected columns like image_name/filename/image.")
    if hc_col is None:
        raise ValueError("Cannot find HC column in training CSV. Expected column like HC/head_circumference.")

    print("\n=== Dataset Overview ===")
    print(f"Train images folder: {paths.train_img_dir}")
    print(f"Test images folder : {paths.test_img_dir}")
    print(f"Train CSV: {paths.train_csv}")
    print(f"Test CSV : {paths.test_csv}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples : {len(test_df)}")

    # -----------------------------
    # 2) Quick EDA (text-based)
    # -----------------------------
    print("\n=== Quick EDA (training) ===")
    print("Columns:", list(train_df.columns))
    print(train_df[[name_col_train, hc_col]].head())

    y = train_df[hc_col].astype(float).values
    print("\nHC stats (mm):")
    print(f"  min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}, std={y.std():.2f}")

    if px_col_train and py_col_train:
        px = train_df[px_col_train].astype(float).values
        py = train_df[py_col_train].astype(float).values
        print("\nPixel size stats:")
        print(f"  px mean={px.mean():.6f}, std={px.std():.6f}")
        print(f"  py mean={py.mean():.6f}, std={py.std():.6f}")
        print(f"  mean |px-py| = {np.mean(np.abs(px - py)):.6f}")

    missing_imgs = 0
    for nm in train_df[name_col_train].astype(str).head(20):
        if not os.path.exists(resolve_image_path(paths.train_img_dir, nm)):
            missing_imgs += 1
    if missing_imgs > 0:
        print("\n[Warning] Some image paths in CSV may not match files. Please check filename column values.")
    else:
        print("\nImage path check (first 20): OK")

    # -----------------------------
    # 3) Build features
    # -----------------------------
    print("\n=== Feature Extraction (train) ===")
    X = make_feature_table(
        train_df,
        img_dir=paths.train_img_dir,
        name_col=name_col_train,
        px_col=px_col_train,
        py_col=py_col_train,
        resize=(256, 256),
        verbose_every=200
    )
    print(f"Feature matrix shape: {X.shape}")

    # -----------------------------
    # 4) Train / Val split + model
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
        n_jobs=-1
    )

    print("\n=== Training RandomForestRegressor ===")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE: {mae:.4f} (mm)")

    # -----------------------------
    # 5) Train on full train, predict test
    # -----------------------------
    print("\n=== Feature Extraction (test) ===")
    X_test = make_feature_table(
        test_df,
        img_dir=paths.test_img_dir,
        name_col=name_col_test,
        px_col=px_col_test,
        py_col=py_col_test,
        resize=(256, 256),
        verbose_every=200
    )
    print(f"Test feature matrix shape: {X_test.shape}")

    print("\n=== Fit full train & Predict test ===")
    model.fit(X, y)
    test_pred = model.predict(X_test)

    out_df = pd.DataFrame({
        name_col_test: test_df[name_col_test].astype(str).values,
        "HC_pred": test_pred.astype(float)
    })
    out_df.to_csv(args.out_pred_csv, index=False)
    print(f"Saved test predictions to: {args.out_pred_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
