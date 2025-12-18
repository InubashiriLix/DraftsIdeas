import argparse
import sys
from pathlib import Path

# allow `from src...` when running as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hashlib
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

from src.ksdd_utils import build_mask_index, guess_mask_path, list_images, read_bgr, read_gray, normalize_mask, resize_image

def is_probably_mask(p: Path) -> bool:
    s = str(p).lower()
    return any(k in s for k in ["mask", "label", "gt", "ground_truth", "annotation", "ann"])

def make_id(p: Path) -> str:
    h = hashlib.md5(str(p).encode("utf-8")).hexdigest()[:12]
    return f"{p.stem}_{h}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="extracted dataset root (folder)")
    ap.add_argument("--out", type=str, default="data/processed", help="output processed dataset folder")
    ap.add_argument("--img_h", type=int, default=1408)
    ap.add_argument("--img_w", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    args = ap.parse_args()

    root = Path(args.data_root)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    size_hw = (args.img_h, args.img_w)

    print("[INFO] Scanning masks index...")
    mask_index = build_mask_index(root)
    print(f"[INFO] Found mask-like files: {len(mask_index)}")

    print("[INFO] Scanning image files...")
    all_imgs = list_images(root)
    imgs = [p for p in all_imgs if not is_probably_mask(p)]
    print(f"[INFO] Candidate images: {len(imgs)} (from {len(all_imgs)} total image files)")

    records = []
    missing_mask = 0
    for img_path in imgs:
        mid = make_id(img_path)
        mask_path = guess_mask_path(img_path, mask_index)

        img = read_bgr(img_path)
        img_r = resize_image(img, size_hw, is_mask=False)
        out_img = out / "images" / f"{mid}.png"
        cv2.imwrite(str(out_img), img_r)

        out_mask = None
        label = 0
        if mask_path is not None and mask_path.exists():
            m = read_gray(mask_path)
            m01 = normalize_mask(m)
            m_r = resize_image((m01 * 255).astype(np.uint8), size_hw, is_mask=True)
            m01_r = (m_r > 0).astype(np.uint8)
            label = int(m01_r.sum() > 0)
            out_mask = out / "masks" / f"{mid}.png"
            cv2.imwrite(str(out_mask), (m01_r * 255).astype(np.uint8))
        else:
            missing_mask += 1

        records.append({
            "id": mid,
            "image_path": str(out_img.resolve()),
            "mask_path": str(out_mask.resolve()) if out_mask else "",
            "label": label,
        })

    df = pd.DataFrame(records)

    if missing_mask > 0:
        print(f"[WARN] {missing_mask} images have no matched mask. Their label defaults to 0 (normal).")

    y = df["label"].values
    idx = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_ratio, random_state=args.seed, stratify=y if y.sum() > 0 else None
    )
    y_train = df.iloc[train_idx]["label"].values
    val_size = args.val_ratio / max(1e-6, (1.0 - args.test_ratio))
    train_idx2, val_idx = train_test_split(
        train_idx, test_size=val_size, random_state=args.seed, stratify=y_train if y_train.sum() > 0 else None
    )

    split = np.array(["train"] * len(df), dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    df["split"] = split

    df["mask_path"] = df["mask_path"].replace({"": np.nan})

    index_csv = out / "index.csv"
    df.to_csv(index_csv, index=False)
    print("[OK] Saved index:", index_csv.resolve())
    print(df.groupby(["split","label"]).size())

if __name__ == "__main__":
    main()
