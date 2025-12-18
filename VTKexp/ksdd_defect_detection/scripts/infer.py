import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from src.unet import UNet

def preprocess_bgr(img_bgr: np.ndarray, size_hw=(1408, 512)) -> torch.Tensor:
    h, w = size_hw
    img_r = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0)  # 1x3xHxW

def load_cls(ckpt: Path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    payload = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model

def load_seg(ckpt: Path, device):
    model = UNet(in_channels=3, out_channels=1, base=32)
    payload = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model

def overlay_contours(img_bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    # mask01: HxW {0,1}
    contours, _ = cv2.findContours((mask01*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img_bgr.copy()
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out, contours

def run_one(image_path: Path, cls_model, seg_model, device, out_dir: Path, size_hw):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)

    x = preprocess_bgr(img, size_hw=size_hw).to(device)

    with torch.no_grad():
        logits = cls_model(x)
        prob = torch.softmax(logits, dim=1)[0, 1].item()  # defect prob

    # visualization base: resized image
    h, w = size_hw
    base = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    pred_mask01 = None
    contours_n = 0

    if seg_model is not None:
        with torch.no_grad():
            mlogits = seg_model(x)
            msig = torch.sigmoid(mlogits)[0, 0].cpu().numpy()
        pred_mask01 = (msig > 0.5).astype(np.uint8)
        mask_path = out_dir / f"{image_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), (pred_mask01 * 255).astype(np.uint8))

        overlay, contours = overlay_contours(base, pred_mask01)
        contours_n = len(contours)
        overlay_path = out_dir / f"{image_path.stem}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

        # also save contours-only view
        contour_img = np.zeros_like(base)
        cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"{image_path.stem}_contours.png"), contour_img)
    else:
        # no seg model - just save resized image
        cv2.imwrite(str(out_dir / f"{image_path.stem}_resized.png"), base)

    return prob, contours_n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="", help="single image path")
    ap.add_argument("--image_dir", type=str, default="", help="folder of images")
    ap.add_argument("--cls_ckpt", type=str, required=True, help="classification checkpoint (best.pt)")
    ap.add_argument("--seg_ckpt", type=str, default="", help="segmentation checkpoint (best.pt) - optional")
    ap.add_argument("--out_dir", type=str, default="outputs/infer")
    ap.add_argument("--img_h", type=int, default=1408)
    ap.add_argument("--img_w", type=int, default=512)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size_hw = (args.img_h, args.img_w)

    cls_model = load_cls(Path(args.cls_ckpt), device)
    seg_model = None
    if args.seg_ckpt and Path(args.seg_ckpt).exists():
        seg_model = load_seg(Path(args.seg_ckpt), device)

    imgs = []
    if args.image:
        imgs = [Path(args.image)]
    elif args.image_dir:
        d = Path(args.image_dir)
        imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]])
    else:
        raise ValueError("Provide --image or --image_dir")

    results_path = out_dir / "results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "defect_prob", "contours_n"])
        for p in imgs:
            prob, cn = run_one(p, cls_model, seg_model, device, out_dir, size_hw)
            writer.writerow([str(p), f"{prob:.6f}", cn])
            print(f"[OK] {p.name} defect_prob={prob:.4f} contours={cn}")

    print("[DONE] Saved:", results_path.resolve())

if __name__ == "__main__":
    main()
