import argparse
from pathlib import Path
import cv2
import numpy as np

def clahe_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs/cv")
    ap.add_argument("--min_area", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = clahe_gray(gray)

    # emphasize defects: use morphological black-hat (good for dark scratches)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(eq, cv2.MORPH_BLACKHAT, kernel)

    # gradient magnitude
    gx = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blackhat, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_u8 = np.clip(grad / (grad.max() + 1e-6) * 255, 0, 255).astype(np.uint8)

    # threshold
    _, th = cv2.threshold(grad_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # clean
    k2 = np.ones((3,3), np.uint8)
    th2 = cv2.morphologyEx(th, cv2.MORPH_OPEN, k2, iterations=1)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, k2, iterations=2)

    contours, _ = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) >= args.min_area]

    overlay = img.copy()
    cv2.drawContours(overlay, kept, -1, (0,255,0), 2)

    cv2.imwrite(str(out/"0_gray.png"), gray)
    cv2.imwrite(str(out/"1_clahe.png"), eq)
    cv2.imwrite(str(out/"2_blackhat.png"), blackhat)
    cv2.imwrite(str(out/"3_grad.png"), grad_u8)
    cv2.imwrite(str(out/"4_thresh.png"), th)
    cv2.imwrite(str(out/"5_clean.png"), th2)
    cv2.imwrite(str(out/"6_overlay.png"), overlay)

    print(f"[DONE] contours kept={len(kept)} -> {out.resolve()}")

if __name__ == "__main__":
    main()
