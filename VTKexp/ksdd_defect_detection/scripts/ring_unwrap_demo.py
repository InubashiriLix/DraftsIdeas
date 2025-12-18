"""
Standalone classic-CV pipeline for ring/annular surface defect highlighting.

Given a roughly circular part image, this script will:
1) auto-detect the ring center + outer radius (fallback to largest contour)
2) choose an inner radius by ratio (configurable)
3) unwrap the annulus to a rectangular polar image
4) enhance local anomalies by background subtraction
5) threshold + clean to get a binary defect mask
6) overlay contours / bounding boxes for quick inspection

Outputs (saved to --out_dir):
 01_roi.png      original with detected center/inner/outer circles
 02_unwrap.png   unwrapped ring (BGR)
 05_diff.png     enhanced difference image (grayscale)
 06_mask.png     binary mask after Otsu + morphology + area filter
 07_overlay.png  unwrapped image with defect contours/boxes/area labels

The pipeline avoids deep learning and runs on CPU-only OpenCV/Numpy.
"""

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def detect_ring(gray: np.ndarray, min_r_ratio: float = 0.2, max_r_ratio: float = 0.48) -> Tuple[Tuple[float, float], float]:
    """
    Detect outer circle of the ring. Tries Hough first, falls back to largest contour.
    Returns (cx, cy), outer_radius.
    """
    h, w = gray.shape
    max_r = int(min(h, w) * max_r_ratio)
    min_r = int(min(h, w) * min_r_ratio)

    # Hough tends to work if edges are present
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 4,
        param1=120,
        param2=40,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is not None and len(circles) > 0:
        circles = np.round(circles[0]).astype(int)
        cx, cy, r = max(circles, key=lambda c: c[2])
        return (float(cx), float(cy)), float(r)

    # Fallback: largest contour enclosing circle
    _, th = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Cannot detect ring: no contours found")
    c_max = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c_max)
    return (float(cx), float(cy)), float(r)


def unwrap_annulus(img: np.ndarray, center: Tuple[float, float], r_in: float, r_out: float, angle_steps: int) -> np.ndarray:
    """Unwrap annulus to polar rectangle using cv2.warpPolar."""
    # warpPolar outputs shape (r_out, angle_steps, 3); radius increases downwards
    polar_full = cv2.warpPolar(
        img,
        (angle_steps, int(np.ceil(r_out))),
        center,
        r_out,
        flags=cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )
    r0 = int(max(0, np.floor(r_in)))
    r1 = int(min(polar_full.shape[0], np.ceil(r_out)))
    annulus = polar_full[r0:r1, :, :]
    return annulus


def enhance_diff(gray: np.ndarray, ksize: int = 21, sigma: float = 0) -> np.ndarray:
    """Subtract smoothed background to highlight local anomalies."""
    blur = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    diff = cv2.absdiff(gray, blur)
    # normalize to 0-255
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return diff_norm.astype(np.uint8)


def clean_mask(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Morphological clean + small component removal."""
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, num):  # skip background 0
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input ring image path")
    ap.add_argument("--out_dir", default="outputs/ring_demo", help="folder to save outputs")
    ap.add_argument("--inner_ratio", type=float, default=0.65, help="inner radius = outer_radius * ratio")
    ap.add_argument("--angle_steps", type=int, default=1024, help="columns in unwrapped image (angle resolution)")
    ap.add_argument("--min_area", type=int, default=50, help="min defect area in polar pixels")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    center, r_out = detect_ring(gray)
    r_in = max(5.0, r_out * args.inner_ratio)

    # 01 ROI visualization
    roi_vis = img.copy()
    cv2.circle(roi_vis, (int(center[0]), int(center[1])), int(r_out), (0, 255, 0), 2)
    cv2.circle(roi_vis, (int(center[0]), int(center[1])), int(r_in), (0, 128, 255), 2)
    cv2.circle(roi_vis, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)
    cv2.imwrite(str(out_dir / "01_roi.png"), roi_vis)

    # 02 unwrap
    unwrap = unwrap_annulus(img, center, r_in, r_out, args.angle_steps)
    cv2.imwrite(str(out_dir / "02_unwrap.png"), unwrap)

    # 05 diff enhancement
    unwrap_gray = cv2.cvtColor(unwrap, cv2.COLOR_BGR2GRAY)
    diff = enhance_diff(unwrap_gray, ksize=21, sigma=0)
    cv2.imwrite(str(out_dir / "05_diff.png"), diff)

    # 06 mask
    _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = clean_mask(th, min_area=args.min_area)
    cv2.imwrite(str(out_dir / "06_mask.png"), mask)

    # 07 overlay
    overlay = unwrap.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < args.min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(
            overlay,
            f"A={int(area)}",
            (x, max(10, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(str(out_dir / "07_overlay.png"), overlay)

    print(f"[DONE] Saved results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
