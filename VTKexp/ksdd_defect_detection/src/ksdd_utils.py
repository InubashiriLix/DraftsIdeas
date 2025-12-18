import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and is_image_file(p)]

def guess_mask_path(img_path: Path, all_masks: Dict[str, Path]) -> Optional[Path]:
    '''
    Try to match mask by basename (without extension) or common suffix patterns.
    all_masks: mapping from key->path where key is lower basename.
    '''
    stem = img_path.stem.lower()
    if stem in all_masks:
        return all_masks[stem]

    candidates = [
        stem + "_mask",
        stem + "_label",
        stem + "_gt",
        stem.replace("img", "mask"),
        stem.replace("image", "mask"),
    ]
    for c in candidates:
        if c in all_masks:
            return all_masks[c]

    for suf in ["_img", "_image", "_rgb"]:
        if stem.endswith(suf):
            base = stem[: -len(suf)]
            if base in all_masks:
                return all_masks[base]
            for c in [base + "_mask", base + "_label", base + "_gt"]:
                if c in all_masks:
                    return all_masks[c]
    return None

def build_mask_index(root: Path) -> Dict[str, Path]:
    '''
    Find probable mask/gt files by directory name heuristic.
    '''
    masks: Dict[str, Path] = {}
    for p in root.rglob("*"):
        if not p.is_file() or not is_image_file(p):
            continue
        lower = str(p).lower()
        if any(k in lower for k in ["mask", "label", "gt", "ground_truth", "annotation", "ann"]):
            masks[p.stem.lower()] = p
    return masks

def resize_image(img: np.ndarray, size_hw: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    h, w = size_hw
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(img, (w, h), interpolation=interp)

def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def normalize_mask(mask: np.ndarray) -> np.ndarray:
    '''
    Convert mask to {0,1} uint8.
    '''
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binm = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
    return binm.astype(np.uint8)

def mask_to_label(mask01: Optional[np.ndarray]) -> int:
    if mask01 is None:
        return 0
    return int(mask01.sum() > 0)

def safe_relpath(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)
