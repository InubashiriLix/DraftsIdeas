from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    # HWC uint8 -> CHW float32 [0,1]
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return t

def mask_to_tensor(mask01: np.ndarray) -> torch.Tensor:
    # HW {0,1} -> 1xHxW float
    t = torch.from_numpy(mask01).unsqueeze(0).float()
    return t

@dataclass
class Record:
    image_path: Path
    mask_path: Optional[Path]
    label: int

class KSDDClsDataset(Dataset):
    def __init__(self, index_csv: Path, split: str):
        self.df = pd.read_csv(index_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row["image_path"])
        img = bgr_to_rgb(img)
        x = to_tensor(img)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y

class KSDDSegDataset(Dataset):
    def __init__(self, index_csv: Path, split: str):
        self.df = pd.read_csv(index_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        # keep only rows with masks
        self.df = self.df[self.df["mask_path"].notna()].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row["image_path"])
        img = bgr_to_rgb(img)
        x = to_tensor(img)

        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(row["mask_path"])
        mask01 = (mask > 0).astype(np.uint8)
        m = mask_to_tensor(mask01)
        return x, m
