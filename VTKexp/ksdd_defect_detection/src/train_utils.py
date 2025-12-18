from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import numpy as np

@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1):
        self.total += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)

def save_checkpoint(path: Path, model: torch.nn.Module, extra: Optional[Dict] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))

def load_checkpoint(path: Path, model: torch.nn.Module, map_location="cpu") -> Dict:
    payload = torch.load(str(path), map_location=map_location)
    model.load_state_dict(payload["model"])
    return payload

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy(pred_logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = torch.argmax(pred_logits, dim=1)
    return float((pred == y).float().mean().item())

def dice_coeff(pred01: torch.Tensor, target01: torch.Tensor, eps: float = 1e-6) -> float:
    # pred01/target01: (N,1,H,W) float {0,1}
    inter = (pred01 * target01).sum(dim=(1,2,3))
    union = pred01.sum(dim=(1,2,3)) + target01.sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (union + eps)
    return float(dice.mean().item())
