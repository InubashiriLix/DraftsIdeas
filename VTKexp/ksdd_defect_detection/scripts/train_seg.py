import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from src.datasets import KSDDSegDataset
from src.unet import UNet
from src.train_utils import AverageMeter, dice_coeff, save_checkpoint, set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="processed dataset folder containing index.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="outputs/seg")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--base", type=int, default=32, help="unet base channels")
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data)
    index_csv = data_dir / "index.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    train_ds = KSDDSegDataset(index_csv, split="train")
    val_ds = KSDDSegDataset(index_csv, split="val")

    if len(train_ds) == 0:
        raise RuntimeError("No training samples with masks were found. Please ensure you downloaded the fine-annotations version.")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    model = UNet(in_channels=3, out_channels=1, base=args.base).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    best_dice = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = AverageMeter()

        for x, m in tqdm(train_loader, desc=f"train epoch {epoch}"):
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = bce(logits, m)
            loss.backward()
            opt.step()
            tr_loss.update(loss.item(), n=x.size(0))

        model.eval()
        va_loss = AverageMeter()
        va_dice = AverageMeter()

        with torch.no_grad():
            for x, m in tqdm(val_loader, desc=f"val epoch {epoch}"):
                x, m = x.to(device), m.to(device)
                logits = model(x)
                loss = bce(logits, m)
                va_loss.update(loss.item(), n=x.size(0))
                pred01 = (torch.sigmoid(logits) > 0.5).float()
                va_dice.update(dice_coeff(pred01, m), n=x.size(0))

        row = {
            "epoch": epoch,
            "train_loss": tr_loss.avg,
            "val_loss": va_loss.avg,
            "val_dice": va_dice.avg,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        print(f"[E{epoch}] train loss={tr_loss.avg:.4f} | val loss={va_loss.avg:.4f} dice={va_dice.avg:.4f}")

        save_checkpoint(out_dir / "last.pt", model, extra={"epoch": epoch, "val_dice": va_dice.avg})

        if va_dice.avg > best_dice:
            best_dice = va_dice.avg
            save_checkpoint(out_dir / "best.pt", model, extra={"epoch": epoch, "val_dice": va_dice.avg})
            print(f"[OK] new best dice={best_dice:.4f}")

    print("[DONE] best val_dice:", best_dice)

if __name__ == "__main__":
    main()
