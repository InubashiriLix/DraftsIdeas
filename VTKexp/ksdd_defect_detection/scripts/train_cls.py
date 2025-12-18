import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import pandas as pd

from src.datasets import KSDDClsDataset
from src.train_utils import AverageMeter, accuracy, save_checkpoint, set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="processed dataset folder containing index.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="outputs/cls")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data)
    index_csv = data_dir / "index.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    train_ds = KSDDClsDataset(index_csv, split="train")
    val_ds = KSDDClsDataset(index_csv, split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = AverageMeter()
        tr_acc = AverageMeter()

        for x, y in tqdm(train_loader, desc=f"train epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            tr_loss.update(loss.item(), n=x.size(0))
            tr_acc.update(accuracy(logits.detach(), y), n=x.size(0))

        model.eval()
        va_loss = AverageMeter()
        va_acc = AverageMeter()
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"val epoch {epoch}"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                va_loss.update(loss.item(), n=x.size(0))
                va_acc.update(accuracy(logits, y), n=x.size(0))

        row = {
            "epoch": epoch,
            "train_loss": tr_loss.avg,
            "train_acc": tr_acc.avg,
            "val_loss": va_loss.avg,
            "val_acc": va_acc.avg,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        print(f"[E{epoch}] train loss={tr_loss.avg:.4f} acc={tr_acc.avg:.4f} | val loss={va_loss.avg:.4f} acc={va_acc.avg:.4f}")

        save_checkpoint(out_dir / "last.pt", model, extra={"epoch": epoch, "val_acc": va_acc.avg})

        if va_acc.avg > best_val_acc:
            best_val_acc = va_acc.avg
            save_checkpoint(out_dir / "best.pt", model, extra={"epoch": epoch, "val_acc": va_acc.avg})
            print(f"[OK] new best val_acc={best_val_acc:.4f}")

    print("[DONE] best val_acc:", best_val_acc)

if __name__ == "__main__":
    main()
