"""
YOLOv8 Classification Training — Cat Neuter Detection
Binary: neutered (ear-tipped) vs intact
"""
from ultralytics import YOLO
import time
import json
import os

PROJECT = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.expanduser("~/08_Drafts/CatNeuter/dataset")

MODEL = "yolov8n-cls.pt"
EPOCHS = 50
IMGSZ = 128  # cat images are bigger than 64x64 defect patches
BATCH = 64
PATIENCE = 15
OPTIMIZER = "AdamW"
LR = 0.001

if __name__ == "__main__":
    print(f"Dataset: {DATASET}")
    print(f"Model: {MODEL}, Epochs: {EPOCHS}, ImgSz: {IMGSZ}, Batch: {BATCH}")

    # Count images
    for split in ["train", "test"]:
        for cls in sorted(os.listdir(os.path.join(DATASET, split))):
            d = os.path.join(DATASET, split, cls)
            if os.path.isdir(d):
                print(f"  {split}/{cls}: {len(os.listdir(d))} images")

    model = YOLO(MODEL)
    run_dir = os.path.join(PROJECT, "runs")

    t0 = time.time()
    model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        optimizer=OPTIMIZER,
        lr0=LR,
        weight_decay=0.01,
        project=run_dir,
        name="train",
        exist_ok=True,
        verbose=True,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        scale=0.3,
        translate=0.1,
    )
    elapsed = time.time() - t0

    log = {
        "model": MODEL,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "training_time_s": round(elapsed, 2),
    }
    log_path = os.path.join(PROJECT, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Log saved to {log_path}")
