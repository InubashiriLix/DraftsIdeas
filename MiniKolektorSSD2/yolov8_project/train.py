"""
YOLOv8 Classification Training on MiniKolektorSSD2
Binary: abnormal vs normal (surface defect detection)
"""
from ultralytics import YOLO
import time
import json
import subprocess
import os

# ── Config ──
DATASET = os.path.expanduser("~/08_Drafts/MiniKolektorSSD2_flat")
PROJECT = os.path.dirname(os.path.abspath(__file__))
MODEL = "yolov8n-cls.pt"  # nano classification model
EPOCHS = 30
IMGSZ = 64
BATCH = 128
DEVICE = 0

def get_gpu_power():
    """Get current GPU power draw in watts."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip().split("\n")[0])
    except:
        return -1.0

if __name__ == "__main__":
    print(f"Dataset: {DATASET}")
    print(f"Model: {MODEL}, Epochs: {EPOCHS}, ImgSz: {IMGSZ}, Batch: {BATCH}")

    # Record GPU power before training
    power_before = get_gpu_power()
    print(f"GPU power before training: {power_before}W")

    model = YOLO(MODEL)

    t_start = time.time()
    results = model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=os.path.join(PROJECT, "runs"),
        name="train",
        exist_ok=True,
        patience=15,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.01,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,  # not useful for small 64x64 classification
        verbose=True,
    )
    t_end = time.time()

    power_after = get_gpu_power()

    train_log = {
        "model": MODEL,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "training_time_s": round(t_end - t_start, 2),
        "gpu_power_before_W": power_before,
        "gpu_power_after_W": power_after,
    }

    log_path = os.path.join(PROJECT, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\nTraining complete in {train_log['training_time_s']}s")
    print(f"Log saved to {log_path}")
    print(f"Best weights: {os.path.join(PROJECT, 'runs/train/weights/best.pt')}")
