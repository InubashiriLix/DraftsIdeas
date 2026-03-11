"""
YOLOv8 Classification Inference + Assessment on MiniKolektorSSD2
Logs: per-sample inference time, GPU power (via external monitor), accuracy, confusion matrix
"""
from ultralytics import YOLO
import time
import json
import subprocess
import os
import sys
import numpy as np
from pathlib import Path
import signal

# ── Config ──
PROJECT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(PROJECT, "runs/train/weights/best.pt")
DATASET = os.path.expanduser("~/08_Drafts/MiniKolektorSSD2_flat")
TEST_DIR = os.path.join(DATASET, "test")
IMGSZ = 64
DEVICE = 0
GPU_MONITOR = os.path.join(PROJECT, "gpu_monitor.py")
GPU_POWER_LOG = os.path.join(PROJECT, "gpu_power_log.json")

def collect_test_images(test_dir):
    images = []
    labels = []
    for class_name in sorted(os.listdir(test_dir)):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_file in sorted(os.listdir(class_dir)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                images.append(os.path.join(class_dir, img_file))
                labels.append(class_name)
    return images, labels

if __name__ == "__main__":
    print(f"Weights: {WEIGHTS}")
    print(f"Test dir: {TEST_DIR}")

    model = YOLO(WEIGHTS)
    class_names = model.names
    print(f"Class names: {class_names}")

    images, gt_labels = collect_test_images(TEST_DIR)
    print(f"Total test samples: {len(images)}")

    # ── Warmup ──
    print("Warming up GPU...")
    for _ in range(10):
        model.predict(images[0], imgsz=IMGSZ, device=DEVICE, verbose=False)

    # ── Start GPU power monitor ──
    print("Starting GPU power monitor...")
    monitor_proc = subprocess.Popen(
        [sys.executable, GPU_MONITOR, GPU_POWER_LOG],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(0.3)  # let monitor start sampling

    # ── Inference ──
    per_sample_times = []
    predictions = []

    print("Running inference...")
    for i, img_path in enumerate(images):
        t0 = time.perf_counter()
        result = model.predict(img_path, imgsz=IMGSZ, device=DEVICE, verbose=False)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        per_sample_times.append(elapsed_ms)

        probs = result[0].probs
        pred_idx = probs.top1
        pred_name = class_names[pred_idx]
        predictions.append(pred_name)

    # ── Stop GPU monitor ──
    print("Stopping GPU power monitor...")
    monitor_proc.send_signal(signal.SIGTERM)
    monitor_proc.wait(timeout=10)
    time.sleep(0.2)

    # ── Read GPU power results ──
    gpu_power = {}
    try:
        with open(GPU_POWER_LOG) as f:
            gpu_power = json.load(f)
        print(f"GPU monitor collected {gpu_power.get('sample_count', '?')} samples")
    except Exception as e:
        print(f"Warning: could not read GPU power log: {e}")
        gpu_power = {"error": str(e)}

    # ── Compute Metrics ──
    correct = sum(1 for p, g in zip(predictions, gt_labels) if p == g)
    accuracy = correct / len(gt_labels) * 100

    class_stats = {}
    for cls in sorted(set(gt_labels)):
        cls_mask = [g == cls for g in gt_labels]
        cls_correct = sum(1 for p, g, m in zip(predictions, gt_labels, cls_mask) if m and p == g)
        cls_total = sum(cls_mask)
        class_stats[cls] = {
            "correct": cls_correct,
            "total": cls_total,
            "accuracy_pct": round(cls_correct / cls_total * 100, 2) if cls_total > 0 else 0
        }

    unique_classes = sorted(set(gt_labels))
    confusion = {gt: {pred: 0 for pred in unique_classes} for gt in unique_classes}
    for p, g in zip(predictions, gt_labels):
        confusion[g][p] += 1

    times_arr = np.array(per_sample_times)

    infer_log = {
        "weights": WEIGHTS,
        "total_samples": len(images),
        "accuracy_pct": round(accuracy, 2),
        "per_class": class_stats,
        "confusion_matrix": confusion,
        "inference_time_ms": {
            "mean": round(float(times_arr.mean()), 3),
            "median": round(float(np.median(times_arr)), 3),
            "std": round(float(times_arr.std()), 3),
            "min": round(float(times_arr.min()), 3),
            "max": round(float(times_arr.max()), 3),
            "p95": round(float(np.percentile(times_arr, 95)), 3),
            "p99": round(float(np.percentile(times_arr, 99)), 3),
        },
        "gpu_power": gpu_power,
        "total_inference_time_s": round(float(times_arr.sum() / 1000), 3),
    }

    log_path = os.path.join(PROJECT, "infer_log.json")
    with open(log_path, "w") as f:
        json.dump(infer_log, f, indent=2)

    # ── Print Summary ──
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    for cls, stats in class_stats.items():
        print(f"  {cls}: {stats['accuracy_pct']}% ({stats['correct']}/{stats['total']})")
    print(f"\nConfusion Matrix (rows=GT, cols=Pred):")
    header = "".ljust(12) + "".join(c.ljust(12) for c in unique_classes)
    print(header)
    for gt in unique_classes:
        row = gt.ljust(12) + "".join(str(confusion[gt][p]).ljust(12) for p in unique_classes)
        print(row)
    print(f"\nInference Time per Sample:")
    print(f"  Mean: {infer_log['inference_time_ms']['mean']} ms")
    print(f"  Median: {infer_log['inference_time_ms']['median']} ms")
    print(f"  P95: {infer_log['inference_time_ms']['p95']} ms")
    print(f"  P99: {infer_log['inference_time_ms']['p99']} ms")
    print(f"\nGPU Power Usage:")
    if "error" not in gpu_power:
        print(f"  Mean power: {gpu_power.get('mean_power_W', '?')} W")
        print(f"  Min power: {gpu_power.get('min_power_W', '?')} W")
        print(f"  Max power: {gpu_power.get('max_power_W', '?')} W")
        print(f"  Total energy: {gpu_power.get('total_energy_J', '?')} J ({gpu_power.get('total_energy_Wh', '?')} Wh)")
        print(f"  Monitor duration: {gpu_power.get('duration_s', '?')} s ({gpu_power.get('sample_count', '?')} samples)")
    else:
        print(f"  Error: {gpu_power.get('error', 'unknown')}")
    print(f"\nTotal inference time: {infer_log['total_inference_time_s']}s for {len(images)} samples")
    print(f"Log saved to {log_path}")
