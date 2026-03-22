"""
YOLOv8 Classification Inference — Cat Neuter Detection
Logs: accuracy, confusion matrix, per-sample time, GPU power
"""
from ultralytics import YOLO
import time
import json
import subprocess
import os
import sys
import signal
import numpy as np

PROJECT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(PROJECT, "runs/train/weights/best.pt")
TEST_DIR = os.path.expanduser("~/08_Drafts/CatNeuter/dataset/test")
IMGSZ = 128
DEVICE = 0
GPU_MONITOR = os.path.join(PROJECT, "gpu_monitor.py")
GPU_POWER_LOG = os.path.join(PROJECT, "gpu_power_log.json")

def collect_test_images(test_dir):
    images, labels = [], []
    for cls in sorted(os.listdir(test_dir)):
        d = os.path.join(test_dir, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                images.append(os.path.join(d, f))
                labels.append(cls)
    return images, labels

if __name__ == "__main__":
    print(f"Weights: {WEIGHTS}")
    print(f"Test dir: {TEST_DIR}")

    model = YOLO(WEIGHTS)
    class_names = model.names
    print(f"Class names: {class_names}")

    images, gt_labels = collect_test_images(TEST_DIR)
    print(f"Total test samples: {len(images)}")

    # Warmup
    print("Warming up GPU...")
    for _ in range(10):
        model.predict(images[0], imgsz=IMGSZ, device=DEVICE, verbose=False)

    # Start GPU monitor
    print("Starting GPU power monitor...")
    monitor_proc = subprocess.Popen(
        [sys.executable, GPU_MONITOR, GPU_POWER_LOG],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(0.3)

    # Inference
    per_sample_times, predictions = [], []
    print("Running inference...")
    for img_path in images:
        t0 = time.perf_counter()
        result = model.predict(img_path, imgsz=IMGSZ, device=DEVICE, verbose=False)
        t1 = time.perf_counter()
        per_sample_times.append((t1 - t0) * 1000)
        pred_idx = result[0].probs.top1
        predictions.append(class_names[pred_idx])

    # Stop monitor
    print("Stopping GPU power monitor...")
    monitor_proc.send_signal(signal.SIGTERM)
    monitor_proc.wait(timeout=10)
    time.sleep(0.2)

    gpu_power = {}
    try:
        with open(GPU_POWER_LOG) as f:
            gpu_power = json.load(f)
        print(f"GPU monitor: {gpu_power.get('sample_count', '?')} samples")
    except Exception as e:
        gpu_power = {"error": str(e)}

    # Metrics
    correct = sum(p == g for p, g in zip(predictions, gt_labels))
    accuracy = correct / len(gt_labels) * 100

    class_stats = {}
    for cls in sorted(set(gt_labels)):
        c = sum(p == g == cls for p, g in zip(predictions, gt_labels))
        t = sum(g == cls for g in gt_labels)
        class_stats[cls] = {"correct": c, "total": t, "accuracy_pct": round(c/t*100, 2) if t else 0}

    unique = sorted(set(gt_labels))
    confusion = {gt: {pred: 0 for pred in unique} for gt in unique}
    for p, g in zip(predictions, gt_labels):
        confusion[g][p] += 1

    # Precision/Recall/F1 for each class
    pr_stats = {}
    for cls in unique:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in unique if other != cls)
        fn = sum(confusion[cls][other] for other in unique if other != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        pr_stats[cls] = {"precision": round(prec*100, 2), "recall": round(rec*100, 2), "f1": round(f1*100, 2)}

    times_arr = np.array(per_sample_times)

    infer_log = {
        "weights": WEIGHTS,
        "total_samples": len(images),
        "accuracy_pct": round(accuracy, 2),
        "per_class": class_stats,
        "precision_recall_f1": pr_stats,
        "confusion_matrix": confusion,
        "inference_time_ms": {
            "mean": round(float(times_arr.mean()), 3),
            "median": round(float(np.median(times_arr)), 3),
            "std": round(float(times_arr.std()), 3),
            "p95": round(float(np.percentile(times_arr, 95)), 3),
            "p99": round(float(np.percentile(times_arr, 99)), 3),
        },
        "gpu_power": gpu_power,
        "total_inference_time_s": round(float(times_arr.sum() / 1000), 3),
        "per_sample_energy_mJ": round(
            gpu_power.get("mean_power_W", 0) * float(times_arr.mean()) if "mean_power_W" in gpu_power else -1, 2
        ),
    }

    log_path = os.path.join(PROJECT, "infer_log.json")
    with open(log_path, "w") as f:
        json.dump(infer_log, f, indent=2)

    # Print
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    for cls, s in class_stats.items():
        pr = pr_stats[cls]
        print(f"  {cls}: acc={s['accuracy_pct']}% P={pr['precision']}% R={pr['recall']}% F1={pr['f1']}% ({s['correct']}/{s['total']})")
    print(f"\nConfusion Matrix (rows=GT, cols=Pred):")
    print("".ljust(12) + "".join(c.ljust(12) for c in unique))
    for gt in unique:
        print(gt.ljust(12) + "".join(str(confusion[gt][p]).ljust(12) for p in unique))
    print(f"\nInference: mean={infer_log['inference_time_ms']['mean']}ms, P95={infer_log['inference_time_ms']['p95']}ms")
    if "error" not in gpu_power:
        print(f"GPU Power: mean={gpu_power.get('mean_power_W')}W, total={gpu_power.get('total_energy_J')}J")
        print(f"Per-sample energy: {infer_log['per_sample_energy_mJ']} mJ")
    print(f"\nLog saved to {log_path}")
