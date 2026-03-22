#!/usr/bin/env python3
"""GPU power monitor — runs in background, samples nvidia-smi power draw."""
import subprocess, time, json, sys, os, signal

NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"
INTERVAL = 0.1  # sample every 100ms
output_file = sys.argv[1] if len(sys.argv) > 1 else "gpu_power_log.json"

samples = []
running = True

def stop(sig, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)

start = time.time()
while running:
    try:
        out = subprocess.check_output(
            [NVIDIA_SMI, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            timeout=2
        ).decode().strip()
        watts = float(out)
        samples.append({"t": round(time.time() - start, 3), "w": watts})
    except:
        pass
    time.sleep(INTERVAL)

# Compute summary
if samples:
    powers = [s["w"] for s in samples]
    duration = samples[-1]["t"] - samples[0]["t"] if len(samples) > 1 else 0
    energy_joules = sum(
        (samples[i+1]["t"] - samples[i]["t"]) * (powers[i] + powers[i+1]) / 2
        for i in range(len(samples) - 1)
    )
    result = {
        "sample_count": len(samples),
        "duration_s": round(duration, 2),
        "mean_power_W": round(sum(powers) / len(powers), 2),
        "min_power_W": round(min(powers), 2),
        "max_power_W": round(max(powers), 2),
        "total_energy_J": round(energy_joules, 2),
        "total_energy_Wh": round(energy_joules / 3600, 4),
    }
else:
    result = {"error": "no samples collected"}

with open(output_file, "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
