"""
Export YOLO .pt model to TensorRT .engine for Jetson Orin Nano
==============================================================
Run this ONCE inside the Docker container on the Jetson.
The .engine file is NOT portable — it must be built on the target device.

Usage:
    python /app/export_tensorrt.py
"""

import os
from ultralytics import YOLO

MODEL_DIR = "/app/model"
PT_MODEL = os.path.join(MODEL_DIR, "best.pt")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "416"))

if not os.path.exists(PT_MODEL):
    print(f"ERROR: {PT_MODEL} not found!")
    print(f"Please place your best.pt model in the ./model/ directory")
    exit(1)

print("=" * 60)
print("  TensorRT Export — Jetson Orin Nano")
print("=" * 60)
print(f"  Source:   {PT_MODEL}")
print(f"  ImgSize:  {IMG_SIZE}")
print(f"  Format:   TensorRT FP16")
print("=" * 60)
print()
print("This will take 5-15 minutes...")
print()

import torch
import gc

# Free memory aggressively before export
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print(f"  GPU memory free: {torch.cuda.mem_get_info()[0] / 1024**2:.0f} MB")
print()

model = YOLO(PT_MODEL)

model.export(
    format="engine",        # TensorRT
    imgsz=IMG_SIZE,         # Must match inference size
    half=True,              # FP16 — supported on Orin (Ampere arch)
    device=0,               # Use GPU
    workspace=1,            # GB — keep LOW to avoid OOM on 8GB Jetson
    simplify=True,          # ONNX simplification before conversion
    batch=1,                # Single batch to reduce memory
)

engine_path = PT_MODEL.replace(".pt", ".engine")
if os.path.exists(engine_path):
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print()
    print("=" * 60)
    print(f"  ✓ SUCCESS! TensorRT engine created")
    print(f"  Path: {engine_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print("=" * 60)
else:
    print("WARNING: Engine file not found at expected path.")
    print("Check the output above for the actual path.")
