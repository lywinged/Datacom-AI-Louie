#!/usr/bin/env python3
"""
Debug script to test embedding speed with different batch sizes.
"""
import time
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from backend.services.onnx_inference import ONNXEmbeddingModel

# Initialize model
model_path = "./models/bge-m3-embed-int8"
print(f"Loading model from: {model_path}")
model = ONNXEmbeddingModel(model_path)

# Test texts (32 copies of "hi")
texts = ["hi"] * 32

print(f"\nTesting with {len(texts)} texts...")
print("=" * 60)

# Test different batch sizes
batch_sizes = [1, 4, 8, 16, 32, 64]

for batch_size in batch_sizes:
    # Warm-up run
    _ = model.encode(texts[:4], batch_size=4)

    # Actual test
    start = time.perf_counter()
    _ = model.encode(texts, batch_size=batch_size)
    elapsed = (time.perf_counter() - start) * 1000

    per_text = elapsed / len(texts)
    print(f"batch_size={batch_size:2d} -> {elapsed:7.1f}ms total, {per_text:5.1f}ms per text")

print("=" * 60)
