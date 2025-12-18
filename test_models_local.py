"""
Quick local test for ReID and Health models.
No database needed - just tests model inference.

Usage: python test_models_local.py <image_path>
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from PIL import Image
import numpy as np

# Set up paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("Tree Identification Backend - Local Model Test")
print("="*60)

# Check for test image
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    # Use a sample image if none provided
    print("\nNo image provided. Creating test image...")
    arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image = Image.fromarray(arr, 'RGB')
    image_path = None

if image_path:
    print(f"\nLoading image: {image_path}")
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

print(f"Image size: {image.size}")

# Test ReID model
print("\n" + "-"*60)
print("Testing Tree Re-ID Model (ConvNeXt-Base + TTA)")
print("-"*60)

from app.services.reid_service import ReIDService

reid_start = time.time()
reid_service = ReIDService("./models/reid/best_model.pth")

load_start = time.time()
loaded = reid_service.load_model()
load_time = time.time() - load_start

if loaded:
    print(f"✓ Model loaded in {load_time:.2f}s")
    print(f"  Device: {reid_service.device}")
    print(f"  Embedding dim: {reid_service.embedding_dim}")

    # Test without TTA
    infer_start = time.time()
    embedding_no_tta = reid_service.extract_embedding(image, use_tta=False)
    no_tta_time = time.time() - infer_start
    print(f"\n✓ Inference (no TTA): {no_tta_time*1000:.1f}ms")
    print(f"  Embedding shape: {embedding_no_tta.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding_no_tta):.4f}")

    # Test with TTA
    infer_start = time.time()
    embedding_tta = reid_service.extract_embedding(image, use_tta=True)
    tta_time = time.time() - infer_start
    print(f"\n✓ Inference (with TTA): {tta_time*1000:.1f}ms")
    print(f"  Embedding shape: {embedding_tta.shape}")
    print(f"  TTA overhead: {tta_time/no_tta_time:.1f}x slower")
else:
    print("✗ Failed to load ReID model")

reid_total = time.time() - reid_start

# Test Health model
print("\n" + "-"*60)
print("Testing Tree Health Model (EfficientNet-B3 Ensemble + TTA)")
print("-"*60)

from app.services.health_service import HealthService

health_start = time.time()
health_service = HealthService("./models/health")

load_start = time.time()
loaded = health_service.load_models()
load_time = time.time() - load_start

if loaded:
    print(f"✓ {len(health_service.models)} models loaded in {load_time:.2f}s")
    print(f"  Device: {health_service.device}")

    # Test health assessment
    infer_start = time.time()
    result = health_service.assess_health(image, use_tta=True)
    health_time = time.time() - infer_start

    print(f"\n✓ Health assessment: {health_time*1000:.1f}ms")
    print(f"  Overall confidence: {result.get('overall_confidence', 'N/A')}")
    print("\n  Predictions:")
    for label, pred in result.get('predictions', {}).items():
        prob = result['probabilities'].get(label, 0)
        conf = result['confidence'].get(label, 0)
        status = "✓" if pred else "✗"
        print(f"    {label}: {status} (prob={prob:.3f}, conf={conf:.3f})")
else:
    print("✗ Failed to load health models")

health_total = time.time() - health_start

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"ReID model test:   {'PASS' if reid_service._loaded else 'FAIL'}")
print(f"Health model test: {'PASS' if health_service._loaded else 'FAIL'}")
print(f"\nTotal time: {reid_total + health_total:.2f}s")
print("\nTo test with your own image:")
print(f"  python {os.path.basename(__file__)} <path_to_image.jpg>")
