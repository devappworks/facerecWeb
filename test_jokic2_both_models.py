#!/usr/bin/env python3
"""Test jokic2.jpg with both VGG-Face and ArcFace models"""

import sys
import time
from deepface import DeepFace

# Configuration
test_image = "/home/photolytics/jokic2.jpg"
db_path = "storage/recognized_faces_prod/serbia"
detector_backend = "retinaface"
distance_metric = "cosine"

print("=" * 80)
print("Testing jokic2.jpg with BOTH models")
print("=" * 80)
print(f"Test Image: {test_image}")
print(f"Database: {db_path}")
print(f"Detector: {detector_backend}")
print("=" * 80)

# Test 1: VGG-Face
print("\n" + "=" * 80)
print("TEST 1: VGG-Face Model")
print("=" * 80)

try:
    start_time = time.time()
    results_vgg = DeepFace.find(
        img_path=test_image,
        db_path=db_path,
        model_name="VGG-Face",
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=True,
        threshold=0.40,
        silent=False
    )
    elapsed_vgg = time.time() - start_time

    print(f"\n✓ VGG-Face search completed in {elapsed_vgg:.2f}s")

    if isinstance(results_vgg, list) and len(results_vgg) > 0:
        for idx, df in enumerate(results_vgg):
            print(f"\nFace #{idx + 1} - Found {len(df)} match(es)")
            if len(df) > 0:
                top_match = df.iloc[0]
                import os
                filename = os.path.basename(top_match['identity'])
                print(f"  Top Match: {filename}")
                print(f"  Distance: {top_match['distance']:.4f}")
            else:
                print("  No matches found")
    else:
        print("No faces detected or no matches found")

except Exception as e:
    print(f"✗ VGG-Face Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: ArcFace
print("\n" + "=" * 80)
print("TEST 2: ArcFace Model")
print("=" * 80)

try:
    start_time = time.time()
    results_arc = DeepFace.find(
        img_path=test_image,
        db_path=db_path,
        model_name="ArcFace",
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=True,
        threshold=0.50,
        silent=False
    )
    elapsed_arc = time.time() - start_time

    print(f"\n✓ ArcFace search completed in {elapsed_arc:.2f}s")

    if isinstance(results_arc, list) and len(results_arc) > 0:
        for idx, df in enumerate(results_arc):
            print(f"\nFace #{idx + 1} - Found {len(df)} match(es)")
            if len(df) > 0:
                top_match = df.iloc[0]
                import os
                filename = os.path.basename(top_match['identity'])
                print(f"  Top Match: {filename}")
                print(f"  Distance: {top_match['distance']:.4f}")
            else:
                print("  No matches found")
    else:
        print("No faces detected or no matches found")

except Exception as e:
    print(f"✗ ArcFace Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"VGG-Face: {elapsed_vgg:.2f}s")
print(f"ArcFace:  {elapsed_arc:.2f}s")
print(f"Speedup:  {((elapsed_vgg - elapsed_arc) / elapsed_vgg * 100):.1f}% {'faster' if elapsed_arc < elapsed_vgg else 'slower'}")
print("=" * 80)
