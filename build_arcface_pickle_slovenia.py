#!/usr/bin/env python3
"""
Script to manually build ArcFace pickle file for Slovenia database
"""
import os
import sys
import time
from deepface import DeepFace

def build_pickle():
    """Build ArcFace pickle file for Slovenia production database"""

    # Configuration
    db_path = "storage/recognized_faces_prod/slovenia"
    model_name = "ArcFace"
    detector_backend = "retinaface"
    distance_metric = "cosine"

    print("="*80)
    print("Building ArcFace Pickle File for Slovenia")
    print("="*80)
    print(f"Database path: {db_path}")
    print(f"Model: {model_name}")
    print(f"Detector: {detector_backend}")
    print(f"Distance metric: {distance_metric}")
    print()

    # Check if database exists
    if not os.path.exists(db_path):
        print(f"ERROR: Database path does not exist: {db_path}")
        sys.exit(1)

    # Count images in database
    image_count = 0
    for root, dirs, files in os.walk(db_path):
        image_count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"Found {image_count} images in database")
    print()

    # Build model first
    print("Step 1: Building ArcFace model...")
    start_model = time.time()
    model = DeepFace.build_model(model_name)
    model_time = time.time() - start_model
    print(f"✓ Model built in {model_time:.2f}s")
    print()

    # Create a dummy image path for testing
    test_image = None
    for root, dirs, files in os.walk(db_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_image = os.path.join(root, f)
                break
        if test_image:
            break

    if not test_image:
        print("ERROR: No test image found in database")
        sys.exit(1)

    print(f"Using test image: {test_image}")
    print()

    # Run DeepFace.find with batched=True to generate pickle
    print("Step 2: Running DeepFace.find to generate pickle file...")
    print("This will take some time as it processes all database images...")
    print()

    start_find = time.time()
    try:
        results = DeepFace.find(
            img_path=test_image,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=False,
            threshold=0.50,
            silent=False,
            batched=True  # This should generate the pickle file
        )
        find_time = time.time() - start_find

        print()
        print(f"✓ DeepFace.find completed in {find_time:.2f}s")
        print(f"✓ Results: {len(results)} face(s) processed")

    except Exception as e:
        find_time = time.time() - start_find
        print()
        print(f"✗ Error after {find_time:.2f}s: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Check if pickle file was created
    print()
    print("Step 3: Checking pickle file...")
    pickle_file = f"{db_path}/ds_model_{model_name.lower()}_detector_{detector_backend}_aligned_normalization_base_expand_0.pkl"

    if os.path.exists(pickle_file):
        size = os.path.getsize(pickle_file)
        size_mb = size / (1024 * 1024)
        print(f"✓ Pickle file exists: {pickle_file}")
        print(f"✓ Size: {size} bytes ({size_mb:.2f} MB)")

        if size < 1000:
            print("⚠ WARNING: Pickle file is very small - may be corrupted!")
        else:
            print("✓ Pickle file appears valid!")
    else:
        print(f"✗ Pickle file not found: {pickle_file}")
        sys.exit(1)

    print()
    print("="*80)
    print(f"Total time: {time.time() - start_model:.2f}s")
    print("="*80)

if __name__ == "__main__":
    build_pickle()
