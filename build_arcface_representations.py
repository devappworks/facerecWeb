#!/usr/bin/env python3
"""
Manually build ArcFace representations pickle file using DeepFace.represent()
This is the proper way according to DeepFace GitHub issues #528 and #875
"""
import os
import sys
import time
import pickle
from tqdm import tqdm
from deepface import DeepFace

def build_representations():
    """Build ArcFace representations and save to pickle file"""

    # Configuration
    db_path = "storage/recognized_faces_prod/serbia"
    model_name = "ArcFace"
    detector_backend = "retinaface"

    print("="*80)
    print("Building ArcFace Representations Pickle File (Manual Method)")
    print("="*80)
    print(f"Database path: {db_path}")
    print(f"Model: {model_name}")
    print(f"Detector: {detector_backend}")
    print()

    # Check if database exists
    if not os.path.exists(db_path):
        print(f"ERROR: Database path does not exist: {db_path}")
        sys.exit(1)

    # Find all image files
    print("Step 1: Scanning database for images...")
    image_files = []
    for root, dirs, files in os.walk(db_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, f))

    print(f"Found {len(image_files)} images")
    print()

    # Build model first
    print("Step 2: Building ArcFace model...")
    start_model = time.time()
    model = DeepFace.build_model(model_name)
    model_time = time.time() - start_model
    print(f"✓ Model built in {model_time:.2f}s")
    print()

    # Generate representations for all images
    print("Step 3: Generating representations using DeepFace.represent()...")
    print("This will take some time...")
    print()

    representations = []
    failed_images = []
    start_repr = time.time()

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Use DeepFace.represent() to get embeddings
            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False,
                align=True
            )

            # Store representation with image path
            if embedding_objs and len(embedding_objs) > 0:
                # Take the first face if multiple detected
                embedding = embedding_objs[0]["embedding"]
                representations.append({
                    "identity": img_path,
                    "embedding": embedding
                })
        except Exception as e:
            failed_images.append((img_path, str(e)))

    repr_time = time.time() - start_repr
    print()
    print(f"✓ Generated {len(representations)} representations in {repr_time:.2f}s")
    print(f"✗ Failed: {len(failed_images)} images")
    print()

    # Save to pickle file
    print("Step 4: Saving representations to pickle file...")
    pickle_filename = f"representations_{model_name.lower()}.pkl"
    pickle_path = os.path.join(db_path, pickle_filename)

    with open(pickle_path, 'wb') as f:
        pickle.dump(representations, f)

    file_size = os.path.getsize(pickle_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"✓ Pickle file saved: {pickle_path}")
    print(f"✓ Size: {file_size} bytes ({file_size_mb:.2f} MB)")
    print()

    # Also check for the alternate naming convention DeepFace might use
    alt_pickle_name = f"ds_model_{model_name.lower()}_detector_{detector_backend}_aligned_normalization_base_expand_0.pkl"
    alt_pickle_path = os.path.join(db_path, alt_pickle_name)

    # Copy to alternate name as well
    import shutil
    shutil.copy(pickle_path, alt_pickle_path)
    print(f"✓ Also saved as: {alt_pickle_name}")
    print()

    if failed_images:
        print("Failed images:")
        for img, error in failed_images[:10]:  # Show first 10
            print(f"  - {img}: {error}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
        print()

    print("="*80)
    print(f"Total time: {time.time() - start_model:.2f}s")
    print(f"Success rate: {len(representations)}/{len(image_files)} ({100*len(representations)/len(image_files):.1f}%)")
    print("="*80)

if __name__ == "__main__":
    build_representations()
