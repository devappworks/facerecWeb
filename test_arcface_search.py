#!/usr/bin/env python3
"""Test ArcFace face search using the newly created pickle file"""

import sys
import time
from deepface import DeepFace

# Configuration
test_image = "/home/photolytics/licensed-image.jpg"
db_path = "storage/recognized_faces_prod/serbia"
model_name = "ArcFace"
detector_backend = "retinaface"
distance_metric = "cosine"
threshold = 0.50

print("=" * 80)
print("ArcFace Face Search Test")
print("=" * 80)
print(f"Test Image: {test_image}")
print(f"Database: {db_path}")
print(f"Model: {model_name}")
print(f"Detector: {detector_backend}")
print(f"Distance Metric: {distance_metric}")
print(f"Threshold: {threshold}")
print("=" * 80)

try:
    # Run the search
    print("\nRunning face search...")
    start_time = time.time()

    results = DeepFace.find(
        img_path=test_image,
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=True,
        threshold=threshold,
        silent=False
    )

    elapsed = time.time() - start_time

    print(f"\n✓ Search completed in {elapsed:.2f}s")
    print("=" * 80)

    # Display results
    if isinstance(results, list) and len(results) > 0:
        for idx, df in enumerate(results):
            print(f"\nFace #{idx + 1} - Found {len(df)} match(es):")
            print("-" * 80)

            if len(df) > 0:
                # Show top 10 matches
                top_matches = df.head(10)
                for i, row in top_matches.iterrows():
                    identity = row['identity']
                    distance = row['distance']
                    threshold_val = row.get('threshold', threshold)

                    # Extract just the filename
                    import os
                    filename = os.path.basename(identity)

                    print(f"  {i+1}. {filename}")
                    print(f"     Distance: {distance:.4f} (threshold: {threshold_val:.4f})")
                    print(f"     Match: {'✓ YES' if distance < threshold_val else '✗ NO'}")
                    print()
            else:
                print("  No matches found above threshold.")
    else:
        print("\nNo faces detected in the test image or no matches found.")

    print("=" * 80)

except Exception as e:
    print(f"\n✗ Error during search: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
