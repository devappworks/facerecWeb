#!/usr/bin/env python3
"""
Test script to check actual distances for multi-person photo.
Shows what threshold would be needed to recognize the people.
"""
import sys
import os
import json
import subprocess
from deepface import DeepFace

# Add app to path
sys.path.insert(0, '/root/facerecognition-backend')

# Load image
image_path = '/root/photoanalytics/borut pahor8.jpg'

print(f"Testing multi-person recognition on: {image_path}\n")

# Extract embeddings
print("1. Extracting embeddings with DeepFace...")
embedding_results = DeepFace.represent(
    img_path=image_path,
    model_name="ArcFace",
    detector_backend="retinaface",
    enforce_detection=False,
    align=True,
    normalization="base"
)

print(f"   Found {len(embedding_results)} faces\n")

# Test each face with different thresholds
worker_path = '/root/facerecognition-backend/scripts/pgvector_search_worker.py'
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]

for face_idx, face_data in enumerate(embedding_results):
    print(f"=" * 60)
    print(f"Face {face_idx + 1}:")
    print(f"=" * 60)

    embedding = face_data["embedding"]
    confidence = face_data.get('face_confidence', 1.0)
    print(f"Face detection confidence: {confidence:.3f}")

    # Test with increasing thresholds
    for threshold in thresholds:
        search_request = {
            'embedding': embedding,
            'domain': 'slovenia',
            'threshold': threshold,
            'top_k': 5
        }

        result = subprocess.run(
            ['/root/facerecognition-backend/venv/bin/python', worker_path],
            input=json.dumps(search_request),
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            worker_response = json.loads(result.stdout)
            if worker_response['status'] == 'success':
                matches = worker_response['matches']

                if matches:
                    print(f"\n  Threshold {threshold:.2f}: {len(matches)} matches")
                    for i, match in enumerate(matches[:3], 1):
                        print(f"    {i}. {match['name']}: distance={match['distance']:.4f}, confidence={match['confidence']*100:.1f}%")

                    # Stop at first threshold that gives matches
                    if threshold == 0.30:
                        continue
                    else:
                        break

    print()

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print("If no matches found at any threshold, the faces may not be in the database")
print("or the photo quality/angle is too different from training images.")
