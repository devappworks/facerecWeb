#!/usr/bin/env python3
"""Test Modal embedding extraction"""
import modal
import base64
import os

APP_NAME = "facereco-gpu"
CLASS_NAME = "FaceRecognitionGPU"

# Test with a real image from the database
TEST_IMAGE = "/home/facereco/facerecWeb/storage/recognized_faces_prod/serbia/Aca_Lukas_2025-05-29_15000311.png"

print("Testing Modal GPU embedding extraction...")
print("-" * 60)

try:
    # Read test image
    with open(TEST_IMAGE, 'rb') as f:
        img_bytes = f.read()

    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    print(f"Loaded test image: {os.path.basename(TEST_IMAGE)}")
    print(f"Image size: {len(img_bytes)} bytes")

    # Connect to Modal
    ServiceClass = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    service = ServiceClass()
    print(f"Connected to Modal service")

    # Extract embeddings
    print("\nExtracting embeddings (this may take a moment on cold start)...")
    result = service.extract_embeddings_batch.remote(
        frames_b64=[img_b64],
        model_name="ArcFace"
    )

    print(f"\nResult:")
    print(f"  Success: {result.get('success')}")
    print(f"  Frames processed: {result.get('frames_processed')}")

    if result.get('results'):
        frame_result = result['results'][0]
        print(f"\nFrame 0 result:")
        print(f"  Success: {frame_result.get('success')}")
        print(f"  Faces detected: {frame_result.get('faces_detected')}")

        if frame_result.get('faces'):
            face = frame_result['faces'][0]
            emb = face.get('embedding', [])
            print(f"  Embedding length: {len(emb)}")
            print(f"  Embedding sample: {emb[:5]}...")
            print(f"  Facial area: {face.get('facial_area')}")

    print("\n✅ Embedding extraction test completed!")

except FileNotFoundError:
    print(f"❌ Test image not found: {TEST_IMAGE}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
