#!/usr/bin/env python3
"""Simple Modal health test"""
import modal

APP_NAME = "facereco-gpu"
CLASS_NAME = "FaceRecognitionGPU"

print("Testing Modal GPU service...")
print("-" * 60)

try:
    ServiceClass = modal.Cls.from_name(APP_NAME, CLASS_NAME)
    service = ServiceClass()
    print(f"Connected to {APP_NAME}/{CLASS_NAME}")

    health = service.health_check.remote()
    print(f"\nHealth Check Result:")
    print(f"  Status: {health.get('status')}")
    print(f"  GPU Available: {health.get('gpu_available')}")
    print(f"  GPU Name: {health.get('gpu_name')}")

    if health.get('status') == 'healthy':
        print("\n✅ Modal GPU service is healthy!")
    else:
        print(f"\n❌ Health check failed: {health}")

except Exception as e:
    print(f"\n❌ Error: {e}")
