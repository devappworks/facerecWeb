#!/usr/bin/env python3
"""Test Modal GPU health check"""
import sys
sys.path.insert(0, '/home/facereco/facerecWeb')

from app.services.modal_service import ModalService

print("Testing Modal GPU service health...")
print("-" * 60)

# Check if Modal is enabled
enabled = ModalService.is_enabled()
print(f"Modal GPU Enabled: {enabled}")

if not enabled:
    print("\nModal GPU is disabled in configuration.")
    sys.exit(0)

# Check if Modal is available
available = ModalService.is_available()
print(f"Modal Service Available: {available}")

if not available:
    print("\nModal service is not available. Check deployment.")
    sys.exit(1)

# Check health
health = ModalService.check_health()
print(f"\nHealth Check Result:")
print(f"  Status: {health.get('status')}")
print(f"  GPU Available: {health.get('gpu_available')}")
print(f"  GPU Name: {health.get('gpu_name')}")

if health.get('status') == 'healthy':
    print("\n✅ Modal GPU service is healthy and ready!")
    sys.exit(0)
else:
    print(f"\n❌ Modal GPU service health check failed: {health}")
    sys.exit(1)
