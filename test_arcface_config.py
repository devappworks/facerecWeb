#!/usr/bin/env python3
"""
Quick verification test for ArcFace configuration
Tests that the new profile is properly configured
"""

import sys
sys.path.insert(0, '/home/user/facerecWeb')

try:
    from app.config.recognition_profiles import ProfileManager

    print("="*60)
    print("ArcFace Profile Configuration Test")
    print("="*60)

    # Test 1: List available profiles
    print("\n1. Available Profiles:")
    profiles = ProfileManager.list_profiles()
    for profile in profiles:
        status = "✓ PRODUCTION" if profile.get("is_production") else "  TEST"
        print(f"   {status} - {profile['name']}: {profile['description']}")

    # Test 2: Check ArcFace profile exists
    print("\n2. ArcFace Profile Check:")
    try:
        arcface_profile = ProfileManager.get_profile("arcface")
        print("   ✓ ArcFace profile found")

        # Test 3: Get ArcFace configuration
        print("\n3. ArcFace Configuration:")
        config = arcface_profile.get_config()

        print(f"   Model: {config.get('model_name')}")
        print(f"   Detector: {config.get('detector_backend')}")
        print(f"   Distance Metric: {config.get('distance_metric')}")
        print(f"   Recognition Threshold: {config.get('recognition_threshold')}")
        print(f"   Detection Confidence: {config.get('detection_confidence_threshold')}")
        print(f"   Batched Mode: {config.get('batched')}")
        print(f"   Profile Version: {config.get('profile_version')}")
        print(f"   Created: {config.get('created')}")
        print(f"   Is Production: {config.get('is_production')}")
        print(f"   Is Test: {config.get('is_test')}")

        # Test 4: Verify critical settings
        print("\n4. Configuration Validation:")
        checks = [
            (config.get('model_name') == 'ArcFace', "Model is ArcFace"),
            (config.get('detector_backend') == 'retinaface', "Using RetinaFace detector"),
            (config.get('recognition_threshold') == 0.50, "Correct threshold (0.50)"),
            (config.get('is_test') == True, "Marked as test profile"),
            (config.get('batched') == True, "Batched mode enabled"),
        ]

        all_passed = True
        for passed, description in checks:
            status = "✓" if passed else "✗"
            print(f"   {status} {description}")
            if not passed:
                all_passed = False

        # Test 5: Check old profile removed
        print("\n5. Old Profile Cleanup Check:")
        try:
            ProfileManager.get_profile("improved")
            print("   ✗ WARNING: 'improved' (Facenet512) profile still exists")
            all_passed = False
        except ValueError:
            print("   ✓ 'improved' (Facenet512) profile removed as expected")

        # Final result
        print("\n" + "="*60)
        if all_passed:
            print("✓ ALL TESTS PASSED - ArcFace configuration is correct!")
            print("="*60)
            print("\nNext steps:")
            print("1. Deploy to testing environment")
            print("2. Run: curl http://localhost:5000/api/test/health")
            print("3. Test A/B endpoint: POST /api/test/recognize")
            sys.exit(0)
        else:
            print("✗ SOME TESTS FAILED - Review configuration")
            print("="*60)
            sys.exit(1)

    except ValueError as e:
        print(f"   ✗ ArcFace profile not found: {e}")
        sys.exit(1)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
