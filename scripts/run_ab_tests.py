"""
Automated test runner for A/B testing
"""

import os
import sys
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.test_recognition_service import TestRecognitionService
from app.services.comparison_service import ComparisonService
from scripts.prepare_test_dataset import load_ground_truth


def run_automated_tests():
    """
    Run automated tests on test dataset
    """
    print("="*60)
    print("AUTOMATED A/B TESTING")
    print("="*60)
    print()

    # Load test dataset
    test_images_path = "storage/test_dataset/images"
    ground_truth = load_ground_truth()

    if not os.path.exists(test_images_path):
        print(f"Error: Test images directory not found: {test_images_path}")
        print("Please run: python scripts/prepare_test_dataset.py")
        return

    # Get all test images
    image_files = [f for f in os.listdir(test_images_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"No test images found in: {test_images_path}")
        return

    print(f"Found {len(image_files)} test images")
    print(f"Ground truth available for: {len(ground_truth)} images")
    print()

    # Run tests
    results = []
    for i, filename in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Testing: {filename}")

        image_path = os.path.join(test_images_path, filename)

        # Read image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Get ground truth if available
        gt = ground_truth.get(filename, {}).get("ground_truth")

        # Run comparison
        try:
            result = TestRecognitionService.recognize_face_comparison(
                image_bytes,
                domain="test",  # Use test domain
                image_id=filename,
                ground_truth=gt
            )

            results.append(result)

            # Print summary
            comp = result["comparison"]
            metrics = comp["comparison_metrics"]

            if metrics["both_succeeded"]:
                if metrics["results_match"]:
                    print(f"  Both agree: {result['pipeline_a_result']['person']}")
                else:
                    print(f"  Disagree: A={result['pipeline_a_result']['person']}, "
                          f"B={result['pipeline_b_result']['person']}")
            elif metrics["only_b_succeeded"]:
                print(f"  Only B succeeded: {result['pipeline_b_result']['person']}")
            elif metrics["only_a_succeeded"]:
                print(f"  Only A succeeded: {result['pipeline_a_result']['person']}")
            else:
                print(f"  Both failed")

            if gt:
                acc = metrics.get("accuracy", {})
                if acc:
                    winner = acc.get("winner", "unknown")
                    print(f"  Ground truth: {gt} | Winner: {winner}")

        except Exception as e:
            print(f"  Error: {str(e)}")

        print()
        time.sleep(0.5)  # Brief pause between tests

    # Generate summary
    print()
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)

    summary = ComparisonService.get_comparison_summary()

    print(f"Total tests: {summary['total_comparisons']}")
    print()

    status = summary["status_breakdown"]
    print("Status Breakdown:")
    print(f"  Both succeeded: {status['both_succeeded']['count']} ({status['both_succeeded']['percentage']}%)")
    print(f"  Both failed: {status['both_failed']['count']} ({status['both_failed']['percentage']}%)")
    print(f"  Only A succeeded: {status['only_a_succeeded']['count']} ({status['only_a_succeeded']['percentage']}%)")
    print(f"  Only B succeeded: {status['only_b_succeeded']['count']} ({status['only_b_succeeded']['percentage']}%)")
    print()

    agreement = summary["agreement"]
    print(f"Agreement rate: {agreement['agreement_rate']}%")
    print()

    if summary.get("accuracy"):
        acc = summary["accuracy"]
        print("Accuracy (vs ground truth):")
        print(f"  Pipeline A: {acc['pipeline_a_accuracy']}%")
        print(f"  Pipeline B: {acc['pipeline_b_accuracy']}%")
        print(f"  Improvement: {acc['improvement']:+.1f}%")
        print()

    # Save results
    results_file = "storage/test_dataset/results/automated_test_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            "test_run": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "individual_results": results
        }, f, indent=2)

    print(f"Results saved to: {results_file}")
    print("="*60)


if __name__ == "__main__":
    run_automated_tests()
