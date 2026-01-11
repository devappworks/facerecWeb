#!/usr/bin/env python3
"""
Test script for batch photo recognition with logging.

This script demonstrates how to use the batch photo recognition API
and retrieve detailed logs with top 3 matches.
"""

import requests
import json
import sys
import os

# Configuration
API_BASE = "https://photolytics.mpanel.app/api/batch-photos"
# You need to provide a valid auth token
# Get it from your login response or database
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")

if not AUTH_TOKEN:
    print("ERROR: Please set AUTH_TOKEN environment variable")
    print("Example: export AUTH_TOKEN='your_token_here'")
    sys.exit(1)


def test_health():
    """Test the health endpoint"""
    print("\n" + "="*60)
    print("TESTING HEALTH ENDPOINT")
    print("="*60)

    url = f"{API_BASE}/health"
    response = requests.get(url)

    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_batch_recognition(photo_paths):
    """Test batch photo recognition"""
    print("\n" + "="*60)
    print("TESTING BATCH RECOGNITION")
    print("="*60)

    url = f"{API_BASE}/recognize"
    headers = {"Authorization": AUTH_TOKEN}

    # Prepare files
    files = []
    for path in photo_paths:
        if not os.path.exists(path):
            print(f"WARNING: File not found: {path}")
            continue

        filename = os.path.basename(path)
        files.append(('photos[]', (filename, open(path, 'rb'), 'image/jpeg')))

    if not files:
        print("ERROR: No valid photos to upload")
        return None

    print(f"Uploading {len(files)} photos...")

    try:
        response = requests.post(url, headers=headers, files=files)

        print(f"Status Code: {response.status_code}")

        # Close file handles
        for _, (_, file_obj, _) in files:
            file_obj.close()

        if response.status_code == 200:
            data = response.json()
            print(f"Response:\n{json.dumps(data, indent=2)}")

            if data.get('success'):
                print(f"\n✓ SUCCESS!")
                print(f"  Batch ID: {data['batch_id']}")
                print(f"  Photos processed: {data['total_processed']}")
                print(f"  Recognition rate: {data['summary']['recognition_rate']}%")
                print(f"  Unique persons: {', '.join(data['summary']['persons_list'])}")

                return data['batch_id']
        else:
            print(f"ERROR: {response.text}")
            return None

    except Exception as e:
        print(f"ERROR: {str(e)}")
        # Close file handles in case of error
        for _, (_, file_obj, _) in files:
            try:
                file_obj.close()
            except:
                pass
        return None


def test_get_log(batch_id):
    """Test retrieving batch log"""
    print("\n" + "="*60)
    print("TESTING GET BATCH LOG")
    print("="*60)

    url = f"{API_BASE}/logs/{batch_id}"
    headers = {"Authorization": AUTH_TOKEN}

    response = requests.get(url, headers=headers)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nBatch Summary:")
        print(f"  Total photos: {data['total_photos']}")
        print(f"  Recognized: {data['summary']['recognized_count']}")
        print(f"  Recognition rate: {data['summary']['recognition_rate']}%")

        print(f"\nDetailed Results:")
        for entry in data['entries']:
            print(f"\n  📷 {entry['filename']}")
            print(f"     Recognized: {'✓ Yes' if entry['recognized'] else '✗ No'}")

            if entry['recognized']:
                print(f"     Primary: {entry['primary_result']['person']} ({entry['primary_result']['confidence']}%)")

                if entry.get('top_3_matches'):
                    print(f"     Top 3 matches:")
                    for match in entry['top_3_matches']:
                        print(f"       {match['rank']}. {match['person']} - {match['confidence']}%")

            if entry.get('metadata'):
                print(f"     Metadata: {entry['metadata']}")

    else:
        print(f"ERROR: {response.text}")


def test_export_csv(batch_id):
    """Test exporting batch to CSV"""
    print("\n" + "="*60)
    print("TESTING CSV EXPORT")
    print("="*60)

    url = f"{API_BASE}/logs/{batch_id}/export"
    headers = {"Authorization": AUTH_TOKEN}

    response = requests.get(url, headers=headers)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        output_file = f"batch_{batch_id}.csv"
        with open(output_file, 'wb') as f:
            f.write(response.content)

        print(f"✓ CSV exported to: {output_file}")
        print(f"  File size: {len(response.content)} bytes")

        # Show first few lines
        with open(output_file, 'r') as f:
            lines = f.readlines()
            print(f"\nFirst {min(5, len(lines))} lines:")
            for line in lines[:5]:
                print(f"  {line.strip()}")

    else:
        print(f"ERROR: {response.text}")


def test_list_batches():
    """Test listing all batches"""
    print("\n" + "="*60)
    print("TESTING LIST BATCHES")
    print("="*60)

    url = f"{API_BASE}/logs?limit=5"
    headers = {"Authorization": AUTH_TOKEN}

    response = requests.get(url, headers=headers)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nTotal batches: {data['total']}")

        for batch in data['batches']:
            print(f"\n  📦 Batch: {batch['batch_id']}")
            print(f"     Photos: {batch['total_photos']}")
            print(f"     Recognition rate: {batch['summary']['recognition_rate']}%")
            print(f"     Created: {batch['created_at']}")

    else:
        print(f"ERROR: {response.text}")


if __name__ == "__main__":
    # Test photos (update paths as needed)
    test_photos = [
        "/root/photoanalytics/jokic2.jpg",
        "/root/photoanalytics/Jokic.jpg",
        "/root/photoanalytics/joki3.jpg"
    ]

    print("="*60)
    print("BATCH PHOTO RECOGNITION TEST SUITE")
    print("="*60)

    # 1. Test health
    if not test_health():
        print("\n✗ Health check failed!")
        sys.exit(1)

    # 2. Test batch recognition
    batch_id = test_batch_recognition(test_photos)

    if not batch_id:
        print("\n✗ Batch recognition failed!")
        sys.exit(1)

    # 3. Test getting log
    test_get_log(batch_id)

    # 4. Test CSV export
    test_export_csv(batch_id)

    # 5. Test list batches
    test_list_batches()

    print("\n" + "="*60)
    print("✓ ALL TESTS COMPLETED!")
    print("="*60)
