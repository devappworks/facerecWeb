#!/usr/bin/env python3
"""Test A/B API endpoint directly to see actual error"""

import requests
import sys

# Test with the same image that worked before
image_path = "/home/photolytics/licensed-image.jpg"

try:
    # Get auth token (you'll need to provide this)
    # For now, let's just try to hit the endpoint and see the error

    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'image_id': 'test_001.jpg'}
        headers = {
            'Authorization': 'Bearer your-token-here'  # You'll need actual token
        }

        print("Sending request to API...")
        response = requests.post(
            'http://localhost:5000/api/test/recognize',
            files=files,
            data=data,
            headers=headers,
            timeout=180
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
