"""
Script to prepare test dataset with ground truth labels
"""

import os
import json
import shutil
from datetime import datetime


def create_test_dataset():
    """
    Create organized test dataset with ground truth

    Directory structure:
    storage/test_dataset/
    ├── ground_truth.json
    ├── images/
    │   ├── person1_001.jpg
    │   ├── person1_002.jpg
    │   ├── person2_001.jpg
    │   └── ...
    └── results/
        └── (comparison results will go here)
    """

    test_dataset_path = "storage/test_dataset"
    images_path = os.path.join(test_dataset_path, "images")
    results_path = os.path.join(test_dataset_path, "results")

    # Create directories
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    print("Test dataset structure created at:", test_dataset_path)
    print("")
    print("Next steps:")
    print("1. Add test images to:", images_path)
    print("2. Create ground_truth.json with format:")
    print("""
    {
      "images": [
        {
          "filename": "person1_001.jpg",
          "ground_truth": "John Doe",
          "description": "Frontal view, good lighting",
          "difficulty": "easy"
        },
        {
          "filename": "person2_001.jpg",
          "ground_truth": "Jane Smith",
          "description": "Side profile, dim lighting",
          "difficulty": "hard"
        }
      ]
    }
    """)
    print("3. Run test suite")


def load_ground_truth():
    """Load ground truth from JSON file"""
    ground_truth_file = "storage/test_dataset/ground_truth.json"

    if not os.path.exists(ground_truth_file):
        return {}

    with open(ground_truth_file, 'r') as f:
        data = json.load(f)

    # Create lookup dictionary
    lookup = {}
    for image_data in data.get("images", []):
        filename = image_data["filename"]
        lookup[filename] = {
            "ground_truth": image_data["ground_truth"],
            "description": image_data.get("description", ""),
            "difficulty": image_data.get("difficulty", "medium")
        }

    return lookup


if __name__ == "__main__":
    create_test_dataset()
