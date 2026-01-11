# Batch Photo Recognition with Detailed Logging

This documentation describes the new batch photo recognition system that allows you to upload multiple photos at once and receive detailed recognition results with top 3 matches for each photo.

## Features

- ✅ **Batch Upload**: Upload up to 100 photos in a single request
- ✅ **Detailed Logging**: Each photo recognition is logged with timestamp, filename, and results
- ✅ **Top 3 Matches**: Get the top 3 recognition matches with confidence scores for each photo
- ✅ **CSV Export**: Export batch results to CSV format for easy analysis
- ✅ **Batch Management**: List, view, and delete batch logs
- ✅ **Summary Statistics**: Get recognition rate, unique persons, and average confidence per batch

## API Endpoints

### 1. Upload Batch Photos for Recognition

**Endpoint:** `POST /api/batch-photos/recognize`

**Description:** Upload multiple photos for face recognition. Each photo will be processed and logged with top 3 matches.

**Headers:**
```
Authorization: YOUR_AUTH_TOKEN
```

**Form Data:**
```
photos[]: file1.jpg
photos[]: file2.jpg
photos[]: file3.jpg
... (up to 100 photos)
```

**Example using cURL:**
```bash
curl -X POST \
  -H "Authorization: YOUR_TOKEN" \
  -F "photos[]=@photo1.jpg" \
  -F "photos[]=@photo2.jpg" \
  -F "photos[]=@photo3.jpg" \
  https://facerecognition.mpanel.app/api/batch-photos/recognize
```

**Example using Python:**
```python
import requests

url = "https://facerecognition.mpanel.app/api/batch-photos/recognize"
headers = {
    "Authorization": "YOUR_TOKEN"
}

files = [
    ('photos[]', open('photo1.jpg', 'rb')),
    ('photos[]', open('photo2.jpg', 'rb')),
    ('photos[]', open('photo3.jpg', 'rb'))
]

response = requests.post(url, headers=headers, files=files)
print(response.json())
```

**Response:**
```json
{
  "success": true,
  "batch_id": "20231215_143022_1234",
  "total_uploaded": 3,
  "total_processed": 3,
  "summary": {
    "total_photos": 3,
    "recognized_count": 2,
    "unrecognized_count": 1,
    "recognition_rate": 66.67,
    "unique_persons": 2,
    "persons_list": ["John Doe", "Jane Smith"],
    "avg_confidence": 78.5
  },
  "message": "Processed 3 photos. Check batch log for details.",
  "log_url": "/api/batch-photos/logs/20231215_143022_1234"
}
```

---

### 2. Get Batch Log Details

**Endpoint:** `GET /api/batch-photos/logs/{batch_id}`

**Description:** Retrieve detailed log for a specific batch with top 3 matches for each photo.

**Headers:**
```
Authorization: YOUR_AUTH_TOKEN
```

**Example:**
```bash
curl -H "Authorization: YOUR_TOKEN" \
  https://facerecognition.mpanel.app/api/batch-photos/logs/20231215_143022_1234
```

**Response:**
```json
{
  "success": true,
  "batch_id": "20231215_143022_1234",
  "total_photos": 3,
  "summary": {
    "total_photos": 3,
    "recognized_count": 2,
    "unrecognized_count": 1,
    "recognition_rate": 66.67,
    "unique_persons": 2,
    "persons_list": ["John Doe", "Jane Smith"],
    "avg_confidence": 78.5
  },
  "entries": [
    {
      "timestamp": "2023-12-15T14:30:22.123456",
      "batch_id": "20231215_143022_1234",
      "filename": "photo1.jpg",
      "domain": "example.com",
      "recognized": true,
      "primary_result": {
        "person": "John Doe",
        "confidence": 85.5
      },
      "top_3_matches": [
        {
          "rank": 1,
          "person": "John Doe",
          "confidence": 85.5
        },
        {
          "rank": 2,
          "person": "John Smith",
          "confidence": 72.3
        },
        {
          "rank": 3,
          "person": "Johnny Depp",
          "confidence": 65.1
        }
      ],
      "metadata": {
        "face_count": 1,
        "processing_time_ms": 245,
        "has_multiple_faces": false
      }
    },
    {
      "timestamp": "2023-12-15T14:30:23.456789",
      "batch_id": "20231215_143022_1234",
      "filename": "photo2.jpg",
      "domain": "example.com",
      "recognized": true,
      "primary_result": {
        "person": "Jane Smith",
        "confidence": 71.5
      },
      "top_3_matches": [
        {
          "rank": 1,
          "person": "Jane Smith",
          "confidence": 71.5
        },
        {
          "rank": 2,
          "person": "Janet Jackson",
          "confidence": 68.2
        },
        {
          "rank": 3,
          "person": "Jennifer Aniston",
          "confidence": 62.7
        }
      ],
      "metadata": {
        "face_count": 1,
        "processing_time_ms": 198
      }
    },
    {
      "timestamp": "2023-12-15T14:30:24.789012",
      "batch_id": "20231215_143022_1234",
      "filename": "photo3.jpg",
      "domain": "example.com",
      "recognized": false,
      "primary_result": {
        "person": "Unknown",
        "confidence": null
      },
      "top_3_matches": [],
      "metadata": {
        "face_count": 0,
        "processing_time_ms": 156
      }
    }
  ]
}
```

---

### 3. Export Batch to CSV

**Endpoint:** `GET /api/batch-photos/logs/{batch_id}/export`

**Description:** Download batch log as CSV file with all details including top 3 matches.

**Headers:**
```
Authorization: YOUR_AUTH_TOKEN
```

**Example:**
```bash
curl -H "Authorization: YOUR_TOKEN" \
  -o batch_results.csv \
  https://facerecognition.mpanel.app/api/batch-photos/logs/20231215_143022_1234/export
```

**CSV Format:**

| Timestamp | Filename | Domain | Recognized | Primary Person | Primary Confidence (%) | Match #1 Person | Match #1 Confidence (%) | Match #2 Person | Match #2 Confidence (%) | Match #3 Person | Match #3 Confidence (%) | Face Count | Processing Time (ms) |
|-----------|----------|--------|------------|----------------|------------------------|-----------------|-------------------------|-----------------|-------------------------|-----------------|-------------------------|------------|----------------------|
| 2023-12-15T14:30:22.123456 | photo1.jpg | example.com | Yes | John Doe | 85.5 | John Doe | 85.5 | John Smith | 72.3 | Johnny Depp | 65.1 | 1 | 245 |
| 2023-12-15T14:30:23.456789 | photo2.jpg | example.com | Yes | Jane Smith | 71.5 | Jane Smith | 71.5 | Janet Jackson | 68.2 | Jennifer Aniston | 62.7 | 1 | 198 |
| 2023-12-15T14:30:24.789012 | photo3.jpg | example.com | No | Unknown | | | | | | | | 0 | 156 |

---

### 4. List All Batches

**Endpoint:** `GET /api/batch-photos/logs`

**Description:** List all batch logs, sorted by date (newest first).

**Headers:**
```
Authorization: YOUR_AUTH_TOKEN
```

**Query Parameters:**
- `limit` (optional): Maximum number of batches to return (default: 50)

**Example:**
```bash
curl -H "Authorization: YOUR_TOKEN" \
  "https://facerecognition.mpanel.app/api/batch-photos/logs?limit=10"
```

**Response:**
```json
{
  "success": true,
  "total": 10,
  "batches": [
    {
      "batch_id": "20231215_143022_1234",
      "total_photos": 5,
      "summary": {
        "total_photos": 5,
        "recognized_count": 4,
        "unrecognized_count": 1,
        "recognition_rate": 80.0,
        "unique_persons": 3,
        "persons_list": ["John Doe", "Jane Smith", "Bob Johnson"],
        "avg_confidence": 78.5
      },
      "created_at": "2023-12-15T14:30:22.123456"
    },
    ...
  ]
}
```

---

### 5. Delete Batch Log

**Endpoint:** `DELETE /api/batch-photos/logs/{batch_id}`

**Description:** Delete a batch log (both JSON and CSV files).

**Headers:**
```
Authorization: YOUR_AUTH_TOKEN
```

**Example:**
```bash
curl -X DELETE \
  -H "Authorization: YOUR_TOKEN" \
  https://facerecognition.mpanel.app/api/batch-photos/logs/20231215_143022_1234
```

**Response:**
```json
{
  "success": true,
  "message": "Batch 20231215_143022_1234 deleted successfully"
}
```

---

### 6. Health Check

**Endpoint:** `GET /api/batch-photos/health`

**Description:** Check if the batch photo recognition service is running.

**Example:**
```bash
curl https://facerecognition.mpanel.app/api/batch-photos/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "batch_photo_recognition",
  "version": "v1",
  "log_directory_exists": true,
  "endpoints": [
    "POST /api/batch-photos/recognize",
    "GET /api/batch-photos/logs/<batch_id>",
    "GET /api/batch-photos/logs/<batch_id>/export",
    "GET /api/batch-photos/logs",
    "DELETE /api/batch-photos/logs/<batch_id>",
    "GET /api/batch-photos/health"
  ]
}
```

---

## Integration Examples

### HTML Form

```html
<!DOCTYPE html>
<html>
<head>
    <title>Batch Photo Recognition</title>
</head>
<body>
    <h1>Upload Photos for Recognition</h1>

    <form id="uploadForm">
        <input type="file" name="photos[]" multiple accept="image/*" id="photoInput">
        <br><br>
        <button type="submit">Upload & Recognize</button>
    </form>

    <div id="results"></div>

    <script>
        const form = document.getElementById('uploadForm');
        const resultsDiv = document.getElementById('results');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData();
            const files = document.getElementById('photoInput').files;

            for (let i = 0; i < files.length; i++) {
                formData.append('photos[]', files[i]);
            }

            resultsDiv.innerHTML = 'Processing...';

            try {
                const response = await fetch('https://facerecognition.mpanel.app/api/batch-photos/recognize', {
                    method: 'POST',
                    headers: {
                        'Authorization': 'YOUR_TOKEN'
                    },
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    resultsDiv.innerHTML = `
                        <h2>Results</h2>
                        <p>Batch ID: ${data.batch_id}</p>
                        <p>Processed: ${data.total_processed} photos</p>
                        <p>Recognized: ${data.summary.recognized_count}</p>
                        <p>Recognition Rate: ${data.summary.recognition_rate}%</p>
                        <p>Unique Persons: ${data.summary.persons_list.join(', ')}</p>
                        <p><a href="${data.log_url}" target="_blank">View Detailed Log</a></p>
                    `;
                } else {
                    resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        });
    </script>
</body>
</html>
```

### Python Script

```python
import requests
import os
from pathlib import Path

def batch_recognize_photos(photo_dir, token):
    """
    Upload all photos in a directory for batch recognition.

    Args:
        photo_dir: Path to directory containing photos
        token: Authentication token

    Returns:
        dict: Batch recognition results
    """
    url = "https://facerecognition.mpanel.app/api/batch-photos/recognize"
    headers = {"Authorization": token}

    # Get all image files
    photo_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        photo_files.extend(Path(photo_dir).glob(ext))

    if not photo_files:
        print(f"No photos found in {photo_dir}")
        return None

    print(f"Found {len(photo_files)} photos")

    # Prepare files for upload
    files = [('photos[]', open(f, 'rb')) for f in photo_files]

    try:
        # Upload and process
        print("Uploading photos...")
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()

        result = response.json()

        if result['success']:
            print(f"\n✓ Batch processing complete!")
            print(f"  Batch ID: {result['batch_id']}")
            print(f"  Photos processed: {result['total_processed']}")
            print(f"  Recognition rate: {result['summary']['recognition_rate']}%")
            print(f"  Unique persons: {', '.join(result['summary']['persons_list'])}")
            print(f"\nView detailed log at: {result['log_url']}")

            return result
        else:
            print(f"Error: {result.get('error')}")
            return None

    finally:
        # Close all file handles
        for _, file_obj in files:
            file_obj.close()

def download_batch_csv(batch_id, token, output_file="batch_results.csv"):
    """Download batch results as CSV"""
    url = f"https://facerecognition.mpanel.app/api/batch-photos/logs/{batch_id}/export"
    headers = {"Authorization": token}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    with open(output_file, 'wb') as f:
        f.write(response.content)

    print(f"CSV downloaded to: {output_file}")

# Example usage
if __name__ == "__main__":
    TOKEN = "your_auth_token_here"
    PHOTO_DIR = "/path/to/your/photos"

    # Process batch
    result = batch_recognize_photos(PHOTO_DIR, TOKEN)

    # Download CSV
    if result and result['success']:
        download_batch_csv(result['batch_id'], TOKEN)
```

---

## Log Storage

All batch logs are stored in:
```
storage/logs/batch_recognition/
├── 20231215_143022_1234.json  # Detailed JSON log
├── 20231215_143022_1234.csv   # CSV export (generated on demand)
├── 20231215_150530_5678.json
└── ...
```

---

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)
- AVIF (.avif)

---

## Limits

- **Maximum batch size**: 100 photos per request
- **Maximum file size**: 10 MB per photo (inherited from Flask config)
- **Supported formats**: See above

---

## Error Handling

### Common Errors

1. **401 Unauthorized**
   - Cause: Missing or invalid auth token
   - Solution: Ensure you're sending the `Authorization` header with a valid token

2. **400 Bad Request - No photos uploaded**
   - Cause: No files in request
   - Solution: Ensure you're sending files with key `photos[]`

3. **400 Bad Request - Batch size exceeds maximum**
   - Cause: More than 100 photos in request
   - Solution: Split your batch into smaller chunks

4. **404 Not Found - Batch not found**
   - Cause: Batch ID doesn't exist
   - Solution: Verify the batch_id is correct

5. **Unsupported format error**
   - Cause: File format not in allowed list
   - Solution: Convert images to supported formats (JPEG, PNG, etc.)

---

## Best Practices

1. **Batch Size**: Upload 20-50 photos per batch for optimal performance
2. **File Names**: Use descriptive filenames to easily identify photos in logs
3. **CSV Export**: Export to CSV for easy analysis in Excel or Google Sheets
4. **Log Management**: Regularly delete old batch logs to save storage space
5. **Error Handling**: Always check the `success` field in responses before accessing data

---

## Support

For issues or questions, please contact the development team or open an issue in the repository.
