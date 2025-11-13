# FacerecWeb - Face Recognition API Service

A production-grade face recognition REST API built with Flask and DeepFace, providing high-accuracy face recognition across multiple client domains with robust validation and scalable batch processing.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [How It Works](#how-it-works)
- [Storage Structure](#storage-structure)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

FacerecWeb is an enterprise-level face recognition system that provides:

- **Face Recognition**: Upload images to recognize people from pre-trained databases
- **Face Training**: Upload and manage images to train the recognition system
- **Batch Processing**: Search through large databases (5000+ images) with parallel processing
- **Multi-Domain Support**: Separate face databases for different clients/organizations
- **Email-based Authentication**: Secure token-based authentication mapped to email addresses
- **Quality Validation**: 7+ quality checks to minimize false positives

### Use Cases

- Security and access control systems
- Event photography (automatic face tagging)
- Media monitoring and person identification
- Customer recognition systems
- Attendance tracking

---

## ✨ Key Features

### Face Recognition
- **High Accuracy**: VGG-Face model with 99.5%+ confidence threshold
- **Multi-Face Detection**: Detects and validates multiple faces in single image
- **Quality Validation Pipeline**:
  - Blur detection (Laplacian variance ≥ 100)
  - Contrast validation (≥ 25)
  - Brightness validation (30-220 range)
  - Edge density validation (≥ 15)
  - Eye coordinate validation
  - Confidence threshold (≥ 99.5%)

### Batch Processing
- Organize databases into batches of 5000 images
- Parallel processing with ThreadPoolExecutor (up to 3 concurrent batches)
- Automatic batch discovery and management
- Performance optimization for large-scale databases

### Multi-Domain Architecture
- Completely isolated databases per domain/client
- Token-based authentication with domain mapping
- Domain-specific storage and processing

### Background Processing
- Asynchronous image processing
- Non-blocking face extraction and validation
- Automatic training data preparation

---

## 🛠 Technology Stack

### AI/ML Models
- **DeepFace 0.0.95** - Face recognition framework
- **VGG-Face** - Primary recognition model
- **RetinaFace** - Face detection backend (primary)
- **MTCNN** - Alternative face detection
- **TensorFlow 2.14.0** - Deep learning framework
- **Keras 2.14.0** - Neural network API

### Image Processing
- **OpenCV** - Image manipulation, face cropping
- **Pillow (PIL)** - Image resizing, format conversion
- **NumPy 1.24.3** - Numerical operations

### Web Framework
- **Flask 2.0.2+** - REST API framework
- **Flask-CORS** - Cross-origin resource sharing
- **python-dotenv** - Environment management

### Storage & Integration
- **Boto3** - AWS S3/Wasabi cloud storage
- **Pandas** - Data processing
- **OpenPyXL** - Excel operations
- **OpenAI API** - Advanced processing
- **Pusher** - Real-time notifications

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Flask REST API Server                     │
│                  (run.py + app/__init__.py)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ├── CORS Enabled (All Origins)
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
  ┌─────▼──────┐    ┌────────▼────────┐   ┌───────▼────────┐
  │  Routes    │    │  Controllers    │   │   Services     │
  │            │───▶│                 │──▶│                │
  │ - auth     │    │ - recognition   │   │ - recognition  │
  │ - image    │    │ - image         │   │ - batch_recog  │
  │ - batch    │    │ - excel         │   │ - face_proc    │
  │ - excel    │    │ - object_det    │   │ - validation   │
  │ - admin    │    │ - sync          │   │ - image        │
  └────────────┘    └─────────────────┘   └────────────────┘
```

### Component Layers

1. **Routes Layer** (`app/routes/`) - API endpoint definitions
2. **Controllers Layer** (`app/controllers/`) - Request handling and validation
3. **Services Layer** (`app/services/`) - Business logic and AI processing
4. **Models Layer** (`app/models/`) - Data structures and storage

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip
- 2GB+ RAM (4GB+ recommended for batch processing)
- GPU support (optional, improves performance)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd facerecWeb
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create environment file**
```bash
cp .env.example .env
```

4. **Configure environment variables** (see [Configuration](#configuration))

5. **Create storage directories**
```bash
mkdir -p storage/uploads
mkdir -p storage/recognized_faces_prod
mkdir -p storage/recognized_faces_batched
mkdir -p storage/objectDetection
mkdir -p storage/training
mkdir -p storage/transfer_images
mkdir -p storage/excel
```

6. **Run the application**
```bash
python run.py
```

The server will start on `http://localhost:5000` by default.

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# Flask Configuration
SECRET_KEY=your-secret-key-here
DEBUG=True

# Authentication
# Format: {"token1": "domain1", "token2": "domain2"}
CLIENTS_TOKENS={"dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD": "domain1", "K8XZ40eX1WF1v49aukU7t0hF0XO57IdZRTh": "domain2"}

# Email to Domain Mapping
# Format: {"email@example.com": "domain1"} or {"email@example.com": ["domain1", "domain2"]}
CLIENTS_EMAILS={"user@example.com": "domain1", "admin@example.com": "domain2"}

# Storage Configuration
IMAGE_STORAGE_PATH=storage/training/default
EXCEL_FILE_PATH=storage/excel/data.xlsx

# External API Keys (Optional)
SERPAPI_SEARCH_API_KEY=your-serpapi-key
GOOGLE_SEARCH_CX=your-search-cx-id
OPENAI_API_KEY=your-openai-key

# Cloud Storage (Optional)
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
WASABI_BUCKET_NAME=your-bucket-name
```

### Application Settings (config.py)

- **MAX_CONTENT_LENGTH**: 30MB (default)
- **UPLOAD_FOLDER**: `storage/uploads`
- **Batch Size**: 5000 images per batch
- **Max Threads**: 3 concurrent batches
- **Recognition Threshold**: 0.35 (cosine distance)
- **Confidence Threshold**: 0.995 (99.5%)

---

## 📚 API Documentation

### Authentication

All endpoints (except auth endpoints) require an `Authorization` header with a valid token.

```bash
Authorization: your-token-here
```

### Authentication Endpoints

#### Get Token by Email
```http
POST /api/auth/token-by-email
Content-Type: application/json

{
  "email": "user@example.com"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "token": "dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD",
    "email": "user@example.com"
  }
}
```

#### Validate Email
```http
POST /api/auth/validate-email
Content-Type: application/json

{
  "email": "user@example.com"
}
```

#### Health Check
```http
GET /api/auth/health
```

---

### Face Recognition Endpoints

#### Standard Face Recognition
```http
POST /recognize
Authorization: your-token-here
Content-Type: multipart/form-data

image: [image file]
```

**Response:**
```json
{
  "status": "success",
  "message": "Face recognized as: John Doe",
  "person": "John Doe",
  "recognized_persons": [
    {
      "name": "John Doe",
      "face_coordinates": {
        "x_percent": 45.2,
        "y_percent": 23.1,
        "width_percent": 15.3,
        "height_percent": 20.7
      }
    }
  ],
  "best_match": {
    "person_name": "John_Doe",
    "display_name": "John Doe",
    "confidence_metrics": {
      "occurrences": 5,
      "average_distance": 0.2341,
      "min_distance": 0.1876,
      "weighted_score": 0.0945,
      "confidence_percentage": 81.24,
      "distances": [0.1876, 0.2134, 0.2456, 0.2389, 0.2850]
    }
  }
}
```

#### Batch Face Recognition (for large databases)
```http
POST /api/batch/recognize
Authorization: your-token-here
Content-Type: multipart/form-data

image: [image file]
domain: example.com
max_threads: 3 (optional)
```

**Response:** Same as standard recognition + batch processing metrics

```json
{
  "status": "success",
  "message": "Face recognized as: John Doe",
  "person": "John Doe",
  "batch_processing": {
    "total_processing_time": 4.52,
    "batch_summary": {
      "total_batches": 3,
      "processed_batches": 3,
      "failed_batches": 0,
      "total_images_searched": 12450
    }
  }
}
```

#### Batch Statistics
```http
GET /api/batch/stats?domain=example.com
```

#### List Batch Domains
```http
GET /api/batch/domains
```

#### Batch Health Check
```http
GET /api/batch/health
```

---

### Image Upload & Training

#### Upload Training Image
```http
POST /upload-with-domain
Authorization: your-token-here
Content-Type: multipart/form-data

image: [image file]
person: John Doe
created_date: 2025-01-15
```

**Response:**
```json
{
  "status": "processing",
  "message": "Image upload started, processing in background"
}
```

#### Manage Images
```http
POST /manage-image
Authorization: your-token-here
Content-Type: application/json

{
  "filename": "John_Doe_20250115_123456.jpg",
  "action": "delete"
}
```

Or for editing:
```json
{
  "filename": "John_Doe_20250115_123456.jpg",
  "action": "edit",
  "person": "John Smith"
}
```

---

### Synchronization Endpoints

#### Sync Faces
```http
POST /sync-faces
Authorization: your-token-here

# Optional parameters
source_dir: storage/recognized_faces (default)
target_dir: storage/recognized_faces_prod (default)
```

#### Sync from Kylo
```http
POST /sync-kylo
Authorization: your-token-here
```

#### Transfer Images
```http
POST /transfer-images
Authorization: your-token-here
```

---

## 🔄 How It Works

### Face Recognition Workflow

```
┌──────────────────┐
│  Upload Image    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Authentication  │ ◄── Validate token against CLIENTS_TOKENS
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Image Preprocessing │
├──────────────────┤
│ - Resize to 1024x1024 (max) │
│ - EXIF orientation fix      │
│ - Save to temp storage      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Face Detection  │
├──────────────────┤
│ DeepFace.extract_faces()    │
│ - RetinaFace backend        │
│ - Normalize & align         │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│  Face Validation (7 checks)  │
├──────────────────────────────┤
│ 1. Confidence ≥ 99.5%        │
│ 2. Eye coordinates validation│
│ 3. Blur detection ≥ 100      │
│ 4. Contrast ≥ 25             │
│ 5. Brightness 30-220         │
│ 6. Edge density ≥ 15         │
│ 7. Size validation           │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────┐
│  Early Exit?     │ ◄── If no valid faces, stop here
└────────┬─────────┘
         │ Valid faces found
         ▼
┌──────────────────────────────┐
│  Face Recognition            │
├──────────────────────────────┤
│ DeepFace.find()              │
│ - VGG-Face model             │
│ - Cosine distance            │
│ - Threshold: 0.35            │
│ - Search in domain DB        │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Result Analysis             │
├──────────────────────────────┤
│ - Group by coordinates       │
│ - Calculate weighted scores  │
│ - Map to original names      │
│ - Select best match          │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────┐
│  Return Result   │
└──────────────────┘
```

### Training Workflow

```
┌──────────────────┐
│  Upload Training │
│     Image        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Save to Uploads │
│  {domain}/{person}_{date}_{timestamp}.jpg
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│  Background Processing       │
├──────────────────────────────┤
│ 1. Extract face              │
│ 2. Validate quality          │
│    - Size ≥ 70x70           │
│    - Blur check             │
│    - Single face only       │
│ 3. Crop with 20% margin     │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────┐
│  Save to Prod DB │
│  storage/recognized_faces_prod/{domain}/
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Rebuild Pickle  │ ◄── DeepFace auto-generates representations
└──────────────────┘
```

### Batch Processing Workflow (for large databases)

```
┌──────────────────┐
│  Batch Structure │
├──────────────────┤
│ batch_0001/      │ ◄── 5000 images + representations_vgg_face.pkl
│ batch_0002/      │ ◄── 5000 images + representations_vgg_face.pkl
│ batch_0003/      │ ◄── 5000 images + representations_vgg_face.pkl
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│  ThreadPoolExecutor (max 3)  │
├──────────────────────────────┤
│ Thread 1 ──▶ batch_0001     │
│ Thread 2 ──▶ batch_0002     │
│ Thread 3 ──▶ batch_0003     │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────┐
│  Combine Results │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Analyze & Return│
└──────────────────┘
```

---

## 📁 Storage Structure

```
storage/
├── uploads/
│   └── {domain}/                    # Temporary uploaded images
│       └── {person}_{date}_{timestamp}.jpg
│
├── recognized_faces/                # Old training data (legacy)
│
├── recognized_faces_prod/           # Production face database
│   └── {domain}/
│       └── {person}_{date}.jpg
│
├── recognized_faces_batched/        # Batch-organized databases
│   └── {domain}/
│       ├── batch_0001/
│       │   ├── representations_vgg_face.pkl
│       │   └── [5000 images]
│       ├── batch_0002/
│       ├── batch_metadata.json
│       └── ...
│
├── objectDetection/                 # Object detection images
│
├── training/                        # Raw training images
│   └── {domain}/
│
├── trainingPass{Domain}/            # Validated training images
│
├── transfer_images/                 # Pending transfer
│
└── excel/                          # Excel data files
    └── data.xlsx
```

---

## ⚡ Performance Optimization

### Image Resizing
- All images resized to max 1024x1024 pixels
- Maintains aspect ratio
- EXIF orientation correction
- Reduces processing time by ~60%

### Early Exit Strategy
- Stops processing if no valid faces detected
- Saves computation on invalid inputs
- Implemented after face validation pipeline

### Batch Processing
- 5000 images per batch
- Parallel processing (3 concurrent threads)
- Individual pickle files per batch
- Estimated speedup: 3x for large databases

### Caching
- DeepFace representations cached in pickle files
- Model weights cached after first load
- Avoids redundant computations

### Quality Thresholds (Optimized for performance vs. accuracy)
- **Confidence**: 99.5% (strict)
- **Blur detection**: Laplacian variance ≥ 100
- **Recognition threshold**: 0.35 cosine distance

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "No faces detected"
**Causes:**
- Low image quality
- Face too small in image
- Extreme lighting conditions
- Face not facing camera

**Solutions:**
- Use high-resolution images (min 640x480)
- Ensure face occupies at least 15% of image
- Good lighting conditions
- Face should be clearly visible

#### 2. "Face rejected - low confidence"
**Causes:**
- Detection confidence < 99.5%
- Blurry image
- Partial face occlusion

**Solutions:**
- Use sharper images
- Ensure full face is visible
- Avoid sunglasses, masks, or heavy shadows

#### 3. "No matches found"
**Causes:**
- Person not in training database
- Different appearance (aging, styling)
- Poor image quality

**Solutions:**
- Upload training images for the person
- Use 5-10 training images per person
- Ensure training images are high quality

#### 4. Slow Recognition
**Causes:**
- Large database (1000+ images)
- Running on CPU instead of GPU

**Solutions:**
- Use batch processing (`/api/batch/recognize`)
- Enable GPU acceleration (CUDA)
- Run batch migration for large databases

#### 5. "Batch structure not found"
**Causes:**
- Batch migration not performed
- Database in old format

**Solutions:**
```bash
# Migrate to batch structure
python scripts/batch_migration_command.py --domain your-domain --delete-originals
```

### Logs

Check application logs for detailed error information:
```bash
# View real-time logs
tail -f /var/log/facerecweb/app.log
```

### Health Checks

```bash
# Check authentication service
curl http://localhost:5000/api/auth/health

# Check batch system
curl http://localhost:5000/api/batch/health
```

---

## 📖 Additional Documentation

- [Batch Recognition System](README_BATCH_RECOGNITION.md) - Detailed batch processing guide
- [Authentication System](README_AUTH_ENDPOINT.md) - Email-to-token authentication

---

## 🤝 Contributing

### Development Setup

1. Install dev dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
python -m pytest tests/
```

3. Format code:
```bash
black app/
flake8 app/
```

### Code Structure

- Follow existing patterns in services/controllers/routes
- Add comprehensive logging
- Include error handling
- Update tests for new features
- Document API changes

---

## 📄 License

[Your License Here]

---

## 🙋 Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Email: support@example.com
- Documentation: [docs-url]

---

## 🔄 Version History

### v1.0.0 (Current)
- Face recognition with VGG-Face
- Batch processing system
- Multi-domain support
- Email authentication
- 7-layer quality validation
- Background processing

### Recent Updates
- Early exit optimization for invalid faces
- DeepFace 0.0.95 update
- Enhanced face validation
- Batched mode enabled by default
