# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flask-based face recognition web application using DeepFace, TensorFlow, and OpenCV. The system supports multi-tenant architecture with domain-based authentication, batch face recognition, and comprehensive image validation.

## Development Commands

### Running the Application
```bash
# Start development server
python run.py

# Production (via WSGI)
# Server uses wsgi.py which imports the app from run.py
```

### Testing
```bash
# Test authentication endpoint
python test_auth_endpoint.py

# Test face recognition
python scripts/test_recognition.py

# Run A/B tests comparing recognition profiles
python scripts/run_ab_tests.py

# Prepare test dataset for A/B testing
python scripts/prepare_test_dataset.py
```

### Training Data Management
```bash
# Process Excel file with celebrity data
python scripts/excel_checker.py

# Process training images and validate faces
python scripts/training_processor.py

# Post images to API (bulk upload)
python scripts/api_post_client.py
```

### Batch Management
```bash
# Migrate images to batch structure (5000 images per batch)
python scripts/batch_migration_command.py --domain example.com

# Dry run migration
python scripts/batch_migration_command.py --domain example.com --dry-run

# Migrate all domains
python scripts/batch_migration_command.py --all-domains --delete-originals

# View batch info
python scripts/batch_migration_command.py --info --domain example.com
```

### Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Key dependencies: Flask, deepface==0.0.95, tensorflow==2.14.0, opencv-python
```

## Architecture

### Service Layer Pattern
The application follows a layered architecture:
- **Services** (`app/services/`) - Core business logic
- **Controllers** (`app/controllers/`) - Request handling and orchestration
- **Routes** (`app/routes/`) - Flask blueprints for API endpoints

### Key Services

**RecognitionService** (`app/services/recognition_service.py`)
- Core face recognition using DeepFace with VGG-Face model
- Face validation with 0.995 confidence threshold
- Quality checks: blur detection, contrast, brightness
- Early exit optimization (returns immediately when face is recognized)

**BatchRecognitionService** (`app/services/batch_recognition_service.py`)
- Parallel processing of batch folders (up to 3 threads)
- Each batch contains 5000 images + `representations_vgg_face.pkl`
- Uses ThreadPoolExecutor for concurrent batch searches

**FaceValidationService** (`app/services/face_validation_service.py`)
- Validates identical eye coordinates (rejects invalid detections)
- Calculates face quality metrics (Laplacian variance for blur)
- Converts coordinates between resized and original image dimensions

**ImageService** (`app/services/image_service.py`)
- Image resizing and format conversion
- Handles PIL/OpenCV format conversions

**ValidationService** (`app/services/validation_service.py`)
- Token-based authentication per domain
- Maps tokens to domains via `CLIENTS_TOKENS` environment variable

**TestRecognitionService** (`app/services/test_recognition_service.py`)
- A/B testing framework for comparing recognition profiles
- Supports current (VGG-Face) vs improved (ArcFace) configurations
- Configurable detection confidence and quality thresholds

**ComparisonService** (`app/services/comparison_service.py`)
- Side-by-side comparison of recognition results
- Analyzes differences between profile outputs
- Generates detailed comparison reports

**MetricsReportingService** (`app/services/metrics_reporting_service.py`)
- Generates recognition performance reports
- Tracks accuracy, precision, recall metrics
- Exports metrics in various formats

**ExcelService** (`app/services/excel_service.py`)
- Generates celebrity name lists by country/occupation
- Integrates with Google Custom Search for image collection
- Manages training data queue

### API Structure

**Face Recognition Endpoints:**
- `POST /upload-with-domain` - Upload face with person name and date
- `POST /recognize` - Recognize face in image (standard)
- `POST /api/batch/recognize` - Batch recognition (parallel search)
- `POST /api/test/recognize` - A/B testing recognition with profile selection

**Authentication Endpoints:**
- `POST /api/auth/token-by-email` - Get auth token by email
- `POST /api/auth/validate-email` - Validate email access
- `GET /api/auth/health` - Authentication service health check

**Training Data Management Endpoints:**
- `POST /api/excel/generate-names` - Generate celebrity names by country
- `POST /api/excel/process-next` - Process next person in queue
- `GET /api/excel/status` - Get training queue status
- `GET /api/excel/stats` - Get training statistics
- `POST /api/excel/sync` - Sync training data to production

**Batch Management Endpoints:**
- `POST /api/batch/recognize` - Parallel batch recognition
- `GET /api/batch/stats` - Batch structure statistics
- `GET /api/batch/domains` - List domains with batch structure
- `GET /api/batch/info` - Detailed batch information
- `GET /api/batch/health` - Batch service health check

**A/B Testing Endpoints:**
- `POST /api/test/recognize` - Test with specific profile (current/improved)
- `POST /api/test/compare` - Compare both profiles side-by-side
- `GET /api/test/profiles` - List available recognition profiles

All endpoints require `Authorization` header with domain token (except health checks).

### Storage Organization

```
storage/
├── recognized_faces/{domain}/           # Regular storage
├── recognized_faces_batched/{domain}/   # Batch system
│   ├── batch_0001/                     # 5000 images + pickle
│   ├── batch_0002/
│   └── batch_metadata.json
├── training/{domain}/                   # Training data
├── uploads/                             # Temporary uploads
└── name_mapping.json                    # Person name mappings
```

### Domain System

- Domains are cleaned for filesystem paths (removes port, invalid chars)
- Each domain has isolated storage directories
- Authentication tokens map to specific domains
- Environment variables:
  - `CLIENTS_TOKENS` - JSON mapping tokens to domain keys
  - `CLIENTS_EMAILS` - JSON mapping emails to domain keys

### Face Recognition Flow

1. **Upload & Validation**: Image received, auth validated
2. **Face Detection**: DeepFace detects faces with RetinaFace
3. **Quality Checks**:
   - Confidence >= 0.995
   - No identical eye coordinates
   - Blur detection (Laplacian variance)
   - Contrast and brightness validation
4. **Recognition**:
   - Standard: Single folder search with DeepFace.find()
   - Batch: Parallel search across batch folders (3 concurrent)
5. **Early Exit**: Recognition stops immediately when match found (performance optimization)

### Batch Recognition Details

The batch system organizes large face databases (>5000 images) for optimized parallel search:

- **Batch Size**: 5000 images per batch folder
- **Pickle Files**: Each batch has `representations_vgg_face.pkl` for fast lookup
- **Threading**: Max 3 concurrent batch searches (configurable)
- **Timeout**: 5 minutes per batch
- **Metadata**: `batch_metadata.json` tracks all batches per domain

Migration creates batch structure from existing recognized faces while preserving original organization.

### Recognition Profiles (A/B Testing)

The system supports multiple recognition configurations for A/B testing comparison:

**Current System Profile** (`app/config/recognition_profiles.py: CurrentSystemProfile`)
- Model: VGG-Face
- Detection confidence: >= 0.995 (99.5%)
- Recognition threshold: 0.35
- Blur threshold: 100 (Laplacian variance)
- Contrast threshold: 25
- Brightness range: 30-220
- Production configuration

**Improved System Profile** (`app/config/recognition_profiles.py: ImprovedSystemProfile`)
- Model: ArcFace
- Detection confidence: >= 0.98 (98%)
- Recognition threshold: 0.40
- Relaxed quality thresholds for testing
- Experimental configuration

Profiles are used via `/api/test/recognize` endpoint by specifying `profile` parameter.

### Training UI System

The backend provides a complete API for automated training data collection (frontend UI not yet implemented):

**Workflow:**
1. **Generate Names**: Generate celebrity lists by country using Excel service
2. **Queue Management**: Names added to processing queue
3. **Image Collection**: Google Custom Search API downloads images per celebrity
4. **Face Processing**: Validates and extracts faces from downloaded images
5. **Storage**: Organized by domain in `storage/training/{domain}/`
6. **Sync to Production**: Validated faces moved to `storage/recognized_faces/`

**Key Features:**
- Automated celebrity image collection
- Background processing with status tracking
- Quality validation before adding to training set
- Bulk upload and synchronization
- Statistics and progress monitoring

**Documentation:**
- [TRAINING_UI_API_DOCS.md](TRAINING_UI_API_DOCS.md) - Complete API documentation
- [FRONTEND_DEVELOPER_README.md](FRONTEND_DEVELOPER_README.md) - Frontend guide
- [TRAINING_DATA_COLLECTION_GUIDE.md](TRAINING_DATA_COLLECTION_GUIDE.md) - Data collection workflow

**Current Status:** Backend APIs fully implemented. Frontend UI needs to be built (currently showing old simple upload interface).

## Configuration

### Environment Variables (.env)
```bash
SECRET_KEY                 # Flask secret key
CLIENTS_TOKENS             # JSON: token -> domain_key mapping
CLIENTS_EMAILS             # JSON: email -> domain_key(s) mapping
SERPAPI_SEARCH_API_KEY     # Google search API key
GOOGLE_SEARCH_CX           # Google Custom Search engine ID
```

### Application Config (config.py)
```python
UPLOAD_FOLDER              # Upload directory path
MAX_CONTENT_LENGTH = 30MB  # Max request size
EXCEL_FILE_PATH            # Excel data location
IMAGE_STORAGE_PATH         # Training images path
```

## Important Implementation Notes

### Face Validation Thresholds
- Confidence: >= 0.995 (very strict to avoid false positives)
- Blur threshold: Laplacian variance checked
- Eye coordinates: Must be different (prevents bad detections)

### Performance Optimizations
- **Early Exit**: Recognition returns immediately on first match
- **Image Resizing**: Large images resized before processing
- **Batch Parallelization**: Multiple batch folders searched concurrently
- **Pickle Caching**: Pre-computed face representations per batch

### Multi-Domain Support
Single email can have multiple domain tokens. Response format:
- Single domain: Returns object `{token, email}`
- Multiple domains: Returns array `[{token, email, domain}, ...]`

### CORS Configuration
All origins accepted in development (configured in run.py).

## Common Patterns

### Adding New Service
1. Create in `app/services/{name}_service.py`
2. Implement as static methods or class
3. Import in relevant controller

### Adding New Endpoint
1. Create controller in `app/controllers/{name}_controller.py`
2. Create route in `app/routes/{name}_routes.py` as Blueprint
3. Register blueprint in `app/__init__.py`

### Domain Path Handling
Always use `RecognitionService.clean_domain_for_path(domain)` to sanitize domain strings for filesystem operations.

### Face Processing
Use `RecognitionService.process_single_face()` for standardized face validation and quality checks.

## Testing Notes

- Test images should be in scripts/ directory
- Auth tests use endpoints defined in README_AUTH_ENDPOINT.md
- Recognition tests validate both single and batch recognition flows
- Always test with valid domain tokens
- A/B testing compares current vs improved recognition profiles
- Use Postman collection (TRAINING_UI_POSTMAN_COLLECTION.json) for API testing

## Additional Documentation

**Core Documentation:**
- [README.md](README.md) - Main project documentation
- [README_AUTH_ENDPOINT.md](README_AUTH_ENDPOINT.md) - Authentication system
- [README_BATCH_RECOGNITION.md](README_BATCH_RECOGNITION.md) - Batch processing

**Training & Testing:**
- [TRAINING_UI_API_DOCS.md](TRAINING_UI_API_DOCS.md) - Training UI API reference (60+ pages)
- [TRAINING_UI_QUICK_REFERENCE.md](TRAINING_UI_QUICK_REFERENCE.md) - Quick reference guide
- [FRONTEND_DEVELOPER_README.md](FRONTEND_DEVELOPER_README.md) - Frontend developer guide
- [TRAINING_DATA_COLLECTION_GUIDE.md](TRAINING_DATA_COLLECTION_GUIDE.md) - Data collection workflow
- [AB_TESTING_PLAN.md](AB_TESTING_PLAN.md) - A/B testing strategy
- [AB_TESTING_UI_DOCS.md](AB_TESTING_UI_DOCS.md) - A/B testing UI documentation

**Analysis:**
- [VIDEO_FACE_RECOGNITION_GUIDE.md](VIDEO_FACE_RECOGNITION_GUIDE.md) - Video processing guide
- [ANALYSIS_AND_RECOMMENDATIONS.md](ANALYSIS_AND_RECOMMENDATIONS.md) - System analysis

## Frontend Implementation Status

**Current State:**
- ✅ Backend APIs fully implemented and tested
- ✅ Authentication system operational
- ✅ Training data collection workflow functional
- ✅ A/B testing infrastructure ready
- ❌ **Frontend UI not implemented** (old simple upload interface still showing)

**Required Frontend Components:**
See [FRONTEND_DEVELOPER_README.md](FRONTEND_DEVELOPER_README.md) for complete requirements. The new UI should include:
1. Login page (email → token authentication)
2. Dashboard (training status, stats)
3. Name Generator (celebrity lists by country)
4. Queue Manager (process training data)
5. Progress Monitor (view training folders)
6. Sync Manager (move to production)
7. Testing page (upload & test recognition)
8. A/B Testing page (compare profiles)

**Implementation Options:**
- React/Vue/Next.js single-page application
- Flask templates with JavaScript
- Separate frontend repository with API integration
