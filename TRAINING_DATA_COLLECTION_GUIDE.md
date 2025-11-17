# Training Data Collection System - How It Works

## Overview

The facerecWeb system has a sophisticated **automated training data collection pipeline** that:
1. Uses **OpenAI GPT** to generate lists of famous people by occupation/country
2. Searches for their images using **RapidAPI Image Search** (SERP)
3. Downloads and validates images using **DeepFace**
4. Stores validated face crops in the training dataset
5. Processes them into the production face recognition database

This document explains the complete workflow.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING DATA COLLECTION PIPELINE                 │
└─────────────────────────────────────────────────────────────────────┘

Step 1: NAME GENERATION (OpenAI GPT)
┌────────────────────────────────────────────┐
│  GET /api/excel/check-excel?country=Serbia │
│                                            │
│  1. Reads storage/excel/occupation.xlsx    │
│  2. For each occupation (Actor, Politician)│
│  3. Calls OpenAI GPT-4.1:                  │
│     "Generate 20+ famous [occupation]      │
│      from [country]"                       │
│  4. Validates names with second GPT call   │
│  5. Saves to storage/excel/data.xlsx       │
└────────────────────────────────────────────┘
                    ↓
Step 2: IMAGE SEARCH (SERP via RapidAPI)
┌────────────────────────────────────────────┐
│  GET /api/excel/process                    │
│                                            │
│  1. Reads first row from data.xlsx         │
│     (name, last_name, occupation)          │
│  2. Removes row from Excel                 │
│  3. Searches images via RapidAPI:          │
│     "real-time-image-search.p.rapidapi.com"│
│     Query: "John Doe Actor"                │
│     Filters: type=face, size=1024x768+     │
│  4. Downloads up to 70 images per person   │
│  5. Saves to storage/training/serbia/      │
└────────────────────────────────────────────┘
                    ↓
Step 3: FACE VALIDATION (DeepFace Background Processing)
┌────────────────────────────────────────────┐
│  Automatic Background Thread               │
│                                            │
│  1. Finds first 3 valid reference images:  │
│     - Face must be detected                │
│     - Not blurry (Laplacian variance)      │
│     - Correct size (≥70x70px)              │
│  2. Extracts & saves to:                   │
│     storage/trainingPassSerbia/Person_Name/│
│  3. Compares remaining images with refs:   │
│     - Uses DeepFace.verify()               │
│     - Model: VGG-Face                      │
│     - Threshold: 0.6 cosine distance       │
│     - If match: extract face & save        │
│     - If no match: delete                  │
│  4. Stops at 40 images per person          │
│  5. Deletes duplicates (perceptual hash)   │
└────────────────────────────────────────────┘
                    ↓
Step 4: TRAINING DATASET PROCESSING (Manual Script)
┌────────────────────────────────────────────┐
│  python scripts/training_processor.py      │
│                                            │
│  1. Scans storage/trainingPassSerbia/      │
│  2. For each person folder:                │
│     - If ≤5 images: delete folder          │
│     - If >5 images:                        │
│       • Copy to storage/recognized_faces_  │
│         prod/serbia/                       │
│       • Transform filenames                │
│       • Run face recognition test          │
│       • Delete source folder               │
└────────────────────────────────────────────┘
                    ↓
Step 5: PRODUCTION DATABASE
┌────────────────────────────────────────────┐
│  storage/recognized_faces_prod/serbia/     │
│                                            │
│  Ready for recognition via:                │
│  POST /api/recognize                       │
└────────────────────────────────────────────┘
```

---

## Detailed Component Breakdown

### 1. OpenAI Celebrity Name Generation

**Endpoint**: `GET /api/excel/check-excel?country=Serbia`

**File**: `app/services/excel_service.py`

**Process**:
```python
# Reads occupations from storage/excel/occupation.xlsx
# Example rows: Actor, Politician, Athlete, Musician

# For each occupation, makes TWO OpenAI calls:

# Call 1: Generation
messages = [{
    "role": "system",
    "content": "Generate more than 20 most famous individuals
                for a given occupation and country."
}]
response = OpenAI.create(
    model="gpt-4.1",
    messages=messages
)
# Returns: ["John Doe", "Jane Smith", ...]

# Call 2: Validation
messages = [{
    "role": "system",
    "content": "Verify whether each person is a real, widely
                recognized individual from the specified country."
}]
response = OpenAI.create(
    model="gpt-4.1",
    messages=messages
)
# Returns validated list with valid=true/false for each

# Saves to storage/excel/data.xlsx
# Format:
# | name      | last_name | occupation |
# |-----------|-----------|------------|
# | John      | Doe       | Actor      |
```

**Key Features**:
- Uses GPT-4.1 for accuracy
- Two-pass validation (generation + verification)
- Saves normalized names (ASCII, no special chars)
- Stores name mappings for original characters

---

### 2. Image Search via SERP API

**Endpoint**: `GET /api/excel/process`

**File**: `app/services/image_service.py` - `fetch_and_save_images()`

**API Used**: RapidAPI Real-Time Image Search
```python
url = "https://real-time-image-search.p.rapidapi.com/search"
querystring = {
    "query": "John Doe Actor",        # Person + occupation
    "limit": "100",                    # Max results to return
    "size": "1024x768_and_more",      # Image size filter
    "type": "face",                    # Only face images
    "region": "us"
}
headers = {
    "x-rapidapi-key": "c3e8343ca0mshe1b719bea5326dbp11db14jsnf52a7fb8ab17",
    "x-rapidapi-host": "real-time-image-search.p.rapidapi.com"
}
```

**Download Process**:
```python
# Limits to 70 images per person
max_images = 70
image_results = data['data'][:max_images]

# For each image URL:
for i, item in enumerate(image_results):
    image_url = item.get('thumbnail_url') or item.get('url')

    # Creates filename:
    # John_Doe_20250113_044003_1.jpg
    # John_Doe_20250113_044003_2.jpg
    # ...

    # Downloads with browser headers to avoid blocking
    headers = {
        'User-Agent': 'Mozilla/5.0 ...',
        'Referer': 'https://www.google.com/'
    }

    # Validates downloaded image:
    # - File size > 100 bytes
    # - Can be opened with PIL
    # - Valid image dimensions

    # Saves to: storage/training/serbia/
```

**Fallback Mechanisms**:
- Tries 4 different download methods if one fails
- Skips invalid/corrupted images
- Logs all failures for debugging

---

### 3. DeepFace Validation (Background Thread)

**File**: `app/services/image_service.py` - `_process_images_with_deepface_thread()`

**This is the most sophisticated part!**

#### Phase 1: Find Reference Images (First 3 valid faces)

```python
# Create person-specific folder
person_dir = "storage/trainingPassSerbia/John_Doe/"

# Find first 3 valid reference images
valid_image_paths = []
max_reference_images = 3

for image_path in image_files:
    if len(valid_image_paths) >= 3:
        break

    # Extract face using DeepFace
    faces = DeepFace.extract_faces(
        img_path=image_path,
        detector_backend="retinaface",
        enforce_detection=False
    )

    # Validate face quality:
    # 1. Size check: width/height ≥ 70px
    if w < 70 or h < 70:
        delete_image()
        continue

    # 2. Blur check: Laplacian variance
    if is_blurred(face_array):
        delete_image()
        continue

    # 3. Multiple faces check
    if len(valid_faces) > 1:
        delete_image()
        continue

    # If valid: extract face with 20% margin
    margin = 0.2
    x1 = max(0, x - int(w * margin))
    y1 = max(0, y - int(h * margin))
    x2 = min(img_width, x + w + int(w * margin))
    y2 = min(img_height, y + h + int(h * margin))

    face_img = img[y1:y2, x1:x2]
    cv2.imwrite(person_dir + filename, face_img)

    valid_image_paths.append(image_path)
    delete_original(image_path)
```

#### Phase 2: Verify Remaining Images Against References

```python
# Calculate perceptual hashes of reference images
processed_hashes = set()
reference_images = []

for ref_path in valid_image_paths:
    img = cv2.imread(ref_path)
    img_hash = calculate_image_hash(img)  # 8x8 perceptual hash
    processed_hashes.add(img_hash)
    reference_images.append({'path': ref_path, 'hash': img_hash})

# Process remaining images (up to 40 total per person)
max_images_per_person = 40
processed_count = 3  # Already have 3 reference images

for image_path in remaining_images:
    if processed_count >= max_images_per_person:
        delete_image(image_path)
        continue

    # Check for duplicate (perceptual hash)
    img = cv2.imread(image_path)
    img_hash = calculate_image_hash(img)

    if img_hash in processed_hashes:
        delete_image(image_path)  # Duplicate
        continue

    # Verify against each reference image
    is_match = False

    for ref_image in reference_images:
        result = DeepFace.verify(
            img1_path=ref_image['path'],
            img2_path=image_path,
            model_name="VGG-Face",
            distance_metric="cosine",
            detector_backend="retinaface",
            threshold=0.6,  # Cosine distance threshold
            enforce_detection=False
        )

        if result["verified"]:
            is_match = True
            break

    if is_match:
        # Extract and save face
        extract_and_save_face(image_path, person_dir)
        processed_count += 1
        processed_hashes.add(img_hash)

    # Always delete original
    delete_image(image_path)
```

**Key Features**:
- **Smart reference selection**: Finds 3 high-quality faces first
- **Duplicate detection**: Perceptual hashing prevents duplicates
- **Face verification**: DeepFace ensures all images are same person
- **Quality filters**: Blur, size, multiple faces
- **Automatic cleanup**: Deletes invalid/processed images
- **Limit enforcement**: Stops at 40 images per person

---

### 4. Training Dataset Processor

**File**: `scripts/training_processor.py`

**Manual Script** (run when ready to move to production):

```bash
python scripts/training_processor.py
```

**Process**:
```python
source_base = "storage/trainingPassSerbia"
target_dir = "storage/recognized_faces_prod/serbia"

# For each person folder in trainingPassSerbia/
for person_folder in os.listdir(source_base):
    image_count = count_images(person_folder)

    # Delete if too few images (quality filter)
    if image_count <= 5:
        delete_folder(person_folder)
        continue

    # Copy images with filename transformation
    for image_file in os.listdir(person_folder):
        # Transform filename:
        # From: Abraham_Nnamdi_Nwankwo_20250524_044003_66.jpg
        # To:   Abraham_Nnamdi_Nwankwo_2025-05-24_04400366.jpg

        transformed_name = transform_filename(image_file)
        shutil.copy2(source_path, target_dir + transformed_name)

    # Run face recognition test
    test_image = first_image_in(target_dir)
    result = RecognitionService.recognize_face(test_image, "serbia")

    if result["status"] == "success":
        # Delete source folder after successful processing
        delete_folder(person_folder)
```

**Key Features**:
- Filters out low-quality batches (≤5 images)
- Transforms filenames for consistent format
- Tests recognition after each batch
- Cleans up after successful processing
- Logging at every step

---

### 5. Production Usage

Once images are in `storage/recognized_faces_prod/serbia/`, they're used by:

```python
# POST /api/recognize
def recognize_face():
    # Compares uploaded image against all stored faces
    db_path = "storage/recognized_faces_prod/serbia"

    result = DeepFace.find(
        img_path=uploaded_image,
        db_path=db_path,
        model_name="VGG-Face",  # Or Facenet512 in A/B testing
        detector_backend="retinaface",
        distance_metric="cosine",
        threshold=0.35  # Or 0.40 in improved system
    )

    return best_match
```

---

## Configuration & API Keys

### Environment Variables

```bash
# OpenAI (for name generation)
OPENAI_API_KEY=<your-key>

# RapidAPI (for image search)
# Hardcoded in image_service.py:
x-rapidapi-key: c3e8343ca0mshe1b719bea5326dbp11db14jsnf52a7fb8ab17

# File paths
EXCEL_FILE_PATH=storage/excel/data.xlsx
EXCEL_FILE_PATH_OCCUPATION=storage/excel/occupation.xlsx
IMAGE_STORAGE_PATH=storage/training/serbia
TRAINING_PASS_PATH=storage/trainingPassSerbia
```

---

## Storage Structure

```
storage/
├── excel/
│   ├── occupation.xlsx          # Input: List of occupations to search
│   └── data.xlsx                # Queue: Generated celebrity names
├── training/
│   └── serbia/                  # Downloaded images (raw, unvalidated)
│       ├── John_Doe_20250113_044003_1.jpg
│       ├── John_Doe_20250113_044003_2.jpg
│       └── ...
├── trainingPassSerbia/          # Validated face crops
│   ├── John_Doe/
│   │   ├── John_Doe_20250113_044003_1.jpg  (face crop)
│   │   ├── John_Doe_20250113_044003_5.jpg  (face crop)
│   │   └── ... (up to 40 images)
│   └── Jane_Smith/
│       └── ...
├── recognized_faces_prod/
│   └── serbia/                  # Production database
│       ├── John_Doe_2025-01-13_04400301.jpg
│       ├── John_Doe_2025-01-13_04400305.jpg
│       └── ...
└── name_mapping.json            # Maps normalized → original names
```

---

## Complete Workflow Example

### Step-by-Step: Adding "Serbian Actors" to Database

1. **Setup occupation.xlsx**:
   ```
   | Occupation |
   |------------|
   | Actor      |
   ```

2. **Trigger name generation**:
   ```bash
   curl "http://localhost:5000/api/excel/check-excel?country=Serbia"
   ```

   **Result**: Generates data.xlsx with ~20 Serbian actors
   ```
   | name      | last_name | occupation |
   |-----------|-----------|------------|
   | Dragan    | Bjelogrlic| Actor      |
   | Nebojsa   | Glogovac  | Actor      |
   | ...       | ...       | ...        |
   ```

3. **Process first person**:
   ```bash
   curl "http://localhost:5000/api/excel/process"
   ```

   **Actions**:
   - Reads first row: "Dragan Bjelogrlic Actor"
   - Removes row from data.xlsx
   - Searches RapidAPI: "Dragan Bjelogrlic Actor"
   - Downloads 70 images → `storage/training/serbia/`
   - **Background thread starts**:
     - Finds 3 valid reference faces
     - Verifies remaining 67 images
     - Saves ~20-40 validated faces → `storage/trainingPassSerbia/Dragan_Bjelogrlic/`
     - Deletes non-matches and originals

4. **Repeat for all names**:
   - Call `/api/excel/process` repeatedly
   - Each call processes one person from data.xlsx
   - Continue until data.xlsx is empty

5. **Move to production**:
   ```bash
   python scripts/training_processor.py
   ```

   **Actions**:
   - Scans `trainingPassSerbia/`
   - Deletes folders with ≤5 images
   - Copies valid batches → `storage/recognized_faces_prod/serbia/`
   - Tests recognition
   - Cleans up source folders

6. **Use in production**:
   ```bash
   curl -X POST http://localhost:5000/api/recognize \
     -H "Authorization: Bearer TOKEN" \
     -F "image=@test_actor.jpg"
   ```

   **Response**:
   ```json
   {
     "status": "success",
     "person": "Dragan Bjelogrlic",
     "confidence": 95.2,
     "best_match": {
       "identity": "Dragan_Bjelogrlic_2025-01-13_04400305.jpg",
       "distance": 0.28
     }
   }
   ```

---

## Performance & Limits

### Processing Times
- **Name generation** (GPT): ~10-30 seconds per occupation
- **Image search** (SERP): ~5-15 seconds per person
- **Image download**: ~30-60 seconds for 70 images
- **DeepFace validation**: ~2-5 minutes per person (background)
- **Total per person**: ~5-10 minutes (mostly background)

### Limits & Quotas
- **RapidAPI**: 100 images per search request (limited to 70)
- **OpenAI**: Rate limits depend on account tier
- **Storage**: ~40 images × ~500KB each = ~20MB per person
- **Max images per person**: 40 (configurable)

### Quality Filters
- **Image size**: ≥1024x768 pixels preferred
- **Face size**: ≥70x70 pixels minimum
- **Blur threshold**: Laplacian variance check
- **Verification threshold**: 0.6 cosine distance (VGG-Face)
- **Min images per person**: >5 for production

---

## Troubleshooting

### Common Issues

**1. No images downloaded**
- Check RapidAPI key validity
- Check person name spelling
- Try more generic search terms
- Check RapidAPI quota

**2. All images rejected by DeepFace**
- Person might not have clear face images online
- Search results might be irrelevant
- Try different search terms
- Check quality thresholds

**3. DeepFace verification too strict**
- Adjust threshold in `_verify_faces()`:
  ```python
  threshold = 0.6  # Try 0.7 or 0.8 for looser matching
  ```

**4. Training processor deletes all folders**
- Not enough valid images per person
- Check `min_images_threshold = 5`
- Review DeepFace validation logs

---

## Future Improvements

Based on the A/B testing framework we just implemented:

1. **Use Facenet512** instead of VGG-Face for verification
   - Better accuracy (97.4% vs ~92%)
   - Update `_verify_faces()` threshold to 0.40

2. **Adjust thresholds** based on A/B test results
   - Detection confidence: 99.5% → 98%
   - Recognition threshold: 0.35 → 0.40

3. **Add video processing** using `VIDEO_FACE_RECOGNITION_GUIDE.md`
   - Frame extraction at 1-8 FPS
   - Quality-based selection
   - Face tracking

4. **Implement ensemble verification**
   - Use multiple models (Facenet512 + ArcFace)
   - Weighted voting for decisions

---

## Summary

The training data collection system is a **fully automated pipeline** that:

✅ **Generates** celebrity names using AI (OpenAI GPT-4.1)
✅ **Searches** for images using SERP API (RapidAPI)
✅ **Downloads** up to 70 images per person
✅ **Validates** faces using DeepFace (blur, size, quality)
✅ **Verifies** all images match the same person
✅ **Filters** duplicates using perceptual hashing
✅ **Limits** to 40 high-quality images per person
✅ **Processes** validated images into production database
✅ **Tests** recognition after each batch

**Key Strength**: The DeepFace validation ensures high-quality, verified training data by comparing all images against trusted reference images.

**Use Case**: Building face recognition databases for public figures (actors, politicians, athletes) where images are publicly available online.
