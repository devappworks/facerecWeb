# Multi-Domain Training Architecture

## Problem Statement

**Scenario:**
- Serbian client needs Serbian celebrities (fast search)
- Greek client needs Greek celebrities (fast search)
- Slovenian client needs Slovenian celebrities (fast search)
- Mixing all celebrities = slow search for everyone

**Goal:** Separate training datasets per domain/client while sharing infrastructure.

---

## Folder Structure Design

### **Current State (Partially Implemented)**

```
storage/
├── trainingPassSerbia/           # ❌ Hardcoded to Serbia
├── training/serbia/              # ❌ Hardcoded to Serbia
├── recognized_faces_prod/        # ✅ Already domain-separated!
│   ├── serbia/
│   ├── greece/
│   └── slovenia/
└── recognized_faces_batched/     # ✅ Already domain-separated!
    ├── serbia/
    ├── greece/
    └── slovenia/
```

### **Proposed Structure (Full Domain Separation)**

```
storage/
├── training/                     # Raw training data per domain
│   ├── serbia/
│   │   ├── novak_djokovic_timestamp_1.jpg
│   │   └── ...
│   ├── greece/
│   │   ├── giannis_antetokounmpo_timestamp_1.jpg
│   │   └── ...
│   └── slovenia/
│       ├── luka_doncic_timestamp_1.jpg
│       └── ...
│
├── trainingPass/                 # Staging area per domain
│   ├── serbia/
│   │   ├── novak_djokovic/
│   │   │   ├── image1.jpg
│   │   │   └── image2.jpg
│   │   └── ...
│   ├── greece/
│   │   └── ...
│   └── slovenia/
│       └── ...
│
├── recognized_faces_prod/        # Production (already domain-separated)
│   ├── serbia/
│   │   ├── novak_djokovic/
│   │   └── ...
│   ├── greece/
│   └── slovenia/
│
├── recognized_faces_batched/     # Batched (already domain-separated)
│   ├── serbia/
│   │   ├── batch_0001/
│   │   ├── batch_0002/
│   │   └── batch_metadata.json
│   ├── greece/
│   └── slovenia/
│
└── training_batches/             # Batch metadata per domain
    ├── serbia_abc123.json
    ├── greece_def456.json
    └── slovenia_ghi789.json
```

**Key Changes:**
1. ❌ Remove hardcoded `trainingPassSerbia` → ✅ `trainingPass/{domain}`
2. ❌ Remove hardcoded `training/serbia` → ✅ `training/{domain}`
3. ✅ Keep existing domain separation in production/batched

---

## Database Schema Updates

### **Add `domains` Configuration Table**

```sql
CREATE TABLE domains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Domain identity
    domain_code VARCHAR(50) UNIQUE NOT NULL,      -- 'serbia', 'greece', 'slovenia'
    display_name VARCHAR(100) NOT NULL,           -- 'Serbia', 'Greece', 'Slovenia'

    -- Default training parameters
    default_country VARCHAR(50),                  -- 'serbia' for Wikidata queries
    default_occupations TEXT,                     -- JSON array: ["actor", "athlete"]

    -- Paths (auto-generated from domain_code)
    training_path VARCHAR(500),                   -- storage/training/serbia
    staging_path VARCHAR(500),                    -- storage/trainingPass/serbia
    production_path VARCHAR(500),                 -- storage/recognized_faces_prod/serbia
    batched_path VARCHAR(500),                    -- storage/recognized_faces_batched/serbia

    -- Statistics
    total_people INTEGER DEFAULT 0,
    total_images INTEGER DEFAULT 0,

    -- Status
    is_active BOOLEAN DEFAULT 1,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_domain_code (domain_code),
    INDEX idx_is_active (is_active)
);
```

**Example data:**
```sql
INSERT INTO domains (domain_code, display_name, default_country) VALUES
('serbia', 'Serbia', 'serbia'),
('greece', 'Greece', 'greece'),
('slovenia', 'Slovenia', 'slovenia'),
('sports_global', 'Global Sports Stars', NULL);  -- Cross-country dataset
```

### **Update `people` Table**

```sql
ALTER TABLE people ADD COLUMN domain VARCHAR(50) NOT NULL DEFAULT 'serbia';
ALTER TABLE people ADD CONSTRAINT fk_people_domain
    FOREIGN KEY (domain) REFERENCES domains(domain_code);

-- Add index for fast domain filtering
CREATE INDEX idx_people_domain ON people(domain);

-- Composite index for common queries
CREATE INDEX idx_people_domain_status ON people(domain, status);
```

### **Update `images` Table**

```sql
ALTER TABLE images ADD COLUMN domain VARCHAR(50) NOT NULL DEFAULT 'serbia';

-- Add index
CREATE INDEX idx_images_domain ON images(domain);
```

### **Update `training_sessions` Table**

```sql
-- Domain already exists in training_sessions, just add FK
ALTER TABLE training_sessions ADD CONSTRAINT fk_sessions_domain
    FOREIGN KEY (domain) REFERENCES domains(domain_code);
```

---

## Code Changes Required

### **1. Update TrainingBatchService Paths**

```python
# OLD (hardcoded):
TRAINING_PASS_PATH = "storage/trainingPassSerbia"

# NEW (domain-based):
@classmethod
def get_training_pass_path(cls, domain: str) -> str:
    """Get staging path for domain"""
    return f"storage/trainingPass/{domain}"

@classmethod
def get_training_path(cls, domain: str) -> str:
    """Get raw training path for domain"""
    return f"storage/training/{domain}"

@classmethod
def get_production_path(cls, domain: str) -> str:
    """Get production path for domain"""
    return f"storage/recognized_faces_prod/{domain}"

@classmethod
def get_batched_path(cls, domain: str) -> str:
    """Get batched path for domain"""
    return f"storage/recognized_faces_batched/{domain}"
```

### **2. Update ImageService**

```python
# OLD (hardcoded):
self.storage_path = os.getenv('IMAGE_STORAGE_PATH', 'storage/training/serbia')
self.training_pass_path = os.getenv('TRAINING_PASS_PATH', 'storage/trainingPassSerbia')

# NEW (domain-based):
def __init__(self, domain: str = 'serbia'):
    self.domain = domain
    self.storage_path = f'storage/training/{domain}'
    self.training_pass_path = f'storage/trainingPass/{domain}'
```

### **3. Update Recognition Service (Already Good!)**

Recognition service already accepts domain:
```python
RecognitionController.recognize_face(image_bytes, domain)
```

Just ensure it searches only in that domain's batched folder. ✅

---

## API Changes

### **All Training Endpoints Must Accept Domain**

```python
# Current:
POST /api/training/generate-candidates
{
  "country": "serbia",
  "occupation": "actor"
}

# New:
POST /api/training/generate-candidates
{
  "country": "serbia",
  "occupation": "actor",
  "domain": "serbia"  # ← REQUIRED
}
```

### **New Domain Management Endpoints**

```python
# List all domains
GET /api/domains
Response: [
  {"code": "serbia", "name": "Serbia", "total_people": 150, "is_active": true},
  {"code": "greece", "name": "Greece", "total_people": 80, "is_active": true}
]

# Create new domain
POST /api/domains
{
  "domain_code": "croatia",
  "display_name": "Croatia",
  "default_country": "croatia"
}

# Get domain stats
GET /api/domains/serbia/stats
Response: {
  "domain": "serbia",
  "total_people": 150,
  "total_images": 5420,
  "images_from_wikimedia": 3800,
  "images_from_serp": 1620,
  "production_batches": 2
}
```

---

## Recognition Performance Optimization

### **Problem: Searching Wrong Domain**

```python
# BAD: Slovenian client searching Serbian + Greek + Slovenian celebrities
all_faces = load_faces('storage/recognized_faces_batched/serbia')  # 10,000 images
all_faces += load_faces('storage/recognized_faces_batched/greece')  # 8,000 images
all_faces += load_faces('storage/recognized_faces_batched/slovenia')  # 3,000 images
# Total: 21,000 images → SLOW!

# GOOD: Slovenian client searching only Slovenian celebrities
slovenia_faces = load_faces('storage/recognized_faces_batched/slovenia')  # 3,000 images
# Total: 3,000 images → FAST! (7x faster)
```

### **Current Recognition Service Already Supports This!**

From `recognition_service.py`:
```python
def recognize_face(image_bytes, domain):
    db_path = f'storage/recognized_faces_batched/{domain}'  # ✅ Already domain-separated!

    results = DeepFace.find(
        img_path=image_bytes,
        db_path=db_path,  # Only searches this domain
        ...
    )
```

**Conclusion:** Recognition is already optimized! Just need to ensure training follows same pattern.

---

## Migration Strategy

### **Phase 1: Create Domain-Agnostic Functions (Non-Breaking)**

```python
# Add new functions alongside old ones
def get_training_path(domain='serbia'):
    return f'storage/training/{domain}'

# Old code still works:
path = 'storage/trainingPassSerbia'

# New code uses:
path = get_training_path('serbia')
```

### **Phase 2: Migrate Existing Data**

```bash
# Move existing Serbian data to new structure
mv storage/trainingPassSerbia storage/trainingPass/serbia
mv storage/training/serbia storage/training/serbia  # Already correct!

# Update name_mapping.json location
mv storage/name_mapping.json storage/trainingPass/serbia/name_mapping.json
```

### **Phase 3: Update All Service Calls**

```python
# Update training_batch_service.py
# Update image_service.py
# Update all endpoints to require domain parameter
```

### **Phase 4: Create Domains Table & Populate**

```sql
CREATE TABLE domains (...);

INSERT INTO domains (domain_code, display_name, default_country) VALUES
('serbia', 'Serbia', 'serbia');

-- Migrate existing people to have domain
UPDATE people SET domain = 'serbia' WHERE domain IS NULL;
```

---

## Advanced: Shared Datasets

### **Use Case: Regional Overlap**

Slovenia might want:
- Slovenian celebrities (primary)
- Croatian celebrities (similar language/culture)
- Serbian celebrities (regional neighbors)

**Solution: Domain Inheritance/References**

```sql
CREATE TABLE domain_datasets (
    id INTEGER PRIMARY KEY,
    domain_code VARCHAR(50),              -- 'slovenia'
    referenced_domain VARCHAR(50),        -- 'croatia', 'serbia'
    priority INTEGER,                     -- Search order: 1, 2, 3

    FOREIGN KEY (domain_code) REFERENCES domains(domain_code),
    FOREIGN KEY (referenced_domain) REFERENCES domains(domain_code)
);

-- Slovenia searches in this order:
INSERT INTO domain_datasets VALUES
(NULL, 'slovenia', 'slovenia', 1),      -- Own celebrities first
(NULL, 'slovenia', 'croatia', 2),       -- Croatian celebrities second
(NULL, 'slovenia', 'serbia', 3);        -- Serbian celebrities third
```

**Recognition code:**
```python
def recognize_face(image_bytes, domain):
    # Get all datasets for this domain
    datasets = get_domain_datasets(domain)  # ['slovenia', 'croatia', 'serbia']

    for dataset in datasets:
        db_path = f'storage/recognized_faces_batched/{dataset}'
        results = DeepFace.find(img_path=image_bytes, db_path=db_path)

        if results.found:
            return results  # Found in this dataset

    return None  # Not found in any dataset
```

---

## Configuration Example

### **Environment Variables**

```bash
# .env file for Slovenian deployment
PRIMARY_DOMAIN=slovenia
ALLOWED_DOMAINS=slovenia,croatia,serbia
DEFAULT_TRAINING_COUNTRIES=slovenia,croatia
```

### **Domain Configuration (JSON)**

```json
{
  "domains": {
    "serbia": {
      "display_name": "Serbia",
      "countries": ["serbia"],
      "occupations": ["actor", "athlete", "politician", "musician"],
      "search_priority": 1
    },
    "greece": {
      "display_name": "Greece",
      "countries": ["greece"],
      "occupations": ["actor", "athlete"],
      "search_priority": 1
    },
    "slovenia": {
      "display_name": "Slovenia",
      "countries": ["slovenia"],
      "occupations": ["actor", "athlete", "musician"],
      "search_priority": 1,
      "inherit_from": ["croatia", "serbia"]
    },
    "sports_global": {
      "display_name": "Global Sports",
      "countries": ["*"],
      "occupations": ["tennis_player", "football_player", "basketball_player"],
      "search_priority": 1,
      "filter": "sitelinks >= 50"
    }
  }
}
```

---

## Client Mapping

### **How to Determine Domain per Client**

```python
# Option 1: Token-based (already implemented)
CLIENTS_TOKENS = {
    "token_serbian_client": "serbia",
    "token_greek_client": "greece",
    "token_slovenian_client": "slovenia"
}

# Option 2: Subdomain-based
# serbia.facerec.example.com → domain='serbia'
# greece.facerec.example.com → domain='greece'

# Option 3: Header-based
# X-Client-Domain: serbia
```

---

## Performance Comparison

### **Scenario: 3 Domains with 150 Celebrities Each**

| Approach | Search Space | Recognition Time |
|----------|-------------|------------------|
| **Single Dataset** | 450 celebrities (22,500 images) | ~3.5 seconds |
| **Domain-Separated** | 150 celebrities (7,500 images) | ~1.2 seconds |
| **Speedup** | 3x smaller | **2.9x faster** ✅ |

### **Scenario: 10 Domains**

| Approach | Search Space | Recognition Time |
|----------|-------------|------------------|
| **Single Dataset** | 1,500 celebrities (75,000 images) | ~12 seconds |
| **Domain-Separated** | 150 celebrities (7,500 images) | ~1.2 seconds |
| **Speedup** | 10x smaller | **10x faster** ✅ |

---

## Recommended Implementation Order

### **Phase 1: Quick Wins (1-2 hours)**
1. ✅ Create `get_training_path(domain)` helper functions
2. ✅ Update `ImageService.__init__(domain)`
3. ✅ Update `training_batch_service.py` to use dynamic paths
4. ✅ Migrate `trainingPassSerbia` → `trainingPass/serbia`

### **Phase 2: Database (2-3 hours)**
1. ✅ Add `domains` table
2. ✅ Add `domain` column to existing tables
3. ✅ Create migration script
4. ✅ Backfill existing data with `domain='serbia'`

### **Phase 3: API Updates (2-3 hours)**
1. ✅ Add domain management endpoints
2. ✅ Update all training endpoints to require domain
3. ✅ Add domain stats endpoints

### **Phase 4: Advanced Features (optional, 3-4 hours)**
1. ⚠️ Domain inheritance (shared datasets)
2. ⚠️ Automatic domain detection from client tokens
3. ⚠️ Domain-specific configuration

---

## Testing Checklist

- [ ] Create new domain via API
- [ ] Train celebrities for domain A
- [ ] Train celebrities for domain B
- [ ] Verify recognition only searches correct domain
- [ ] Benchmark recognition speed (should be 3-10x faster)
- [ ] Test domain inheritance (if implemented)
- [ ] Verify existing Serbian data still works

---

## Summary

### **Folder Structure**
```
storage/
├── training/{domain}/          # Raw downloads
├── trainingPass/{domain}/      # Staging
├── recognized_faces_prod/{domain}/    # Production (already ✅)
└── recognized_faces_batched/{domain}/ # Batches (already ✅)
```

### **Database**
- `domains` table for configuration
- `domain` column on all tables
- Foreign keys for referential integrity

### **Benefits**
- 🚀 **3-10x faster recognition** (smaller search space)
- 🎯 **Client isolation** (no data leakage)
- 💰 **Cost optimization** (only train what's needed)
- 📊 **Better analytics** (per-domain stats)
- 🔧 **Flexibility** (shared datasets, inheritance)

### **Next Steps?**
1. Implement Phase 1 (path helpers + migration)?
2. Add `domains` table to database schema?
3. Update API endpoints to require domain?
4. All of the above?

Let me know what to implement first! 🚀
