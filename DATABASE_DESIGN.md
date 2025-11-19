# Simple Training Database Design

## Philosophy: "As Simple As Possible"

- **SQLite** - No server, just a file
- **3 core tables** - People, Images, Training Sessions
- **Files stay on disk** - DB tracks metadata only
- **SQLAlchemy ORM** - Python-friendly, Flask-integrated

---

## Schema Design

### **Table 1: `people`** - Who we're training on

```sql
CREATE TABLE people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Identity
    full_name VARCHAR(255) NOT NULL,              -- "Novak Djokovic"
    normalized_name VARCHAR(255) UNIQUE NOT NULL, -- "novak_djokovic" (folder name)

    -- Wikidata info
    wikidata_id VARCHAR(50),                      -- "Q5812"
    primary_image_url TEXT,                       -- Wikidata P18 image
    sitelinks INTEGER DEFAULT 0,                  -- Notability: 152 for Djokovic

    -- Metadata
    occupation VARCHAR(100),                      -- "tennis_player"
    country VARCHAR(50),                          -- "serbia"
    description TEXT,                             -- From Wikidata

    -- Training status
    status VARCHAR(50) DEFAULT 'pending',         -- pending, in_training, completed, deployed
    folder_path VARCHAR(500),                     -- storage/trainingPassSerbia/novak_djokovic

    -- Statistics
    total_images INTEGER DEFAULT 0,               -- How many photos we have
    images_from_wikimedia INTEGER DEFAULT 0,      -- Source breakdown
    images_from_serp INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    trained_at TIMESTAMP,                         -- When training completed
    deployed_at TIMESTAMP,                        -- When deployed to production

    INDEX idx_normalized_name (normalized_name),
    INDEX idx_wikidata_id (wikidata_id),
    INDEX idx_status (status)
);
```

**Key Points:**
- One row = one celebrity
- `normalized_name` is unique (matches folder name)
- Tracks source breakdown (wikimedia vs serp)
- Status tracking for workflow

---

### **Table 2: `images`** - Individual photos

```sql
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,                   -- FK to people.id

    -- File info
    filename VARCHAR(255) NOT NULL,               -- "novak_djokovic_20250117_1.jpg"
    file_path VARCHAR(500) NOT NULL,              -- Full path on disk
    file_size INTEGER,                            -- Bytes

    -- Source tracking
    source VARCHAR(50) NOT NULL,                  -- 'wikimedia_p18', 'wikimedia_category', 'serp'
    source_url TEXT,                              -- Original download URL

    -- Quality metrics
    is_validated BOOLEAN DEFAULT 0,               -- Passed DeepFace validation?
    width INTEGER,                                -- Image dimensions
    height INTEGER,
    validation_error TEXT,                        -- Why it failed (if applicable)

    -- Timestamps
    downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validated_at TIMESTAMP,

    FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE,
    INDEX idx_person_id (person_id),
    INDEX idx_source (source),
    INDEX idx_validated (is_validated)
);
```

**Key Points:**
- Tracks every single image
- Source attribution (wikimedia/serp)
- Validation status from DeepFace
- Cascade delete: if person deleted, images deleted too

---

### **Table 3: `training_sessions`** - Batch operations

```sql
CREATE TABLE training_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(50) UNIQUE NOT NULL,       -- "a4f3c21d" (batch UUID)

    -- Session info
    country VARCHAR(50),                          -- "serbia"
    occupation VARCHAR(100),                      -- "actor"
    domain VARCHAR(100),                          -- "serbia" (storage domain)

    -- Progress tracking
    status VARCHAR(50) DEFAULT 'processing',      -- processing, completed, failed
    total_people INTEGER DEFAULT 0,
    people_completed INTEGER DEFAULT 0,
    people_failed INTEGER DEFAULT 0,

    -- Statistics
    total_images_downloaded INTEGER DEFAULT 0,
    images_from_wikimedia INTEGER DEFAULT 0,
    images_from_serp INTEGER DEFAULT 0,

    -- Cost tracking (optional but useful!)
    estimated_serp_cost DECIMAL(10,2),            -- How much SERP API cost

    -- Timestamps
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,

    -- Error tracking
    error_message TEXT,

    INDEX idx_session_id (session_id),
    INDEX idx_status (status)
);
```

**Key Points:**
- One row = one batch training operation
- Tracks overall progress and statistics
- Cost tracking for ROI analysis

---

## Minimal Relationships

```
training_sessions (1) ----< (many) people
people (1) ----< (many) images
```

**That's it.** No complex joins, no many-to-many tables.

---

## Migration Strategy: File-based → Database

### **Phase 1: Add database alongside files (non-breaking)**

```python
# Existing code still works with files
# New code also writes to database

def save_person_to_db(person_data):
    person = Person(
        full_name=person_data['full_name'],
        normalized_name=person_data['folder_name'],
        wikidata_id=person_data.get('wikidata_id'),
        ...
    )
    db.session.add(person)
    db.session.commit()

    # Still create folder (backward compatible)
    os.makedirs(folder_path, exist_ok=True)
```

### **Phase 2: Query database instead of scanning folders**

```python
# OLD: Scan filesystem
for folder in os.listdir('storage/trainingPassSerbia'):
    image_count = len(os.listdir(folder))

# NEW: Query database
people = Person.query.filter_by(status='completed').all()
for person in people:
    image_count = person.total_images
```

### **Phase 3: Backfill existing data (optional)**

```python
# Scan existing folders and populate database
def backfill_database():
    for folder in os.listdir('storage/trainingPassSerbia'):
        person = Person(
            normalized_name=folder,
            full_name=name_mapping.get(folder, folder),
            folder_path=folder_path,
            total_images=count_images(folder_path),
            status='completed'
        )
        db.session.add(person)
```

---

## Example Queries (Why This Helps)

### **1. How many people trained per source?**
```python
# Without DB: Impossible without scanning all folders
# With DB: One query
stats = db.session.query(
    func.count(People.id).label('total'),
    func.sum(People.images_from_wikimedia).label('wikimedia_total'),
    func.sum(People.images_from_serp).label('serp_total')
).first()

print(f"Wikimedia images: {stats.wikimedia_total}")
print(f"SERP images: {stats.serp_total}")
print(f"Cost savings: {(stats.wikimedia_total / (stats.wikimedia_total + stats.serp_total)) * 100}%")
```

### **2. Find people who need more training images**
```python
# Without DB: Scan every folder, count files
# With DB: Simple filter
people_needing_more = People.query.filter(
    People.total_images < 40,
    People.status == 'in_training'
).all()
```

### **3. Track cost over time**
```python
# Without DB: No way to know
# With DB: Aggregate by month
monthly_costs = db.session.query(
    func.strftime('%Y-%m', TrainingSession.started_at).label('month'),
    func.sum(TrainingSession.estimated_serp_cost).label('cost'),
    func.sum(TrainingSession.images_from_wikimedia).label('free_images')
).group_by('month').all()
```

### **4. Find duplicates or people with same Wikidata ID**
```python
# Without DB: Manual comparison
# With DB: GROUP BY query
duplicates = db.session.query(
    People.wikidata_id,
    func.count(People.id).label('count')
).group_by(People.wikidata_id).having(func.count(People.id) > 1).all()
```

---

## Implementation Checklist

### **Setup (30 minutes)**
- [ ] Add SQLAlchemy to requirements.txt
- [ ] Create `app/models/` directory
- [ ] Create `app/models/person.py`, `image.py`, `training_session.py`
- [ ] Create `app/database.py` with db connection
- [ ] Create migration script `migrations/init_db.sql`

### **Integration (2 hours)**
- [ ] Update `training_batch_service.py` to save to DB
- [ ] Update `wikimedia_image_service.py` to record images
- [ ] Add DB queries to existing endpoints

### **Testing (1 hour)**
- [ ] Test database creation
- [ ] Test training session with DB recording
- [ ] Verify data accuracy

### **Optional: Backfill (1 hour)**
- [ ] Create backfill script for existing folders
- [ ] Run on production data
- [ ] Verify counts match

---

## Configuration

### **Connection String (SQLite)**

```python
# config.py
SQLALCHEMY_DATABASE_URI = os.getenv(
    'DATABASE_URL',
    'sqlite:///storage/training.db'  # Simple file-based database
)
SQLALCHEMY_TRACK_MODIFICATIONS = False
```

**That's it!** No PostgreSQL server, no Docker, no complexity.

---

## File Structure After Implementation

```
facerecWeb/
├── storage/
│   ├── training.db                    # ← NEW: SQLite database
│   ├── trainingPassSerbia/            # Existing: actual images
│   │   ├── novak_djokovic/
│   │   │   ├── image1.jpg
│   │   │   └── image2.jpg
│   └── recognized_faces_prod/         # Existing: production
│
├── app/
│   ├── models/                        # ← NEW
│   │   ├── __init__.py
│   │   ├── person.py
│   │   ├── image.py
│   │   └── training_session.py
│   ├── database.py                    # ← NEW: DB connection
│   └── services/
│       └── training_batch_service.py  # Updated: saves to DB
```

---

## Why This Is "Simple"

✅ **No server** - SQLite is just a file
✅ **3 tables** - Not over-engineered
✅ **Keeps files** - Images stay on disk (no BLOBs)
✅ **Non-breaking** - Works alongside existing code
✅ **Backfill friendly** - Can populate from existing data
✅ **Easy queries** - No need to scan folders anymore
✅ **Flask-native** - SQLAlchemy integrates seamlessly

---

## Alternatives Considered (and why we didn't choose them)

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **JSON files** | Super simple | No queries, slow, no concurrency | ❌ Too limited |
| **PostgreSQL** | Production-grade | Server setup, overkill for this | ❌ Over-engineered |
| **MongoDB** | Flexible schema | Another dependency, overkill | ❌ Unnecessary |
| **SQLite** | Simple file, SQL queries, good enough | Not for high concurrency | ✅ **Perfect fit** |

---

## Next Steps?

1. Want me to implement the SQLAlchemy models?
2. Want me to create the migration script?
3. Want me to integrate DB writes into training_batch_service.py?
4. Want me to create a backfill script for existing data?

Let me know what to build first! 🚀
