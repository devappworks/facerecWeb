# pgvector Investigation Required

**Date:** December 25, 2025, 14:50 UTC
**Status:** ⚠️ **REVERTED TO PKL** - pgvector causes worker crashes
**Current Mode:** PKL-only (stable)

---

## What Happened

Attempted to switch from PKL to pgvector database search, but encountered persistent worker crashes with memory corruption errors.

### Timeline

1. **14:41 UTC** - Deployed pgvector-only service (replaced PKL)
2. **14:45 UTC** - First crash: `free(): invalid pointer` during database search
3. **14:46 UTC** - Fixed: Removed dual DeepFace calls (extract_faces + represent)
4. **14:48 UTC** - Second crash: Same error, still at database search step
5. **14:50 UTC** - **Reverted to PKL mode** to restore service

---

## Error Details

### Crash Pattern

**Consistent crash point:** Database search step, NOT during DeepFace processing

```
[pgvector] Saved temp image: storage/uploads/slovenia/temp_recognition_*.jpg
[pgvector] Extracting embeddings with ArcFace model
[pgvector] Found 1 faces with embeddings
[pgvector] Searching database for face 0, domain=slovenia
free(): invalid pointer  ← CRASH HERE
```

**Exit code:** 139 (segmentation fault)

### What This Tells Us

1. ✅ **DeepFace works fine** - Embeddings extracted successfully
2. ✅ **Database connection works** - Gets to search step
3. ❌ **Crash during vector search** - Something in pgvector query causes memory corruption

### Possible Root Causes

#### Theory 1: pgvector Python Library Issue
- The `pgvector` Python package may have C extension conflicts with TensorFlow
- Both use low-level memory operations
- Potential fix: Use different pgvector client library

#### Theory 2: psycopg2 + TensorFlow Conflict
- `psycopg2-binary` and TensorFlow may conflict in same process
- Both use C extensions
- Potential fix: Use `psycopg2` (compile from source) instead of `psycopg2-binary`

#### Theory 3: NumPy Array Conversion Issue
- Converting embedding to list for database insert may cause issues
- Potential fix: Test different serialization methods

#### Theory 4: Database Connection Pool Issue
- Opening new connections during TensorFlow operations
- Potential fix: Pre-initialize connection pool at startup

---

## Investigation Steps Needed

### Step 1: Test pgvector Standalone

Create a minimal test that:
1. Loads TensorFlow/DeepFace
2. Extracts an embedding
3. Searches pgvector database

```python
# Test script: scripts/test_pgvector_with_tensorflow.py
import numpy as np
from deepface import DeepFace
from app.services.vector_db_service import get_vector_db

# Extract embedding
embedding_result = DeepFace.represent(
    img_path="test_image.jpg",
    model_name="ArcFace"
)
embedding = np.array(embedding_result[0]["embedding"], dtype=np.float32)

# Search database
db = get_vector_db()
matches = db.find_matches(
    query_embedding=embedding,
    domain="slovenia",
    threshold=0.30
)

print(f"Found {len(matches)} matches")
```

**Expected:** If this crashes, it confirms the TensorFlow + pgvector conflict.

### Step 2: Test Without TensorFlow

Test pgvector search with a **pre-generated** embedding (no DeepFace):

```python
# Test script: scripts/test_pgvector_only.py
import numpy as np
from app.services.vector_db_service import get_vector_db

# Use a known embedding from database
embedding = np.random.rand(512).astype(np.float32)

# Search database
db = get_vector_db()
matches = db.find_matches(
    query_embedding=embedding,
    domain="slovenia",
    threshold=0.30
)

print(f"Found {len(matches)} matches")
```

**Expected:** If this works, TensorFlow is the issue.

### Step 3: Different psycopg2 Version

Try using `psycopg2` (source) instead of `psycopg2-binary`:

```bash
pip uninstall psycopg2-binary
pip install psycopg2

# Restart service
systemctl restart facerecweb
```

### Step 4: Different pgvector Client

Try using raw SQL instead of pgvector Python library:

```python
# In vector_db_service.py
def find_matches(self, query_embedding, domain, threshold, top_k):
    cursor = self._conn.cursor()

    # Use raw SQL instead of pgvector operators
    cursor.execute("""
        SELECT p.name, (f.embedding <=> %s::vector) as distance
        FROM face_embeddings f
        JOIN persons p ON f.person_id = p.id
        WHERE p.domain = %s
        AND (f.embedding <=> %s::vector) < %s
        ORDER BY distance
        LIMIT %s
    """, (
        query_embedding.tolist(),  # Convert to Python list
        domain,
        query_embedding.tolist(),
        threshold,
        top_k
    ))

    results = cursor.fetchall()
    cursor.close()
    return results
```

---

## Alternative Approaches

### Option A: Separate Service for pgvector

Run pgvector recognition in a **separate Python process** (not gunicorn worker):

```
Main gunicorn process (PKL recognition)
    ↓
    Calls separate Flask/FastAPI service
    ↓
pgvector service (isolated Python process)
    ↓
    Returns results via HTTP
```

**Pros:**
- Complete isolation (no memory conflicts)
- Can restart pgvector service independently
- Easy to monitor and debug

**Cons:**
- More complex architecture
- Network overhead (though minimal on localhost)

### Option B: Use PostgreSQL Stored Procedure

Move face recognition logic into PostgreSQL:

```sql
CREATE OR REPLACE FUNCTION recognize_face(
    query_emb vector(512),
    search_domain text,
    match_threshold float,
    result_limit int
)
RETURNS TABLE(person_name text, distance float) AS $$
BEGIN
    RETURN QUERY
    SELECT p.name, (f.embedding <=> query_emb) as dist
    FROM face_embeddings f
    JOIN persons p ON f.person_id = p.id
    WHERE p.domain = search_domain
    AND (f.embedding <=> query_emb) < match_threshold
    ORDER BY dist
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;
```

Call from Python:
```python
cursor.execute(
    "SELECT * FROM recognize_face(%s::vector, %s, %s, %s)",
    (embedding.tolist(), domain, threshold, top_k)
)
```

**Pros:**
- Minimal Python code
- Database handles all vector operations
- May avoid Python/C extension conflicts

**Cons:**
- Logic in database (harder to test/debug)
- Less flexible

### Option C: Stick with PKL, Optimize It

Instead of switching to pgvector, optimize PKL:

1. **Use smaller PKL files** (filter by domain before search)
2. **Implement caching** (keep hot domains in memory)
3. **Use faster storage** (SSD, tmpfs)
4. **Parallel search** (search multiple PKL files simultaneously)

**Pros:**
- Proven stable (no crashes)
- Less migration risk
- Simpler architecture

**Cons:**
- Still slower than pgvector (O(n) vs O(log n))
- Still large memory footprint

---

## Recommended Next Steps

### Immediate (Restore Service)
- ✅ **Reverted to PKL** - Service stable at 14:50 UTC
- ✅ **VECTOR_DB_ENABLED=false**
- ✅ **Frontend working** at https://photolytics.mpanel.app

### Short-term (Investigation)
1. Run **Step 1** test (TensorFlow + pgvector standalone)
2. If crashes, run **Step 2** test (pgvector without TensorFlow)
3. Identify exact conflict source

### Medium-term (Fix Approach)
Based on investigation results:
- **If TensorFlow conflict:** Implement Option A (separate service)
- **If psycopg2 issue:** Try Option B (stored procedure) or different library
- **If unfixable:** Consider Option C (optimize PKL instead)

---

## Current Configuration

```bash
# /root/facerecognition-backend/.env
VECTOR_DB_ENABLED=false  # Disabled - pgvector causes crashes
```

**Service:** Running stable in PKL mode
**Performance:** 500ms+ queries (slow but stable)
**Database:** Still available and populated (23,640 Slovenia, 933 Serbia)

---

## Files for Investigation

### Test Scripts to Create
- `scripts/test_pgvector_with_tensorflow.py` - Test TensorFlow + pgvector together
- `scripts/test_pgvector_only.py` - Test pgvector without TensorFlow
- `scripts/test_different_psycopg2.py` - Test with compiled psycopg2

### Code to Review
- [app/services/vector_db_service.py](app/services/vector_db_service.py) - Database connection and queries
- [app/services/recognition_service_pgvector.py](app/services/recognition_service_pgvector.py) - pgvector recognition logic
- [requirements.txt](requirements.txt) - Python package versions

### Logs to Analyze
- `/root/facerecognition-backend/storage/logs/gunicorn-error.log` - Crash details
- `journalctl -u facerecweb` - System-level errors

---

## Summary

**Problem:** pgvector database search causes worker crashes with memory corruption, even with only one DeepFace call.

**Root Cause:** Unknown - likely conflict between TensorFlow/DeepFace and pgvector/psycopg2 C extensions.

**Current Status:** Reverted to PKL mode (stable, but slower).

**Next Steps:** Investigate root cause with standalone tests, then choose fix approach (separate service, stored procedures, or optimize PKL).

**Impact:** Migration to pgvector blocked until memory corruption issue resolved.

---

**Service is stable and operational in PKL mode.** Face recognition works at https://photolytics.mpanel.app.
