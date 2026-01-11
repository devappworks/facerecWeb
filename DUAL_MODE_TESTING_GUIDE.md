# Dual-Mode Testing Guide

**Status:** ⚠️ DUAL-MODE TEMPORARILY DISABLED
**Date:** December 25, 2025
**Last Update:** Service restarted at 14:37:08 UTC - dual-mode disabled due to DeepFace memory conflict
**Issue:** See [DUAL_MODE_ISSUE_AND_SOLUTION.md](DUAL_MODE_ISSUE_AND_SOLUTION.md) for details

---

## System Configuration

```bash
VECTOR_DB_ENABLED=true
VECTOR_DB_DUAL_MODE=false  # Disabled due to DeepFace memory conflict
```

**Service:** `facerecweb.service` - **RUNNING STABLE** at 14:37:08 UTC (PKL-only mode)

---

## ⚠️ Dual-Mode Issue (14:35 UTC)

**Problem:** Worker crashed when running dual-mode due to DeepFace being called twice in the same process (memory corruption: `free(): invalid pointer`).

**Temporary Fix:** Disabled `VECTOR_DB_DUAL_MODE` to restore stability.

**Permanent Solution:** Implement separate worker processes for PKL and pgvector. See [DUAL_MODE_ISSUE_AND_SOLUTION.md](DUAL_MODE_ISSUE_AND_SOLUTION.md) for implementation plan.

---

## 🔧 Critical Fix Applied (14:30 UTC)

**Problem Found:** The `/analyze` endpoint (used by photolytics.mpanel.app) was calling `RecognitionService.recognize_face()` **directly**, bypassing the `RecognitionController` where dual-mode logic was implemented.

**Fix Applied:** Modified [app/services/vision/vision_service.py](app/services/vision/vision_service.py:188-192) to call through `RecognitionController` instead:

```python
# OLD (bypassed dual-mode):
from app.services.recognition_service import RecognitionService
result = RecognitionService.recognize_face(image_data, domain)

# NEW (enables dual-mode):
from app.controllers.recognition_controller import RecognitionController
result = RecognitionController.recognize_face(image_data, domain)
```

**Result:** Dual-mode now works for both `/recognize` AND `/analyze` endpoints!

---

## What's Happening Now

When you upload an image to https://photolytics.mpanel.app:

1. **PKL Recognition** runs (original behavior)
2. **pgvector Recognition** runs (NEW!)
3. **Results are compared**
4. **PKL results returned** (primary) + pgvector data

---

## Testing Slovenia Dataset

**Dataset Status:**
- ✅ **Slovenia:** 2,625 persons, 23,640 embeddings (READY TO TEST!)
- ⏳ **Serbia:** Migration in progress (217/29,599)

**Recommended:** Test with Slovenia domain for now!

---

## How to Test

### 1. Open Frontend
Navigate to: https://photolytics.mpanel.app

### 2. Upload Image
- Use an image of a Slovenian politician/celebrity
- The system will automatically use Slovenia domain (based on your auth token)

### 3. Check Response

The API response will now include:

```json
{
  "status": "success",
  "recognized_faces": [
    {
      "identity": "Person Name",
      "distance": 0.25,
      "confidence": 0.75,
      ...
    }
  ],

  // NEW FIELDS:
  "mode": "dual",

  "pgvector_results": [
    {
      "identity": "Person Name",
      "distance": 0.24,
      "confidence": 0.76,
      "source": "pgvector"
    }
  ],

  "comparison": {
    "top1_match": true,
    "pkl_top1": "Person Name",
    "pgvector_top1": "Person Name",
    "top5_overlap": 5,
    "top5_overlap_pct": 100.0,
    "pkl_count": 5,
    "pgvector_count": 5
  }
}
```

---

## What to Look For

### ✅ Success Indicators

1. **Response includes `mode: "dual"`**
2. **Both `recognized_faces` and `pgvector_results` are present**
3. **`comparison.top1_match === true`** (same top result from both)
4. **`comparison.top5_overlap_pct > 80%`** (good overlap)

### ⚠️ Warning Signs

1. **`mode: "pkl_only (pgvector failed)"`** - pgvector search failed
2. **`comparison.top1_match === false`** - Different top results
3. **`comparison.top5_overlap_pct < 80%`** - Low overlap
4. **`pgvector_error` field present** - Error during pgvector search

---

## Monitoring Logs

### Real-time Log Monitoring

```bash
# Terminal 1: Monitor dual-mode activity
tail -f /root/facerecognition-backend/storage/logs/gunicorn-error.log | grep -E "DUAL-MODE|pgvector"

# Terminal 2: Monitor all errors
tail -f /root/facerecognition-backend/storage/logs/gunicorn-error.log

# Terminal 3: Check database queries
tail -f /var/log/postgresql/postgresql-16-main.log
```

### Key Log Messages

**Dual-mode activated:**
```
[DUAL-MODE] Saved image for pgvector: storage/uploads/pgvector_temp/pgvector_test_*.jpg
[DUAL-MODE] Running pgvector recognition
```

**Success:**
```
[pgvector] Extracting ArcFace embedding from ...
[pgvector] Searching database for domain=slovenia, threshold=0.30
[pgvector] Found N matches in XX.XXms
[DUAL-MODE] PKL found N matches, pgvector found N matches
```

**Comparison results:**
```
[DUAL-MODE] Top-1 mismatch: PKL=Person A vs pgvector=Person B
[DUAL-MODE] Low top-5 overlap: XX.X%
```

---

## Performance Expectations

### pgvector Performance
- **Query time:** ~17-20ms (for 23,640 embeddings)
- **Embedding extraction:** ~2-3 seconds (DeepFace)
- **Total pgvector time:** ~2-3 seconds

### Comparison with PKL
- **PKL query:** ~500ms (brute-force search)
- **pgvector query:** ~20ms (**25x faster!**)

---

## Testing Checklist

### Basic Functionality
- [ ] Upload image of known Slovenian person
- [ ] Response includes `mode: "dual"`
- [ ] Both PKL and pgvector results present
- [ ] Top-1 results match

### Edge Cases
- [ ] Upload image with NO face detected
- [ ] Upload image with MULTIPLE faces
- [ ] Upload image of UNKNOWN person (not in database)
- [ ] Upload very large image (>5MB)
- [ ] Upload low-quality image

### Performance
- [ ] Response time < 5 seconds
- [ ] pgvector search time < 100ms (check logs)
- [ ] No memory leaks after 10+ requests

### Accuracy
- [ ] Test with 10 different known persons
- [ ] Calculate top-1 match rate (should be >95%)
- [ ] Check top-5 overlap percentage (should be >80%)

---

## Database Stats

Check current migration progress:

```bash
PGPASSWORD='1Un5J20uOLstxUxJYzvb76GCdAVOjZ5eN17Cq/Q/brU=' psql -U facerecadmin -d facerecognition -h localhost -c "
SELECT domain, COUNT(DISTINCT p.id) as persons, COUNT(f.id) as embeddings
FROM persons p
LEFT JOIN face_embeddings f ON p.id = f.person_id
GROUP BY domain;
"
```

---

## Troubleshooting

### Problem: `pgvector_error` in response

**Check:**
1. Database connection: `python scripts/test_vector_db.py`
2. Environment variables: `grep VECTOR_DB /root/facerecognition-backend/.env`
3. Error logs: `tail -100 storage/logs/gunicorn-error.log | grep pgvector`

### Problem: `mode: "pkl_only"`

**Means:** `VECTOR_DB_ENABLED=false`

**Fix:**
```bash
# Edit .env and set:
VECTOR_DB_ENABLED=true

# Restart service:
systemctl restart facerecweb
```

### Problem: Response missing `comparison` field

**Means:** Either PKL or pgvector returned 0 results

**Check:**
- Image quality (is face detected?)
- Domain setting (are you using the right domain?)
- Database has data for that domain

### Problem: Top-1 mismatch

**This is EXPECTED during testing!**

**What to do:**
1. Log the mismatch (check logs)
2. Manually verify which result is correct
3. Check confidence scores
4. If pgvector is consistently wrong, may need to adjust threshold

---

## Sample Test Images

### Good Test Cases (Slovenia)

Try uploading images of:
- **Politicians:** Janez Janša, Robert Golob, Borut Pahor
- **Celebrities:** Luka Dončić, Tadej Pogačar, Primož Roglič

### Where to Find Test Images

1. **Google Images** (search: "Janez Janša official photo")
2. **Wikimedia Commons** (free, high-quality)
3. **News websites** (recent photos)

---

## Next Steps After Testing

### If Tests Pass (>95% accuracy)

1. **Monitor for 24-48 hours** in dual-mode
2. **Collect metrics:**
   - Top-1 match rate
   - Top-5 overlap percentage
   - Average pgvector query time
   - Any errors/failures

3. **Switch to pgvector-only:**
   ```bash
   # Edit .env:
   VECTOR_DB_ENABLED=true
   VECTOR_DB_DUAL_MODE=false  # <-- Turn off PKL

   # Restart:
   systemctl restart facerecweb
   ```

### If Tests Fail (<95% accuracy)

1. **Check database integrity:**
   ```bash
   python scripts/migrate_pkl_to_postgres_standalone.py --validate-only --domain slovenia
   ```

2. **Adjust threshold:** May need to tune the `threshold` parameter

3. **Check embedding quality:** Verify ArcFace model is working correctly

4. **Investigate discrepancies:** Compare PKL vs pgvector for specific test cases

---

## Quick Commands Reference

```bash
# Restart service
systemctl restart facerecweb

# Check service status
systemctl status facerecweb

# Monitor logs
tail -f storage/logs/gunicorn-error.log | grep DUAL-MODE

# Check database
python scripts/test_vector_db.py

# Check migration progress
python scripts/migrate_pkl_to_postgres_standalone.py --validate-only --domain slovenia

# Disable dual-mode (revert to PKL only)
# Edit .env: VECTOR_DB_ENABLED=false
# Then: systemctl restart facerecweb
```

---

## Support

If you encounter issues:

1. **Check logs** in `storage/logs/gunicorn-error.log`
2. **Run database tests:** `python scripts/test_vector_db.py`
3. **Verify configuration:** `grep VECTOR_DB .env`
4. **Check service status:** `systemctl status facerecweb`

---

**Ready to test! Go to https://photolytics.mpanel.app and upload an image!** 🚀

The system will now:
- Run BOTH PKL and pgvector recognition
- Compare the results
- Return comprehensive data for validation
- Log any discrepancies

**Expected response time:** 2-5 seconds (most of that is DeepFace processing, pgvector search is only ~20ms!)
