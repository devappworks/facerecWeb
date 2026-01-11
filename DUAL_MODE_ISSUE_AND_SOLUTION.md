# Dual-Mode Issue: DeepFace Memory Conflict

**Date:** December 25, 2025
**Status:** ⚠️ DUAL-MODE TEMPORARILY DISABLED
**Reason:** TensorFlow/DeepFace memory corruption when called twice in same process

---

## Problem Description

### What Happened

When dual-mode was activated and a user uploaded an image:

1. ✅ PKL recognition ran successfully (using `DeepFace.find()`)
2. ✅ pgvector recognition started (using `DeepFace.represent()`)
3. ❌ **Worker crashed** with `free(): invalid pointer` error
4. ❌ Service returned 502 Bad Gateway to client

### Root Cause

**DeepFace/TensorFlow cannot be called twice in the same Python process** due to:
- TensorFlow model singleton patterns
- GPU/CPU memory allocation conflicts
- C++ layer memory management issues

### Evidence from Logs

```
Dec 25 14:35:45 INFO: [DUAL-MODE] Running pgvector recognition
Dec 25 14:35:45 INFO: [pgvector] Extracting ArcFace embedding from storage/uploads/pgvector_temp/pgvector_test_1766673310569.jpg
Dec 25 14:35:49 INFO: [pgvector] Searching database for domain=slovenia, threshold=0.3
Dec 25 14:35:49 free(): invalid pointer
[Worker crashed, new worker spawned]
```

### Call Chain That Failed

```
/analyze endpoint
  → VisionService.analyze_image_with_face_recognition()
    → RecognitionController.recognize_face()
      → RecognitionService.recognize_face()  # FIRST DeepFace call
        → DeepFace.extract_faces()
        → DeepFace.find()  # Searches PKL database
      → PgVectorRecognitionService.recognize_face_pgvector()  # SECOND DeepFace call ❌
        → DeepFace.represent()  # CRASH: Memory corruption
```

---

## Temporary Solution (Current)

**Disabled dual-mode** to restore service stability:

```bash
VECTOR_DB_ENABLED=true
VECTOR_DB_DUAL_MODE=false  # Disabled due to DeepFace memory conflict
```

Service restarted at: 14:37 UTC

**Result:** Service runs PKL-only mode (stable, no crashes)

---

## Permanent Solutions (Choose One)

### Option 1: Separate Worker Processes ⭐ RECOMMENDED

**Approach:** Run PKL and pgvector in separate gunicorn workers, compare results in main process.

**Implementation:**

1. **Create dedicated pgvector worker pool:**
   ```python
   # In recognition_controller.py
   from multiprocessing import Pool

   # Create worker pool at module level
   pgvector_pool = Pool(processes=2)  # Separate processes

   def _pgvector_worker(image_path, domain, threshold, top_k):
       """Worker function running in separate process"""
       from app.services.recognition_pgvector import PgVectorRecognitionService
       return PgVectorRecognitionService.recognize_face_pgvector(
           image_path=image_path,
           domain=domain,
           threshold=threshold,
           top_k=top_k
       )

   class RecognitionController:
       @staticmethod
       def recognize_face(image_bytes, domain):
           # Run PKL in main process
           pkl_result = RecognitionService.recognize_face(image_bytes, domain)

           if VECTOR_DB_ENABLED and VECTOR_DB_DUAL_MODE:
               # Save image for pgvector
               image_path = save_temp_image(image_bytes)

               # Run pgvector in SEPARATE PROCESS (no memory conflict!)
               async_result = pgvector_pool.apply_async(
                   _pgvector_worker,
                   args=(image_path, domain, 0.30, 10)
               )

               try:
                   # Wait for result with timeout
                   pgvector_results = async_result.get(timeout=10)

                   # Compare and merge results
                   pkl_result['pgvector_results'] = pgvector_results
                   pkl_result['mode'] = 'dual'
               except Exception as e:
                   logger.error(f"pgvector worker failed: {e}")
                   pkl_result['mode'] = 'pkl_only (pgvector failed)'

           return pkl_result
   ```

**Pros:**
- ✅ Completely isolates DeepFace calls (no memory conflicts)
- ✅ Can run both in parallel (faster!)
- ✅ Easy to roll back if pgvector fails

**Cons:**
- ⚠️ Requires more RAM (2x model loading)
- ⚠️ Slightly more complex code

---

### Option 2: Extract Embedding Once, Reuse It

**Approach:** Extract the ArcFace embedding once during PKL recognition, then reuse it for pgvector search.

**Implementation:**

1. **Modify RecognitionService to return embedding:**
   ```python
   # In recognition_service.py
   def recognize_face(image_bytes, domain):
       # ... existing code ...

       # After DeepFace.find(), extract the embedding used
       # (DeepFace caches it internally)
       embedding = get_cached_embedding()  # Need to implement

       return {
           'status': 'success',
           'recognized_faces': faces,
           'embedding': embedding  # ADD THIS
       }
   ```

2. **Modify PgVectorRecognitionService to accept pre-computed embedding:**
   ```python
   # In recognition_pgvector.py
   @staticmethod
   def recognize_face_pgvector(
       embedding: np.ndarray,  # Accept embedding directly
       domain: str,
       threshold: float = 0.30,
       top_k: int = 10
   ) -> List[Dict[str, Any]]:
       # Skip DeepFace.represent(), use provided embedding
       db = get_vector_db()
       matches = db.find_matches(
           query_embedding=embedding,
           domain=domain,
           threshold=threshold,
           top_k=top_k
       )
       return format_results(matches)
   ```

3. **Update RecognitionController:**
   ```python
   # In recognition_controller.py
   def recognize_face(image_bytes, domain):
       pkl_result = RecognitionService.recognize_face(image_bytes, domain)

       if VECTOR_DB_ENABLED and VECTOR_DB_DUAL_MODE:
           embedding = pkl_result.get('embedding')

           if embedding is not None:
               pgvector_results = PgVectorRecognitionService.recognize_face_pgvector(
                   embedding=embedding,  # Reuse embedding!
                   domain=domain
               )
               pkl_result['pgvector_results'] = pgvector_results
               pkl_result['mode'] = 'dual'

       return pkl_result
   ```

**Pros:**
- ✅ No memory conflicts (single DeepFace call)
- ✅ Faster (no redundant embedding extraction)
- ✅ Lower RAM usage

**Cons:**
- ⚠️ Requires refactoring RecognitionService
- ⚠️ Need to access DeepFace internal cache

---

### Option 3: pgvector-Only Mode (Skip Dual-Mode Entirely)

**Approach:** Skip dual-mode testing, switch directly to pgvector after manual validation.

**Implementation:**

1. **Validate pgvector independently:**
   - Run standalone tests with `scripts/test_vector_db.py`
   - Compare PKL vs pgvector results manually for 50-100 test images
   - Calculate accuracy metrics offline

2. **Switch to pgvector-only:**
   ```bash
   VECTOR_DB_ENABLED=true
   VECTOR_DB_DUAL_MODE=false
   VECTOR_DB_USE_PGVECTOR_ONLY=true  # NEW flag
   ```

3. **Modify RecognitionController:**
   ```python
   def recognize_face(image_bytes, domain):
       if VECTOR_DB_USE_PGVECTOR_ONLY:
           # Skip PKL entirely, use pgvector only
           image_path = save_temp_image(image_bytes)
           results = PgVectorRecognitionService.recognize_face_pgvector(
               image_path=image_path,
               domain=domain
           )
           return format_response(results)
       else:
           # PKL-only mode
           return RecognitionService.recognize_face(image_bytes, domain)
   ```

**Pros:**
- ✅ Simplest implementation
- ✅ No dual-mode complexity
- ✅ No memory conflicts

**Cons:**
- ⚠️ No real-time validation in production
- ⚠️ Riskier cutover (all-or-nothing)
- ⚠️ Harder to debug discrepancies

---

## Recommended Approach

**Use Option 1: Separate Worker Processes**

Why:
1. **Safety:** Completely isolates DeepFace calls (no crashes)
2. **Performance:** Can run both in parallel (faster dual-mode)
3. **Flexibility:** Easy to disable pgvector if issues arise
4. **Validation:** Real-time production comparison for confidence

---

## Implementation Plan (Option 1)

### Phase 1: Worker Pool Setup

```bash
# 1. Create pgvector worker module
touch app/workers/__init__.py
touch app/workers/pgvector_worker.py

# 2. Implement worker function
# (See code above)

# 3. Update recognition_controller.py
# (See code above)
```

### Phase 2: Testing

```bash
# 1. Start service with dual-mode
VECTOR_DB_DUAL_MODE=true
systemctl restart facerecweb

# 2. Upload test image
curl -X POST https://facerecognition.mpanel.app/analyze \
  -H "Authorization: $TOKEN" \
  -F "image=@test_image.jpg" \
  -G --data-urlencode "language=serbian"

# 3. Check response for dual-mode fields
# Should include: mode, pgvector_results, comparison

# 4. Monitor logs for crashes
tail -f storage/logs/gunicorn-error.log | grep -E "DUAL-MODE|pgvector|invalid pointer"
```

### Phase 3: Validation

```bash
# 1. Test with 50-100 images
# 2. Calculate top-1 match rate
# 3. Check top-5 overlap percentage
# 4. Monitor for ANY crashes (should be 0)
```

### Phase 4: Cutover

```bash
# If validation passes:
VECTOR_DB_DUAL_MODE=false
VECTOR_DB_USE_PGVECTOR_ONLY=true
systemctl restart facerecweb
```

---

## Quick Recovery

If dual-mode causes issues again:

```bash
# 1. Disable dual-mode immediately
sed -i 's/VECTOR_DB_DUAL_MODE=true/VECTOR_DB_DUAL_MODE=false/' /root/facerecognition-backend/.env

# 2. Restart service
systemctl restart facerecweb

# 3. Verify stability
systemctl status facerecweb

# 4. Service returns to PKL-only mode (stable)
```

---

## Related Files

- [recognition_controller.py](app/controllers/recognition_controller.py:1-114) - Dual-mode orchestration
- [recognition_pgvector.py](app/services/recognition_pgvector.py:1-210) - pgvector search logic
- [vision_service.py](app/services/vision/vision_service.py:188-192) - Vision analysis integration
- [.env](.env) - Configuration flags

---

## Status

**Current:** Dual-mode disabled, PKL-only mode (stable)
**Next:** Implement Option 1 (separate worker processes)
**Target:** Dual-mode working without crashes by end of day

---

**Contact:** Check logs at `/root/facerecognition-backend/storage/logs/gunicorn-error.log`
