"""
pgvector-based recognition service - drop-in replacement for PKL recognition.

This service mimics the PKL RecognitionService interface but uses PostgreSQL + pgvector
for face matching instead of DeepFace.find() with PKL files.

IMPORTANT: Uses only ONE DeepFace call (represent) to avoid memory corruption.
"""

import os
import time
import logging
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from deepface import DeepFace
from typing import Dict, List, Any, Tuple

from app.services.vector_db_service import get_vector_db

logger = logging.getLogger(__name__)


class PgVectorRecognitionService:
    """
    Face recognition using PostgreSQL + pgvector (PKL replacement).

    This service provides the same interface as RecognitionService
    but uses database vector search instead of PKL files.
    """

    @staticmethod
    def validate_face_quality(face_pixels, face_idx, source_type="image"):
        """
        Validate face quality using blur, contrast, brightness and edge density.

        Args:
            face_pixels: Cropped face image as numpy array (BGR uint8)
            face_idx: Face index for logging
            source_type: "image" (strict) or "video" (lenient blur threshold)

        Returns:
            (is_valid, quality_metrics) tuple
        """
        quality_metrics = {
            "blur_score": 0.0,
            "blur_threshold": 100,
            "contrast": 0.0,
            "brightness": 0.0,
            "edge_density": 0.0
        }
        try:
            if face_pixels.dtype in ('float64', 'float32'):
                face_uint8 = (face_pixels * 255).astype(np.uint8)
            else:
                face_uint8 = face_pixels.astype(np.uint8)

            gray = cv2.cvtColor(face_uint8, cv2.COLOR_BGR2GRAY)

            # Blur detection (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_threshold = 75 if source_type == "video" else 100
            quality_metrics["blur_score"] = round(float(laplacian_var), 2)
            quality_metrics["blur_threshold"] = blur_threshold

            # Contrast (standard deviation of grayscale)
            contrast = float(gray.std())
            quality_metrics["contrast"] = round(contrast, 2)

            # Brightness (mean of grayscale)
            mean_brightness = float(gray.mean())
            quality_metrics["brightness"] = round(mean_brightness, 2)

            # Edge density (Sobel gradient magnitude)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_density = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
            quality_metrics["edge_density"] = round(edge_density, 2)

            # Check thresholds
            if laplacian_var < blur_threshold:
                logger.info(f"[pgvector] Face {face_idx} rejected - too blurry (Laplacian: {laplacian_var:.2f} < {blur_threshold})")
                quality_metrics["reject_reason"] = "too_blurry"
                return False, quality_metrics

            if contrast < 25.0:
                logger.info(f"[pgvector] Face {face_idx} rejected - low contrast ({contrast:.2f})")
                quality_metrics["reject_reason"] = "low_contrast"
                return False, quality_metrics

            if mean_brightness < 30 or mean_brightness > 220:
                logger.info(f"[pgvector] Face {face_idx} rejected - poor brightness ({mean_brightness:.2f})")
                quality_metrics["reject_reason"] = "poor_brightness"
                return False, quality_metrics

            if edge_density < 15.0:
                logger.info(f"[pgvector] Face {face_idx} rejected - low edge density ({edge_density:.2f})")
                quality_metrics["reject_reason"] = "low_edge_density"
                return False, quality_metrics

            logger.info(f"[pgvector] Face {face_idx} passed quality checks - blur:{laplacian_var:.0f} contrast:{contrast:.1f} bright:{mean_brightness:.0f} edge:{edge_density:.1f}")
            return True, quality_metrics

        except Exception as e:
            logger.error(f"[pgvector] Quality validation error for face {face_idx}: {e}")
            quality_metrics["reject_reason"] = f"error: {str(e)}"
            return False, quality_metrics

    @staticmethod
    def filter_faces_by_size(face_areas, size_threshold=0.30):
        """
        Filter out faces that are too small relative to the largest face.

        Args:
            face_areas: List of dicts with 'index', 'w', 'h', 'area'
            size_threshold: Minimum ratio to largest face area (0.30 = 30%)

        Returns:
            Set of face indices to keep
        """
        if len(face_areas) <= 1:
            return {fa['index'] for fa in face_areas}

        largest_area = max(fa['area'] for fa in face_areas)
        min_required = largest_area * size_threshold
        keep = set()
        for fa in face_areas:
            if fa['area'] >= min_required:
                keep.add(fa['index'])
                logger.info(f"[pgvector] Face {fa['index']} size OK ({fa['area']}px, {fa['area']/largest_area*100:.0f}% of largest)")
            else:
                logger.info(f"[pgvector] Face {fa['index']} too small ({fa['area']}px, {fa['area']/largest_area*100:.0f}% of largest, need {size_threshold*100:.0f}%)")
        return keep

    @staticmethod
    def recognize_face(image_bytes, domain, source_type="image", collect_diagnostics=False):
        """
        Recognize faces in an image using pgvector database search.

        This method mimics RecognitionService.recognize_face() interface
        to be a drop-in replacement.

        CRITICAL: Uses only DeepFace.represent() (not extract_faces) to avoid
        calling DeepFace twice which causes memory corruption.

        Args:
            image_bytes: Image data as bytes or BytesIO
            domain: Domain to search ('serbia', 'slovenia', etc.)
            source_type: "image" or "video" (for logging)
            collect_diagnostics: If True, attach detailed diagnostics to response

        Returns:
            Dict with same structure as PKL RecognitionService:
            {
                'status': 'success',
                'recognized_persons': [...],
                'all_detected_matches': [...],
                'best_match': {...},
                'mode': 'pgvector'
            }
        """
        start_time = time.time()
        diag = None
        if collect_diagnostics:
            diag = {
                "model": "ArcFace",
                "detector": "retinaface",
                "source_type": source_type,
                "mode": "pgvector",
                "threshold_used": 0.45,
                "confidence_gate": 0.65,
                "image_dimensions": {},
                "pipeline_summary": {},
                "per_face_details": [],
                "rejected_faces": [],
                "near_misses": [],
                "timing_ms": {}
            }

        try:
            # Step 1: Save image to temp file (DeepFace requires file path)
            temp_folder = os.path.join('storage/uploads', domain)
            os.makedirs(temp_folder, exist_ok=True)
            image_path = os.path.join(temp_folder, f"temp_recognition_{int(time.time() * 1000)}.jpg")

            # Get bytes from BytesIO if needed
            if hasattr(image_bytes, 'getvalue'):
                actual_bytes = image_bytes.getvalue()
            else:
                actual_bytes = image_bytes

            # Save to file
            with open(image_path, 'wb') as f:
                f.write(actual_bytes)

            logger.info(f"[pgvector] Saved temp image: {image_path}")

            # Capture image dimensions for diagnostics
            if diag:
                try:
                    img = Image.open(BytesIO(actual_bytes))
                    diag["image_dimensions"] = {
                        "original": {"width": img.width, "height": img.height}
                    }
                except Exception:
                    pass

            # Step 2: Extract embeddings using DeepFace.represent
            # This detects faces AND extracts embeddings in ONE call (no memory conflict!)
            logger.info(f"[pgvector] Extracting embeddings with ArcFace model")

            detection_start = time.time()
            embedding_results = DeepFace.represent(
                img_path=image_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False,  # Don't fail if no face
                align=True,
                normalization="base"
            )
            if diag:
                diag["timing_ms"]["face_detection"] = round((time.time() - detection_start) * 1000)

            logger.info(f"[pgvector] Found {len(embedding_results)} faces with embeddings")

            if not embedding_results or len(embedding_results) == 0:
                # Cleanup temp file
                if os.path.exists(image_path):
                    os.remove(image_path)

                no_faces_result = {
                    'status': 'success',
                    'recognized_persons': [],
                    'all_detected_matches': [],
                    'best_match': None,
                    'mode': 'pgvector',
                    'message': 'No faces detected in image'
                }
                if diag:
                    diag["pipeline_summary"] = {"total_faces_detected": 0, "faces_matched": 0}
                    diag["timing_ms"]["total"] = round((time.time() - start_time) * 1000)
                    no_faces_result["diagnostics"] = diag
                return no_faces_result

            # Step 3: Quality validation and size filtering before DB search
            # Load image for cropping face regions for quality checks
            img_cv2 = cv2.imread(image_path)

            recognized_persons = []
            all_detected_matches = []
            best_match = None
            best_score = -1
            recognition_start = time.time()
            faces_skipped_confidence = 0
            faces_skipped_embedding = 0
            faces_skipped_quality = 0
            faces_skipped_size = 0

            # First pass: validate confidence, embedding, and quality; collect face areas
            valid_faces = []
            for face_idx, face_data in enumerate(embedding_results):
                embedding = np.array(face_data.get("embedding", []), dtype=np.float32)

                if len(embedding) != 512:
                    logger.error(f"[pgvector] Invalid embedding size for face {face_idx}: {len(embedding)}")
                    faces_skipped_embedding += 1
                    if diag:
                        diag["rejected_faces"].append({
                            "face_index": face_idx + 1,
                            "reason": "invalid_embedding_size",
                            "detection_confidence": face_data.get('face_confidence', None),
                            "quality_metrics": {"embedding_size": len(embedding)}
                        })
                    continue

                facial_area = face_data.get('facial_area', {})
                confidence = face_data.get('face_confidence', 1.0)

                if confidence < 0.65:
                    logger.info(f"[pgvector] Face {face_idx} low confidence ({confidence:.3f}) - skipping")
                    faces_skipped_confidence += 1
                    if diag:
                        diag["rejected_faces"].append({
                            "face_index": face_idx + 1,
                            "reason": "low_detection_confidence",
                            "detection_confidence": round(confidence, 4),
                            "quality_metrics": {"confidence_gate": 0.65}
                        })
                    continue

                # Quality validation: crop face region and check blur/contrast/brightness
                quality_metrics = {}
                if img_cv2 is not None:
                    x = facial_area.get('x', 0)
                    y = facial_area.get('y', 0)
                    w = facial_area.get('w', 0)
                    h = facial_area.get('h', 0)
                    if w > 0 and h > 0:
                        # Ensure bounds are within image
                        ih, iw = img_cv2.shape[:2]
                        x1 = max(0, x)
                        y1 = max(0, y)
                        x2 = min(iw, x + w)
                        y2 = min(ih, y + h)
                        cropped = img_cv2[y1:y2, x1:x2]

                        if cropped.size > 0:
                            is_valid, quality_metrics = PgVectorRecognitionService.validate_face_quality(
                                cropped, face_idx, source_type
                            )
                            if not is_valid:
                                faces_skipped_quality += 1
                                if diag:
                                    diag["rejected_faces"].append({
                                        "face_index": face_idx + 1,
                                        "reason": quality_metrics.get("reject_reason", "quality_check_failed"),
                                        "detection_confidence": round(confidence, 4),
                                        "quality_metrics": quality_metrics
                                    })
                                continue

                face_w = facial_area.get('w', 0)
                face_h = facial_area.get('h', 0)
                valid_faces.append({
                    'index': face_idx,
                    'face_data': face_data,
                    'embedding': embedding,
                    'facial_area': facial_area,
                    'confidence': confidence,
                    'quality_metrics': quality_metrics,
                    'w': face_w,
                    'h': face_h,
                    'area': face_w * face_h
                })

            # Size filtering: remove faces too small relative to the largest
            valid_after_quality = len(valid_faces)
            if len(valid_faces) > 1:
                keep_indices = PgVectorRecognitionService.filter_faces_by_size(valid_faces, size_threshold=0.30)
                before_count = len(valid_faces)
                filtered_out = [vf for vf in valid_faces if vf['index'] not in keep_indices]
                valid_faces = [vf for vf in valid_faces if vf['index'] in keep_indices]
                faces_skipped_size = before_count - len(valid_faces)

                if diag:
                    for vf in filtered_out:
                        diag["rejected_faces"].append({
                            "face_index": vf['index'] + 1,
                            "reason": "too_small",
                            "detection_confidence": round(vf['confidence'], 4),
                            "quality_metrics": {
                                "face_width": vf['w'],
                                "face_height": vf['h'],
                                "face_area": vf['area']
                            }
                        })

            logger.info(f"[pgvector] After filtering: {len(valid_faces)} faces (skipped: {faces_skipped_confidence} confidence, {faces_skipped_quality} quality, {faces_skipped_size} size)")

            # Second pass: search database for each valid face
            for vf in valid_faces:
                face_idx = vf['index']
                face_data = vf['face_data']
                embedding = vf['embedding']
                facial_area = vf['facial_area']
                confidence = vf['confidence']

                try:
                    # Search database via SEPARATE PROCESS (avoid TensorFlow/psycopg2 conflicts)
                    logger.info(f"[pgvector] Searching database for face {face_idx}, domain={domain}")

                    import subprocess
                    import json

                    # Prepare request
                    # When diagnostics enabled, use wide threshold to see ALL nearby candidates
                    search_threshold = 1.0 if collect_diagnostics else 0.45
                    search_request = {
                        'embedding': embedding.tolist(),
                        'domain': domain,
                        'threshold': search_threshold,
                        'top_k': 10 if not collect_diagnostics else 20
                    }

                    # Call worker in separate process
                    worker_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        'scripts', 'pgvector_search_worker.py'
                    )

                    try:
                        result = subprocess.run(
                            ['/root/facerecognition-backend/venv/bin/python', worker_path],
                            input=json.dumps(search_request),
                            capture_output=True,
                            text=True,
                            timeout=10
                        )

                        if result.returncode != 0:
                            logger.error(f"[pgvector] Worker failed: {result.stderr}")
                            matches = []
                        else:
                            worker_response = json.loads(result.stdout)
                            if worker_response['status'] == 'success':
                                matches = worker_response['matches']
                            else:
                                logger.error(f"[pgvector] Worker error: {worker_response.get('message')}")
                                matches = []

                    except subprocess.TimeoutExpired:
                        logger.error("[pgvector] Worker timeout")
                        matches = []
                    except Exception as e:
                        logger.error(f"[pgvector] Worker exception: {e}")
                        matches = []

                    logger.info(f"[pgvector] Found {len(matches)} matches for face {face_idx}")

                    # Build diagnostics for this face
                    face_diag = None
                    if diag:
                        # Calculate face area as percentage of image
                        face_area_pct = None
                        orig_dims = diag.get("image_dimensions", {}).get("original", {})
                        if orig_dims.get("width") and orig_dims.get("height"):
                            img_area = orig_dims["width"] * orig_dims["height"]
                            face_area_pct = round((vf['area'] / img_area) * 100, 1) if img_area > 0 else None

                        face_diag = {
                            "face_index": face_idx + 1,
                            "detection_confidence": round(confidence, 4),
                            "facial_area": facial_area,
                            "face_area_percent": face_area_pct,
                            "quality_metrics": vf.get('quality_metrics', {}),
                            "status": "passed",
                            "top_matches": []
                        }

                    # Split results: actual matches (below threshold) vs all candidates
                    actual_threshold = 0.45
                    actual_matches = [m for m in matches if m['distance'] < actual_threshold]

                    # Collect diagnostics from ALL candidates (matches + non-matches)
                    if face_diag and matches:
                        # Deduplicate by person name, keeping best (lowest) distance
                        seen_persons = {}
                        for m in matches:
                            name = m['name']
                            if name not in seen_persons or m['distance'] < seen_persons[name]['distance']:
                                seen_persons[name] = m
                        unique_by_person = sorted(seen_persons.values(), key=lambda x: x['distance'])

                        for m in unique_by_person[:10]:
                            entry = {
                                "person": m['name'],
                                "distance": round(m['distance'], 4),
                                "confidence_pct": round(m['confidence'] * 100, 1),
                                "reference_image": m.get('image_path', ''),
                                "matched": m['distance'] < actual_threshold
                            }
                            face_diag["top_matches"].append(entry)

                        # Near-misses: persons close to but above threshold
                        near_miss_ceiling = actual_threshold * 1.4  # 40% above threshold
                        for m in unique_by_person:
                            if m['distance'] >= actual_threshold and m['distance'] <= near_miss_ceiling:
                                diag["near_misses"].append({
                                    "person": m['name'],
                                    "min_distance": round(m['distance'], 4),
                                    "confidence_pct": round(m['confidence'] * 100, 1),
                                    "distance_above_threshold": round(m['distance'] - actual_threshold, 4)
                                })

                    if actual_matches:
                        # Get top match
                        top_match = actual_matches[0]
                        person_name = top_match['name']
                        distance = top_match['distance']
                        confidence_pct = top_match['confidence'] * 100

                        # Add to recognized persons (PKL format)
                        recognized_persons.append({
                            'name': person_name,
                            'face_coordinates': {
                                'x': facial_area.get('x', 0),
                                'y': facial_area.get('y', 0),
                                'w': facial_area.get('w', 0),
                                'h': facial_area.get('h', 0)
                            }
                        })

                        # Add to all_detected_matches (PKL format)
                        all_detected_matches.append({
                            'person_name': person_name,
                            'metrics': {
                                'confidence_percentage': confidence_pct,
                                'occurrences': 1,
                                'min_distance': distance,
                                'weighted_score': confidence_pct
                            }
                        })

                        # Track best match
                        if confidence_pct > best_score:
                            best_score = confidence_pct
                            best_match = {
                                'person_name': person_name,
                                'distance': distance,
                                'confidence': confidence_pct
                            }
                    else:
                        if face_diag:
                            face_diag["status"] = "no_db_match"

                    if face_diag:
                        diag["per_face_details"].append(face_diag)

                except Exception as e:
                    logger.error(f"[pgvector] Error processing face {face_idx}: {e}", exc_info=True)
                    continue

            # Cleanup temp file
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    logger.info(f"[pgvector] Cleaned up temp file: {image_path}")
            except Exception as e:
                logger.warning(f"[pgvector] Failed to cleanup {image_path}: {e}")

            elapsed = time.time() - start_time
            logger.info(f"[pgvector] Recognition completed in {elapsed:.2f}s, found {len(recognized_persons)} persons")

            result = {
                'status': 'success',
                'recognized_persons': recognized_persons,
                'all_detected_matches': all_detected_matches,
                'best_match': best_match,
                'mode': 'pgvector',
                'elapsed_time': elapsed
            }

            if diag:
                diag["pipeline_summary"] = {
                    "total_faces_detected": len(embedding_results),
                    "valid_after_confidence": len(embedding_results) - faces_skipped_confidence - faces_skipped_embedding,
                    "valid_after_quality": valid_after_quality,
                    "valid_after_size_filter": len(valid_faces),
                    "faces_matched": len(recognized_persons)
                }
                diag["timing_ms"]["recognition_search"] = round((time.time() - recognition_start) * 1000)
                diag["timing_ms"]["total"] = round(elapsed * 1000)
                # Deduplicate near-misses by person name
                seen_near = set()
                unique_near = []
                for nm in diag["near_misses"]:
                    if nm["person"] not in seen_near:
                        seen_near.add(nm["person"])
                        unique_near.append(nm)
                diag["near_misses"] = sorted(unique_near, key=lambda x: x["min_distance"])
                result["diagnostics"] = diag

            return result

        except Exception as e:
            logger.error(f"[pgvector] Recognition failed: {e}", exc_info=True)

            # Cleanup temp file on error
            try:
                if 'image_path' in locals() and os.path.exists(image_path):
                    os.remove(image_path)
            except:
                pass

            # Return error in PKL-compatible format
            return {
                'status': 'error',
                'message': str(e),
                'recognized_persons': [],
                'all_detected_matches': [],
                'best_match': None,
                'mode': 'pgvector (error)'
            }
