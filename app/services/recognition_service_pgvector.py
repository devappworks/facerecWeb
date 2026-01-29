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
from PIL import Image
from io import BytesIO
from deepface import DeepFace
from typing import Dict, List, Any

from app.services.vector_db_service import get_vector_db

logger = logging.getLogger(__name__)


class PgVectorRecognitionService:
    """
    Face recognition using PostgreSQL + pgvector (PKL replacement).

    This service provides the same interface as RecognitionService
    but uses database vector search instead of PKL files.
    """

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
                "threshold_used": 0.50,
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

            # Step 3: Search database for each detected face
            recognized_persons = []
            all_detected_matches = []
            best_match = None
            best_score = -1
            recognition_start = time.time()
            faces_skipped_confidence = 0
            faces_skipped_embedding = 0

            for face_idx, face_data in enumerate(embedding_results):
                try:
                    # Get embedding
                    embedding = np.array(face_data["embedding"], dtype=np.float32)

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

                    # Get facial area (bounding box)
                    facial_area = face_data.get('facial_area', {})

                    # Check confidence if available
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

                    # Search database via SEPARATE PROCESS (avoid TensorFlow/psycopg2 conflicts)
                    logger.info(f"[pgvector] Searching database for face {face_idx}, domain={domain}")

                    import subprocess
                    import json

                    # Prepare request
                    search_request = {
                        'embedding': embedding.tolist(),
                        'domain': domain,
                        'threshold': 0.50,  # Match training threshold (was 0.30, too strict)
                        'top_k': 10
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
                        face_diag = {
                            "face_index": face_idx + 1,
                            "detection_confidence": round(confidence, 4),
                            "facial_area": facial_area,
                            "quality_metrics": {},
                            "status": "passed",
                            "top_matches": []
                        }

                    if matches and len(matches) > 0:
                        # Get top match
                        top_match = matches[0]
                        person_name = top_match['name']
                        distance = top_match['distance']
                        confidence_pct = top_match['confidence'] * 100

                        # Collect per-face top matches for diagnostics
                        if face_diag:
                            for m in matches[:5]:
                                face_diag["top_matches"].append({
                                    "person": m['name'],
                                    "distance": round(m['distance'], 4),
                                    "confidence_pct": round(m['confidence'] * 100, 1),
                                    "reference_image": m.get('image', '')
                                })
                            # Collect near-misses: matches above threshold but close
                            near_miss_threshold = 0.50 * 1.2  # 20% above threshold
                            for m in matches:
                                if m['distance'] >= 0.50 and m['distance'] <= near_miss_threshold:
                                    diag["near_misses"].append({
                                        "person": m['name'],
                                        "min_distance": round(m['distance'], 4),
                                        "confidence_pct": round(m['confidence'] * 100, 1),
                                        "distance_above_threshold": round(m['distance'] - 0.50, 4)
                                    })

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
                    "valid_after_quality": len(embedding_results) - faces_skipped_confidence - faces_skipped_embedding,
                    "valid_after_size_filter": len(embedding_results) - faces_skipped_confidence - faces_skipped_embedding,
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
