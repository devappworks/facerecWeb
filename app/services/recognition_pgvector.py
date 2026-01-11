"""
pgvector-based recognition service for dual-mode operation.

This module provides face recognition using PostgreSQL + pgvector
as an alternative/complement to PKL-based recognition.
"""

import os
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from deepface import DeepFace

from app.services.vector_db_service import get_vector_db

logger = logging.getLogger(__name__)


class PgVectorRecognitionService:
    """Face recognition using PostgreSQL + pgvector."""

    @staticmethod
    def recognize_face_pgvector(
        image_path: str,
        domain: str,
        threshold: float = 0.30,
        top_k: int = 10,
        model_name: str = "ArcFace"
    ) -> List[Dict[str, Any]]:
        """
        Recognize faces using pgvector similarity search.

        Args:
            image_path: Path to uploaded image
            domain: Domain to search ('serbia', 'slovenia', etc.)
            threshold: Maximum cosine distance (default 0.30 = 70% confidence)
            top_k: Maximum number of results to return
            model_name: DeepFace model name (default: ArcFace)

        Returns:
            List of recognition results, each containing:
            - identity: Person's name
            - distance: Cosine distance (0-1, lower is better)
            - confidence: 1 - distance (0-1, higher is better)
            - source: 'pgvector'
        """
        start_time = time.time()

        try:
            # Step 1: Extract embedding using DeepFace
            logger.info(f"[pgvector] Extracting {model_name} embedding from {image_path}")
            embedding_result = DeepFace.represent(
                img_path=image_path,
                model_name=model_name,
                detector_backend="retinaface",
                enforce_detection=False,
                align=True,
                normalization="base"
            )

            if not embedding_result or len(embedding_result) == 0:
                logger.warning(f"[pgvector] No face detected in {image_path}")
                return []

            # Get first face's embedding
            embedding = np.array(embedding_result[0]["embedding"], dtype=np.float32)

            # Validate embedding dimensions
            if len(embedding) != 512:
                logger.error(f"[pgvector] Invalid embedding size: {len(embedding)} (expected 512)")
                return []

            # Step 2: Search database using pgvector
            logger.info(f"[pgvector] Searching database for domain={domain}, threshold={threshold}")
            db = get_vector_db()
            matches = db.find_matches(
                query_embedding=embedding,
                domain=domain,
                threshold=threshold,
                top_k=top_k
            )

            # Step 3: Format results to match PKL format
            results = []
            for match in matches:
                results.append({
                    'identity': match['name'],
                    'distance': match['distance'],
                    'confidence': match['confidence'],
                    'source': 'pgvector',
                    'image_path': match.get('image_path', ''),
                    'person_id': match.get('person_id')
                })

            elapsed = (time.time() - start_time) * 1000
            logger.info(f"[pgvector] Found {len(results)} matches in {elapsed:.2f}ms")

            return results

        except Exception as e:
            logger.error(f"[pgvector] Recognition failed: {e}", exc_info=True)
            return []


    @staticmethod
    def recognize_face_dual_mode(
        image_path: str,
        domain: str,
        threshold: float = 0.30,
        pkl_results: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Run both PKL and pgvector recognition, compare results.

        Args:
            image_path: Path to uploaded image
            domain: Domain to search
            threshold: Recognition threshold
            pkl_results: Pre-computed PKL results (optional, to avoid double processing)

        Returns:
            Dict with:
            - primary_results: PKL results (or pgvector if PKL unavailable)
            - pgvector_results: pgvector results
            - comparison: Comparison metrics
            - mode: 'dual', 'pkl_only', or 'pgvector_only'
        """
        start_time = time.time()

        # Get pgvector results
        pgvector_results = PgVectorRecognitionService.recognize_face_pgvector(
            image_path=image_path,
            domain=domain,
            threshold=threshold
        )

        # Compare if we have both
        comparison = None
        if pkl_results and len(pkl_results) > 0 and len(pgvector_results) > 0:
            comparison = PgVectorRecognitionService.compare_results(
                pkl_results=pkl_results,
                pgvector_results=pgvector_results
            )

        elapsed = (time.time() - start_time) * 1000

        return {
            'primary_results': pkl_results if pkl_results else pgvector_results,
            'pgvector_results': pgvector_results,
            'comparison': comparison,
            'mode': 'dual' if pkl_results else 'pgvector_only',
            'elapsed_ms': elapsed
        }


    @staticmethod
    def compare_results(pkl_results: List[Dict], pgvector_results: List[Dict]) -> Dict[str, Any]:
        """
        Compare PKL and pgvector results.

        Args:
            pkl_results: Results from PKL search
            pgvector_results: Results from pgvector search

        Returns:
            Comparison metrics
        """
        try:
            # Extract top-1 names
            pkl_top1 = pkl_results[0]['identity'] if len(pkl_results) > 0 else None
            pgvector_top1 = pgvector_results[0]['identity'] if len(pgvector_results) > 0 else None

            # Extract top-5 names
            pkl_top5 = {r['identity'] for r in pkl_results[:5]}
            pgvector_top5 = {r['identity'] for r in pgvector_results[:5]}

            # Calculate overlap
            top5_overlap = len(pkl_top5 & pgvector_top5)
            top5_union = len(pkl_top5 | pgvector_top5)
            overlap_pct = (top5_overlap / top5_union * 100) if top5_union > 0 else 0

            comparison = {
                'top1_match': pkl_top1 == pgvector_top1,
                'pkl_top1': pkl_top1,
                'pgvector_top1': pgvector_top1,
                'top5_overlap': top5_overlap,
                'top5_overlap_pct': overlap_pct,
                'pkl_count': len(pkl_results),
                'pgvector_count': len(pgvector_results)
            }

            # Log discrepancies
            if not comparison['top1_match']:
                logger.warning(
                    f"[DUAL-MODE] Top-1 mismatch: "
                    f"PKL={pkl_top1} vs pgvector={pgvector_top1}"
                )

            if overlap_pct < 80:
                logger.warning(
                    f"[DUAL-MODE] Low top-5 overlap: {overlap_pct:.1f}%"
                )

            return comparison

        except Exception as e:
            logger.error(f"[DUAL-MODE] Comparison failed: {e}")
            return {'error': str(e)}
