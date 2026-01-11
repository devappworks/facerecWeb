"""
Automated Training Service - Clean implementation for celebrity face training.

This service handles the complete automated training pipeline:
1. Query Wikidata for celebrities by country/occupation
2. Build reference images using SERP consensus (3/4 majority voting)
   - Download first 4 SERP images, extract faces
   - Compare all pairs to find matching faces
   - If 3+ agree, use them as references (self-validated)
   - Fall back to P18 (Wikidata) if consensus fails
3. Download additional images from SERP (Google Images)
4. Extract faces and validate against consensus references
5. Store validated faces for training approval
6. Generate galleries for human review

Key design decisions:
- SERP consensus (3/4 majority) is the PRIMARY reference method
- P18 is fallback when consensus fails
- Multiple references = more robust validation
- Original downloads are preserved for audit trail
- Structured metadata tracks the full lineage
"""

import os
import json
import uuid
import shutil
import logging
import requests
import numpy as np
import cv2
from datetime import datetime
from threading import Thread
from typing import Dict, List, Optional, Tuple
from flask import current_app

from app.services.wikidata_service import WikidataService
from app.services.face_processing_service import FaceProcessingService
from app.services.image_rejection_logger import ImageRejectionLogger

logger = logging.getLogger(__name__)


class AutomatedTrainingService:
    """Service for automated celebrity face training."""

    # Storage paths
    STORAGE_BASE = 'storage'

    def __init__(self, domain: str = 'serbia'):
        """
        Initialize service for a specific domain.

        Args:
            domain: Domain code (e.g., 'serbia', 'greece')
        """
        self.domain = domain

        # Path configuration
        self.serp_originals_path = os.path.join(self.STORAGE_BASE, 'serp_originals', domain)
        self.training_path = os.path.join(self.STORAGE_BASE, 'training', domain)
        self.training_pass_path = os.path.join(self.STORAGE_BASE, 'trainingPass', domain)
        self.galleries_path = os.path.join(self.STORAGE_BASE, 'training-galleries', domain)
        self.batches_path = os.path.join(self.STORAGE_BASE, 'training_batches')
        self.checkpoints_path = os.path.join(self.STORAGE_BASE, 'training_checkpoints', domain)

        # Ensure directories exist
        for path in [self.serp_originals_path, self.training_path,
                     self.training_pass_path, self.galleries_path, self.batches_path,
                     self.checkpoints_path]:
            os.makedirs(path, exist_ok=True)

        # API configuration
        self.rapidapi_key = os.getenv(
            'RAPIDAPI_KEY',
            'c3e8343ca0mshe1b719bea5326dbp11db14jsnf52a7fb8ab17'
        )

        # Embedding cache for vectorized operations (in-memory)
        # Maps image path -> embedding vector
        # Cleared after each person to prevent unbounded growth
        self.embedding_cache = {}

    # ==================== Checkpoint System ====================
    # Saves progress after each stage to enable resume after OOM/crash
    # Stages: serp_downloaded -> consensus_built -> validated -> completed

    def _get_checkpoint_path(self, person_name: str) -> str:
        """Get checkpoint file path for a person."""
        safe_name = self._safe_folder_name(person_name)
        return os.path.join(self.checkpoints_path, f'{safe_name}.checkpoint.json')

    def _save_checkpoint(self, person_name: str, stage: str, data: Dict) -> None:
        """
        Save checkpoint for a person at a specific stage.

        Stages:
            - serp_downloaded: SERP images downloaded, ready for consensus
            - consensus_built: References established, ready for validation
            - validated: Validation complete, ready for finalization
            - completed: Fully done (checkpoint will be deleted)
        """
        checkpoint_path = self._get_checkpoint_path(person_name)
        checkpoint = {
            'person_name': person_name,
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        logger.debug(f"Checkpoint saved for {person_name} at stage: {stage}")

    def _load_checkpoint(self, person_name: str) -> Optional[Dict]:
        """Load checkpoint for a person if exists."""
        checkpoint_path = self._get_checkpoint_path(person_name)
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                logger.info(f"Found checkpoint for {person_name} at stage: {checkpoint.get('stage')}")
                return checkpoint
            except Exception as e:
                logger.warning(f"Failed to load checkpoint for {person_name}: {e}")
        return None

    def _clear_checkpoint(self, person_name: str) -> None:
        """Clear checkpoint after successful completion."""
        checkpoint_path = self._get_checkpoint_path(person_name)
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                logger.debug(f"Checkpoint cleared for {person_name}")
            except Exception as e:
                logger.warning(f"Failed to clear checkpoint for {person_name}: {e}")

    def get_incomplete_persons(self) -> List[Dict]:
        """
        Get list of persons with incomplete checkpoints (interrupted training).

        Returns:
            List of checkpoint dicts for persons that need resume
        """
        incomplete = []
        if not os.path.exists(self.checkpoints_path):
            return incomplete

        for filename in os.listdir(self.checkpoints_path):
            if filename.endswith('.checkpoint.json'):
                filepath = os.path.join(self.checkpoints_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        checkpoint = json.load(f)
                    if checkpoint.get('stage') != 'completed':
                        incomplete.append(checkpoint)
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint {filename}: {e}")

        # Sort by timestamp (oldest first)
        incomplete.sort(key=lambda x: x.get('timestamp', ''))
        return incomplete

    # ==================== End Checkpoint System ====================

    # ==================== Vectorized Embedding & Distance Computation ====================
    # Replaces pairwise O(n²) DeepFace.verify calls with fast vectorized numpy operations
    # ~140x speedup: 14 minutes -> 6 seconds for 20-image consensus building

    def _extract_embedding(self, image_path: str, batch_id: str = "", use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Extract face embedding from image using ArcFace model (with caching).

        Args:
            image_path: Path to image file
            batch_id: Batch identifier for logging
            use_cache: Whether to use/update embedding cache (default True)

        Returns:
            Embedding vector (128-dim for ArcFace) or None if extraction fails
        """
        # Check cache first
        if use_cache and image_path in self.embedding_cache:
            return self.embedding_cache[image_path]

        try:
            from deepface import DeepFace
            result = DeepFace.represent(
                img_path=image_path,
                model_name='ArcFace',
                detector_backend='retinaface',
                enforce_detection=False
            )
            if result and len(result) > 0:
                embedding = np.array(result[0]['embedding'], dtype=np.float32)
                # Cache the embedding
                if use_cache:
                    self.embedding_cache[image_path] = embedding
                return embedding
            return None
        except Exception as e:
            if batch_id:
                logger.debug(f"[{batch_id}] Failed to extract embedding from {image_path}: {e}")
            return None

    def _clear_embedding_cache(self) -> None:
        """Clear embedding cache to free memory after processing each person."""
        self.embedding_cache.clear()

    def _batch_cosine_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine distances between all embeddings (vectorized using numpy).

        Vectorized computation using numpy broadcasting (no scipy dependency).
        Much faster than calling DeepFace.verify for each pair.

        Args:
            embeddings: Array of shape (n, 128) where n is number of faces

        Returns:
            Distance matrix of shape (n, n) where distances[i,j] is cosine distance
        """
        if len(embeddings) == 0:
            return np.array([])
        if len(embeddings) == 1:
            return np.array([[0.0]])

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero

        # Compute cosine similarity: dot product of normalized vectors
        # Shape: (n, n)
        similarity_matrix = np.dot(normalized, normalized.T)

        # Cosine distance = 1 - similarity
        distance_matrix = 1.0 - similarity_matrix

        # Clamp to [0, 1] (to handle numerical errors)
        distance_matrix = np.clip(distance_matrix, 0.0, 2.0)

        return distance_matrix

    def _cosine_distance_single(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute single cosine distance between two embeddings (for fallback)."""
        if emb1 is None or emb2 is None:
            return 1.0
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        similarity = dot / (norm1 * norm2)
        return max(0.0, 1.0 - similarity)  # Clamp to [0, 1]

    # ==================== End Vectorized Embedding & Distance Computation ====================

    def start_training_batch(
        self,
        country: str,
        occupation: str,
        limit: int = 10,
        images_per_person: int = 50
    ) -> Dict:
        """
        Start a new training batch for celebrities from Wikidata.

        Args:
            country: Country code (e.g., 'serbia')
            occupation: Occupation code (e.g., 'actor')
            limit: Maximum number of celebrities to process
            images_per_person: Target images per person

        Returns:
            Dict with batch_id and initial status
        """
        batch_id = str(uuid.uuid4())[:8]

        logger.info(f"[Batch {batch_id}] Starting training batch: {country} {occupation}s (limit: {limit})")

        # Query Wikidata for celebrities
        celebrities = WikidataService.query_celebrities(country, occupation, limit)

        if not celebrities:
            return {
                'success': False,
                'error': 'No celebrities found in Wikidata',
                'batch_id': batch_id
            }

        logger.info(f"[Batch {batch_id}] Found {len(celebrities)} celebrities")

        # Create batch status file
        batch_status = {
            'batch_id': batch_id,
            'domain': self.domain,
            'country': country,
            'occupation': occupation,
            'created_at': datetime.now().isoformat(),
            'status': 'processing',
            'total_celebrities': len(celebrities),
            'processed': 0,
            'people': []
        }

        # Initialize people list
        for celeb in celebrities:
            batch_status['people'].append({
                'wikidata_id': celeb.get('wikidata_id'),
                'name': celeb.get('full_name'),
                'status': 'pending',
                'p18_url': celeb.get('image_url'),
                'images_found': 0,
                'images_accepted': 0,
                'gallery_url': None
            })

        self._save_batch_status(batch_id, batch_status)

        # Get app context for background thread
        app_context = current_app._get_current_object().app_context()

        # Start background processing
        thread = Thread(
            target=self._process_batch_background,
            args=(batch_id, celebrities, images_per_person, app_context)
        )
        thread.daemon = True
        thread.start()

        return {
            'success': True,
            'batch_id': batch_id,
            'message': f'Started processing {len(celebrities)} celebrities',
            'status_url': f'/api/training/batch/{batch_id}/status'
        }

    def _process_batch_background(
        self,
        batch_id: str,
        celebrities: List[Dict],
        images_per_person: int,
        app_context
    ):
        """
        Background thread to process all celebrities in a batch.
        """
        with app_context:
            try:
                batch_status = self._load_batch_status(batch_id)

                for i, celeb in enumerate(celebrities):
                    person_name = celeb.get('full_name', 'Unknown')
                    wikidata_id = celeb.get('wikidata_id')
                    p18_url = celeb.get('image_url')
                    p18_urls = celeb.get('image_urls', [])  # Multiple P18 references

                    num_refs = len(p18_urls) if p18_urls else (1 if p18_url else 0)
                    logger.info(f"[Batch {batch_id}] Processing {i+1}/{len(celebrities)}: {person_name} ({num_refs} P18 references)")

                    # Update status
                    batch_status['people'][i]['status'] = 'processing'
                    self._save_batch_status(batch_id, batch_status)

                    try:
                        # Process this person with all P18 references
                        result = self._process_person(
                            batch_id=batch_id,
                            person_name=person_name,
                            wikidata_id=wikidata_id,
                            p18_url=p18_url,
                            images_per_person=images_per_person,
                            p18_urls=p18_urls if p18_urls else None
                        )

                        # Update status
                        batch_status['people'][i]['status'] = 'completed'
                        batch_status['people'][i]['images_found'] = result.get('images_found', 0)
                        batch_status['people'][i]['images_accepted'] = result.get('images_accepted', 0)
                        batch_status['people'][i]['gallery_url'] = result.get('gallery_url')
                        batch_status['processed'] = i + 1

                    except Exception as e:
                        logger.error(f"[Batch {batch_id}] Error processing {person_name}: {str(e)}")
                        batch_status['people'][i]['status'] = 'failed'
                        batch_status['people'][i]['error'] = str(e)
                        batch_status['processed'] = i + 1

                    self._save_batch_status(batch_id, batch_status)

                # Mark batch as complete
                batch_status['status'] = 'completed'
                batch_status['completed_at'] = datetime.now().isoformat()
                self._save_batch_status(batch_id, batch_status)

                logger.info(f"[Batch {batch_id}] Batch processing complete")

            except Exception as e:
                logger.error(f"[Batch {batch_id}] Fatal error in batch processing: {str(e)}")
                try:
                    batch_status = self._load_batch_status(batch_id)
                    batch_status['status'] = 'failed'
                    batch_status['error'] = str(e)
                    self._save_batch_status(batch_id, batch_status)
                except:
                    pass

    def _process_person(
        self,
        batch_id: str,
        person_name: str,
        wikidata_id: str,
        p18_url: str,
        images_per_person: int,
        search_name: str = None,
        p18_urls: List[str] = None
    ) -> Dict:
        """
        Process a single person: download images, extract faces, validate.
        Supports checkpoint-based resume for crash recovery.

        Args:
            batch_id: Batch identifier
            person_name: Full name of person (used for folder names)
            wikidata_id: Wikidata entity ID
            p18_url: URL of Wikidata P18 image (backward compatibility)
            images_per_person: Target number of images
            search_name: International name to use for SERP search (if different from person_name)
            p18_urls: List of all P18 URLs for multi-reference validation

        Returns:
            Dict with processing results
        """
        import gc

        # Use search_name for SERP queries if provided, otherwise use person_name
        serp_search_name = search_name or person_name
        # Create safe folder name
        folder_name = self._safe_folder_name(person_name)
        person_batch_id = f"{batch_id}_{folder_name}"

        # Create directories
        originals_dir = os.path.join(self.serp_originals_path, folder_name, person_batch_id)
        faces_dir = os.path.join(self.training_pass_path, folder_name)
        os.makedirs(originals_dir, exist_ok=True)
        os.makedirs(faces_dir, exist_ok=True)

        # Check for existing checkpoint to resume from
        checkpoint = self._load_checkpoint(person_name)
        resume_stage = checkpoint.get('stage') if checkpoint else None
        checkpoint_data = checkpoint.get('data', {}) if checkpoint else {}

        # Initialize or restore metadata
        if resume_stage and checkpoint_data.get('metadata'):
            metadata = checkpoint_data['metadata']
            logger.info(f"[{person_batch_id}] Resuming from checkpoint stage: {resume_stage}")
        else:
            metadata = {
                'person_name': person_name,
                'wikidata_id': wikidata_id,
                'batch_id': person_batch_id,
                'created_at': datetime.now().isoformat(),
                'reference': None,
                'images': {}
            }

        # Variables for tracking
        reference = None
        serp_images = []
        images_found = 0
        images_accepted = 0

        # ============================================================
        # Step 1: Build SERP consensus references (skip if resuming past this)
        # ============================================================
        if resume_stage in ['consensus_built', 'serp_downloaded', 'validated']:
            # Restore reference from checkpoint
            reference = checkpoint_data.get('reference')
            logger.info(f"[{person_batch_id}] Restored reference from checkpoint")
        else:
            logger.info(f"[{person_batch_id}] Building SERP consensus references...")
            reference = self._build_serp_consensus_references(
                person_name=serp_search_name,
                folder_name=folder_name,
                originals_dir=originals_dir,
                faces_dir=faces_dir,
                batch_id=person_batch_id
            )

            if not reference:
                logger.warning(f"[{person_batch_id}] Failed to build SERP consensus references, skipping person")
                self._clear_checkpoint(person_name)
                return {'images_found': 0, 'images_accepted': 0, 'gallery_url': None}

            logger.info(f"[{person_batch_id}] Using {reference.get('source', 'unknown')} reference with {len(reference.get('reference_paths', []))} images")
            metadata['reference'] = reference

            # Save checkpoint after consensus built
            self._save_checkpoint(person_name, 'consensus_built', {
                'metadata': metadata,
                'reference': reference,
                'originals_dir': originals_dir,
                'faces_dir': faces_dir,
                'serp_search_name': serp_search_name,
                'images_per_person': images_per_person
            })

        # ============================================================
        # Step 2 & 3: INCREMENTAL download and validation
        # Download in batches, validate, stop when target reached
        # This saves time by not downloading unnecessary images
        # ============================================================

        # Get reference info
        reference_paths = reference.get('reference_paths', [reference['face_path']])
        num_references = len(reference_paths)
        consensus_urls = reference.get('consensus_urls', set())

        # ============================================================
        # Copy consensus images to main training folder as accepted images
        # These are already validated via clustering, no need to re-validate
        # ============================================================
        consensus_training_images = 0
        if reference.get('source') == 'serp_consensus' and reference.get('all_references'):
            logger.info(f"[{person_batch_id}] Copying {len(reference['all_references'])} consensus images to training folder...")
            for idx, ref in enumerate(reference['all_references']):
                src_path = ref.get('face_path')
                if src_path and os.path.exists(src_path):
                    # Create new filename for training (001, 002, etc.)
                    ext = os.path.splitext(src_path)[1]
                    new_filename = f"{folder_name}_{idx+1:03d}{ext}"
                    dest_path = os.path.join(faces_dir, new_filename)

                    # Copy (don't move - keep refs for validation)
                    shutil.copy2(src_path, dest_path)
                    consensus_training_images += 1
                    logger.debug(f"[{person_batch_id}] Copied consensus image {idx+1}: {new_filename}")

            logger.info(f"[{person_batch_id}] ✓ Added {consensus_training_images} consensus images to training set")

        # Count consensus images as already accepted
        images_accepted = consensus_training_images
        images_found = consensus_training_images

        # Target: 30-40 accepted images (configurable via images_per_person)
        target_accepted = images_per_person

        # Batch settings
        INITIAL_BATCH = 50  # First batch size
        RETRY_BATCH = 30    # Additional batch size if needed
        MAX_TOTAL = 150     # Maximum total images to try

        validation_stats = {
            'total_processed': 0,
            'accepted': consensus_training_images,  # Include consensus images in stats
            'rejected_reasons': {}
        }

        # Check for resume
        if resume_stage in ['serp_downloaded', 'validated']:
            serp_images = checkpoint_data.get('serp_images', [])
            if not serp_images:
                serp_images = self._reconstruct_serp_images(originals_dir)
            logger.info(f"[{person_batch_id}] Restored {len(serp_images)} SERP images from checkpoint")

            # Process restored images
            for idx, img_info in enumerate(serp_images, 1):
                if images_accepted >= target_accepted:
                    break

                validation_stats['total_processed'] += 1
                result = self._process_and_validate_image(
                    image_path=img_info['path'],
                    reference_path=reference['face_path'],
                    folder_name=folder_name,
                    faces_dir=faces_dir,
                    batch_id=person_batch_id,
                    reference_paths=reference_paths
                )

                img_filename = os.path.basename(img_info['path'])
                metadata['images'][img_filename] = {
                    'sequence': img_info.get('sequence'),
                    'source_url': img_info.get('source_url'),
                    'extraction': result['extraction'],
                    'validation': result.get('validation'),
                    'archived_at': datetime.now().isoformat()
                }

                if result['extraction']['status'] == 'accepted':
                    images_accepted += 1
                    validation_stats['accepted'] += 1
                    logger.info(f"[{person_batch_id}] ✓ Accepted: {img_filename} ({images_accepted}/{target_accepted})")
                else:
                    rejection_reason = result['extraction'].get('reason', 'unknown')
                    validation_stats['rejected_reasons'][rejection_reason] = validation_stats['rejected_reasons'].get(rejection_reason, 0) + 1

                if idx % 10 == 0:
                    gc.collect()

            images_found = len(serp_images) + consensus_training_images
        else:
            # Fresh download - use incremental batching
            logger.info(f"[{person_batch_id}] Starting incremental download (target: {target_accepted} accepted images, already have {consensus_training_images} from consensus)")

            total_downloaded = 0
            batch_num = 0
            all_serp_images = []

            while images_accepted < target_accepted and total_downloaded < MAX_TOTAL:
                batch_num += 1
                batch_size = INITIAL_BATCH if batch_num == 1 else RETRY_BATCH

                logger.info(f"[{person_batch_id}] Batch {batch_num}: Downloading {batch_size} images...")

                # Download batch - skip URLs already used for consensus
                batch_images = self._download_serp_images(
                    person_name=serp_search_name,
                    folder_name=folder_name,
                    originals_dir=originals_dir,
                    max_images=batch_size,
                    batch_id=person_batch_id,
                    skip_urls=consensus_urls
                )

                if not batch_images:
                    logger.warning(f"[{person_batch_id}] No more images available from SERP")
                    break

                total_downloaded += len(batch_images)
                all_serp_images.extend(batch_images)

                # Validate this batch
                batch_accepted = 0
                for idx, img_info in enumerate(batch_images, 1):
                    if images_accepted >= target_accepted:
                        logger.info(f"[{person_batch_id}] Target reached! Stopping validation.")
                        break

                    validation_stats['total_processed'] += 1
                    result = self._process_and_validate_image(
                        image_path=img_info['path'],
                        reference_path=reference['face_path'],
                        folder_name=folder_name,
                        faces_dir=faces_dir,
                        batch_id=person_batch_id,
                        reference_paths=reference_paths
                    )

                    img_filename = os.path.basename(img_info['path'])
                    metadata['images'][img_filename] = {
                        'sequence': img_info.get('sequence'),
                        'source_url': img_info.get('source_url'),
                        'extraction': result['extraction'],
                        'validation': result.get('validation'),
                        'archived_at': datetime.now().isoformat()
                    }

                    if result['extraction']['status'] == 'accepted':
                        images_accepted += 1
                        batch_accepted += 1
                        validation_stats['accepted'] += 1
                        logger.info(f"[{person_batch_id}] ✓ Accepted: {img_filename} ({images_accepted}/{target_accepted})")
                    else:
                        rejection_reason = result['extraction'].get('reason', 'unknown')
                        validation_stats['rejected_reasons'][rejection_reason] = validation_stats['rejected_reasons'].get(rejection_reason, 0) + 1

                    if idx % 10 == 0:
                        gc.collect()

                # Log batch summary
                acceptance_rate = batch_accepted / len(batch_images) * 100 if batch_images else 0
                logger.info(f"[{person_batch_id}] Batch {batch_num} complete: {batch_accepted}/{len(batch_images)} accepted ({acceptance_rate:.0f}%), total: {images_accepted}/{target_accepted}")

                # Check if we should continue
                if images_accepted >= target_accepted:
                    break

                # Estimate if more batches are needed
                if batch_accepted == 0:
                    logger.warning(f"[{person_batch_id}] No images accepted in batch, stopping")
                    break

            images_found = total_downloaded + consensus_training_images
            serp_images = all_serp_images

            # Save checkpoint
            self._save_checkpoint(person_name, 'serp_downloaded', {
                'metadata': metadata,
                'reference': reference,
                'serp_images': serp_images,
                'originals_dir': originals_dir,
                'faces_dir': faces_dir,
                'serp_search_name': serp_search_name,
                'images_per_person': images_per_person
            })

        # Save metadata (convert numpy types to native Python types for JSON serialization)
        metadata_path = os.path.join(originals_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # ============================================================
        # Step 4: Generate gallery and finalize
        # ============================================================
        gallery_url = self._generate_gallery(
            folder_name=folder_name,
            batch_id=person_batch_id,
            metadata=metadata
        )

        # Log summary
        logger.info(f"[{person_batch_id}] ============================================================")
        logger.info(f"[{person_batch_id}] Training complete: {images_accepted}/{images_per_person} target ({images_accepted}/{images_found} total)")
        logger.info(f"[{person_batch_id}] Validation stats: {validation_stats['accepted']} accepted, {validation_stats['total_processed'] - validation_stats['accepted']} rejected")
        if validation_stats['rejected_reasons']:
            logger.info(f"[{person_batch_id}] Rejection reasons: {validation_stats['rejected_reasons']}")
        logger.info(f"[{person_batch_id}] Gallery: {gallery_url}")
        logger.info(f"[{person_batch_id}] ============================================================")

        # Clear checkpoint on successful completion
        self._clear_checkpoint(person_name)

        # Clear embedding cache to free memory after processing this person
        self._clear_embedding_cache()

        return {
            'images_found': images_found,
            'images_accepted': images_accepted,
            'gallery_url': gallery_url
        }

    def _reconstruct_serp_images(self, originals_dir: str) -> List[Dict]:
        """Reconstruct serp_images list from files on disk after crash."""
        serp_images = []
        if not os.path.exists(originals_dir):
            return serp_images

        for filename in sorted(os.listdir(originals_dir)):
            if filename.startswith('download_') and filename.endswith(('.jpg', '.png', '.webp')):
                filepath = os.path.join(originals_dir, filename)
                # Extract sequence number from filename like download_002.jpg
                try:
                    seq = int(filename.split('_')[1].split('.')[0])
                except:
                    seq = len(serp_images) + 1
                serp_images.append({
                    'path': filepath,
                    'sequence': seq,
                    'source_url': None  # Lost after crash, but not needed for validation
                })

        logger.info(f"Reconstructed {len(serp_images)} SERP images from disk")
        return serp_images

    def _download_and_process_p18(
        self,
        p18_url: str,
        person_name: str,
        folder_name: str,
        originals_dir: str,
        faces_dir: str,
        batch_id: str,
        p18_urls: List[str] = None
    ) -> Optional[Dict]:
        """
        Download Wikidata P18 image(s) and extract face(s) as reference(s).
        Supports multiple P18 images for better validation coverage.

        Args:
            p18_url: Primary P18 URL (backward compatibility)
            p18_urls: List of all P18 URLs (preferred if provided)

        Returns:
            Dict with reference info including list of all valid reference paths
        """
        # Build list of URLs to process
        urls_to_process = []
        if p18_urls and len(p18_urls) > 0:
            urls_to_process = p18_urls
        elif p18_url:
            urls_to_process = [p18_url]

        if not urls_to_process:
            logger.warning(f"[{batch_id}] No P18 URL(s) provided")
            return None

        references = []
        primary_reference = None

        for idx, url in enumerate(urls_to_process):
            try:
                # Download P18 image
                response = requests.get(url, timeout=30, headers={
                    'User-Agent': 'FaceRecognitionTrainingApp/1.0'
                })

                if response.status_code != 200:
                    logger.warning(f"[{batch_id}] Failed to download P18 #{idx+1}: HTTP {response.status_code}")
                    continue

                # Validate that content is actually an image
                if not self._is_valid_image_content(response.content, batch_id):
                    logger.warning(f"[{batch_id}] P18 #{idx+1} is not a valid image (possibly HTML redirect)")
                    continue

                # Save original with unique sequence
                ext = self._get_extension(url)
                seq = f"{idx+1:03d}"
                original_path = os.path.join(originals_dir, f"p18_{seq}{ext}")
                with open(original_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"[{batch_id}] P18 #{idx+1} downloaded: {original_path}")

                # Extract face - save to a reference subfolder to avoid collision
                ref_faces_dir = os.path.join(faces_dir, '_references')
                os.makedirs(ref_faces_dir, exist_ok=True)
                face_filename = f"{folder_name}_ref_{seq}{ext}"
                face_path = os.path.join(ref_faces_dir, face_filename)

                face_extracted = self._extract_face(original_path, face_path, batch_id)

                if not face_extracted:
                    logger.warning(f"[{batch_id}] Failed to extract face from P18 #{idx+1}")
                    continue

                ref_info = {
                    'source': 'wikidata_p18',
                    'source_url': url,
                    'original_path': original_path,
                    'face_path': face_path,
                    'face_filename': face_filename,
                    'is_reference': True,
                    'index': idx
                }
                references.append(ref_info)

                # First successful reference is the primary
                if primary_reference is None:
                    primary_reference = ref_info

            except Exception as e:
                logger.warning(f"[{batch_id}] Error processing P18 #{idx+1}: {str(e)}")
                continue

        if not references:
            logger.warning(f"[{batch_id}] Failed to extract face from any of {len(urls_to_process)} P18 images")
            return None

        logger.info(f"[{batch_id}] Successfully processed {len(references)}/{len(urls_to_process)} P18 references")

        # Return primary reference with all reference paths for multi-reference validation
        result = primary_reference.copy()
        result['all_references'] = references
        result['reference_paths'] = [ref['face_path'] for ref in references]
        return result

    def _count_faces_in_image(self, image_path: str) -> int:
        """
        Count REAL faces in an image for filtering multi-person photos.

        Strategy: Count ALL real faces (confidence > 0, has eyes) to detect multi-person photos.
        We reject any image with multiple faces, even if some are small.
        The MIN_FACE_SIZE is only used to ensure at least one face is large enough for training.

        Returns:
            Number of real faces (>=20px), or:
            -1 on error
            -2 if exactly 1 face but too small for training (<50px)
        """
        try:
            from deepface import DeepFace

            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='retinaface',
                enforce_detection=False
            )

            if not faces:
                return 0

            MIN_FACE_SIZE = 50  # Minimum face size for training quality
            MIN_DETECT_SIZE = 20  # Minimum size to count as a real face (not noise)

            real_faces = 0
            has_large_face = False

            for face in faces:
                confidence = face.get('confidence', 0)
                facial_area = face.get('facial_area', {})
                left_eye = facial_area.get('left_eye')
                right_eye = facial_area.get('right_eye')
                width = facial_area.get('w', 0)
                height = facial_area.get('h', 0)

                # Skip fake detections (confidence=0, no eyes, tiny noise)
                if confidence <= 0 or left_eye is None or right_eye is None:
                    continue
                if width < MIN_DETECT_SIZE or height < MIN_DETECT_SIZE:
                    continue

                # This is a real face - count it
                real_faces += 1

                # Check if it's large enough for training
                if width >= MIN_FACE_SIZE and height >= MIN_FACE_SIZE:
                    has_large_face = True

            # Return -2 if we have exactly 1 face but it's too small for training
            if real_faces == 1 and not has_large_face:
                return -2

            return real_faces

        except Exception as e:
            logger.debug(f"Face count error: {str(e)}")
            return -1

    def _download_serp_images(
        self,
        person_name: str,
        folder_name: str,
        originals_dir: str,
        max_images: int,
        batch_id: str,
        skip_urls: set = None,
        start_sequence: int = None
    ) -> List[Dict]:
        """
        Download images from Google Images via RapidAPI with face filtering.

        Only keeps images with exactly 1 face of sufficient size for training.
        Retries with different query variations if not enough valid images found.

        Args:
            skip_urls: Set of URLs to skip (e.g., already used for consensus)
            start_sequence: Starting sequence number for filenames (default: auto-detect from existing files)

        Returns:
            List of dicts with image paths and metadata
        """
        downloaded = []
        skip_urls = skip_urls or set()

        # Use domain-specific region for better results
        region_map = {
            'serbia': 'rs',
            'croatia': 'hr',
            'bosnia': 'ba',
            'montenegro': 'me',
            'slovenia': 'si',
            'macedonia': 'mk'
        }
        region = region_map.get(self.domain.lower(), 'us')

        # Query variations to try if first attempt doesn't get enough images
        query_variations = [
            person_name,
            f"{person_name} portrait",
            f"{person_name} photo",
        ]

        # Determine starting sequence number
        if start_sequence is not None:
            sequence_counter = start_sequence
        else:
            # Auto-detect from existing files in originals_dir
            sequence_counter = 2  # Default start at 2 since P18 is 1
            if os.path.exists(originals_dir):
                for fname in os.listdir(originals_dir):
                    if fname.startswith('download_'):
                        try:
                            seq = int(fname.split('_')[1].split('.')[0])
                            sequence_counter = max(sequence_counter, seq + 1)
                        except:
                            pass

        rejected_faces = 0
        failed_downloads = 0

        for query_idx, query in enumerate(query_variations):
            if len(downloaded) >= max_images:
                break

            try:
                # Query SERP API - request 5x more to account for face filtering + failures
                url = "https://real-time-image-search.p.rapidapi.com/search"
                request_limit = min(max_images * 5, 200)

                querystring = {
                    "query": query,
                    "limit": str(request_limit),
                    "size": "large",
                    "type": "photo",
                    "region": region
                }
                headers = {
                    "x-rapidapi-key": self.rapidapi_key,
                    "x-rapidapi-host": "real-time-image-search.p.rapidapi.com"
                }

                response = requests.get(url, headers=headers, params=querystring, timeout=30)

                if response.status_code != 200:
                    logger.error(f"[{batch_id}] SERP API error: HTTP {response.status_code}")
                    continue

                data = response.json()
                results = data.get('data', [])

                logger.info(f"[{batch_id}] SERP query '{query}' returned {len(results)} results (region={region})")

                # Download images until we have max_images successful downloads
                for i, item in enumerate(results):
                    if len(downloaded) >= max_images:
                        break

                    try:
                        image_url = item.get('thumbnail_url') or item.get('url')
                        if not image_url:
                            continue

                        # Skip URLs already used for consensus references
                        if image_url in skip_urls:
                            logger.debug(f"[{batch_id}] Skipping URL already used for consensus: {image_url[:50]}...")
                            continue

                        # Download image
                        img_response = requests.get(image_url, timeout=15, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        })

                        if img_response.status_code != 200:
                            failed_downloads += 1
                            continue

                        # Validate that content is actually an image (not HTML redirect)
                        if not self._is_valid_image_content(img_response.content, batch_id):
                            logger.debug(f"[{batch_id}] Skipping invalid image from {image_url[:50]}...")
                            failed_downloads += 1
                            continue

                        # Save to originals temporarily
                        ext = self._get_extension(image_url)
                        filename = f"download_{sequence_counter:03d}{ext}"
                        filepath = os.path.join(originals_dir, filename)

                        with open(filepath, 'wb') as f:
                            f.write(img_response.content)

                        # Check face count - only keep images with exactly 1 face
                        face_count = self._count_faces_in_image(filepath)

                        if face_count != 1:
                            # Remove file - wrong number of faces
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            rejected_faces += 1

                            if face_count == 0:
                                reason = "no faces"
                            elif face_count == -2:
                                reason = "face too small"
                            elif face_count == -1:
                                reason = "detection error"
                            else:
                                reason = f"{face_count} faces (multi-person)"
                            logger.debug(f"[{batch_id}] Rejected image: {reason}")
                            continue

                        # Image passed face check - keep it
                        downloaded.append({
                            'path': filepath,
                            'sequence': sequence_counter,
                            'source_url': image_url
                        })
                        sequence_counter += 1
                        logger.debug(f"[{batch_id}] Downloaded {len(downloaded)}/{max_images}: {filename} (1 face ✓)")

                    except Exception as e:
                        logger.warning(f"[{batch_id}] Failed to download image {i+1}: {str(e)}")
                        failed_downloads += 1
                        continue

                # Log progress after each query variation
                if query_idx > 0:
                    logger.info(f"[{batch_id}] After retry '{query}': {len(downloaded)}/{max_images} valid images")

            except Exception as e:
                logger.error(f"[{batch_id}] Error with SERP query '{query}': {str(e)}")
                continue

        logger.info(f"[{batch_id}] SERP download complete: {len(downloaded)} valid images "
                   f"(rejected {rejected_faces} wrong face count, {failed_downloads} failed)")

        return downloaded

    def _process_and_validate_image(
        self,
        image_path: str,
        reference_path: str,
        folder_name: str,
        faces_dir: str,
        batch_id: str,
        reference_paths: List[str] = None
    ) -> Dict:
        """
        Extract face from image and validate against reference(s).
        Supports multiple reference images - passes if ANY reference matches.

        Args:
            reference_path: Primary reference path (backward compatibility)
            reference_paths: List of all reference paths (preferred if provided)

        Returns:
            Dict with extraction and validation results
        """
        result = {
            'extraction': {
                'status': 'rejected',
                'face_file': None,
                'reason': None
            },
            'validation': None
        }

        try:
            # Get sequence number from filename
            filename = os.path.basename(image_path)
            seq_match = self._extract_sequence(filename)

            # Create destination path for extracted face
            ext = os.path.splitext(image_path)[1]
            face_filename = f"{folder_name}_{seq_match:03d}{ext}" if seq_match else filename
            face_path = os.path.join(faces_dir, face_filename)

            # Extract face
            face_extracted = self._extract_face(image_path, face_path, batch_id)

            if not face_extracted:
                result['extraction']['reason'] = 'Face extraction failed (no face, multiple faces, too small, or blurry)'
                return result

            # Build list of reference paths to check against
            refs_to_check = reference_paths if reference_paths else [reference_path]

            # Validate using VECTORIZED embedding comparison
            # Extract candidate face embedding once
            candidate_emb = self._extract_embedding(face_path, batch_id)

            best_distance = 999.0
            best_match = False
            matched_ref_idx = None

            if candidate_emb is not None:
                # Extract reference embeddings and compute distances vectorized
                ref_embeddings = []
                valid_ref_indices = []

                for ref_idx, ref_path in enumerate(refs_to_check):
                    ref_emb = self._extract_embedding(ref_path, batch_id)
                    if ref_emb is not None:
                        ref_embeddings.append(ref_emb)
                        valid_ref_indices.append(ref_idx)

                if len(ref_embeddings) > 0:
                    # Compute distances from candidate to all references (vectorized using numpy)
                    ref_embeddings_array = np.array(ref_embeddings, dtype=np.float32)

                    # Normalize candidate and reference embeddings
                    candidate_norm = np.linalg.norm(candidate_emb)
                    candidate_normalized = candidate_emb / (candidate_norm + 1e-8)

                    ref_norms = np.linalg.norm(ref_embeddings_array, axis=1, keepdims=True)
                    ref_normalized = ref_embeddings_array / (ref_norms + 1e-8)

                    # Compute cosine distances: candidate vs each reference
                    # Shape: (n_refs,)
                    similarities = np.dot(ref_normalized, candidate_normalized)
                    distances = 1.0 - similarities
                    distances = np.clip(distances, 0.0, 2.0)

                    # Find best match
                    for dist_idx, distance in enumerate(distances):
                        ref_idx = valid_ref_indices[dist_idx]
                        if distance < best_distance:
                            best_distance = distance
                            best_match = (distance < 0.50)  # ArcFace threshold
                            if best_match:
                                matched_ref_idx = ref_idx

                        # Early exit if we find a match
                        if best_match:
                            logger.info(f"[{batch_id}] Matched against reference #{ref_idx+1}/{len(refs_to_check)} with distance={distance:.4f}")
                            break

                    # Log all distances if no match found
                    if not best_match:
                        logger.info(f"[{batch_id}] Compared against {len(distances)} references, best distance={best_distance:.4f}")
            else:
                # Fallback to original method if embedding extraction fails
                logger.warning(f"[{batch_id}] Could not extract embedding, falling back to DeepFace.verify")
                for ref_idx, ref_path in enumerate(refs_to_check):
                    is_match, distance = self._verify_faces(ref_path, face_path, batch_id)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = is_match
                        if is_match:
                            matched_ref_idx = ref_idx
                    if is_match:
                        logger.info(f"[{batch_id}] Matched against reference #{ref_idx+1}/{len(refs_to_check)} with distance={distance:.4f}")
                        break

            # ArcFace threshold: 0.50 (balanced threshold for quality training data)
            # Same-person distances are typically 0.2-0.5, different-person 0.6+
            arcface_threshold = 0.50
            result['validation'] = {
                'reference_distance': round(best_distance, 4),
                'threshold': arcface_threshold,
                'model': 'ArcFace',
                'passed': best_match,
                'references_checked': len(refs_to_check),
                'matched_reference': matched_ref_idx
            }

            if best_match:
                result['extraction']['status'] = 'accepted'
                result['extraction']['face_file'] = face_filename
                logger.info(f"[{batch_id}] Accepted: distance={best_distance:.4f} (checked {len(refs_to_check)} references)")
            else:
                result['extraction']['reason'] = f'Distance {best_distance:.4f} > {arcface_threshold} threshold (checked {len(refs_to_check)} references)'
                result['validation']['reason'] = f'Distance {best_distance:.4f} exceeds ArcFace threshold {arcface_threshold} for all {len(refs_to_check)} references'
                # Remove rejected face file
                if os.path.exists(face_path):
                    os.remove(face_path)
                logger.info(f"[{batch_id}] Rejected: best distance={best_distance:.4f} across {len(refs_to_check)} references")

        except Exception as e:
            result['extraction']['reason'] = f'Processing error: {str(e)}'
            logger.error(f"[{batch_id}] Error processing image: {str(e)}")

        return result

    def _extract_face(self, source_path: str, dest_path: str, batch_id: str) -> bool:
        """
        Extract face from image using DeepFace.

        Only accepts images with exactly one valid face to avoid ambiguity.

        Returns:
            True if face extracted successfully, False otherwise
        """
        import cv2

        try:
            if not os.path.exists(source_path):
                return False

            # Extract faces
            face_objs = FaceProcessingService.extract_faces_with_timeout(source_path, None, "unknown")

            if not face_objs:
                logger.warning(f"[{batch_id}] No faces detected in {source_path}")
                return False

            # Validate faces
            valid_faces = []
            for i, item in enumerate(face_objs):
                face_image_array = item['face']
                w, h = face_image_array.shape[1], face_image_array.shape[0]

                # Check size
                if w < 70 or h < 70:
                    continue

                # Check blur
                if FaceProcessingService.is_blurred(face_image_array, len(face_objs)):
                    continue

                valid_faces.append((face_image_array, i))

            # Reject if multiple valid faces or no valid faces
            if len(valid_faces) != 1:
                logger.warning(f"[{batch_id}] Invalid face count: {len(valid_faces)} valid faces")
                return False

            # Get the single valid face
            _, face_index = valid_faces[0]
            face = face_objs[face_index]

            if "facial_area" not in face:
                return False

            # Load image and crop face
            img = cv2.imread(source_path)
            if img is None:
                return False

            facial_area = face["facial_area"]
            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]

            # Add margin
            margin = 0.2
            height, width = img.shape[:2]
            x1 = max(0, x - int(w * margin))
            y1 = max(0, y - int(h * margin))
            x2 = min(width, x + w + int(w * margin))
            y2 = min(height, y + h + int(h * margin))

            face_img = img[y1:y2, x1:x2]

            # Save
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            cv2.imwrite(dest_path, face_img)

            return True

        except Exception as e:
            logger.error(f"[{batch_id}] Face extraction error: {str(e)}")
            return False

    def _verify_faces(self, img1_path: str, img2_path: str, batch_id: str) -> Tuple[bool, float]:
        """
        Verify if two face images are the same person using ArcFace.

        ArcFace (2019) has 99.8% LFW accuracy vs VGG-Face's 97%.
        ArcFace cosine distances are typically lower (tighter clusters):
        - Same person: 0.2-0.5 (vs VGG-Face 0.3-0.7)
        - Different person: 0.6+ (vs VGG-Face 0.8+)

        Returns:
            Tuple of (is_match, distance)
        """
        from deepface import DeepFace

        try:
            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                return False, 999.0

            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name="ArcFace",  # Upgraded from VGG-Face (99.8% vs 97% accuracy)
                distance_metric="cosine",
                detector_backend="retinaface",
                threshold=0.50,  # ArcFace threshold for training (balanced for quality)
                enforce_detection=False
            )

            return result["verified"], result["distance"]

        except Exception as e:
            logger.error(f"[{batch_id}] Face verification error: {str(e)}")
            return False, 999.0

    def _build_serp_consensus_references(
        self,
        person_name: str,
        folder_name: str,
        originals_dir: str,
        faces_dir: str,
        batch_id: str,
        consensus_threshold: float = 0.50,  # Slightly relaxed ArcFace threshold
        min_consensus: int = 4  # Accept 4+ matching images (reduced from 5 for better success rate)
    ) -> Optional[Dict]:
        """
        Build reference images using SERP consensus with improved strategy.

        Now uses ArcFace embeddings which have tighter distance distributions:
        - ArcFace same-person: typically 0.2-0.50
        - ArcFace different-person: typically 0.6+

        Strategy:
        1. Download first 10 SERP images (increased from 4 for better coverage)
        2. Extract faces from each
        3. Compare all pairs to find which faces are "same person"
        4. If 4+ images agree (form a cluster), use them as references
        5. Return None if no consensus (ambiguous search results)

        Args:
            person_name: Name to search for
            folder_name: Safe folder name
            originals_dir: Where to save original downloads
            faces_dir: Where to save extracted faces
            batch_id: Batch identifier
            consensus_threshold: Distance threshold for "same person" (default 0.50 for ArcFace)
            min_consensus: Minimum images that must agree (default 4 for better coverage)

        Returns:
            Dict with reference info or None if no consensus
        """
        logger.info(f"[{batch_id}] Building SERP consensus references for: {person_name}")

        # Download first 20 images from SERP (increased from 10 for better coverage)
        serp_images = self._download_serp_images(
            person_name=person_name,
            folder_name=folder_name,
            originals_dir=originals_dir,
            max_images=20,
            batch_id=batch_id
        )

        if len(serp_images) < 4:
            logger.warning(f"[{batch_id}] Only got {len(serp_images)} SERP images, need at least 4 for consensus")
            return None

        # Extract faces from each image (process up to 20)
        import gc
        candidate_faces = []
        ref_faces_dir = os.path.join(faces_dir, '_consensus_refs')
        os.makedirs(ref_faces_dir, exist_ok=True)

        for idx, img_info in enumerate(serp_images[:20]):
            ext = os.path.splitext(img_info['path'])[1]
            face_filename = f"{folder_name}_candidate_{idx+1:02d}{ext}"
            face_path = os.path.join(ref_faces_dir, face_filename)

            face_extracted = self._extract_face(img_info['path'], face_path, batch_id)

            if face_extracted:
                candidate_faces.append({
                    'index': idx,
                    'face_path': face_path,
                    'source_path': img_info['path'],
                    'source_url': img_info.get('source_url')
                })
                logger.info(f"[{batch_id}] Extracted face from candidate {idx+1}")
            else:
                logger.warning(f"[{batch_id}] Failed to extract face from candidate {idx+1}")

            # Garbage collect every 5 images during consensus building
            if (idx + 1) % 5 == 0:
                gc.collect()

        if len(candidate_faces) < 4:
            logger.warning(f"[{batch_id}] Only {len(candidate_faces)} valid faces, need at least 4 for consensus")
            # Cleanup
            for cf in candidate_faces:
                if os.path.exists(cf['face_path']):
                    os.remove(cf['face_path'])
            return None

        # Build similarity matrix using VECTORIZED embedding computation (140x faster)
        # Extract all embeddings at once instead of computing pairwise DeepFace distances
        logger.info(f"[{batch_id}] Extracting embeddings for {len(candidate_faces)} candidate faces...")
        embeddings = []
        valid_indices = []

        for idx, cf in enumerate(candidate_faces):
            emb = self._extract_embedding(cf['face_path'], batch_id)
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(idx)
            else:
                logger.warning(f"[{batch_id}] Could not extract embedding for candidate {idx+1}")

        if len(embeddings) < 4:
            logger.warning(f"[{batch_id}] Only {len(embeddings)} valid embeddings, need at least 4 for consensus")
            return None

        # Compute all pairwise distances at once (vectorized)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        distance_matrix_vectorized = self._batch_cosine_distances(embeddings_array)

        # Map vectorized distances back to original indices
        n = len(candidate_faces)
        similarity_matrix = [[False] * n for _ in range(n)]
        distance_matrix = [[999.0] * n for _ in range(n)]

        # Fill in valid distances from vectorized computation
        for i_idx, i in enumerate(valid_indices):
            similarity_matrix[i][i] = True
            distance_matrix[i][i] = 0.0

            for j_idx, j in enumerate(valid_indices):
                distance = float(distance_matrix_vectorized[i_idx][j_idx])
                is_same = distance < consensus_threshold
                similarity_matrix[i][j] = is_same
                distance_matrix[i][j] = distance
                if i < j:  # Log only once per pair
                    logger.info(f"[{batch_id}] Candidate {i+1} vs {j+1}: distance={distance:.4f}, same={is_same}")

        # Mark invalid indices as isolated (no matches)
        for idx in range(n):
            if idx not in valid_indices:
                for j in range(n):
                    similarity_matrix[idx][j] = False
                    similarity_matrix[j][idx] = False

        # Find the largest cluster of matching faces
        # For each face, count how many others match it
        match_counts = []
        for i in range(n):
            matches = sum(1 for j in range(n) if similarity_matrix[i][j])
            match_counts.append((i, matches))

        # Sort by match count descending
        match_counts.sort(key=lambda x: -x[1])

        # Get the cluster around the face with most matches
        best_face_idx = match_counts[0][0]
        cluster_indices = [i for i in range(n) if similarity_matrix[best_face_idx][i]]

        logger.info(f"[{batch_id}] Best cluster: face {best_face_idx+1} with {len(cluster_indices)} matches: {[i+1 for i in cluster_indices]}")

        if len(cluster_indices) < min_consensus:
            logger.warning(f"[{batch_id}] Cluster size {len(cluster_indices)} < {min_consensus}, no consensus reached")
            # Cleanup
            for cf in candidate_faces:
                if os.path.exists(cf['face_path']):
                    os.remove(cf['face_path'])
            return None

        # Build reference from consensus cluster
        consensus_refs = [candidate_faces[i] for i in cluster_indices]

        # Move consensus faces to proper reference location (rename from candidate to ref)
        reference_paths = []
        for idx, ref in enumerate(consensus_refs):
            ext = os.path.splitext(ref['face_path'])[1]
            new_face_filename = f"{folder_name}_ref_{idx+1:03d}{ext}"
            new_face_path = os.path.join(ref_faces_dir, new_face_filename)
            if ref['face_path'] != new_face_path:
                os.rename(ref['face_path'], new_face_path)
                ref['face_path'] = new_face_path
                ref['face_filename'] = new_face_filename
            reference_paths.append(new_face_path)

        # Remove non-consensus candidate faces
        for cf in candidate_faces:
            if cf['index'] not in cluster_indices and os.path.exists(cf['face_path']):
                os.remove(cf['face_path'])

        logger.info(f"[{batch_id}] ✓ SERP consensus reached: {len(consensus_refs)}/{len(candidate_faces)} images agree")

        # Collect URLs of consensus images to skip in future downloads
        consensus_urls = set()
        for ref in consensus_refs:
            if ref.get('source_url'):
                consensus_urls.add(ref['source_url'])

        # Return in same format as P18 reference for compatibility
        primary_ref = consensus_refs[0]
        return {
            'source': 'serp_consensus',
            'source_url': primary_ref.get('source_url'),
            'original_path': primary_ref.get('source_path'),
            'face_path': primary_ref['face_path'],
            'face_filename': primary_ref.get('face_filename'),
            'is_reference': True,
            'consensus_size': len(consensus_refs),
            'all_references': consensus_refs,
            'reference_paths': reference_paths,
            'consensus_urls': consensus_urls  # URLs to skip in subsequent downloads
        }

    def _generate_gallery(self, folder_name: str, batch_id: str, metadata: Dict) -> Optional[str]:
        """
        Generate HTML gallery for review.

        Returns:
            Gallery URL or None if failed
        """
        try:
            gallery_dir = os.path.join(self.galleries_path, folder_name.lower(), batch_id)
            os.makedirs(gallery_dir, exist_ok=True)

            person_name = metadata.get('person_name', folder_name)
            reference = metadata.get('reference', {})
            images = metadata.get('images', {})

            # Count stats
            total = len(images) + 1  # +1 for reference
            accepted = sum(1 for img in images.values()
                         if img.get('extraction', {}).get('status') == 'accepted') + 1

            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Gallery - {person_name}</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ text-align: center; color: #333; }}
        .stats {{ display: flex; gap: 15px; justify-content: center; margin: 20px 0; }}
        .stat-box {{ padding: 15px 25px; border-radius: 8px; text-align: center; }}
        .stat-box.total {{ background: #d1ecf1; border: 2px solid #17a2b8; }}
        .stat-box.accepted {{ background: #d4edda; border: 2px solid #28a745; }}
        .stat-box.rejected {{ background: #f8d7da; border: 2px solid #dc3545; }}
        .stat-value {{ font-size: 28px; font-weight: bold; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .reference-section {{ background: #e3f2fd; border: 2px solid #1976d2; padding: 20px;
                            border-radius: 8px; margin: 20px 0; text-align: center; }}
        .reference-section h2 {{ color: #1976d2; margin-top: 0; }}
        .reference-section img {{ max-height: 300px; border-radius: 8px; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
        .image-card {{ background: white; border-radius: 8px; padding: 10px; text-align: center; }}
        .image-card.accepted {{ border: 2px solid #28a745; }}
        .image-card.rejected {{ border: 2px solid #dc3545; opacity: 0.7; }}
        .image-card img {{ max-width: 100%; max-height: 180px; border-radius: 4px; }}
        .image-card .status {{ font-weight: bold; margin: 5px 0; }}
        .image-card .distance {{ font-size: 12px; color: #666; }}
        .image-card .reason {{ font-size: 11px; color: #dc3545; }}
    </style>
</head>
<body>
    <h1>Training Gallery: {person_name}</h1>
    <p style="text-align: center; color: #666;">Batch: {batch_id}</p>

    <div class="stats">
        <div class="stat-box total">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Total Processed</div>
        </div>
        <div class="stat-box accepted">
            <div class="stat-value" style="color: #28a745;">{accepted}</div>
            <div class="stat-label">Accepted</div>
        </div>
        <div class="stat-box rejected">
            <div class="stat-value" style="color: #dc3545;">{total - accepted}</div>
            <div class="stat-label">Rejected</div>
        </div>
    </div>

    <div class="reference-section">
        <h2>Reference Image (Wikidata P18)</h2>
        <img src="/serp_originals/{self.domain}/{folder_name}/{batch_id}/download_001.jpg"
             onerror="this.style.display='none'" />
        <p>Source: Wikidata P18 (Primary Image)</p>
    </div>

    <h2>Processed Images</h2>
    <div class="image-grid">
"""

            # Add image cards
            for filename, info in sorted(images.items(), key=lambda x: (x[1] or {}).get('sequence', 999)):
                if not info:
                    continue
                extraction = info.get('extraction') or {}
                validation = info.get('validation') or {}
                status = extraction.get('status', 'unknown')
                distance = validation.get('reference_distance') if validation else None
                reason = extraction.get('reason') or (validation.get('reason', '') if validation else '')

                status_text = 'ACCEPTED' if status == 'accepted' else 'REJECTED'
                distance_text = f"Distance: {distance:.4f}" if distance else ""

                html += f"""
        <div class="image-card {status}">
            <img src="/serp_originals/{self.domain}/{folder_name}/{batch_id}/{filename}"
                 onerror="this.src='/static/no-image.png'" />
            <div class="status" style="color: {'#28a745' if status == 'accepted' else '#dc3545'}">
                {status_text}
            </div>
            <div class="distance">{distance_text}</div>
            <div class="reason">{reason}</div>
        </div>
"""

            html += """
    </div>
</body>
</html>
"""

            # Save gallery
            gallery_path = os.path.join(gallery_dir, 'index.html')
            with open(gallery_path, 'w') as f:
                f.write(html)

            gallery_url = f"/training-galleries/{self.domain}/{folder_name.lower()}/{batch_id}/index.html"
            logger.info(f"[{batch_id}] Gallery generated: {gallery_url}")

            return gallery_url

        except Exception as e:
            logger.error(f"[{batch_id}] Error generating gallery: {str(e)}")
            return None

    # Utility methods

    def _safe_folder_name(self, name: str) -> str:
        """Convert name to safe folder name with proper Serbian transliteration."""
        import unicodedata
        import re

        # Only đ/Đ needs manual transliteration - other Serbian chars (ž,š,č,ć)
        # decompose properly with NFKD normalization
        transliteration_map = {
            'đ': 'dj', 'Đ': 'Dj',
            'ß': 'ss',  # German
            'ø': 'o', 'Ø': 'O',  # Nordic
            'æ': 'ae', 'Æ': 'AE',
            'œ': 'oe', 'Œ': 'OE',
        }

        # First apply manual transliteration for chars that don't decompose
        result = name
        for char, replacement in transliteration_map.items():
            result = result.replace(char, replacement)

        # Then normalize remaining unicode (handles ž→z, š→s, č→c, ć→c, é→e, etc.)
        normalized = unicodedata.normalize('NFKD', result)
        ascii_name = ''.join([c for c in normalized if not unicodedata.combining(c)])
        ascii_name = ascii_name.encode('ascii', 'ignore').decode('ascii')

        # Replace spaces with underscores, remove unsafe chars
        safe = re.sub(r'[^\w_-]', '', ascii_name.replace(' ', '_'))

        return safe

    def _get_extension(self, url: str) -> str:
        """Get file extension from URL."""
        lower_url = url.lower()
        if '.png' in lower_url:
            return '.png'
        elif '.webp' in lower_url:
            return '.webp'
        return '.jpg'

    def _is_valid_image_content(self, content: bytes, batch_id: str = None) -> bool:
        """
        Validate that downloaded content is actually an image, not HTML/redirect.

        SERP images often return HTML redirect pages from social media sites
        (Facebook, Instagram, Pinterest) instead of actual image data.

        Uses multiple validation strategies:
        1. Minimum size check (real images > 1KB)
        2. Magic bytes check (JPEG, PNG, WebP, GIF headers)
        3. OpenCV decode attempt (definitive validation)

        Returns:
            True if content is a valid, decodable image
        """
        # Check 1: Minimum size - HTML redirects are typically 300-500 bytes
        if len(content) < 1000:
            if batch_id:
                logger.debug(f"[{batch_id}] Image too small ({len(content)} bytes), likely HTML redirect")
            return False

        # Check 2: Magic bytes - quick rejection of obvious non-images
        is_jpeg = content[:3] == b'\xff\xd8\xff'
        is_png = content[:8] == b'\x89PNG\r\n\x1a\n'
        is_webp = content[:4] == b'RIFF' and len(content) > 12 and content[8:12] == b'WEBP'
        is_gif = content[:6] in (b'GIF87a', b'GIF89a')

        if not (is_jpeg or is_png or is_webp or is_gif):
            # Check if it's HTML (common for redirect pages)
            if content[:5] == b'<html' or content[:5] == b'<!DOC' or content[:1] == b'<':
                if batch_id:
                    logger.debug(f"[{batch_id}] Content is HTML, not an image")
                return False
            # Unknown format, try OpenCV anyway

        # Check 3: Actually try to decode the image (definitive test)
        try:
            nparr = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                if batch_id:
                    logger.debug(f"[{batch_id}] OpenCV failed to decode image")
                return False

            # Additional sanity check: image should have reasonable dimensions
            h, w = img.shape[:2]
            if w < 10 or h < 10:
                if batch_id:
                    logger.debug(f"[{batch_id}] Image too small: {w}x{h}")
                return False

            return True

        except Exception as e:
            if batch_id:
                logger.debug(f"[{batch_id}] Image decode error: {str(e)}")
            return False

    def _extract_sequence(self, filename: str) -> Optional[int]:
        """Extract sequence number from filename."""
        import re
        match = re.search(r'_(\d+)\.\w+$', filename)
        if match:
            return int(match.group(1))
        return None

    def _save_batch_status(self, batch_id: str, status: Dict):
        """Save batch status to file."""
        status_path = os.path.join(self.batches_path, f"{batch_id}.json")
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)

    def _load_batch_status(self, batch_id: str) -> Dict:
        """Load batch status from file."""
        status_path = os.path.join(self.batches_path, f"{batch_id}.json")
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                return json.load(f)
        return {}

    # Public API methods

    def get_batch_status(self, batch_id: str) -> Dict:
        """Get status of a training batch."""
        return self._load_batch_status(batch_id)

    def list_batches(self) -> List[Dict]:
        """List all training batches."""
        batches = []
        for filename in os.listdir(self.batches_path):
            if filename.endswith('.json'):
                batch_id = filename[:-5]
                status = self._load_batch_status(batch_id)
                batches.append({
                    'batch_id': batch_id,
                    'status': status.get('status'),
                    'created_at': status.get('created_at'),
                    'total': status.get('total_celebrities', 0),
                    'processed': status.get('processed', 0)
                })
        return sorted(batches, key=lambda x: x.get('created_at', ''), reverse=True)
