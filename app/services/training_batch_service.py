"""
Training Batch Service for managing batch celebrity training workflows.
Handles candidate generation, batch processing, and deployment.
"""

import os
import json
import uuid
import shutil
import threading
from datetime import datetime
from typing import List, Dict, Optional
from flask import current_app
import logging

from app.services.wikidata_service import WikidataService
from app.services.image_service import ImageService

logger = logging.getLogger(__name__)


class TrainingBatchService:
    """Service for managing batch training of celebrities"""

    BATCH_STORAGE = "storage/training_batches"
    TRAINING_PASS_PATH = "storage/trainingPassSerbia"
    PRODUCTION_PATH = "storage/recognized_faces_prod"

    @classmethod
    def generate_candidates(cls, country: str, occupation: str, domain: str = "serbia") -> Dict:
        """
        Generate celebrity candidates from Wikidata and check against existing DB.

        Args:
            country: Country code (e.g., "serbia")
            occupation: Occupation code (e.g., "actor")
            domain: Domain for checking existing faces

        Returns:
            Dictionary with candidates list and statistics
        """
        try:
            logger.info(f"Generating candidates for {country} {occupation}s")

            # Query Wikidata
            celebrities = WikidataService.query_celebrities(country, occupation, limit=500)

            if not celebrities:
                return {
                    "success": False,
                    "message": "No celebrities found in Wikidata",
                    "candidates": []
                }

            # Check which ones already exist in production DB
            production_dir = os.path.join(cls.PRODUCTION_PATH, domain)
            existing_folders = {}  # Map normalized name -> actual folder name

            if os.path.exists(production_dir):
                for folder in os.listdir(production_dir):
                    if os.path.isdir(os.path.join(production_dir, folder)):
                        normalized = cls._normalize_name(folder)
                        existing_folders[normalized] = folder

            # Process candidates
            candidates = []
            for celeb in celebrities:
                # Create normalized folder name
                folder_name = f"{celeb['name']}_{celeb['last_name']}".lower().replace(' ', '_')
                folder_name_normalized = cls._normalize_name(folder_name)

                # Check if exists (exact match or fuzzy match)
                actual_folder_name = None
                exists = False

                # 1. Try exact match
                if folder_name_normalized in existing_folders:
                    exists = True
                    actual_folder_name = existing_folders[folder_name_normalized]
                else:
                    # 2. Try fuzzy match (check if normalized name is similar)
                    # This handles cases like "Novak Djokovic" vs "Novak_Djokovic_Tennis"
                    for norm_name, actual_name in existing_folders.items():
                        # Check if either contains the other
                        if (folder_name_normalized in norm_name or
                            norm_name in folder_name_normalized or
                            cls._name_similarity(folder_name_normalized, norm_name) > 0.8):
                            exists = True
                            actual_folder_name = actual_name
                            logger.info(f"Fuzzy match: '{folder_name_normalized}' matched with '{actual_name}'")
                            break

                # Count existing photos if exists
                photo_count = 0
                if exists and actual_folder_name:
                    person_path = os.path.join(production_dir, actual_folder_name)
                    if os.path.exists(person_path):
                        photo_count = len([
                            f for f in os.listdir(person_path)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
                        ])

                candidates.append({
                    "full_name": celeb['full_name'],
                    "name": celeb['name'],
                    "last_name": celeb['last_name'],
                    "occupation": celeb['occupation'],
                    "country": celeb['country'],
                    "description": celeb['description'],
                    "wikidata_id": celeb['wikidata_id'],
                    "has_wikipedia_image": celeb['has_image'],
                    "exists_in_db": exists,
                    "existing_photo_count": photo_count,
                    "folder_name": folder_name_normalized
                })

            # Statistics
            total = len(candidates)
            new_count = len([c for c in candidates if not c['exists_in_db']])
            existing_count = total - new_count

            logger.info(f"Generated {total} candidates: {new_count} new, {existing_count} existing")

            return {
                "success": True,
                "message": f"Found {total} {country} {occupation}s",
                "candidates": candidates,
                "statistics": {
                    "total": total,
                    "new": new_count,
                    "existing": existing_count
                }
            }

        except Exception as e:
            logger.error(f"Error generating candidates: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "candidates": []
            }

    @classmethod
    def start_batch_training(cls, candidates: List[Dict], domain: str = "serbia") -> Dict:
        """
        Start batch training for selected candidates.

        Args:
            candidates: List of candidate dictionaries to train
            domain: Domain for image storage

        Returns:
            Batch information with batch_id for tracking
        """
        try:
            # Create batch ID
            batch_id = str(uuid.uuid4())[:8]

            # Create batch metadata
            batch_meta = {
                "batch_id": batch_id,
                "created_at": datetime.now().isoformat(),
                "domain": domain,
                "total_people": len(candidates),
                "status": "processing",
                "people": [
                    {
                        "full_name": c['full_name'],
                        "name": c['name'],
                        "last_name": c['last_name'],
                        "occupation": c['occupation'],
                        "folder_name": c.get('folder_name', f"{c['name']}_{c['last_name']}".lower()),
                        "status": "queued",
                        "current_step": None,
                        "photos_downloaded": 0,
                        "photos_validated": 0,
                        "error": None,
                        "started_at": None,
                        "completed_at": None
                    }
                    for c in candidates
                ]
            }

            # Save batch metadata
            os.makedirs(cls.BATCH_STORAGE, exist_ok=True)
            batch_file = os.path.join(cls.BATCH_STORAGE, f"{batch_id}.json")

            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(batch_meta, f, ensure_ascii=False, indent=2)

            logger.info(f"Created training batch {batch_id} with {len(candidates)} people")

            # Start background processing
            app_context = current_app.app_context()
            thread = threading.Thread(
                target=cls._process_batch_thread,
                args=(batch_id, app_context)
            )
            thread.daemon = True
            thread.start()

            return {
                "success": True,
                "message": f"Started batch training for {len(candidates)} people",
                "batch_id": batch_id,
                "total_people": len(candidates)
            }

        except Exception as e:
            logger.error(f"Error starting batch training: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }

    @classmethod
    def get_batch_status(cls, batch_id: str) -> Dict:
        """
        Get current status of a training batch.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch status with progress information
        """
        try:
            batch_file = os.path.join(cls.BATCH_STORAGE, f"{batch_id}.json")

            if not os.path.exists(batch_file):
                return {
                    "success": False,
                    "message": "Batch not found"
                }

            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_meta = json.load(f)

            # Calculate statistics
            people = batch_meta.get('people', [])
            completed = len([p for p in people if p['status'] == 'completed'])
            processing = len([p for p in people if p['status'] == 'processing'])
            failed = len([p for p in people if p['status'] == 'failed'])
            queued = len([p for p in people if p['status'] == 'queued'])

            return {
                "success": True,
                "batch_id": batch_id,
                "status": batch_meta.get('status', 'unknown'),
                "created_at": batch_meta.get('created_at'),
                "total": batch_meta.get('total_people', 0),
                "completed": completed,
                "processing": processing,
                "failed": failed,
                "queued": queued,
                "progress_percentage": int((completed / max(len(people), 1)) * 100),
                "people": people
            }

        except Exception as e:
            logger.error(f"Error getting batch status: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }

    @classmethod
    def deploy_to_production(cls, people: List[str], domain: str = "serbia") -> Dict:
        """
        Deploy trained people from trainingPass to production.

        Args:
            people: List of folder names to deploy
            domain: Target domain

        Returns:
            Deployment result
        """
        try:
            source_base = cls.TRAINING_PASS_PATH
            target_dir = os.path.join(cls.PRODUCTION_PATH, domain)

            os.makedirs(target_dir, exist_ok=True)

            deployed = []
            skipped = []
            errors = []

            for person_folder in people:
                try:
                    source_path = os.path.join(source_base, person_folder)

                    if not os.path.exists(source_path):
                        skipped.append({
                            "folder": person_folder,
                            "reason": "Source folder not found"
                        })
                        continue

                    # Count images
                    image_files = [
                        f for f in os.listdir(source_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
                    ]

                    if len(image_files) < 5:
                        skipped.append({
                            "folder": person_folder,
                            "reason": f"Too few images ({len(image_files)}), minimum is 5"
                        })
                        # Delete folder with too few images
                        shutil.rmtree(source_path)
                        continue

                    # Move to production
                    target_path = os.path.join(target_dir, person_folder)

                    if os.path.exists(target_path):
                        # Merge folders
                        for image_file in image_files:
                            src = os.path.join(source_path, image_file)
                            dst = os.path.join(target_path, image_file)
                            shutil.copy2(src, dst)
                        shutil.rmtree(source_path)
                    else:
                        # Move entire folder
                        shutil.move(source_path, target_path)

                    deployed.append({
                        "folder": person_folder,
                        "image_count": len(image_files)
                    })

                    logger.info(f"Deployed {person_folder} with {len(image_files)} images")

                except Exception as person_error:
                    errors.append({
                        "folder": person_folder,
                        "error": str(person_error)
                    })
                    logger.error(f"Error deploying {person_folder}: {str(person_error)}")

            return {
                "success": True,
                "message": f"Deployed {len(deployed)} people to production",
                "deployed": deployed,
                "skipped": skipped,
                "errors": errors,
                "statistics": {
                    "deployed_count": len(deployed),
                    "skipped_count": len(skipped),
                    "error_count": len(errors)
                }
            }

        except Exception as e:
            logger.error(f"Error deploying to production: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }

    @classmethod
    def _process_batch_thread(cls, batch_id: str, app_context):
        """Background thread to process batch training"""
        with app_context:
            try:
                logger.info(f"Started processing batch {batch_id}")

                batch_file = os.path.join(cls.BATCH_STORAGE, f"{batch_id}.json")

                # Load batch
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_meta = json.load(f)

                image_service = ImageService()

                # Process each person
                for i, person in enumerate(batch_meta['people']):
                    try:
                        logger.info(f"Processing {person['full_name']} ({i+1}/{len(batch_meta['people'])})")

                        # Update status to processing
                        person['status'] = 'processing'
                        person['started_at'] = datetime.now().isoformat()
                        person['current_step'] = 'downloading_images'
                        cls._save_batch(batch_file, batch_meta)

                        # Fetch images
                        result = image_service.fetch_and_save_images(
                            person['name'],
                            person['last_name'],
                            person['occupation']
                        )

                        person['photos_downloaded'] = result.get('count', 0)
                        person['current_step'] = 'validating_faces'
                        cls._save_batch(batch_file, batch_meta)

                        # Note: DeepFace validation happens automatically in background
                        # We'll mark as completed immediately, validation continues async

                        person['status'] = 'completed'
                        person['completed_at'] = datetime.now().isoformat()
                        person['current_step'] = 'completed'
                        cls._save_batch(batch_file, batch_meta)

                        logger.info(f"Completed {person['full_name']}: {person['photos_downloaded']} photos")

                    except Exception as person_error:
                        person['status'] = 'failed'
                        person['error'] = str(person_error)
                        person['completed_at'] = datetime.now().isoformat()
                        cls._save_batch(batch_file, batch_meta)
                        logger.error(f"Failed processing {person['full_name']}: {str(person_error)}")

                # Mark batch as completed
                batch_meta['status'] = 'completed'
                batch_meta['completed_at'] = datetime.now().isoformat()
                cls._save_batch(batch_file, batch_meta)

                logger.info(f"Finished processing batch {batch_id}")

            except Exception as e:
                logger.error(f"Error in batch processing thread: {str(e)}")

    @staticmethod
    def _save_batch(batch_file: str, batch_meta: Dict):
        """Save batch metadata to file"""
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_meta, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize name for folder matching"""
        import unicodedata
        import re

        # Convert to lowercase
        name = name.lower()

        # Normalize unicode
        name = unicodedata.normalize('NFKD', name)
        name = ''.join([c for c in name if not unicodedata.combining(c)])

        # Remove non-alphanumeric except underscores and spaces
        name = re.sub(r'[^\w\s_]', '', name)

        # Replace spaces with underscores
        name = name.replace(' ', '_')

        return name.strip('_')

    @staticmethod
    def _name_similarity(name1: str, name2: str) -> float:
        """
        Calculate similarity between two names using Levenshtein distance.
        Returns a value between 0 (completely different) and 1 (identical).
        """
        if name1 == name2:
            return 1.0

        # Simple character-based similarity
        len1, len2 = len(name1), len(name2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Calculate Levenshtein distance (simple implementation)
        if len1 > len2:
            name1, name2 = name2, name1
            len1, len2 = len2, len1

        current_row = range(len1 + 1)
        for i in range(1, len2 + 1):
            previous_row = current_row
            current_row = [i] + [0] * len1
            for j in range(1, len1 + 1):
                add = previous_row[j] + 1
                delete = current_row[j - 1] + 1
                change = previous_row[j - 1]
                if name1[j - 1] != name2[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)

        distance = current_row[len1]
        max_len = max(len1, len2)
        similarity = 1 - (distance / max_len)

        return similarity
