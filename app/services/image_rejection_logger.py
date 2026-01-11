"""
Image Rejection Logger Service
Logs and stores rejected images for debugging face validation issues.
"""

import os
import json
import shutil
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageRejectionLogger:
    """
    Service for logging image rejections during training.
    Stores rejected images temporarily for debugging purposes.
    """

    # Configuration
    REJECTION_LOG_PATH = "storage/logs/image_rejections"
    REJECTED_IMAGES_PATH = "storage/rejected_images"
    RETENTION_DAYS = 7  # Keep rejected images for 7 days
    LOG_FILE = "rejection_log.json"

    # Rejection reasons
    REASON_NO_FACE = "no_face_detected"
    REASON_FACE_TOO_SMALL = "face_too_small"
    REASON_BLURRY = "face_blurry"
    REASON_MULTIPLE_FACES = "multiple_faces"
    REASON_MULTIPLE_INVALID = "multiple_invalid_faces"
    REASON_LOW_CONFIDENCE = "low_confidence"
    REASON_NO_EYES = "no_eyes_detected"
    REASON_DOWNLOAD_FAILED = "download_failed"
    REASON_INVALID_FORMAT = "invalid_image_format"
    REASON_OTHER = "other"

    @classmethod
    def initialize(cls):
        """Create necessary directories for logging"""
        os.makedirs(cls.REJECTION_LOG_PATH, exist_ok=True)
        os.makedirs(cls.REJECTED_IMAGES_PATH, exist_ok=True)
        logger.info(f"Image rejection logger initialized. Logs: {cls.REJECTION_LOG_PATH}, Images: {cls.REJECTED_IMAGES_PATH}")

    @classmethod
    def log_rejection(cls,
                      image_path: str,
                      person_name: str,
                      reason: str,
                      details: Optional[Dict[str, Any]] = None,
                      source: str = "wikimedia",
                      batch_id: Optional[str] = None,
                      store_image: bool = True) -> Dict:
        """
        Log an image rejection with details.

        Args:
            image_path: Path to the rejected image
            person_name: Name of the person being trained
            reason: Rejection reason (use class constants)
            details: Additional details about the rejection
            source: Image source (wikimedia, serp, etc.)
            batch_id: Batch ID if applicable
            store_image: Whether to copy the image to rejected folder

        Returns:
            Dict with rejection log entry
        """
        cls.initialize()

        timestamp = datetime.now().isoformat()
        date_folder = datetime.now().strftime('%Y-%m-%d')

        # Create rejection entry
        rejection_entry = {
            "timestamp": timestamp,
            "person_name": person_name,
            "original_path": image_path,
            "reason": reason,
            "reason_description": cls._get_reason_description(reason),
            "details": details or {},
            "source": source,
            "batch_id": batch_id,
            "stored_path": None
        }

        # Store the rejected image if requested and file exists
        if store_image and os.path.exists(image_path):
            try:
                # Create person folder in rejected images
                person_folder = os.path.join(cls.REJECTED_IMAGES_PATH, date_folder, person_name)
                os.makedirs(person_folder, exist_ok=True)

                # Generate unique filename
                original_filename = os.path.basename(image_path)
                reason_prefix = reason[:20].replace(" ", "_")
                stored_filename = f"{reason_prefix}_{original_filename}"
                stored_path = os.path.join(person_folder, stored_filename)

                # Copy the image
                shutil.copy2(image_path, stored_path)
                rejection_entry["stored_path"] = stored_path

                logger.info(f"Stored rejected image: {stored_path}")
            except Exception as e:
                logger.warning(f"Failed to store rejected image: {e}")

        # Append to daily log file
        cls._append_to_log(date_folder, rejection_entry)

        # Log to standard logger
        logger.info(f"[REJECTION] {person_name}: {reason} - {image_path}")
        if details:
            logger.debug(f"[REJECTION DETAILS] {json.dumps(details)}")

        return rejection_entry

    @classmethod
    def _get_reason_description(cls, reason: str) -> str:
        """Get human-readable description for rejection reason"""
        descriptions = {
            cls.REASON_NO_FACE: "No face was detected in the image",
            cls.REASON_FACE_TOO_SMALL: "Detected face is too small (< 70x70 pixels)",
            cls.REASON_BLURRY: "Face image is too blurry for training",
            cls.REASON_MULTIPLE_FACES: "Multiple valid faces detected - ambiguous identity",
            cls.REASON_MULTIPLE_INVALID: "Multiple faces detected but none valid",
            cls.REASON_LOW_CONFIDENCE: "Face detection confidence too low",
            cls.REASON_NO_EYES: "Eyes not detected in face region",
            cls.REASON_DOWNLOAD_FAILED: "Failed to download image from source",
            cls.REASON_INVALID_FORMAT: "Invalid or corrupted image format",
            cls.REASON_OTHER: "Other rejection reason"
        }
        return descriptions.get(reason, reason)

    @classmethod
    def _append_to_log(cls, date_folder: str, entry: Dict):
        """Append rejection entry to daily log file"""
        log_folder = os.path.join(cls.REJECTION_LOG_PATH, date_folder)
        os.makedirs(log_folder, exist_ok=True)

        log_file = os.path.join(log_folder, cls.LOG_FILE)

        # Read existing entries or create new list
        entries = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                entries = []

        # Append new entry
        entries.append(entry)

        # Write back
        with open(log_file, 'w') as f:
            json.dump(entries, f, indent=2)

    @classmethod
    def get_rejection_summary(cls, date: Optional[str] = None, person_name: Optional[str] = None) -> Dict:
        """
        Get summary of rejections for a date or person.

        Args:
            date: Date string (YYYY-MM-DD) or None for today
            person_name: Filter by person name

        Returns:
            Dict with rejection summary
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        log_file = os.path.join(cls.REJECTION_LOG_PATH, date, cls.LOG_FILE)

        if not os.path.exists(log_file):
            return {
                "date": date,
                "total_rejections": 0,
                "by_reason": {},
                "by_person": {},
                "entries": []
            }

        try:
            with open(log_file, 'r') as f:
                entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            entries = []

        # Filter by person if specified
        if person_name:
            entries = [e for e in entries if e.get('person_name') == person_name]

        # Build summary
        by_reason = {}
        by_person = {}

        for entry in entries:
            reason = entry.get('reason', 'unknown')
            person = entry.get('person_name', 'unknown')

            by_reason[reason] = by_reason.get(reason, 0) + 1
            by_person[person] = by_person.get(person, 0) + 1

        return {
            "date": date,
            "total_rejections": len(entries),
            "by_reason": by_reason,
            "by_person": by_person,
            "entries": entries
        }

    @classmethod
    def cleanup_old_rejections(cls, days: Optional[int] = None):
        """
        Remove rejection logs and images older than retention period.

        Args:
            days: Number of days to retain (default: RETENTION_DAYS)
        """
        if days is None:
            days = cls.RETENTION_DAYS

        cutoff_date = datetime.now() - timedelta(days=days)
        removed_count = 0

        # Cleanup log folders
        if os.path.exists(cls.REJECTION_LOG_PATH):
            for folder_name in os.listdir(cls.REJECTION_LOG_PATH):
                try:
                    folder_date = datetime.strptime(folder_name, '%Y-%m-%d')
                    if folder_date < cutoff_date:
                        folder_path = os.path.join(cls.REJECTION_LOG_PATH, folder_name)
                        shutil.rmtree(folder_path)
                        removed_count += 1
                        logger.info(f"Removed old rejection log folder: {folder_name}")
                except ValueError:
                    continue  # Skip non-date folders

        # Cleanup image folders
        if os.path.exists(cls.REJECTED_IMAGES_PATH):
            for folder_name in os.listdir(cls.REJECTED_IMAGES_PATH):
                try:
                    folder_date = datetime.strptime(folder_name, '%Y-%m-%d')
                    if folder_date < cutoff_date:
                        folder_path = os.path.join(cls.REJECTED_IMAGES_PATH, folder_name)
                        shutil.rmtree(folder_path)
                        removed_count += 1
                        logger.info(f"Removed old rejected images folder: {folder_name}")
                except ValueError:
                    continue

        logger.info(f"Cleanup complete. Removed {removed_count} old folders.")
        return removed_count

    @classmethod
    def get_rejected_images_for_person(cls, person_name: str, date: Optional[str] = None) -> list:
        """
        Get list of rejected images for a person.

        Args:
            person_name: Name of the person
            date: Date string (YYYY-MM-DD) or None for today

        Returns:
            List of rejected image paths
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        person_folder = os.path.join(cls.REJECTED_IMAGES_PATH, date, person_name)

        if not os.path.exists(person_folder):
            return []

        images = []
        for filename in os.listdir(person_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                images.append(os.path.join(person_folder, filename))

        return images
