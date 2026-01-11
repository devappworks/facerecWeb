"""
Storage Management Service
Manages video storage, calculates disk usage, and provides cleanup operations.
"""

import os
import json
import shutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Get project root directory (parent of 'app' directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class StorageService:
    """Service for managing video storage and disk usage"""

    VIDEO_STORAGE = str(PROJECT_ROOT / "storage" / "videos")
    FRAMES_STORAGE = str(PROJECT_ROOT / "storage" / "video_frames")
    RESULTS_STORAGE = str(PROJECT_ROOT / "storage" / "video_results")

    def __init__(self):
        """Initialize storage service"""
        os.makedirs(self.VIDEO_STORAGE, exist_ok=True)
        os.makedirs(self.FRAMES_STORAGE, exist_ok=True)
        os.makedirs(self.RESULTS_STORAGE, exist_ok=True)

    def get_directory_size(self, directory: str) -> float:
        """
        Calculate total size of directory in MB.

        Args:
            directory: Path to directory

        Returns:
            Size in MB
        """
        total_size = 0
        if os.path.exists(directory):
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        if os.path.exists(filepath):
                            total_size += os.path.getsize(filepath)
                    except OSError as e:
                        logger.warning(f"Could not get size of {filepath}: {e}")
                        continue
        return total_size / (1024 * 1024)  # Convert to MB

    def extract_video_id_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract video_id from filename.

        Format: YYYYMMDD_HHMMSS_video_id.ext
        Example: 20250520_143052_a1b2c3d4e5f6.mp4 -> a1b2c3d4e5f6

        Args:
            filename: Video filename

        Returns:
            Video ID or None
        """
        try:
            # Remove extension
            name_without_ext = os.path.splitext(filename)[0]
            # Split by underscore
            parts = name_without_ext.split('_')
            # video_id is the last part
            if len(parts) >= 3:
                return parts[-1]
            return None
        except Exception as e:
            logger.warning(f"Could not extract video_id from {filename}: {e}")
            return None

    def get_video_metadata(self, video_id: str, video_filepath: str) -> Dict:
        """
        Extract metadata for a video.

        Args:
            video_id: Video identifier
            video_filepath: Full path to video file

        Returns:
            Dictionary with video metadata
        """
        try:
            # Get file stats
            stat_info = os.stat(video_filepath)
            size_mb = stat_info.st_size / (1024 * 1024)
            uploaded_at = datetime.fromtimestamp(stat_info.st_mtime).isoformat()

            # Check for frames directory
            frames_dir = os.path.join(self.FRAMES_STORAGE, video_id)
            frames_count = 0
            if os.path.exists(frames_dir):
                frames_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

            # Check for results file
            results_file = None
            results_available = False

            # Try different result file naming patterns
            possible_results = [
                os.path.join(self.RESULTS_STORAGE, f"{video_id}.json"),
                os.path.join(self.RESULTS_STORAGE, f"{video_id}_results.json"),
            ]

            for result_path in possible_results:
                if os.path.exists(result_path):
                    results_file = result_path
                    results_available = True
                    break

            # Load result metadata if available
            duration_seconds = None
            statistics = None
            performance = None
            domain = None
            if results_available and results_file:
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)

                        # Handle both old format (statistics at root) and new format (under face_recognition)
                        if 'face_recognition' in results and results['face_recognition']:
                            face_rec = results['face_recognition']
                            # New format: face_recognition contains statistics and performance
                            statistics = face_rec.get('statistics')
                            performance = face_rec.get('performance')
                            # Duration may be in extraction_info within face_recognition
                            extraction_info = face_rec.get('extraction_info', {})
                            video_info = extraction_info.get('video_info', {})
                            duration_seconds = video_info.get('duration')
                        else:
                            # Old format: statistics and performance at root level
                            statistics = results.get('statistics')
                            performance = results.get('performance')
                            if 'extraction_info' in results and 'video_info' in results['extraction_info']:
                                duration_seconds = results['extraction_info']['video_info'].get('duration')

                        domain = results.get('domain')
                except Exception as e:
                    logger.warning(f"Could not load results for {video_id}: {e}")

            return {
                "video_id": video_id,
                "filename": os.path.basename(video_filepath),
                "size_mb": round(size_mb, 2),
                "uploaded_at": uploaded_at,
                "duration_seconds": duration_seconds,
                "frames_extracted": frames_count,
                "results_available": results_available,
                "statistics": statistics,
                "performance": performance,
                "domain": domain
            }

        except Exception as e:
            logger.error(f"Error getting metadata for {video_id}: {e}")
            return {
                "video_id": video_id,
                "filename": os.path.basename(video_filepath),
                "error": str(e)
            }

    def get_storage_stats(self) -> Dict:
        """
        Get storage statistics including disk usage and video list.

        Returns:
            Dictionary with storage stats
        """
        try:
            # Calculate disk usage
            videos_size = self.get_directory_size(self.VIDEO_STORAGE)
            frames_size = self.get_directory_size(self.FRAMES_STORAGE)
            results_size = self.get_directory_size(self.RESULTS_STORAGE)
            total_size = videos_size + frames_size + results_size

            # Get list of all videos
            videos = []
            if os.path.exists(self.VIDEO_STORAGE):
                for filename in os.listdir(self.VIDEO_STORAGE):
                    filepath = os.path.join(self.VIDEO_STORAGE, filename)
                    if os.path.isfile(filepath):
                        video_id = self.extract_video_id_from_filename(filename)
                        if video_id:
                            metadata = self.get_video_metadata(video_id, filepath)
                            videos.append(metadata)

            # Sort by upload date (newest first)
            videos.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)

            return {
                "success": True,
                "total_videos": len(videos),
                "total_size_mb": round(total_size, 2),
                "videos": videos,
                "disk_usage": {
                    "videos_mb": round(videos_size, 2),
                    "frames_mb": round(frames_size, 2),
                    "results_mb": round(results_size, 2),
                    "total_mb": round(total_size, 2)
                }
            }

        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {
                "success": False,
                "message": f"Error getting storage stats: {str(e)}"
            }

    def get_videos_list(self) -> Dict:
        """
        Get list of all stored videos with metadata.

        Returns:
            Dictionary with videos list
        """
        try:
            videos = []

            if os.path.exists(self.VIDEO_STORAGE):
                for filename in os.listdir(self.VIDEO_STORAGE):
                    filepath = os.path.join(self.VIDEO_STORAGE, filename)
                    if os.path.isfile(filepath):
                        video_id = self.extract_video_id_from_filename(filename)
                        if video_id:
                            metadata = self.get_video_metadata(video_id, filepath)
                            videos.append(metadata)

            # Sort by upload date (newest first)
            videos.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)

            return {
                "success": True,
                "count": len(videos),
                "videos": videos
            }

        except Exception as e:
            logger.error(f"Error getting videos list: {e}")
            return {
                "success": False,
                "message": f"Error getting videos list: {str(e)}"
            }

    def delete_video(self, video_id: str) -> Dict:
        """
        Delete video and all associated files (frames, results).

        Args:
            video_id: Video identifier

        Returns:
            Dictionary with deletion status
        """
        try:
            deleted_items = []
            errors = []

            # Find and delete video file
            video_deleted = False
            if os.path.exists(self.VIDEO_STORAGE):
                for filename in os.listdir(self.VIDEO_STORAGE):
                    if video_id in filename:
                        video_path = os.path.join(self.VIDEO_STORAGE, filename)
                        try:
                            os.remove(video_path)
                            deleted_items.append(f"Video: {filename}")
                            video_deleted = True
                        except Exception as e:
                            errors.append(f"Failed to delete video {filename}: {e}")

            if not video_deleted:
                return {
                    "success": False,
                    "message": f"Video with ID {video_id} not found"
                }

            # Delete frames directory
            frames_dir = os.path.join(self.FRAMES_STORAGE, video_id)
            if os.path.exists(frames_dir):
                try:
                    frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
                    shutil.rmtree(frames_dir)
                    deleted_items.append(f"Frames directory: {frame_count} frames")
                except Exception as e:
                    errors.append(f"Failed to delete frames directory: {e}")

            # Delete results file
            possible_results = [
                os.path.join(self.RESULTS_STORAGE, f"{video_id}.json"),
                os.path.join(self.RESULTS_STORAGE, f"{video_id}_results.json"),
            ]

            for result_path in possible_results:
                if os.path.exists(result_path):
                    try:
                        os.remove(result_path)
                        deleted_items.append(f"Results file: {os.path.basename(result_path)}")
                    except Exception as e:
                        errors.append(f"Failed to delete results file: {e}")

            logger.info(f"Deleted video {video_id}: {deleted_items}")

            return {
                "success": True,
                "video_id": video_id,
                "deleted_items": deleted_items,
                "errors": errors if errors else None,
                "message": f"Successfully deleted video {video_id}"
            }

        except Exception as e:
            logger.error(f"Error deleting video {video_id}: {e}")
            return {
                "success": False,
                "message": f"Error deleting video: {str(e)}"
            }

    def cleanup_old_videos(self, days: int) -> Dict:
        """
        Delete videos older than specified number of days.

        Args:
            days: Delete videos older than this many days

        Returns:
            Dictionary with cleanup results
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            deleted_videos = []
            errors = []
            total_size_freed = 0

            if os.path.exists(self.VIDEO_STORAGE):
                for filename in os.listdir(self.VIDEO_STORAGE):
                    filepath = os.path.join(self.VIDEO_STORAGE, filename)

                    if os.path.isfile(filepath):
                        # Check file modification time
                        stat_info = os.stat(filepath)
                        file_date = datetime.fromtimestamp(stat_info.st_mtime)

                        if file_date < cutoff_date:
                            video_id = self.extract_video_id_from_filename(filename)

                            if video_id:
                                # Get size before deletion
                                file_size = stat_info.st_size / (1024 * 1024)

                                # Delete video and associated files
                                result = self.delete_video(video_id)

                                if result.get('success'):
                                    deleted_videos.append({
                                        "video_id": video_id,
                                        "filename": filename,
                                        "upload_date": file_date.isoformat(),
                                        "size_mb": round(file_size, 2)
                                    })
                                    total_size_freed += file_size
                                else:
                                    errors.append(f"Failed to delete {filename}: {result.get('message')}")

            logger.info(f"Cleanup complete: Deleted {len(deleted_videos)} videos older than {days} days, freed {total_size_freed:.2f} MB")

            return {
                "success": True,
                "days_threshold": days,
                "cutoff_date": cutoff_date.isoformat(),
                "videos_deleted": len(deleted_videos),
                "size_freed_mb": round(total_size_freed, 2),
                "deleted_videos": deleted_videos,
                "errors": errors if errors else None
            }

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {
                "success": False,
                "message": f"Error during cleanup: {str(e)}"
            }
