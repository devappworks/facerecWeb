"""
Video Face Recognition Service
Extracts frames from video and performs face recognition on each frame.
"""

import os
import cv2
import uuid
import json
import time
import psutil
import logging
import traceback
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from flask import current_app
from threading import Thread
from contextlib import nullcontext

from app.services.recognition_service import RecognitionService
from app.services.modal_service import ModalService
from app.services.embedding_matcher import EmbeddingMatcher
from app.services.face_tracker import (
    FaceTracker,
    FaceDetection,
    FaceTrack,
    convert_gpu_result_to_detections
)
from app.services.track_identity_resolver import TrackIdentityResolver
from app.services.gemini_video_service import get_gemini_service, GeminiVideoService
from app.config import HISTORICAL_PERSONS_BLACKLIST

# Configure dedicated video processing logger with file handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'storage', 'logs')
os.makedirs(_log_dir, exist_ok=True)

# Add file handler for video processing logs (10MB max, keep 5 backups)
_video_log_file = os.path.join(_log_dir, 'video_processing.log')
_file_handler = RotatingFileHandler(_video_log_file, maxBytes=10*1024*1024, backupCount=5)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(_file_handler)

# Also ensure console output for gunicorn
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(_console_handler)


def apply_super_resolution_if_small(frame: np.ndarray, min_face_size: int = 100) -> np.ndarray:
    """
    Apply super-resolution upscaling to frames with small faces.

    For faces smaller than min_face_size pixels, upscale using bicubic interpolation
    to improve recognition accuracy. This is a lightweight approach that doesn't
    require deep learning models.

    Args:
        frame: Input frame (BGR)
        min_face_size: Minimum face dimension before upscaling (default: 100px)

    Returns:
        Upscaled frame if faces are small, otherwise original frame
    """
    try:
        h, w = frame.shape[:2]

        # Simple heuristic: If frame is smaller than 640x480, likely has small faces
        # More sophisticated: Use face detection, but that's expensive
        # For video processing, we use a simple size check
        if min(h, w) < 480:
            # Upscale by 2x using bicubic interpolation (good quality, fast)
            upscaled = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            logger.debug(f"Applied super-resolution upscaling: {w}x{h} → {upscaled.shape[1]}x{upscaled.shape[0]}")
            return upscaled

        return frame

    except Exception as e:
        logger.error(f"Error in super-resolution: {e}")
        return frame  # Return original if error


def calculate_frame_quality(frame: np.ndarray) -> Dict:
    """
    Calculate quality metrics for a video frame.

    Returns scores for blur, contrast, brightness that can be used to:
    1. Filter out low-quality frames before recognition
    2. Weight recognition results by frame quality

    Args:
        frame: OpenCV image (BGR format)

    Returns:
        Dictionary with quality metrics and overall score
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Blur detection (Laplacian variance)
        # Higher = sharper, lower = more blurry
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()

        # 2. Contrast (standard deviation of pixel intensities)
        contrast_score = gray.std()

        # 3. Brightness (mean pixel intensity)
        brightness = gray.mean()
        # Optimal brightness is around 127 (middle of 0-255)
        # Score decreases as brightness deviates from optimal
        brightness_score = 100 - abs(brightness - 127) * (100 / 127)

        # 4. Edge density (Canny edge detection)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = (edges > 0).sum() / edges.size * 100

        # Calculate overall quality score (0-100)
        # Weights based on importance for face recognition
        # - Blur is most important (faces must be sharp)
        # - Contrast helps distinguish features
        # - Brightness affects overall visibility
        # - Edge density indicates detail level

        # Normalize scores to 0-100 range
        blur_normalized = min(100, blur_score / 5)  # 500+ var = 100 score
        contrast_normalized = min(100, contrast_score * 2)  # 50+ std = 100 score
        edge_normalized = min(100, edge_density * 2)  # 50%+ density = 100 score

        overall_score = (
            blur_normalized * 0.4 +      # 40% weight on sharpness
            contrast_normalized * 0.25 +  # 25% weight on contrast
            brightness_score * 0.15 +     # 15% weight on brightness
            edge_normalized * 0.20        # 20% weight on edge detail
        )

        # Convert numpy types to Python native types for JSON serialization
        overall_score_float = float(overall_score)

        return {
            "blur_score": round(float(blur_score), 2),
            "blur_normalized": round(float(blur_normalized), 2),
            "contrast_score": round(float(contrast_score), 2),
            "contrast_normalized": round(float(contrast_normalized), 2),
            "brightness": round(float(brightness), 2),
            "brightness_score": round(float(brightness_score), 2),
            "edge_density": round(float(edge_density), 2),
            "edge_normalized": round(float(edge_normalized), 2),
            "overall_score": round(overall_score_float, 2),
            "is_good_quality": bool(overall_score_float >= 50)  # Threshold for "good" quality
        }

    except Exception as e:
        logger.error(f"Error calculating frame quality: {str(e)}")
        return {
            "blur_score": 0,
            "blur_normalized": 0,
            "contrast_score": 0,
            "contrast_normalized": 0,
            "brightness": 0,
            "brightness_score": 0,
            "edge_density": 0,
            "edge_normalized": 0,
            "overall_score": 0,
            "is_good_quality": False,
            "error": str(e)
        }


def select_best_frames(frames_with_quality: List[Dict],
                       top_percent: float = 0.5,
                       min_frames: int = 3) -> List[Dict]:
    """
    Select the best quality frames from a list.

    Args:
        frames_with_quality: List of frame info dicts with 'quality' key
        top_percent: Select top N% of frames (default: 50%)
        min_frames: Minimum number of frames to select (default: 3)

    Returns:
        List of selected frame info dicts, sorted by quality (best first)
    """
    if not frames_with_quality:
        return []

    # Sort by overall quality score (descending)
    sorted_frames = sorted(
        frames_with_quality,
        key=lambda x: x.get('quality', {}).get('overall_score', 0),
        reverse=True
    )

    # Calculate how many frames to select
    num_to_select = max(
        min_frames,
        int(len(sorted_frames) * top_percent)
    )

    # Don't select more than available
    num_to_select = min(num_to_select, len(sorted_frames))

    selected = sorted_frames[:num_to_select]

    logger.info(f"Selected {len(selected)}/{len(frames_with_quality)} best quality frames "
                f"(top {top_percent*100:.0f}%, min {min_frames})")

    return selected


def check_temporal_clustering(frame_numbers: List[int], total_frames: int,
                              max_gap: int = 50, min_cluster_size: int = 3) -> bool:
    """
    Check if detections are temporally clustered (consecutive/nearby frames) or scattered.

    Scattered detections across random frames are suspicious and indicate false positives.
    Real persons tend to appear in clusters (consecutive scenes).

    Args:
        frame_numbers: List of frame numbers where person was detected
        total_frames: Total frames in video
        max_gap: Maximum frame gap to consider part of same cluster (default: 50 frames = ~25 seconds at 2fps)
        min_cluster_size: Minimum frames in a cluster to be valid (default: 3)

    Returns:
        True if detections are clustered, False if scattered

    Examples:
        - [10, 11, 12, 50, 51, 52]: Two clusters of 3 → True (clustered)
        - [10, 100, 200, 300, 400]: Single frames scattered → False
        - [10, 15, 20, 25, 30]: Close sequence within max_gap → True
    """
    if not frame_numbers or len(frame_numbers) < min_cluster_size:
        return False

    # Sort frame numbers
    sorted_frames = sorted(frame_numbers)

    # Find clusters (groups of frames within max_gap)
    clusters = []
    current_cluster = [sorted_frames[0]]

    for i in range(1, len(sorted_frames)):
        gap = sorted_frames[i] - sorted_frames[i-1]

        if gap <= max_gap:
            # Same cluster
            current_cluster.append(sorted_frames[i])
        else:
            # New cluster - save current and start new
            if len(current_cluster) >= min_cluster_size:
                clusters.append(current_cluster)
            current_cluster = [sorted_frames[i]]

    # Don't forget the last cluster
    if len(current_cluster) >= min_cluster_size:
        clusters.append(current_cluster)

    # Check if we have at least one valid cluster
    has_valid_cluster = len(clusters) > 0

    # Calculate percentage of frames that are in clusters
    frames_in_clusters = sum(len(cluster) for cluster in clusters)
    cluster_percentage = frames_in_clusters / len(frame_numbers) if frame_numbers else 0

    # Pass if:
    # 1. At least one valid cluster exists AND
    # 2. At least 60% of detections are part of clusters
    is_clustered = has_valid_cluster and cluster_percentage >= 0.6

    if not is_clustered:
        logger.debug(f"Temporal clustering FAILED: {len(clusters)} clusters, "
                    f"{cluster_percentage*100:.1f}% in clusters, "
                    f"frames: {sorted_frames[:10]}{'...' if len(sorted_frames) > 10 else ''}")

    return is_clustered


def calculate_dynamic_min_occurrence(total_frames: int,
                                      min_absolute: int = 5,
                                      max_absolute: int = 15,
                                      percentage: float = 0.03) -> int:
    """
    Calculate dynamic min_frame_occurrence based on video length.

    Balances between:
    - Short videos: Need at least 5 frames to confirm identity (stricter to reduce false positives)
    - Long videos: Need more frames to filter noise
    - Collage videos: Cap at 15 so people appearing for ~45+ seconds are included

    Args:
        total_frames: Total number of frames in video
        min_absolute: Minimum threshold (default: 5, increased from 2 to reduce false positives)
        max_absolute: Maximum threshold for collage videos (default: 15, increased from 10)
        percentage: Percentage of frames required (default: 3%, reduced from 5% but with higher min)

    Returns:
        Calculated min_frame_occurrence value

    Examples:
        - 10 frames (30s video): max(5, min(15, 0.3)) = 5
        - 40 frames (2min video): max(5, min(15, 1.2)) = 5
        - 100 frames (5min video): max(5, min(15, 3)) = 5
        - 200 frames (10min video): max(5, min(15, 6)) = 6
        - 500 frames (25min video): max(5, min(15, 15)) = 15
        - 600 frames (30min video): max(5, min(15, 18)) = 15 (capped)
    """
    calculated = int(total_frames * percentage)
    return max(min_absolute, min(max_absolute, calculated))


def aggregate_frame_results(results: List[Dict],
                           min_frame_occurrence: int = None,
                           use_weighted_voting: bool = True) -> Dict:
    """
    Aggregate recognition results across multiple frames for multi-frame voting.

    This function implements Phase 2 of video recognition improvement:
    - Collects per-person statistics across all processed frames
    - Filters out persons who appear in fewer than min_frame_occurrence frames
    - Uses weighted voting by confidence (higher confidence = more weight)
    - Returns aggregated confidence metrics for reliable person detection

    Args:
        results: List of per-frame recognition results
        min_frame_occurrence: Minimum frames required. If None, calculated dynamically
                              based on video length (5% of frames, min 2, max 10)
        use_weighted_voting: If True, weight each frame by its confidence score

    Returns:
        Dictionary with aggregated person statistics and filtered results
    """
    total_frames = len(results)

    # Calculate dynamic min_frame_occurrence if not specified
    min_frame_occurrence_dynamic = min_frame_occurrence is None
    if min_frame_occurrence_dynamic:
        min_frame_occurrence = calculate_dynamic_min_occurrence(total_frames)
        logger.info(f"Dynamic min_frame_occurrence calculated: {min_frame_occurrence} "
                   f"(based on {total_frames} total frames)")

    # Collect per-person statistics across frames
    person_stats = defaultdict(lambda: {
        "frame_count": 0,
        "confidences": [],
        "frame_numbers": [],
        "timestamps": [],
        "occurrences_per_frame": []  # Track per-frame match counts
    })

    recognized_frames = 0

    for result in results:
        if result.get("recognized") and result.get("person"):
            recognized_frames += 1
            person = result["person"]
            confidence = result.get("confidence", 0) or 0  # Handle None
            frame_number = result.get("frame_number", 0)
            timestamp = result.get("timestamp", 0)

            # Get occurrence count from raw_result if available
            occurrences = 1
            raw_result = result.get("raw_result", {})
            if raw_result.get("best_match"):
                occurrences = raw_result["best_match"].get("confidence_metrics", {}).get("occurrences", 1)

            person_stats[person]["frame_count"] += 1
            person_stats[person]["confidences"].append(confidence)
            person_stats[person]["frame_numbers"].append(frame_number)
            person_stats[person]["timestamps"].append(timestamp)
            person_stats[person]["occurrences_per_frame"].append(occurrences)

    # Calculate aggregated metrics for each person
    aggregated_persons = {}
    filtered_persons = {}  # Persons that pass min_frame_occurrence filter

    for person, stats in person_stats.items():
        confidences = stats["confidences"]
        frame_count = stats["frame_count"]

        # Basic statistics
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        best_confidence = max(confidences) if confidences else 0
        avg_occurrences = sum(stats["occurrences_per_frame"]) / len(stats["occurrences_per_frame"]) if stats["occurrences_per_frame"] else 0
        occurrence_rate = (frame_count / total_frames) * 100 if total_frames > 0 else 0

        # Weighted voting: sum of squared confidence scores
        # Squaring gives much more weight to high-confidence matches
        # This means quality matters more than quantity:
        #
        # Example comparison:
        #   Person A: 3 frames at 70% → sum(70² + 75² + 72²)/100 = 156.89
        #   Person B: 5 frames at 46% → sum(45² + 48² + 42² + 50² + 46²)/100 = 107.09
        #   Person A wins despite fewer frames because of higher confidence
        #
        # The squared approach ensures:
        # - 1 frame at 80% (64) beats 2 frames at 40% (32)
        # - But 2 frames at 60% (72) beats 1 frame at 80% (64)
        if use_weighted_voting:
            # Sum of squared confidences, normalized
            weighted_score = sum(c * c for c in confidences) / 100.0
            # Weighted frame equivalent: how many "perfect 100%" frames this equals
            weighted_frame_equivalent = weighted_score / 100.0
        else:
            weighted_score = frame_count * 100  # Just use frame count
            weighted_frame_equivalent = float(frame_count)

        person_data = {
            "frame_count": frame_count,
            "occurrence_rate": round(occurrence_rate, 2),
            "avg_confidence": round(avg_confidence, 2),
            "best_confidence": round(best_confidence, 2),
            "avg_occurrences_per_frame": round(avg_occurrences, 1),
            "total_training_matches": sum(stats["occurrences_per_frame"]),
            "weighted_score": round(weighted_score, 2),
            "weighted_frame_equivalent": round(weighted_frame_equivalent, 2),
            "frame_numbers": stats["frame_numbers"],
            "timestamps": stats["timestamps"]
        }

        aggregated_persons[person] = person_data

        # Apply min_frame_occurrence filter
        if frame_count >= min_frame_occurrence:
            filtered_persons[person] = person_data

    # Sort by weighted_score (descending) if using weighted voting,
    # otherwise by frame_count then avg_confidence
    if use_weighted_voting:
        sorted_filtered = dict(sorted(
            filtered_persons.items(),
            key=lambda x: x[1]["weighted_score"],
            reverse=True
        ))
        sorted_all = dict(sorted(
            aggregated_persons.items(),
            key=lambda x: x[1]["weighted_score"],
            reverse=True
        ))
    else:
        sorted_filtered = dict(sorted(
            filtered_persons.items(),
            key=lambda x: (x[1]["frame_count"], x[1]["avg_confidence"]),
            reverse=True
        ))
        sorted_all = dict(sorted(
            aggregated_persons.items(),
            key=lambda x: (x[1]["frame_count"], x[1]["avg_confidence"]),
            reverse=True
        ))

    # Determine primary person (highest weighted score that passes filter)
    primary_person = None
    primary_score = 0
    if sorted_filtered:
        primary_person = list(sorted_filtered.keys())[0]
        primary_score = sorted_filtered[primary_person]["weighted_score"]

    # Determine "confirmed" persons - those who are likely real detections
    # Criteria:
    # 1. Must pass min_frame_occurrence filter
    # 2. Weighted score must be >= 10% of primary person's score (relative threshold)
    # 3. OR weighted score must be >= 50 (absolute minimum for confidence)
    # 4. Must have at least ONE frame with >= 65% confidence (best_confidence threshold)
    #    - This filters out persistent low-confidence matches (e.g., historical photos matching modern faces)
    #
    # This helps distinguish between:
    # - Real persons: appear consistently with decent confidence
    # - False positives: appear in few frames or with very low confidence
    confirmed_persons = {}
    relative_threshold = primary_score * 0.10  # 10% of primary person's score
    absolute_minimum = 50.0  # Minimum weighted score to be considered
    best_confidence_minimum = 75.0  # At least one frame must have this confidence (IMPROVED: raised from 70% to 75% to reduce false positives)
    min_confidence_std_threshold = 20.0  # Maximum std deviation in confidence (NEW: detects inconsistent/suspicious detections)

    for person, stats in sorted_filtered.items():
        score = stats["weighted_score"]
        best_conf = stats.get("best_confidence", 0)
        confidences = stats.get("confidences", [])

        # NEW: Calculate confidence consistency (standard deviation)
        # High variance = inconsistent detection across frames (red flag for false positives)
        confidence_std = 0
        if len(confidences) > 1:
            mean_conf = sum(confidences) / len(confidences)
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
            confidence_std = variance ** 0.5

        # NEW: Check temporal clustering (are detections consecutive or scattered?)
        # Scattered detections across random frames are suspicious
        frame_numbers = stats.get("frame_numbers", [])
        is_clustered = check_temporal_clustering(frame_numbers, total_frames)

        # IMPROVED CRITERIA:
        # 1. Score threshold (unchanged)
        # 2. Best confidence >= 75% (raised from 70%)
        # 3. Confidence std <= 20% (NEW: consistency check)
        # 4. Temporal clustering check (NEW: consecutive or clustered frames)
        passes_score = (score >= relative_threshold or score >= absolute_minimum)
        passes_confidence = best_conf >= best_confidence_minimum
        passes_consistency = confidence_std <= min_confidence_std_threshold
        passes_clustering = is_clustered

        if passes_score and passes_confidence and passes_consistency and passes_clustering:
            confirmed_persons[person] = stats
            # Add diagnostic info
            confirmed_persons[person]["confidence_std"] = round(confidence_std, 2)
            confirmed_persons[person]["is_clustered"] = is_clustered
        else:
            # Log why person was filtered out
            reasons = []
            if not passes_score:
                reasons.append(f"low_score({score:.1f})")
            if not passes_confidence:
                reasons.append(f"low_best_conf({best_conf:.1f}%)")
            if not passes_consistency:
                reasons.append(f"high_variance(std={confidence_std:.1f}%)")
            if not passes_clustering:
                reasons.append("scattered_frames")
            logger.debug(f"Filtered out {person}: {', '.join(reasons)}")

    # Log confirmation criteria
    if sorted_filtered:
        logger.info(f"Confirmation thresholds: relative={relative_threshold:.1f} (10% of {primary_score:.1f}), "
                   f"absolute={absolute_minimum}, best_confidence>={best_confidence_minimum}%")
        logger.info(f"Confirmed persons: {len(confirmed_persons)}/{len(sorted_filtered)} passed filter")

    # Filter out historical persons (blacklist)
    # These are historical figures whose photos should not appear in modern video recognition
    # Check both underscore and space versions since names may be formatted differently
    def is_blacklisted(person_name):
        """Check if person is in blacklist, handling both underscore and space formats"""
        if not person_name:
            return False
        # Check exact match
        if person_name in HISTORICAL_PERSONS_BLACKLIST:
            return True
        # Check with underscores replaced by spaces
        if person_name.replace(' ', '_') in HISTORICAL_PERSONS_BLACKLIST:
            return True
        # Check with spaces replaced by underscores
        if person_name.replace('_', ' ') in [p.replace('_', ' ') for p in HISTORICAL_PERSONS_BLACKLIST]:
            return True
        return False

    blacklisted_persons = []
    for person in list(confirmed_persons.keys()):
        if is_blacklisted(person):
            blacklisted_persons.append(person)
            logger.info(f"Removing blacklisted historical person from results: {person}")
            del confirmed_persons[person]

    for person in list(sorted_filtered.keys()):
        if is_blacklisted(person):
            if person not in blacklisted_persons:
                blacklisted_persons.append(person)
            del sorted_filtered[person]

    # Update primary person if it was blacklisted
    if is_blacklisted(primary_person):
        logger.warning(f"Primary person '{primary_person}' was blacklisted! Selecting new primary from remaining persons.")
        if sorted_filtered:
            primary_person = list(sorted_filtered.keys())[0]
            primary_score = sorted_filtered[primary_person]["weighted_score"]
            logger.info(f"New primary person: {primary_person} (score: {primary_score:.2f})")
        else:
            primary_person = None
            primary_score = 0
            logger.warning("No persons remain after blacklist filtering!")

    return {
        "total_frames": total_frames,
        "recognized_frames": recognized_frames,
        "min_frame_occurrence": min_frame_occurrence,
        "min_frame_occurrence_dynamic": min_frame_occurrence_dynamic,  # Was it auto-calculated?
        "use_weighted_voting": use_weighted_voting,
        "primary_person": primary_person,
        "primary_person_score": round(primary_score, 2),
        "confirmed_persons": confirmed_persons,  # Persons confirmed as real detections
        "confirmation_threshold_relative": round(relative_threshold, 2),
        "confirmation_threshold_absolute": absolute_minimum,
        "filtered_persons": sorted_filtered,  # Only persons meeting min_frame_occurrence
        "all_persons": sorted_all,  # All detected persons (for debugging)
        "persons_filtered_out": [p for p in sorted_all if p not in sorted_filtered],
        "blacklisted_persons": blacklisted_persons,  # Historical persons removed from results
    }


class VideoService:
    """Service for processing videos and recognizing faces in frames"""

    VIDEO_STORAGE = "storage/videos"
    FRAMES_STORAGE = "storage/video_frames"
    RESULTS_STORAGE = "storage/video_results"

    def __init__(self):
        """Initialize video service"""
        os.makedirs(self.VIDEO_STORAGE, exist_ok=True)
        os.makedirs(self.FRAMES_STORAGE, exist_ok=True)
        os.makedirs(self.RESULTS_STORAGE, exist_ok=True)

    def save_video(self, video_bytes: bytes, original_filename: str) -> Dict:
        """
        Save uploaded video file.

        Args:
            video_bytes: Video file bytes
            original_filename: Original filename

        Returns:
            Dictionary with video_id and file path
        """
        try:
            video_id = str(uuid.uuid4())[:12]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Get file extension
            ext = os.path.splitext(original_filename)[1] or '.mp4'
            filename = f"{timestamp}_{video_id}{ext}"

            video_path = os.path.join(self.VIDEO_STORAGE, filename)

            # Save video file
            with open(video_path, 'wb') as f:
                f.write(video_bytes)

            # Get video info
            video_info = self._get_video_info(video_path)

            logger.info(f"Saved video {video_id}: {video_info}")

            return {
                "success": True,
                "video_id": video_id,
                "video_path": video_path,
                "filename": filename,
                "size_mb": len(video_bytes) / (1024 * 1024),
                **video_info
            }

        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
            return {
                "success": False,
                "message": f"Error saving video: {str(e)}"
            }

    def _get_video_info(self, video_path: str) -> Dict:
        """
        Get video metadata using OpenCV.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video info
        """
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return {
                    "error": "Could not open video"
                }

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_seconds": round(duration, 2),
                "duration_formatted": self._format_duration(duration)
            }

        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return {"error": str(e)}

    def _format_duration(self, seconds: float) -> str:
        """Format duration as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def extract_frames(self, video_path: str, video_id: str,
                      interval_seconds: float = 3.0) -> Dict:
        """
        Extract frames from video at specified interval.

        Args:
            video_path: Path to video file
            video_id: Unique video identifier
            interval_seconds: Extract 1 frame every N seconds (default 3)

        Returns:
            Dictionary with extraction results
        """
        start_time = time.time()

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return {
                    "success": False,
                    "message": "Could not open video"
                }

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Calculate frame interval
            frame_interval = int(fps * interval_seconds)

            logger.info(f"Video: {video_id}, FPS: {fps}, Total frames: {total_frames}, "
                       f"Duration: {duration:.2f}s, Extracting every {frame_interval} frames "
                       f"({interval_seconds}s)")

            # Create directory for frames
            frames_dir = os.path.join(self.FRAMES_STORAGE, video_id)
            os.makedirs(frames_dir, exist_ok=True)

            extracted_frames = []
            frame_number = 0
            extracted_count = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Extract frame at interval
                if frame_number % frame_interval == 0:
                    timestamp = frame_number / fps if fps > 0 else 0

                    frame_filename = f"frame_{frame_number:06d}_t{timestamp:.2f}s.jpg"
                    frame_path = os.path.join(frames_dir, frame_filename)

                    # Save frame as JPEG
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

                    extracted_frames.append({
                        "frame_number": frame_number,
                        "timestamp": round(timestamp, 2),
                        "filename": frame_filename,
                        "path": frame_path
                    })

                    extracted_count += 1

                    if extracted_count % 10 == 0:
                        logger.info(f"Extracted {extracted_count} frames...")

                frame_number += 1

            cap.release()

            extraction_time = time.time() - start_time

            logger.info(f"Extracted {extracted_count} frames from {total_frames} total "
                       f"in {extraction_time:.2f}s")

            return {
                "success": True,
                "video_id": video_id,
                "total_frames": total_frames,
                "extracted_count": extracted_count,
                "extraction_time": round(extraction_time, 2),
                "frames": extracted_frames,
                "frames_dir": frames_dir,
                "video_info": {
                    "fps": fps,
                    "duration": round(duration, 2),
                    "interval_seconds": interval_seconds
                }
            }

        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return {
                "success": False,
                "message": f"Error extracting frames: {str(e)}"
            }

    def recognize_frames(self, video_id: str, domain: str,
                        extraction_result: Dict,
                        min_frame_occurrence: int = None,
                        quality_filter: bool = True,
                        quality_threshold: float = 40.0,
                        use_weighted_voting: bool = True) -> Dict:
        """
        Perform face recognition on extracted frames with multi-frame voting.

        Args:
            video_id: Unique video identifier
            domain: Domain for face recognition
            extraction_result: Result from extract_frames()
            min_frame_occurrence: Minimum frames a person must appear in to be included
                                  in final results. If None (default), calculated dynamically
                                  based on video length (5% of frames, min 2, max 10).
            quality_filter: If True, skip low-quality frames (default: True)
            quality_threshold: Minimum quality score (0-100) for a frame to be processed (default: 40)
            use_weighted_voting: If True, weight votes by confidence (default: True)

        Returns:
            Dictionary with recognition results including:
            - Per-frame results
            - Aggregated multi-frame voting results (with weighted scores)
            - Filtered persons (meeting min_frame_occurrence threshold)
            - Quality statistics
        """
        start_time = time.time()

        try:
            frames = extraction_result.get('frames', [])

            if not frames:
                return {
                    "success": False,
                    "message": "No frames to process"
                }

            logger.info(f"Starting recognition on {len(frames)} frames for video {video_id}")
            logger.info(f"Quality filter: {quality_filter}, threshold: {quality_threshold}")

            # System monitoring
            process = psutil.Process()
            initial_cpu = psutil.cpu_percent(interval=None)
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

            results = []
            recognized_count = 0
            failed_count = 0
            skipped_quality_count = 0
            persons_detected = set()
            quality_scores = []

            for i, frame_info in enumerate(frames):
                frame_path = frame_info['path']
                frame_number = frame_info['frame_number']
                timestamp = frame_info['timestamp']

                try:
                    # Read frame as image for quality check
                    frame_img = cv2.imread(frame_path)
                    if frame_img is None:
                        raise ValueError(f"Could not read frame: {frame_path}")

                    # Phase 3: Calculate frame quality
                    quality = calculate_frame_quality(frame_img)
                    quality_scores.append(quality['overall_score'])

                    # Skip low-quality frames if filter is enabled
                    if quality_filter and quality['overall_score'] < quality_threshold:
                        skipped_quality_count += 1
                        logger.debug(f"Skipping frame {frame_number} - quality {quality['overall_score']:.1f} < {quality_threshold}")
                        results.append({
                            "frame_number": frame_number,
                            "timestamp": timestamp,
                            "filename": frame_info['filename'],
                            "recognized": False,
                            "person": None,
                            "confidence": None,
                            "quality": quality,
                            "skipped_reason": "low_quality"
                        })
                        continue

                    # Read frame as bytes for recognition
                    with open(frame_path, 'rb') as f:
                        frame_bytes = f.read()

                    # Recognize faces (use video threshold - more lenient than image threshold)
                    recognition_result = RecognitionService.recognize_face(
                        frame_bytes,
                        domain,
                        source_type="video"
                    )

                    # Parse result
                    recognized = False
                    person_name = None
                    confidence = None

                    # Check for both old format ('success': True) and new format ('status': 'success')
                    if recognition_result.get('success') or recognition_result.get('status') == 'success':
                        person_name = recognition_result.get('person')
                        # Get confidence from best_match if available
                        if not recognition_result.get('confidence') and recognition_result.get('best_match'):
                            confidence = recognition_result['best_match'].get('confidence_metrics', {}).get('confidence_percentage')
                        else:
                            confidence = recognition_result.get('confidence')

                        if person_name and person_name != 'Unknown':
                            recognized = True
                            recognized_count += 1
                            persons_detected.add(person_name)

                    results.append({
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "filename": frame_info['filename'],
                        "recognized": recognized,
                        "person": person_name,
                        "confidence": confidence,
                        "quality": quality,
                        "raw_result": recognition_result
                    })

                    # Log progress every 10 frames
                    if (i + 1) % 10 == 0:
                        current_cpu = psutil.cpu_percent(interval=None)
                        current_memory = process.memory_info().rss / (1024 * 1024)

                        logger.info(
                            f"Processed {i + 1}/{len(frames)} frames | "
                            f"Recognized: {recognized_count} | "
                            f"Skipped (quality): {skipped_quality_count} | "
                            f"CPU: {current_cpu:.1f}% | "
                            f"Memory: {current_memory:.1f} MB"
                        )

                except Exception as frame_error:
                    logger.error(f"Error processing frame {frame_number}: {str(frame_error)}")
                    failed_count += 1

                    results.append({
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "filename": frame_info['filename'],
                        "recognized": False,
                        "person": None,
                        "confidence": None,
                        "quality": quality if 'quality' in dir() else None,
                        "error": str(frame_error)
                    })

            # Final system stats
            final_cpu = psutil.cpu_percent(interval=None)
            final_memory = process.memory_info().rss / (1024 * 1024)
            avg_cpu = (initial_cpu + final_cpu) / 2
            memory_used = final_memory - initial_memory

            processing_time = time.time() - start_time
            fps_processed = len(frames) / processing_time if processing_time > 0 else 0

            # Phase 2: Multi-frame voting aggregation with weighted voting
            # Aggregate results across frames and filter by min_frame_occurrence
            aggregated = aggregate_frame_results(
                results,
                min_frame_occurrence=min_frame_occurrence,
                use_weighted_voting=use_weighted_voting
            )

            # Get the actual min_frame_occurrence used (may have been calculated dynamically)
            actual_min_occurrence = aggregated['min_frame_occurrence']

            # Log aggregation results
            logger.info(f"\n{'='*60}")
            dynamic_note = " (dynamic)" if aggregated['min_frame_occurrence_dynamic'] else ""
            weighted_note = " [weighted]" if use_weighted_voting else ""
            logger.info(f"MULTI-FRAME VOTING RESULTS{weighted_note} (min_occurrence={actual_min_occurrence}{dynamic_note}):")
            logger.info(f"{'='*60}")
            logger.info(f"Primary person detected: {aggregated['primary_person']}")
            logger.info(f"Persons passing filter ({actual_min_occurrence}+ frames):")
            for person, stats in aggregated['filtered_persons'].items():
                weighted_info = f", weighted={stats['weighted_score']:.1f}" if use_weighted_voting else ""
                logger.info(f"  ✅ {person}: {stats['frame_count']} frames, "
                           f"avg conf: {stats['avg_confidence']}%, "
                           f"best: {stats['best_confidence']}%{weighted_info}")
            if aggregated['persons_filtered_out']:
                logger.info(f"Persons filtered out (<{actual_min_occurrence} frames):")
                for person in aggregated['persons_filtered_out']:
                    stats = aggregated['all_persons'][person]
                    logger.info(f"  ❌ {person}: {stats['frame_count']} frame(s)")
            logger.info(f"{'='*60}\n")

            # Save results to JSON
            result_file = os.path.join(
                self.RESULTS_STORAGE,
                f"{video_id}_results.json"
            )

            # Phase 3: Calculate quality statistics
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            min_quality = min(quality_scores) if quality_scores else 0
            max_quality = max(quality_scores) if quality_scores else 0
            frames_processed = len(frames) - skipped_quality_count

            complete_result = {
                "video_id": video_id,
                "domain": domain,
                "processed_at": datetime.now().isoformat(),
                "use_weighted_voting": use_weighted_voting,
                "extraction_info": extraction_result,
                "statistics": {
                    "total_frames": len(frames),
                    "frames_processed": frames_processed,
                    "frames_skipped_quality": skipped_quality_count,
                    "recognized_frames": recognized_count,
                    "failed_frames": failed_count,
                    "recognition_rate": round((recognized_count / frames_processed) * 100, 2) if frames_processed > 0 else 0,
                    "unique_persons": len(persons_detected),
                    "persons_list": sorted(list(persons_detected))
                },
                "quality_statistics": {
                    "filter_enabled": quality_filter,
                    "quality_threshold": quality_threshold,
                    "avg_quality_score": round(avg_quality, 2),
                    "min_quality_score": round(min_quality, 2),
                    "max_quality_score": round(max_quality, 2),
                    "frames_above_threshold": len([q for q in quality_scores if q >= quality_threshold]),
                    "frames_below_threshold": skipped_quality_count
                },
                "multi_frame_voting": {
                    "min_frame_occurrence": min_frame_occurrence,
                    "min_frame_occurrence_dynamic": aggregated["min_frame_occurrence_dynamic"],
                    "use_weighted_voting": aggregated["use_weighted_voting"],
                    "primary_person": aggregated["primary_person"],
                    "primary_person_score": aggregated["primary_person_score"],
                    "confirmed_persons": aggregated["confirmed_persons"],
                    "confirmation_threshold_relative": aggregated["confirmation_threshold_relative"],
                    "confirmation_threshold_absolute": aggregated["confirmation_threshold_absolute"],
                    "filtered_persons": aggregated["filtered_persons"],
                    "all_detected_persons": aggregated["all_persons"],
                    "persons_filtered_out": aggregated["persons_filtered_out"],
                    "blacklisted_persons": aggregated["blacklisted_persons"]
                },
                "performance": {
                    "processing_time_seconds": round(processing_time, 2),
                    "frames_per_second": round(fps_processed, 2),
                    "avg_cpu_percent": round(avg_cpu, 2),
                    "memory_used_mb": round(memory_used, 2),
                    "final_memory_mb": round(final_memory, 2)
                },
                "results": results
            }

            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(complete_result, f, ensure_ascii=False, indent=2)

            # Build enhanced log message with quality and multi-frame voting info
            filtered_count = len(aggregated['filtered_persons'])
            total_detected = len(aggregated['all_persons'])

            logger.info(
                f"Video {video_id} processing complete:\n"
                f"  Total frames: {len(frames)}\n"
                f"  Processed: {frames_processed} (skipped {skipped_quality_count} low-quality)\n"
                f"  Recognized: {recognized_count} ({(recognized_count/frames_processed*100):.1f}% of processed)\n"
                f"  Quality: avg={avg_quality:.1f}, min={min_quality:.1f}, max={max_quality:.1f}\n"
                f"  Unique persons (raw): {len(persons_detected)}\n"
                f"  Persons after voting filter: {filtered_count}/{total_detected}\n"
                f"  Primary person: {aggregated['primary_person']}\n"
                f"  Processing time: {processing_time:.2f}s ({fps_processed:.2f} FPS)\n"
                f"  CPU usage: {avg_cpu:.1f}%\n"
                f"  Memory used: {memory_used:.1f} MB"
            )

            return {
                "success": True,
                **complete_result
            }

        except Exception as e:
            logger.error(f"Error recognizing frames: {str(e)}")
            return {
                "success": False,
                "message": f"Error recognizing frames: {str(e)}"
            }

    def process_video(self, video_bytes: bytes, original_filename: str,
                     domain: str, interval_seconds: float = 3.0,
                     min_frame_occurrence: int = None,
                     quality_filter: bool = True,
                     quality_threshold: float = 40.0,
                     use_weighted_voting: bool = True) -> Dict:
        """
        Complete video processing pipeline: save, extract, recognize.

        Args:
            video_bytes: Video file bytes
            original_filename: Original filename
            domain: Domain for face recognition
            interval_seconds: Extract 1 frame every N seconds
            min_frame_occurrence: Minimum frames a person must appear in.
                                  If None (default), calculated dynamically based on video length.
            quality_filter: If True, skip low-quality frames (default: True)
            quality_threshold: Minimum quality score for frame processing (default: 40)
            use_weighted_voting: If True, weight votes by confidence (default: True)

        Returns:
            Dictionary with complete processing results
        """
        try:
            # Step 1: Save video
            save_result = self.save_video(video_bytes, original_filename)

            if not save_result.get('success'):
                return save_result

            video_id = save_result['video_id']
            video_path = save_result['video_path']

            # Step 2: Extract frames
            extraction_result = self.extract_frames(
                video_path,
                video_id,
                interval_seconds
            )

            if not extraction_result.get('success'):
                return extraction_result

            # Step 3: Recognize faces with quality filtering and weighted multi-frame voting
            recognition_result = self.recognize_frames(
                video_id,
                domain,
                extraction_result,
                min_frame_occurrence=min_frame_occurrence,
                quality_filter=quality_filter,
                quality_threshold=quality_threshold,
                use_weighted_voting=use_weighted_voting
            )

            return recognition_result

        except Exception as e:
            logger.error(f"Error in video processing pipeline: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing video: {str(e)}"
            }

    def process_video_async(self, video_bytes: bytes, original_filename: str,
                           domain: str, interval_seconds: float = 3.0) -> str:
        """
        Start async video processing in background thread.

        Args:
            video_bytes: Video file bytes
            original_filename: Original filename
            domain: Domain for face recognition
            interval_seconds: Extract 1 frame every N seconds

        Returns:
            video_id for tracking progress
        """
        # First save the video to get video_id
        save_result = self.save_video(video_bytes, original_filename)

        if not save_result.get('success'):
            raise Exception(save_result.get('message', 'Failed to save video'))

        video_id = save_result['video_id']
        video_path = save_result['video_path']

        # Start background processing
        app_context = current_app.app_context()
        thread = Thread(
            target=self._process_video_thread,
            args=(video_id, video_path, domain, interval_seconds, app_context)
        )
        thread.daemon = True
        thread.start()

        return video_id

    def retry_face_recognition(self, video_id: str) -> Dict:
        """
        Retry face recognition for a video stuck in 'gemini_complete' stage.

        This is useful when the background thread was killed (e.g., server restart)
        before face recognition could complete.

        Args:
            video_id: Video identifier

        Returns:
            Dict with success status and message
        """
        # Check if results file exists
        result_file = os.path.join(self.RESULTS_STORAGE, f"{video_id}_results.json")
        if not os.path.exists(result_file):
            return {
                "success": False,
                "message": f"No results file found for video {video_id}"
            }

        # Load current results
        with open(result_file, 'r', encoding='utf-8') as f:
            current_results = json.load(f)

        # Check if video is in a retry-able state
        stage = current_results.get('processing_stage', '')
        if stage == 'complete':
            return {
                "success": False,
                "message": "Video processing is already complete"
            }

        if stage not in ['gemini_complete', 'gemini_failed', 'extraction_failed', 'face_recognition_failed']:
            return {
                "success": False,
                "message": f"Video is in stage '{stage}', cannot retry"
            }

        # Find the video file
        video_path = None
        domain = current_results.get('domain', 'serbia')

        for filename in os.listdir(self.VIDEO_STORAGE):
            if video_id in filename:
                video_path = os.path.join(self.VIDEO_STORAGE, filename)
                break

        if not video_path or not os.path.exists(video_path):
            return {
                "success": False,
                "message": f"Video file not found for {video_id}"
            }

        # Update status to show retry is in progress
        current_results['message'] = "Retrying face recognition..."
        current_results['updated_at'] = datetime.now().isoformat()
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(current_results, f, ensure_ascii=False, indent=2)

        # Start face recognition in background thread
        try:
            app_context = current_app.app_context()
        except RuntimeError:
            # Running outside Flask context - run synchronously
            self._retry_face_recognition_thread(video_id, video_path, domain, current_results, None)
            return {
                "success": True,
                "message": "Face recognition completed (sync)",
                "video_id": video_id
            }

        thread = Thread(
            target=self._retry_face_recognition_thread,
            args=(video_id, video_path, domain, current_results, app_context)
        )
        thread.daemon = True
        thread.start()

        return {
            "success": True,
            "message": "Face recognition retry started",
            "video_id": video_id
        }

    def _retry_face_recognition_thread(self, video_id: str, video_path: str,
                                       domain: str, existing_results: Dict, app_context):
        """Background thread for retrying face recognition only."""
        # Handle both Flask context and direct calls
        context_manager = app_context if app_context else nullcontext()
        with context_manager:
            try:
                logger.info(f"Retrying face recognition for video {video_id}")
                total_start = time.time()

                # Extract frames
                extraction = self.extract_frames_inmemory(video_path, video_id, 3.0)

                if not extraction.get('success'):
                    logger.error(f"Frame extraction failed: {extraction}")
                    existing_results['processing_stage'] = 'extraction_failed'
                    existing_results['message'] = f"Frame extraction failed: {extraction.get('message')}"
                    self._save_partial_results(video_id, existing_results, "extraction_failed")
                    return

                # Run face recognition
                face_rec_result = self.recognize_frames_with_tracking(
                    video_id=video_id,
                    domain=domain,
                    frames=extraction['frames'],
                    metadata=extraction['metadata'],
                    quality_threshold=40.0,
                    use_tracking=True,
                    iou_threshold=0.3,
                    min_track_length=3,
                    min_vote_ratio=0.20,
                    min_consistency=0.25
                )

                if face_rec_result.get('success'):
                    face_rec_result['extraction_info']['video_info'] = extraction['video_info']
                    face_rec_result['extraction_info']['extraction_time'] = extraction['extraction_time']

                    total_time = time.time() - total_start

                    # Build final result, preserving Gemini analysis from existing results
                    final_result = {
                        "success": True,
                        "video_id": video_id,
                        "domain": domain,
                        "processed_at": datetime.now().isoformat(),
                        "processing_stage": "complete",
                        "total_processing_time_seconds": round(total_time, 2),
                        "gemini_analysis": existing_results.get("gemini_analysis"),
                        "gemini_metadata": existing_results.get("gemini_metadata"),
                        "face_recognition": {
                            "method": face_rec_result.get("method"),
                            "extraction_info": face_rec_result.get("extraction_info"),
                            "statistics": face_rec_result.get("statistics"),
                            "tracking_results": face_rec_result.get("tracking_results"),
                            "multi_frame_voting": face_rec_result.get("multi_frame_voting"),
                            "performance": face_rec_result.get("performance"),
                            "frame_results": face_rec_result.get("frame_results", [])
                        },
                        "person_correlation": self._correlate_persons(
                            existing_results.get("gemini_analysis"),
                            face_rec_result
                        )
                    }

                    result_file = os.path.join(self.RESULTS_STORAGE, f"{video_id}_results.json")
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(final_result, f, ensure_ascii=False, indent=2)

                    logger.info(f"Face recognition retry complete for video {video_id} in {total_time:.1f}s")
                else:
                    logger.error(f"Face recognition retry failed: {face_rec_result}")
                    existing_results['processing_stage'] = 'face_recognition_failed'
                    existing_results['message'] = f"Face recognition failed: {face_rec_result.get('message')}"
                    self._save_partial_results(video_id, existing_results, "face_recognition_failed")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in face recognition retry for {video_id}: {error_msg}")
                existing_results['processing_stage'] = 'error'
                existing_results['message'] = f"Retry failed: {error_msg}"
                existing_results['updated_at'] = datetime.now().isoformat()
                result_file = os.path.join(self.RESULTS_STORAGE, f"{video_id}_results.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)

    def _run_gemini_analysis(self, video_id: str, video_path: str) -> Dict:
        """
        Run Gemini video analysis (fast, ~20-30s).

        Returns initial insights before face recognition completes.
        Results are saved immediately so users can see them.
        """
        gemini_service = get_gemini_service()

        if not gemini_service.is_available():
            logger.warning("Gemini video service not available, skipping video analysis")
            return {"success": False, "error": "Gemini service not available"}

        logger.info(f"Starting Gemini video analysis for {video_id}")
        start_time = time.time()

        try:
            result = gemini_service.analyze_video(video_path)

            if result.get('success'):
                elapsed = time.time() - start_time
                logger.info(f"Gemini analysis complete for {video_id} in {elapsed:.1f}s")
                logger.info(f"  Video type: {result.get('video_type')}")
                if result.get('usage'):
                    logger.info(f"  Cost: ${result['usage'].get('estimated_cost_usd', 0):.4f}")
            else:
                logger.warning(f"Gemini analysis failed for {video_id}: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Gemini analysis error for {video_id}: {e}")
            return {"success": False, "error": str(e)}

    def _save_partial_results(self, video_id: str, results: Dict, stage: str):
        """
        Save partial results during processing.

        Allows frontend to show Gemini results before face recognition completes.
        """
        result_file = os.path.join(self.RESULTS_STORAGE, f"{video_id}_results.json")
        results['processing_stage'] = stage
        results['updated_at'] = datetime.now().isoformat()

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved partial results for {video_id} at stage: {stage}")

    def _process_video_thread(self, video_id: str, video_path: str,
                             domain: str, interval_seconds: float, app_context):
        """Background thread for video processing (GPU with CPU fallback)"""
        with app_context:
            try:
                logger.info(f"Starting background processing for video {video_id}")
                total_start = time.time()

                # PHASE 1: Gemini Analysis (FAST - runs first)
                # This gives users initial insights within ~20-30 seconds
                gemini_result = self._run_gemini_analysis(video_id, video_path)

                # Save Gemini results immediately so frontend can show them
                if gemini_result.get('success'):
                    initial_results = {
                        "success": True,
                        "video_id": video_id,
                        "domain": domain,
                        "processing_stage": "gemini_complete",
                        "gemini_analysis": gemini_result.get("analysis"),
                        "gemini_metadata": {
                            "video_type": gemini_result.get("video_type"),
                            "model": gemini_result.get("model"),
                            "processing_time_seconds": gemini_result.get("processing_time_seconds"),
                            "usage": gemini_result.get("usage")
                        },
                        "face_recognition": None,  # Not ready yet
                        "message": "Gemini analysis complete. Face recognition in progress..."
                    }
                    self._save_partial_results(video_id, initial_results, "gemini_complete")
                else:
                    # Gemini failed, but continue with face recognition
                    initial_results = {
                        "success": True,
                        "video_id": video_id,
                        "domain": domain,
                        "processing_stage": "gemini_failed",
                        "gemini_analysis": None,
                        "gemini_metadata": {
                            "error": gemini_result.get("error", "Unknown error")
                        },
                        "face_recognition": None,
                        "message": "Gemini analysis failed. Face recognition in progress..."
                    }
                    self._save_partial_results(video_id, initial_results, "gemini_failed")

                # PHASE 2: Face Recognition (SLOWER - enriches results)
                logger.info(f"Starting face recognition for video {video_id}")

                # Extract frames to memory for GPU processing
                extraction = self.extract_frames_inmemory(video_path, video_id, interval_seconds)

                if not extraction.get('success'):
                    logger.error(f"Frame extraction failed: {extraction}")
                    # Update results with extraction failure
                    initial_results['processing_stage'] = 'extraction_failed'
                    initial_results['message'] = f"Frame extraction failed: {extraction.get('message')}"
                    self._save_partial_results(video_id, initial_results, "extraction_failed")
                    return

                # GPU recognition with CPU fallback
                face_rec_result = self.recognize_frames_with_tracking(
                    video_id=video_id,
                    domain=domain,
                    frames=extraction['frames'],
                    metadata=extraction['metadata'],
                    quality_threshold=40.0,
                    use_tracking=True,
                    iou_threshold=0.3,
                    min_track_length=3,
                    min_vote_ratio=0.20,  # Lowered from 0.5 to allow confirmation with 20% vote share
                    min_consistency=0.25  # Lowered from 0.4 to allow confirmation with 25% consistency
                )

                if face_rec_result.get('success'):
                    # Add video info to result
                    face_rec_result['extraction_info']['video_info'] = extraction['video_info']
                    face_rec_result['extraction_info']['extraction_time'] = extraction['extraction_time']

                    # PHASE 3: Merge Gemini + Face Recognition results
                    total_time = time.time() - total_start

                    final_result = {
                        "success": True,
                        "video_id": video_id,
                        "domain": domain,
                        "processed_at": datetime.now().isoformat(),
                        "processing_stage": "complete",
                        "total_processing_time_seconds": round(total_time, 2),

                        # Gemini analysis (visual, quotes, body language, etc.)
                        "gemini_analysis": gemini_result.get("analysis") if gemini_result.get("success") else None,
                        "gemini_metadata": {
                            "success": gemini_result.get("success", False),
                            "video_type": gemini_result.get("video_type"),
                            "model": gemini_result.get("model"),
                            "processing_time_seconds": gemini_result.get("processing_time_seconds"),
                            "usage": gemini_result.get("usage"),
                            "error": gemini_result.get("error") if not gemini_result.get("success") else None
                        },

                        # Face recognition results
                        "face_recognition": {
                            "method": face_rec_result.get("method"),
                            "extraction_info": face_rec_result.get("extraction_info"),
                            "statistics": face_rec_result.get("statistics"),
                            "tracking_results": face_rec_result.get("tracking_results"),
                            "multi_frame_voting": face_rec_result.get("multi_frame_voting"),
                            "performance": face_rec_result.get("performance"),
                            "frame_results": face_rec_result.get("frame_results", [])  # Per-frame recognition for UI
                        },

                        # Combined person correlation
                        "person_correlation": self._correlate_persons(
                            gemini_result.get("analysis") if gemini_result.get("success") else None,
                            face_rec_result
                        )
                    }

                    # Save final merged results
                    result_file = os.path.join(self.RESULTS_STORAGE, f"{video_id}_results.json")
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(final_result, f, ensure_ascii=False, indent=2)

                    gpu_status = "GPU" if face_rec_result.get('performance', {}).get('gpu_used') else "CPU fallback"
                    logger.info(f"Processing complete for video {video_id}")
                    logger.info(f"  Gemini: {gemini_result.get('processing_time_seconds', 0):.1f}s")
                    logger.info(f"  Face recognition ({gpu_status}): {face_rec_result.get('performance', {}).get('processing_time_seconds', 0):.1f}s")
                    logger.info(f"  Total: {total_time:.1f}s")
                else:
                    logger.error(f"Face recognition failed: {face_rec_result}")
                    # Save partial result with Gemini only
                    initial_results['processing_stage'] = 'face_recognition_failed'
                    initial_results['message'] = f"Face recognition failed: {face_rec_result.get('message')}"
                    self._save_partial_results(video_id, initial_results, "face_recognition_failed")

            except Exception as e:
                error_msg = str(e)
                error_traceback = traceback.format_exc()
                logger.error(f"Error in background video processing for {video_id}: {error_msg}")
                logger.error(f"Full traceback:\n{error_traceback}")

                # Save error state to result file so users know what went wrong
                try:
                    error_result = {
                        "success": False,
                        "video_id": video_id,
                        "domain": domain,
                        "processing_stage": "error",
                        "error": error_msg,
                        "error_traceback": error_traceback,
                        "updated_at": datetime.now().isoformat(),
                        "message": f"Processing failed: {error_msg}"
                    }
                    result_file = os.path.join(self.RESULTS_STORAGE, f"{video_id}_results.json")
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(error_result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved error state for video {video_id}")
                except Exception as save_error:
                    logger.error(f"Failed to save error state: {save_error}")

    def _correlate_persons(self, gemini_analysis: Optional[Dict], face_rec_result: Dict) -> Dict:
        """
        Cross-reference Gemini-identified people with face recognition results.

        This helps correlate visual descriptions with database matches.
        """
        if not gemini_analysis:
            return {"note": "Gemini analysis not available for correlation"}

        # Get people from Gemini analysis (field name varies by video type)
        gemini_people = []
        for field in ["speakers", "on_screen_people", "participants", "key_people", "people"]:
            if field in gemini_analysis:
                gemini_people.extend(gemini_analysis[field])

        # Get confirmed persons from face recognition
        face_rec_persons = {}

        # Try tracking results format first
        tracking_results = face_rec_result.get("tracking_results", {})
        identity_results = tracking_results.get("identity_results", {})
        confirmed = identity_results.get("confirmed_persons", {})

        if confirmed:
            face_rec_persons = confirmed
        else:
            # Fall back to multi_frame_voting format
            multi_frame = face_rec_result.get("multi_frame_voting", {})
            confirmed = multi_frame.get("confirmed_persons", {})
            face_rec_persons = confirmed

        return {
            "gemini_identified": [
                {
                    "person_id": p.get("person_id"),
                    "identified_name": p.get("identified_name"),
                    "name_source": p.get("name_source"),
                    "role": p.get("role"),
                    "physical_description": p.get("physical_description")
                }
                for p in gemini_people
            ],
            "face_recognition_confirmed": {
                name: {
                    "total_frames": stats.get("total_frames", stats.get("frame_count")),
                    "avg_confidence": stats.get("avg_confidence"),
                    "total_tracks": stats.get("total_tracks", 1)
                }
                for name, stats in face_rec_persons.items()
            },
            "summary": {
                "gemini_people_count": len(gemini_people),
                "face_rec_people_count": len(face_rec_persons),
                "gemini_identified_names": [
                    p.get("identified_name") for p in gemini_people if p.get("identified_name")
                ],
                "face_rec_names": list(face_rec_persons.keys())
            }
        }

    def get_video_result(self, video_id: str) -> Optional[Dict]:
        """
        Get processing results for a video.

        Args:
            video_id: Video identifier

        Returns:
            Dictionary with results or None if not found
        """
        try:
            result_file = os.path.join(
                self.RESULTS_STORAGE,
                f"{video_id}_results.json"
            )

            if not os.path.exists(result_file):
                return None

            with open(result_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error getting video result: {str(e)}")
            return None

    def extract_frames_inmemory(self, video_path: str, video_id: str,
                                interval_seconds: float = 3.0) -> Dict:
        """
        Extract frames to memory (no disk I/O for frames).
        Returns frames as JPEG bytes with quality metrics.
        """
        start_time = time.time()

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return {"success": False, "message": "Could not open video"}

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            frame_interval = int(fps * interval_seconds)

            frames = []
            metadata = []
            frame_number = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_number % frame_interval == 0:
                    timestamp = frame_number / fps if fps > 0 else 0

                    # IMPROVEMENT: Apply super-resolution for small frames
                    # Upscale frames < 480p to improve face recognition on distant/small faces
                    frame_processed = apply_super_resolution_if_small(frame)

                    # Calculate quality (on processed frame)
                    quality = calculate_frame_quality(frame_processed)

                    # Encode to JPEG bytes (in memory)
                    _, buffer = cv2.imencode('.jpg', frame_processed, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    frame_bytes = buffer.tobytes()

                    frames.append(frame_bytes)
                    metadata.append({
                        "frame_number": frame_number,
                        "timestamp": round(timestamp, 2),
                        "quality": quality
                    })

                frame_number += 1

            cap.release()

            return {
                "success": True,
                "frames": frames,
                "metadata": metadata,
                "total_frames": total_frames,
                "extracted_count": len(frames),
                "extraction_time": round(time.time() - start_time, 2),
                "video_info": {
                    "fps": fps,
                    "duration": round(duration, 2),
                    "width": width,
                    "height": height,
                    "interval_seconds": interval_seconds
                }
            }

        except Exception as e:
            logger.error(f"Error extracting frames to memory: {e}")
            return {"success": False, "message": str(e)}

    def recognize_frames_gpu(
        self,
        video_id: str,
        domain: str,
        frames: List[bytes],
        metadata: List[Dict],
        quality_threshold: float = 40.0,
        min_frame_occurrence: int = None,
        use_weighted_voting: bool = True
    ) -> Dict:
        """
        Hybrid face recognition: GPU extracts embeddings, CPU matches against database.

        Architecture:
        - Modal GPU: Face detection + embedding extraction (neural network heavy)
        - Server CPU: Embedding comparison (simple vector math)

        Falls back to full CPU processing if GPU unavailable.
        """
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)

        # Filter by quality
        filtered_frames = []
        filtered_metadata = []
        skipped_count = 0

        for frame, meta in zip(frames, metadata):
            if meta['quality']['overall_score'] >= quality_threshold:
                filtered_frames.append(frame)
                filtered_metadata.append(meta)
            else:
                skipped_count += 1

        logger.info(f"Quality filter: {len(filtered_frames)} pass, {skipped_count} skipped")

        # Select model
        model_name = "ArcFace"

        # Try hybrid GPU+CPU processing
        gpu_used = False

        # Step 1: Extract embeddings using Modal GPU
        gpu_result = ModalService.extract_embeddings(
            frames=filtered_frames,
            model_name=model_name
        )

        if gpu_result.get('success'):
            gpu_used = True
            logger.info(f"GPU embedding extraction: {gpu_result['fps']:.1f} FPS, "
                       f"{gpu_result['frames_processed']} frames")

            # Step 2: Match embeddings on CPU against local database
            results = self._match_embeddings_cpu(
                gpu_result['results'],
                filtered_metadata,
                domain
            )

            # If no pkl database available, fall back to full CPU processing
            if results is None:
                logger.warning("No embedding database available, using full CPU fallback")
                gpu_used = False
                results = self._process_frames_cpu(video_id, domain, filtered_frames, filtered_metadata)

        elif gpu_result.get('use_fallback'):
            logger.warning("GPU unavailable, falling back to full CPU processing")
            results = self._process_frames_cpu(video_id, domain, filtered_frames, filtered_metadata)
        else:
            return {"success": False, "message": gpu_result.get('message')}

        # Multi-frame voting
        aggregated = aggregate_frame_results(
            results,
            min_frame_occurrence=min_frame_occurrence,
            use_weighted_voting=use_weighted_voting
        )

        processing_time = time.time() - start_time
        fps = len(frames) / processing_time if processing_time > 0 else 0
        final_memory = process.memory_info().rss / (1024 * 1024)

        return {
            "success": True,
            "video_id": video_id,
            "domain": domain,
            "processed_at": datetime.now().isoformat(),
            "extraction_info": {
                "total_frames": len(frames),
                "extracted_count": len(filtered_frames),
            },
            "statistics": {
                "total_frames": len(frames),
                "frames_processed": len(filtered_frames),
                "frames_skipped_quality": skipped_count,
                "recognized_frames": sum(1 for r in results if r.get('recognized')),
                "recognition_rate": round(sum(1 for r in results if r.get('recognized')) / len(filtered_frames) * 100, 2) if filtered_frames else 0,
                "unique_persons": len(set(r.get('person') for r in results if r.get('person'))),
                "persons_list": sorted(list(set(r.get('person') for r in results if r.get('person'))))
            },
            "multi_frame_voting": aggregated,
            "performance": {
                "processing_time_seconds": round(processing_time, 2),
                "frames_per_second": round(fps, 2),
                "gpu_used": gpu_used,
                "hybrid_mode": gpu_used,  # True = GPU embeddings + CPU matching
                "memory_used_mb": round(final_memory - initial_memory, 2),
                "final_memory_mb": round(final_memory, 2)
            },
            "results": results
        }

    def recognize_frames_with_tracking(
        self,
        video_id: str,
        domain: str,
        frames: List[bytes],
        metadata: List[Dict],
        quality_threshold: float = 40.0,
        use_tracking: bool = True,
        iou_threshold: float = 0.3,
        min_track_length: int = 3,
        min_vote_ratio: float = 0.5,
        min_consistency: float = 0.4
    ) -> Dict:
        """
        Face recognition with temporal tracking and per-track voting.

        This is the new pipeline that reduces false positives by:
        1. Tracking faces across frames using bounding box IoU
        2. Applying majority voting within each track
        3. Filtering tracks by consistency

        Args:
            video_id: Video identifier
            domain: Database domain
            frames: List of frame bytes (JPEG)
            metadata: List of frame metadata with quality info
            quality_threshold: Minimum quality score to process frame
            use_tracking: If True, use face tracking (if False, use legacy method)
            iou_threshold: Minimum IoU for same track
            min_track_length: Minimum frames for valid track
            min_vote_ratio: Minimum vote ratio for identity
            min_consistency: Minimum consistency for valid identity

        Returns:
            Recognition results with tracking info
        """
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)

        # Filter by quality
        filtered_frames = []
        filtered_metadata = []
        skipped_count = 0

        for frame, meta in zip(frames, metadata):
            if meta['quality']['overall_score'] >= quality_threshold:
                filtered_frames.append(frame)
                filtered_metadata.append(meta)
            else:
                skipped_count += 1

        logger.info(f"Quality filter: {len(filtered_frames)} pass, {skipped_count} skipped")

        if not filtered_frames:
            return {"success": False, "message": "No frames passed quality filter"}

        # If tracking disabled, fall back to legacy method
        if not use_tracking:
            return self.recognize_frames_gpu(
                video_id=video_id,
                domain=domain,
                frames=frames,
                metadata=metadata,
                quality_threshold=quality_threshold
            )

        # Step 1: Extract embeddings using GPU
        model_name = "ArcFace"
        gpu_result = ModalService.extract_embeddings(
            frames=filtered_frames,
            model_name=model_name
        )

        if not gpu_result.get('success'):
            if gpu_result.get('use_fallback'):
                logger.warning("GPU unavailable, falling back to CPU without tracking")
                return self._process_frames_cpu_legacy(video_id, domain, filtered_frames, filtered_metadata)
            return {"success": False, "message": gpu_result.get('message')}

        gpu_used = True
        logger.info(f"GPU extraction: {gpu_result['fps']:.1f} FPS, {gpu_result['frames_processed']} frames")

        # Step 2: Initialize face tracker
        tracker = FaceTracker(
            iou_threshold=iou_threshold,
            max_frames_missing=2,  # Allow gap of 2 frames
            min_track_length=min_track_length,
            use_embedding_similarity=False  # Start simple, enable if needed
        )

        # Step 3: Process each frame through tracker
        frame_results = []  # Store per-frame results for debugging/compatibility

        for gpu_res, meta in zip(gpu_result['results'], filtered_metadata):
            frame_number = meta['frame_number']
            timestamp = meta['timestamp']
            frame_quality = meta['quality']['overall_score']

            # Convert GPU result to FaceDetection objects
            detections = convert_gpu_result_to_detections(
                gpu_result=gpu_res,
                frame_number=frame_number,
                timestamp=timestamp,
                frame_quality=frame_quality
            )

            # Process frame through tracker
            tracker.process_frame(frame_number, detections)

            # Store frame result for compatibility
            frame_results.append({
                "frame_number": frame_number,
                "timestamp": timestamp,
                "faces_detected": len(detections),
                "quality": meta['quality']
            })

        # Step 4: Close all tracks (end of video)
        tracker.close_all_tracks()

        # Step 5: Get valid tracks
        valid_tracks = tracker.get_valid_tracks()
        track_stats = tracker.get_track_statistics()

        logger.info(f"Tracking complete: {track_stats['valid_tracks']} valid tracks "
                    f"(filtered {track_stats['filtered_tracks']} short tracks)")

        # Step 6: Resolve identities for each track
        resolver = TrackIdentityResolver(
            domain=domain,
            embedding_threshold=0.50,
            min_vote_ratio=min_vote_ratio,
            min_consistency=min_consistency,
            min_best_confidence=70.0
        )

        identity_results = resolver.resolve_all_tracks(valid_tracks)

        # Step 7: Build detailed frame results from resolved tracks
        # This provides per-frame recognition data for the UI
        detailed_frame_results = self._build_frame_results_from_tracks(
            frame_results, valid_tracks, identity_results
        )

        # Step 8: Build final results
        processing_time = time.time() - start_time
        fps = len(frames) / processing_time if processing_time > 0 else 0
        final_memory = process.memory_info().rss / (1024 * 1024)

        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"FACE TRACKING RESULTS:")
        logger.info(f"{'='*60}")
        logger.info(f"Primary person: {identity_results['primary_person']}")
        logger.info(f"Confirmed persons:")
        for person, stats in identity_results['confirmed_persons'].items():
            logger.info(f"  ✅ {person}: {stats['total_frames']} frames across {stats['total_tracks']} tracks, "
                       f"avg conf: {stats['avg_confidence']}%")
        logger.info(f"{'='*60}\n")

        return {
            "success": True,
            "video_id": video_id,
            "domain": domain,
            "processed_at": datetime.now().isoformat(),
            "method": "tracking",
            "extraction_info": {
                "total_frames": len(frames),
                "extracted_count": len(filtered_frames),
            },
            "statistics": {
                "total_frames": len(frames),
                "frames_processed": len(filtered_frames),
                "frames_skipped_quality": skipped_count,
                "unique_persons": len(identity_results['confirmed_persons']),
                "persons_list": list(identity_results['confirmed_persons'].keys()),
                # Calculate recognized frames from confirmed persons' total frames
                "recognized_frames": sum(
                    stats.get('total_frames', 0)
                    for stats in identity_results['confirmed_persons'].values()
                ),
                "recognition_rate": round(
                    (sum(stats.get('total_frames', 0) for stats in identity_results['confirmed_persons'].values())
                     / len(filtered_frames) * 100) if filtered_frames else 0, 2
                )
            },
            "tracking_results": {
                "track_statistics": track_stats,
                "identity_results": identity_results,
                "parameters": {
                    "iou_threshold": iou_threshold,
                    "min_track_length": min_track_length,
                    "min_vote_ratio": min_vote_ratio,
                    "min_consistency": min_consistency
                }
            },
            "performance": {
                "processing_time_seconds": round(processing_time, 2),
                "frames_per_second": round(fps, 2),
                "gpu_used": gpu_used,
                "memory_used_mb": round(final_memory - initial_memory, 2)
            },
            "frame_results": detailed_frame_results  # Per-frame recognition data for UI
        }

    def _build_frame_results_from_tracks(
        self,
        basic_frame_results: List[Dict],
        valid_tracks: List['FaceTrack'],
        identity_results: Dict
    ) -> List[Dict]:
        """
        Build detailed per-frame recognition results from tracking data.

        This extracts per-frame recognition info from resolved tracks so the UI
        can display frame-by-frame results (person, confidence, etc.).

        Args:
            basic_frame_results: Basic frame info (frame_number, timestamp, faces_detected)
            valid_tracks: List of FaceTrack objects with detections
            identity_results: Resolved identities including confirmed_persons

        Returns:
            List of frame results with recognition data
        """
        # Build a mapping of frame_number -> recognition data
        frame_recognition_map = {}

        # Get resolved tracks (those with confirmed identity)
        resolved_tracks = identity_results.get('resolved_tracks', [])

        for track_info in resolved_tracks:
            person = track_info.get('person')  # Field name is 'person', not 'identity'
            track_id = track_info.get('track_id')
            # Use the avg_confidence from track_info (this is the RECOGNITION confidence)
            track_avg_confidence = track_info.get('avg_confidence', 0)

            # Find the corresponding track object
            for track in valid_tracks:
                if track.track_id == track_id:
                    # Extract per-frame data from track's detections
                    for detection in track.detections:
                        frame_num = detection.frame_number

                        # Get RECOGNITION confidence from identity_confidences (not detection.confidence which is face DETECTION confidence)
                        # If the person was voted for this detection, get their confidence
                        identity_confs = getattr(detection, 'identity_confidences', {}) or {}
                        if person in identity_confs:
                            confidences = identity_confs[person]
                            confidence = max(confidences) if confidences else track_avg_confidence
                        else:
                            # Fallback to track average confidence
                            confidence = track_avg_confidence

                        # If multiple faces in same frame, keep highest confidence
                        if frame_num not in frame_recognition_map:
                            frame_recognition_map[frame_num] = {
                                'person': person,
                                'confidence': confidence,
                                'recognized': True,
                                'track_id': track_id
                            }
                        elif confidence > frame_recognition_map[frame_num]['confidence']:
                            frame_recognition_map[frame_num] = {
                                'person': person,
                                'confidence': confidence,
                                'recognized': True,
                                'track_id': track_id
                            }
                    break

        # Also include unresolved tracks (faces detected but identity not confirmed)
        unresolved_tracks = identity_results.get('unresolved_tracks', [])

        for track_info in unresolved_tracks:
            track_id = track_info.get('track_id')
            # Get the top vote candidate even if not confirmed
            votes = track_info.get('votes', {})
            top_candidate = max(votes.items(), key=lambda x: x[1])[0] if votes else None
            track_avg_confidence = track_info.get('avg_confidence', 0)

            for track in valid_tracks:
                if track.track_id == track_id:
                    for detection in track.detections:
                        frame_num = detection.frame_number

                        # Get RECOGNITION confidence from identity_confidences
                        identity_confs = getattr(detection, 'identity_confidences', {}) or {}
                        if top_candidate and top_candidate in identity_confs:
                            confidences = identity_confs[top_candidate]
                            confidence = max(confidences) if confidences else track_avg_confidence
                        else:
                            confidence = track_avg_confidence

                        # Only add if not already in map (resolved tracks take priority)
                        if frame_num not in frame_recognition_map:
                            frame_recognition_map[frame_num] = {
                                'person': top_candidate,  # Top candidate, but unconfirmed
                                'confidence': confidence,
                                'recognized': False,  # Mark as unconfirmed
                                'track_id': track_id,
                                'unconfirmed': True
                            }
                    break

        # Merge recognition data with basic frame info
        detailed_results = []

        for frame_info in basic_frame_results:
            frame_num = frame_info['frame_number']
            recognition = frame_recognition_map.get(frame_num, {})

            detailed_results.append({
                'frame_number': frame_num,
                'timestamp': frame_info['timestamp'],
                'faces_detected': frame_info['faces_detected'],
                'quality': frame_info['quality'],
                'person': recognition.get('person'),
                'confidence': recognition.get('confidence'),
                'recognized': recognition.get('recognized', False),
                'track_id': recognition.get('track_id'),
                'unconfirmed': recognition.get('unconfirmed', False)
            })

        return detailed_results

    def _process_frames_cpu_legacy(self, video_id: str, domain: str,
                                    frames: List[bytes], metadata: List[Dict]) -> Dict:
        """Legacy CPU fallback without tracking - just for emergencies"""
        results = self._process_frames_cpu(video_id, domain, frames, metadata)

        # Use existing aggregation
        aggregated = aggregate_frame_results(results, min_frame_occurrence=None, use_weighted_voting=True)

        return {
            "success": True,
            "video_id": video_id,
            "domain": domain,
            "method": "legacy_cpu",
            "multi_frame_voting": aggregated,
            "results": results
        }

    def _match_embeddings_cpu(
        self,
        gpu_results: List[Dict],
        metadata: List[Dict],
        domain: str
    ) -> List[Dict]:
        """
        Match GPU-extracted embeddings against local database on CPU.

        Args:
            gpu_results: Results from Modal with embeddings
            metadata: Frame metadata with quality info
            domain: Domain for database lookup

        Returns:
            List of recognition results, or None if matching unavailable
        """
        # Get embedding matcher for this domain
        matcher = EmbeddingMatcher(domain)
        if not matcher.load_database():
            # No pkl file available - return None to trigger CPU fallback
            logger.warning(f"No embedding database for {domain}, cannot do hybrid matching")
            return None

        results = []
        for gpu_res, meta in zip(gpu_results, metadata):
            frame_number = meta['frame_number']
            timestamp = meta['timestamp']

            if not gpu_res.get('success'):
                results.append({
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "recognized": False,
                    "person": None,
                    "confidence": None,
                    "quality": meta['quality'],
                    "error": gpu_res.get('error', 'GPU extraction failed')
                })
                continue

            faces = gpu_res.get('faces', [])

            if not faces:
                results.append({
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "recognized": False,
                    "person": None,
                    "confidence": None,
                    "quality": meta['quality'],
                    "faces_detected": 0
                })
                continue

            # Match each face's embedding against database
            match_result = matcher.match_frame_embeddings(faces)

            # IMPROVEMENT: Apply quality-weighted confidence adjustment
            # High confidence on low-quality frames is suspicious (potential false positive)
            # Quality score ranges 0-100, we normalize to 0.5-1.0 multiplier
            raw_confidence = match_result['confidence']
            quality_score = meta['quality']['overall_score']

            # Quality weight calculation:
            # - Quality 100 → weight = 1.0 (no penalty)
            # - Quality 80  → weight = 0.9 (10% penalty)
            # - Quality 60  → weight = 0.8 (20% penalty)
            # - Quality 40  → weight = 0.7 (30% penalty - minimum, since frames below 40 are filtered)
            quality_weight = 0.5 + (quality_score / 200.0)  # Maps 0-100 to 0.5-1.0

            # Apply quality weighting ONLY if confidence is suspiciously high (>85%)
            # This penalizes high-confidence matches on low-quality frames
            if raw_confidence and raw_confidence > 85.0 and quality_score < 80:
                adjusted_confidence = raw_confidence * quality_weight
                logger.debug(f"Frame {frame_number}: Quality-adjusted confidence "
                           f"{raw_confidence:.1f}% → {adjusted_confidence:.1f}% "
                           f"(quality={quality_score:.1f}, weight={quality_weight:.2f})")
            else:
                adjusted_confidence = raw_confidence

            results.append({
                "frame_number": frame_number,
                "timestamp": timestamp,
                "recognized": match_result['recognized'],
                "person": match_result['person'],
                "confidence": adjusted_confidence,  # Use quality-adjusted confidence
                "confidence_raw": raw_confidence,   # Keep original for debugging
                "quality": meta['quality'],
                "quality_weight_applied": quality_weight if (raw_confidence and raw_confidence > 85 and quality_score < 80) else 1.0,
                "faces_detected": len(faces),
                "all_matches": match_result.get('all_matches', [])
            })

        return results

    def _convert_gpu_results(self, gpu_results: List[Dict], metadata: List[Dict]) -> List[Dict]:
        """Convert GPU results to standard format"""
        results = []
        for gpu_res, meta in zip(gpu_results, metadata):
            results.append({
                "frame_number": meta['frame_number'],
                "timestamp": meta['timestamp'],
                "recognized": gpu_res.get('recognized', False),
                "person": gpu_res.get('person'),
                "confidence": gpu_res.get('confidence'),
                "quality": meta['quality'],
                "raw_result": gpu_res
            })
        return results

    def _process_frames_cpu(self, video_id: str, domain: str,
                            frames: List[bytes], metadata: List[Dict]) -> List[Dict]:
        """CPU fallback - process frames sequentially"""
        results = []
        for frame_bytes, meta in zip(frames, metadata):
            try:
                recognition_result = RecognitionService.recognize_face(
                    frame_bytes, domain, source_type="video"
                )

                recognized = False
                person_name = None
                confidence = None

                if recognition_result.get('success') or recognition_result.get('status') == 'success':
                    person_name = recognition_result.get('person')
                    if recognition_result.get('best_match'):
                        confidence = recognition_result['best_match'].get('confidence_metrics', {}).get('confidence_percentage')
                    else:
                        confidence = recognition_result.get('confidence')

                    if person_name and person_name != 'Unknown':
                        recognized = True

                results.append({
                    "frame_number": meta['frame_number'],
                    "timestamp": meta['timestamp'],
                    "recognized": recognized,
                    "person": person_name,
                    "confidence": confidence,
                    "quality": meta['quality'],
                    "raw_result": recognition_result
                })
            except Exception as e:
                logger.error(f"CPU frame error: {e}")
                results.append({
                    "frame_number": meta['frame_number'],
                    "timestamp": meta['timestamp'],
                    "recognized": False,
                    "person": None,
                    "confidence": None,
                    "quality": meta['quality'],
                    "error": str(e)
                })
        return results

    def process_video_gpu(self, video_bytes: bytes, original_filename: str,
                          domain: str, interval_seconds: float = 3.0,
                          min_frame_occurrence: int = None,
                          quality_filter: bool = True,
                          quality_threshold: float = 40.0,
                          use_weighted_voting: bool = True) -> Dict:
        """
        GPU-accelerated video processing pipeline.
        Falls back to CPU if GPU unavailable.
        """
        try:
            # Step 1: Save video
            save_result = self.save_video(video_bytes, original_filename)
            if not save_result.get('success'):
                return save_result

            video_id = save_result['video_id']
            video_path = save_result['video_path']

            # Step 2: Extract frames to memory (no disk I/O)
            extraction = self.extract_frames_inmemory(video_path, video_id, interval_seconds)
            if not extraction.get('success'):
                return extraction

            # Step 3: GPU recognition with face tracking
            result = self.recognize_frames_with_tracking(
                video_id=video_id,
                domain=domain,
                frames=extraction['frames'],
                metadata=extraction['metadata'],
                quality_threshold=quality_threshold,
                use_tracking=True,  # Enable face tracking
                iou_threshold=0.3,
                min_track_length=3,
                min_vote_ratio=0.5,
                min_consistency=0.4
            )

            # Add video info to result
            if result.get('success'):
                result['extraction_info']['video_info'] = extraction['video_info']
                result['extraction_info']['extraction_time'] = extraction['extraction_time']

                # Save results to JSON (same as original)
                result_file = os.path.join(self.RESULTS_STORAGE, f"{video_id}_results.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

            return result

        except Exception as e:
            logger.error(f"GPU video processing error: {e}")
            return {"success": False, "message": str(e)}
