"""
Face Tracker Module - Track faces across video frames using bounding box IoU.

This module implements spatial tracking of faces across video frames to enable
per-track identity resolution and reduce false positives.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class FaceDetection:
    """Single face detection in a frame"""
    frame_number: int
    timestamp: float
    bbox: tuple  # (x1, y1, x2, y2) - top-left and bottom-right corners
    embedding: np.ndarray  # 512-dim vector for ArcFace
    confidence: float  # Detection confidence from face detector
    frame_quality: float  # Overall frame quality score


@dataclass
class FaceTrack:
    """A sequence of face detections representing the same person across frames"""
    track_id: int
    detections: List[FaceDetection] = field(default_factory=list)
    identity_votes: Dict[str, int] = field(default_factory=dict)  # person_name -> vote count
    identity_confidences: Dict[str, List[float]] = field(default_factory=dict)  # person_name -> [confidences]
    is_closed: bool = False

    def add_detection(self, detection: FaceDetection):
        """Add a detection to this track"""
        self.detections.append(detection)

    def get_length(self) -> int:
        """Number of frames in this track"""
        return len(self.detections)

    def get_frame_range(self) -> tuple:
        """Return (first_frame, last_frame) numbers"""
        if not self.detections:
            return (0, 0)
        frames = [d.frame_number for d in self.detections]
        return (min(frames), max(frames))

    def get_last_bbox(self) -> Optional[tuple]:
        """Get bounding box from most recent detection"""
        if not self.detections:
            return None
        return self.detections[-1].bbox

    def get_last_embedding(self) -> Optional[np.ndarray]:
        """Get embedding from most recent detection"""
        if not self.detections:
            return None
        return self.detections[-1].embedding

    def add_identity_vote(self, person_name: str, confidence: float):
        """Record an identity match for this track"""
        if person_name not in self.identity_votes:
            self.identity_votes[person_name] = 0
            self.identity_confidences[person_name] = []
        self.identity_votes[person_name] += 1
        self.identity_confidences[person_name].append(confidence)

    def get_final_identity(self, min_vote_ratio: float = 0.5) -> Optional[Dict]:
        """
        Determine final identity using majority voting.

        Args:
            min_vote_ratio: Minimum ratio of votes required (0.5 = majority)

        Returns:
            Dict with 'person', 'confidence', 'vote_count', 'vote_ratio', 'consistency'
            or None if no identity meets threshold
        """
        if not self.identity_votes:
            return None

        total_votes = sum(self.identity_votes.values())
        if total_votes == 0:
            return None

        # Find person with most votes
        best_person = max(self.identity_votes, key=self.identity_votes.get)
        vote_count = self.identity_votes[best_person]
        vote_ratio = vote_count / total_votes

        # Check if meets minimum threshold
        if vote_ratio < min_vote_ratio:
            return None

        # Calculate average confidence for this person
        confidences = self.identity_confidences.get(best_person, [])
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        best_confidence = max(confidences) if confidences else 0

        # Consistency = what % of track frames identified this person
        consistency = vote_count / len(self.detections) if self.detections else 0

        return {
            "person": best_person,
            "avg_confidence": round(avg_confidence, 2),
            "best_confidence": round(best_confidence, 2),
            "vote_count": vote_count,
            "total_votes": total_votes,
            "vote_ratio": round(vote_ratio, 3),
            "consistency": round(consistency, 3),
            "track_length": len(self.detections),
            "frame_range": self.get_frame_range(),
            "all_votes": dict(self.identity_votes)
        }


class FaceTracker:
    """
    Tracks faces across video frames using IoU (bounding box overlap).

    Algorithm:
    1. For each new frame, try to match each detected face to an existing track
    2. Matching is done by IoU between bounding boxes
    3. If IoU > threshold, extend the track
    4. If no match found, create new track
    5. Close tracks that haven't been updated for N frames
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_frames_missing: int = 2,
        min_track_length: int = 3,
        use_embedding_similarity: bool = False,
        embedding_threshold: float = 0.6
    ):
        """
        Args:
            iou_threshold: Minimum IoU to consider same track (0.3 = 30% overlap)
            max_frames_missing: Close track if no match for this many frames
            min_track_length: Minimum detections for a valid track
            use_embedding_similarity: Also use embedding distance for matching
            embedding_threshold: Max embedding distance for same track (if enabled)
        """
        self.iou_threshold = iou_threshold
        self.max_frames_missing = max_frames_missing
        self.min_track_length = min_track_length
        self.use_embedding_similarity = use_embedding_similarity
        self.embedding_threshold = embedding_threshold

        self.tracks: List[FaceTrack] = []
        self.next_track_id: int = 0
        self.current_frame: int = -1
        self.frames_since_update: Dict[int, int] = {}  # track_id -> frames since last update

    def calculate_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.

        Args:
            bbox1, bbox2: Tuples of (x1, y1, x2, y2)

        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def calculate_embedding_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine distance between two embeddings"""
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 1.0

        similarity = dot / (norm1 * norm2)
        return 1 - similarity

    def process_frame(self, frame_number: int, detections: List[FaceDetection]) -> None:
        """
        Process all face detections from a single frame.

        Args:
            frame_number: The frame number being processed
            detections: List of FaceDetection objects for this frame
        """
        self.current_frame = frame_number

        # Get active (non-closed) tracks
        active_tracks = [t for t in self.tracks if not t.is_closed]

        # Track which detections have been matched
        matched_detection_indices = set()
        matched_track_ids = set()

        # Try to match each detection to an existing track
        # Use Hungarian algorithm for optimal matching, or greedy for simplicity

        # Build cost matrix (negative IoU, so lower is better)
        matches = []  # List of (track_idx, detection_idx, iou_score)

        for t_idx, track in enumerate(active_tracks):
            last_bbox = track.get_last_bbox()
            if last_bbox is None:
                continue

            for d_idx, detection in enumerate(detections):
                iou = self.calculate_iou(last_bbox, detection.bbox)

                # Optionally also check embedding similarity
                if self.use_embedding_similarity and iou > 0:
                    last_emb = track.get_last_embedding()
                    if last_emb is not None:
                        emb_dist = self.calculate_embedding_distance(last_emb, detection.embedding)
                        if emb_dist > self.embedding_threshold:
                            iou = 0  # Reject match if embedding too different

                if iou >= self.iou_threshold:
                    matches.append((t_idx, d_idx, iou))

        # Greedy matching: sort by IoU descending, take best matches
        matches.sort(key=lambda x: x[2], reverse=True)

        for t_idx, d_idx, iou in matches:
            if t_idx in matched_track_ids or d_idx in matched_detection_indices:
                continue

            # Match found - extend track
            track = active_tracks[t_idx]
            track.add_detection(detections[d_idx])
            self.frames_since_update[track.track_id] = 0

            matched_track_ids.add(t_idx)
            matched_detection_indices.add(d_idx)

        # Create new tracks for unmatched detections
        for d_idx, detection in enumerate(detections):
            if d_idx not in matched_detection_indices:
                new_track = FaceTrack(track_id=self.next_track_id)
                new_track.add_detection(detection)
                self.tracks.append(new_track)
                self.frames_since_update[self.next_track_id] = 0
                self.next_track_id += 1

        # Update frames_since_update for tracks not updated this frame
        for track in active_tracks:
            if track.track_id not in [active_tracks[i].track_id for i in matched_track_ids]:
                self.frames_since_update[track.track_id] = \
                    self.frames_since_update.get(track.track_id, 0) + 1

                # Close track if missing for too long
                if self.frames_since_update[track.track_id] > self.max_frames_missing:
                    track.is_closed = True

    def close_all_tracks(self) -> None:
        """Close all remaining open tracks (call at end of video)"""
        for track in self.tracks:
            track.is_closed = True

    def get_valid_tracks(self) -> List[FaceTrack]:
        """Return tracks that meet minimum length requirement"""
        return [t for t in self.tracks if t.get_length() >= self.min_track_length]

    def get_all_tracks(self) -> List[FaceTrack]:
        """Return all tracks (including short ones, for debugging)"""
        return self.tracks

    def get_track_statistics(self) -> Dict:
        """Return statistics about tracking"""
        all_tracks = self.tracks
        valid_tracks = self.get_valid_tracks()

        return {
            "total_tracks": len(all_tracks),
            "valid_tracks": len(valid_tracks),
            "filtered_tracks": len(all_tracks) - len(valid_tracks),
            "avg_track_length": sum(t.get_length() for t in valid_tracks) / len(valid_tracks) if valid_tracks else 0,
            "max_track_length": max((t.get_length() for t in all_tracks), default=0),
            "min_track_length_threshold": self.min_track_length
        }


def extract_bbox_from_detection(face_data: Dict) -> Optional[tuple]:
    """
    Extract bounding box from face detection result.

    Different detectors return bbox in different formats. Handle all cases.

    Args:
        face_data: Dict from face detector with facial_area or bbox

    Returns:
        Tuple (x1, y1, x2, y2) or None if not found
    """
    # DeepFace format: 'facial_area' with x, y, w, h
    if 'facial_area' in face_data:
        fa = face_data['facial_area']
        if isinstance(fa, dict):
            x = fa.get('x', 0)
            y = fa.get('y', 0)
            w = fa.get('w', 0)
            h = fa.get('h', 0)
            return (x, y, x + w, y + h)
        elif isinstance(fa, (list, tuple)) and len(fa) == 4:
            x, y, w, h = fa
            return (x, y, x + w, y + h)

    # RetinaFace format: 'bbox' with [x1, y1, x2, y2]
    if 'bbox' in face_data:
        bbox = face_data['bbox']
        if len(bbox) == 4:
            return tuple(bbox)

    # Direct x1, y1, x2, y2 keys
    if all(k in face_data for k in ['x1', 'y1', 'x2', 'y2']):
        return (face_data['x1'], face_data['y1'], face_data['x2'], face_data['y2'])

    # Region format from some detectors
    if 'region' in face_data:
        region = face_data['region']
        if isinstance(region, dict):
            x = region.get('x', 0)
            y = region.get('y', 0)
            w = region.get('w', 0)
            h = region.get('h', 0)
            return (x, y, x + w, y + h)

    return None


def convert_gpu_result_to_detections(
    gpu_result: Dict,
    frame_number: int,
    timestamp: float,
    frame_quality: float
) -> List[FaceDetection]:
    """
    Convert GPU extraction result to list of FaceDetection objects.

    Args:
        gpu_result: Result from Modal GPU with 'faces' list
        frame_number: Current frame number
        timestamp: Timestamp in seconds
        frame_quality: Quality score of this frame

    Returns:
        List of FaceDetection objects
    """
    detections = []

    faces = gpu_result.get('faces', [])
    for face in faces:
        embedding = face.get('embedding', [])
        if not embedding:
            continue

        bbox = extract_bbox_from_detection(face)
        if bbox is None:
            # If no bbox, we can't track this face spatially
            # Create a dummy bbox based on frame (will create new track each time)
            bbox = (0, 0, 100, 100)

        detection = FaceDetection(
            frame_number=frame_number,
            timestamp=timestamp,
            bbox=bbox,
            embedding=np.array(embedding),
            confidence=face.get('confidence', face.get('det_score', 0.0)),
            frame_quality=frame_quality
        )
        detections.append(detection)

    return detections
