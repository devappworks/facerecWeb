# Face Tracking Implementation Plan

## Problem Statement

The video face recognition system produces false positives where people are incorrectly identified in frames. For example, in a video of Rasim_Ljajic (44 frames), the system also falsely detected:
- Mihajlo_Mudrik - 6 frames (95.9% confidence)
- Vladan_Petrov - 3 frames (96.8% confidence)
- Mehdi_Taremi - 3 frames (90.7% confidence)

**Root cause**: Each frame is processed independently. There is no temporal continuity - the system doesn't know that the face in frame 10 is the same face in frame 11.

**Solution**: Implement face tracking across frames with per-track identity voting. False positives are typically non-consecutive, so if frame 15 incorrectly matches "Vladan_Petrov" but frames 14 and 16 match "Rasim_Ljajic", track-level voting filters it out.

---

## Architecture Overview

```
CURRENT PIPELINE:
Frame → Detect Face → Extract Embedding → Match to DB → Result per frame
                                                              ↓
                                                    Aggregate all frames
                                                              ↓
                                                    Multi-frame voting

NEW PIPELINE:
Frame → Detect Face → Extract Embedding ──┐
                                          ↓
                              FACE TRACKER (NEW)
                              - Track faces across frames
                              - Group detections into "tracks"
                                          ↓
                              PER-TRACK VOTING (NEW)
                              - Match embeddings within each track
                              - Majority voting per track
                                          ↓
                              TRACK FILTERING (NEW)
                              - Filter short/inconsistent tracks
                                          ↓
                                    Final Results
```

---

## Implementation Tasks

### Task 1: Create Face Tracker Module

**File to create**: `/home/facereco/facerecWeb/app/services/face_tracker.py`

**Purpose**: Track faces across video frames using bounding box IoU (Intersection over Union) and optionally embedding similarity.

#### 1.1 Data Structures

Create these classes/dataclasses:

```python
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
```

#### 1.2 FaceTracker Class

```python
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
```

#### 1.3 Helper Functions

Add these helper functions to the module:

```python
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
```

---

### Task 2: Create Track Identity Resolver

**File to create**: `/home/facereco/facerecWeb/app/services/track_identity_resolver.py`

**Purpose**: Resolve identities for each track by matching embeddings to database and applying majority voting.

```python
"""
Track Identity Resolver - Assigns identities to face tracks using majority voting.

For each track:
1. Match each detection's embedding against the database
2. Collect votes for each person
3. Apply majority voting to determine final identity
4. Filter tracks by consistency and confidence
"""

import logging
from typing import List, Dict, Optional, Any
from .face_tracker import FaceTrack, FaceDetection
from .embedding_matcher import EmbeddingMatcher

logger = logging.getLogger(__name__)


class TrackIdentityResolver:
    """
    Resolves identities for face tracks using the embedding database.
    """

    def __init__(
        self,
        domain: str,
        embedding_threshold: float = 0.50,
        min_vote_ratio: float = 0.5,
        min_consistency: float = 0.4,
        min_best_confidence: float = 70.0
    ):
        """
        Args:
            domain: Database domain (e.g., 'serbia', 'slovenia')
            embedding_threshold: Max cosine distance for a match (lower = stricter)
            min_vote_ratio: Minimum ratio of votes for winning identity (0.5 = majority)
            min_consistency: Minimum % of track frames with winning identity
            min_best_confidence: Minimum best confidence % for a valid identity
        """
        self.domain = domain
        self.embedding_threshold = embedding_threshold
        self.min_vote_ratio = min_vote_ratio
        self.min_consistency = min_consistency
        self.min_best_confidence = min_best_confidence

        # Get embedding matcher
        self.matcher = EmbeddingMatcher(domain)
        self.matcher_loaded = self.matcher.load_database()

        if not self.matcher_loaded:
            logger.warning(f"Could not load embedding database for {domain}")

    def resolve_track_identity(self, track: FaceTrack) -> Optional[Dict]:
        """
        Determine identity for a single track.

        Args:
            track: FaceTrack with detections

        Returns:
            Identity result dict or None if no valid identity found
        """
        if not self.matcher_loaded:
            return None

        if track.get_length() == 0:
            return None

        # Match each detection in the track
        for detection in track.detections:
            matches = self.matcher.find_matches(
                query_embedding=detection.embedding.tolist(),
                threshold=self.embedding_threshold,
                top_k=3
            )

            # Record vote for best match
            if matches:
                best_match = matches[0]
                track.add_identity_vote(
                    person_name=best_match['person'],
                    confidence=best_match['confidence']
                )

        # Get final identity using majority voting
        identity = track.get_final_identity(min_vote_ratio=self.min_vote_ratio)

        if identity is None:
            return None

        # Apply additional filters
        if identity['consistency'] < self.min_consistency:
            logger.debug(f"Track {track.track_id} rejected: consistency {identity['consistency']:.2f} < {self.min_consistency}")
            return None

        if identity['best_confidence'] < self.min_best_confidence:
            logger.debug(f"Track {track.track_id} rejected: best_confidence {identity['best_confidence']:.1f} < {self.min_best_confidence}")
            return None

        identity['track_id'] = track.track_id
        return identity

    def resolve_all_tracks(self, tracks: List[FaceTrack]) -> Dict[str, Any]:
        """
        Resolve identities for all tracks and aggregate results.

        Args:
            tracks: List of FaceTrack objects

        Returns:
            Dict with resolved identities and statistics
        """
        resolved_tracks = []
        unresolved_tracks = []
        person_aggregation = {}  # person_name -> aggregated stats

        for track in tracks:
            identity = self.resolve_track_identity(track)

            if identity:
                resolved_tracks.append(identity)

                # Aggregate by person
                person = identity['person']
                if person not in person_aggregation:
                    person_aggregation[person] = {
                        "person": person,
                        "total_frames": 0,
                        "total_tracks": 0,
                        "confidences": [],
                        "track_ids": [],
                        "frame_ranges": []
                    }

                agg = person_aggregation[person]
                agg['total_frames'] += identity['track_length']
                agg['total_tracks'] += 1
                agg['confidences'].append(identity['avg_confidence'])
                agg['track_ids'].append(identity['track_id'])
                agg['frame_ranges'].append(identity['frame_range'])
            else:
                unresolved_tracks.append({
                    "track_id": track.track_id,
                    "track_length": track.get_length(),
                    "frame_range": track.get_frame_range(),
                    "votes": dict(track.identity_votes) if track.identity_votes else {}
                })

        # Calculate final stats for each person
        confirmed_persons = {}
        for person, agg in person_aggregation.items():
            avg_conf = sum(agg['confidences']) / len(agg['confidences']) if agg['confidences'] else 0
            confirmed_persons[person] = {
                "person": person,
                "total_frames": agg['total_frames'],
                "total_tracks": agg['total_tracks'],
                "avg_confidence": round(avg_conf, 2),
                "track_ids": agg['track_ids'],
                "frame_ranges": agg['frame_ranges']
            }

        # Sort by total_frames descending
        confirmed_persons = dict(sorted(
            confirmed_persons.items(),
            key=lambda x: x[1]['total_frames'],
            reverse=True
        ))

        # Determine primary person
        primary_person = None
        if confirmed_persons:
            primary_person = list(confirmed_persons.keys())[0]

        return {
            "primary_person": primary_person,
            "confirmed_persons": confirmed_persons,
            "resolved_tracks_count": len(resolved_tracks),
            "unresolved_tracks_count": len(unresolved_tracks),
            "resolved_tracks": resolved_tracks,
            "unresolved_tracks": unresolved_tracks,
            "parameters": {
                "embedding_threshold": self.embedding_threshold,
                "min_vote_ratio": self.min_vote_ratio,
                "min_consistency": self.min_consistency,
                "min_best_confidence": self.min_best_confidence
            }
        }
```

---

### Task 3: Modify Video Service to Use Face Tracking

**File to modify**: `/home/facereco/facerecWeb/app/services/video_service.py`

**Changes needed**:

#### 3.1 Add imports at the top of the file

Add these imports after the existing imports (around line 24):

```python
from app.services.face_tracker import (
    FaceTracker,
    FaceDetection,
    FaceTrack,
    convert_gpu_result_to_detections
)
from app.services.track_identity_resolver import TrackIdentityResolver
```

#### 3.2 Add new method: `recognize_frames_with_tracking`

Add this new method to the `VideoService` class (after `recognize_frames_gpu` method, around line 1275):

```python
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

    # Step 7: Build final results
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
            "persons_list": list(identity_results['confirmed_persons'].keys())
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
        "frame_results": frame_results  # For debugging
    }
```

#### 3.3 Update `process_video_gpu` method to use tracking

Modify the `process_video_gpu` method (around line 1408) to use the new tracking method.

Find this section in `process_video_gpu`:

```python
# Step 3: GPU recognition with fallback
result = self.recognize_frames_gpu(
    video_id=video_id,
    domain=domain,
    frames=extraction['frames'],
    metadata=extraction['metadata'],
    quality_threshold=quality_threshold,
    min_frame_occurrence=min_frame_occurrence,
    use_weighted_voting=use_weighted_voting
)
```

Replace it with:

```python
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
```

#### 3.4 Add fallback method for CPU processing without tracking

Add this method after `recognize_frames_with_tracking`:

```python
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
```

---

### Task 4: Update Modal Service to Return Bounding Boxes

**File to modify**: `/home/facereco/facerecWeb/app/services/modal_service.py` (and Modal app files)

**Changes needed**:

The GPU service must return bounding box coordinates along with embeddings. Check if the current `ModalService.extract_embeddings()` returns bounding boxes.

If not, modify the Modal app to include `facial_area` or `bbox` in the response:

```python
# In the Modal app face processing code, ensure each face result includes:
{
    "embedding": [...],  # 512-dim vector
    "facial_area": {     # Bounding box
        "x": int,
        "y": int,
        "w": int,
        "h": int
    },
    "confidence": float  # Detection confidence
}
```

**Location**: Check `/home/facereco/facerecWeb/modal_app/` for the Modal deployment code.

---

### Task 5: Add Unit Tests

**File to create**: `/home/facereco/facerecWeb/test_face_tracking.py`

```python
"""
Unit tests for face tracking implementation.
"""

import unittest
import numpy as np
from app.services.face_tracker import (
    FaceTracker,
    FaceTrack,
    FaceDetection,
    extract_bbox_from_detection
)


class TestFaceTracker(unittest.TestCase):

    def test_iou_calculation(self):
        """Test IoU calculation"""
        tracker = FaceTracker()

        # Perfect overlap
        bbox1 = (0, 0, 100, 100)
        bbox2 = (0, 0, 100, 100)
        self.assertAlmostEqual(tracker.calculate_iou(bbox1, bbox2), 1.0)

        # No overlap
        bbox1 = (0, 0, 50, 50)
        bbox2 = (100, 100, 150, 150)
        self.assertEqual(tracker.calculate_iou(bbox1, bbox2), 0.0)

        # 50% overlap (one dimension)
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 0, 150, 100)
        iou = tracker.calculate_iou(bbox1, bbox2)
        self.assertGreater(iou, 0.3)
        self.assertLess(iou, 0.4)

    def test_track_creation(self):
        """Test that new tracks are created for unmatched faces"""
        tracker = FaceTracker(iou_threshold=0.3, min_track_length=1)

        # Frame 1: one face
        det1 = FaceDetection(
            frame_number=0,
            timestamp=0.0,
            bbox=(100, 100, 200, 200),
            embedding=np.random.rand(512),
            confidence=0.99,
            frame_quality=80.0
        )
        tracker.process_frame(0, [det1])

        self.assertEqual(len(tracker.tracks), 1)

        # Frame 2: face in completely different location -> new track
        det2 = FaceDetection(
            frame_number=1,
            timestamp=1.0,
            bbox=(500, 500, 600, 600),  # No overlap with det1
            embedding=np.random.rand(512),
            confidence=0.99,
            frame_quality=80.0
        )
        tracker.process_frame(1, [det2])

        self.assertEqual(len(tracker.tracks), 2)

    def test_track_extension(self):
        """Test that faces with high IoU extend existing track"""
        tracker = FaceTracker(iou_threshold=0.3, min_track_length=1)

        # Frame 1
        det1 = FaceDetection(
            frame_number=0,
            timestamp=0.0,
            bbox=(100, 100, 200, 200),
            embedding=np.random.rand(512),
            confidence=0.99,
            frame_quality=80.0
        )
        tracker.process_frame(0, [det1])

        # Frame 2: face slightly moved (high IoU)
        det2 = FaceDetection(
            frame_number=1,
            timestamp=1.0,
            bbox=(110, 110, 210, 210),  # Shifted by 10px, still high overlap
            embedding=np.random.rand(512),
            confidence=0.99,
            frame_quality=80.0
        )
        tracker.process_frame(1, [det2])

        # Should still be 1 track with 2 detections
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.tracks[0].get_length(), 2)

    def test_majority_voting(self):
        """Test per-track majority voting"""
        track = FaceTrack(track_id=0)

        # Add 5 detections
        for i in range(5):
            track.add_detection(FaceDetection(
                frame_number=i,
                timestamp=float(i),
                bbox=(100, 100, 200, 200),
                embedding=np.random.rand(512),
                confidence=0.99,
                frame_quality=80.0
            ))

        # Add votes: 3 for "PersonA", 2 for "PersonB"
        track.add_identity_vote("PersonA", 80.0)
        track.add_identity_vote("PersonA", 85.0)
        track.add_identity_vote("PersonA", 90.0)
        track.add_identity_vote("PersonB", 95.0)
        track.add_identity_vote("PersonB", 92.0)

        identity = track.get_final_identity(min_vote_ratio=0.5)

        self.assertIsNotNone(identity)
        self.assertEqual(identity['person'], "PersonA")
        self.assertEqual(identity['vote_count'], 3)
        self.assertGreaterEqual(identity['vote_ratio'], 0.5)

    def test_min_track_length_filter(self):
        """Test that short tracks are filtered"""
        tracker = FaceTracker(min_track_length=3)

        # Add 2 detections (below threshold)
        for i in range(2):
            det = FaceDetection(
                frame_number=i,
                timestamp=float(i),
                bbox=(100, 100, 200, 200),
                embedding=np.random.rand(512),
                confidence=0.99,
                frame_quality=80.0
            )
            tracker.process_frame(i, [det])

        tracker.close_all_tracks()

        self.assertEqual(len(tracker.get_all_tracks()), 1)
        self.assertEqual(len(tracker.get_valid_tracks()), 0)  # Filtered out


class TestBboxExtraction(unittest.TestCase):

    def test_deepface_format(self):
        """Test extracting bbox from DeepFace format"""
        face_data = {
            'facial_area': {'x': 10, 'y': 20, 'w': 100, 'h': 100}
        }
        bbox = extract_bbox_from_detection(face_data)
        self.assertEqual(bbox, (10, 20, 110, 120))

    def test_retinaface_format(self):
        """Test extracting bbox from RetinaFace format"""
        face_data = {
            'bbox': [10, 20, 110, 120]
        }
        bbox = extract_bbox_from_detection(face_data)
        self.assertEqual(bbox, (10, 20, 110, 120))


if __name__ == '__main__':
    unittest.main()
```

---

## Testing Plan

### Test 1: Unit Tests
```bash
cd /home/facereco/facerecWeb
python -m pytest test_face_tracking.py -v
```

### Test 2: Integration Test with Real Video
```bash
# Use the same video that showed false positives
python -c "
from app.services.video_service import VideoService

service = VideoService()
with open('path/to/test_video.mp4', 'rb') as f:
    video_bytes = f.read()

result = service.process_video_gpu(
    video_bytes=video_bytes,
    original_filename='test.mp4',
    domain='serbia',
    interval_seconds=3.0
)

print('Primary person:', result['tracking_results']['identity_results']['primary_person'])
print('Confirmed persons:', list(result['tracking_results']['identity_results']['confirmed_persons'].keys()))
"
```

### Test 3: Compare Before/After
1. Run the old method on a video with known false positives
2. Run the new tracking method on the same video
3. Compare results - false positives should be eliminated or reduced

---

## Parameter Tuning Guide

| Parameter | Default | Effect of Increasing | Effect of Decreasing |
|-----------|---------|---------------------|---------------------|
| `iou_threshold` | 0.3 | More likely to create new tracks | More likely to merge faces |
| `min_track_length` | 3 | Filters more short appearances | Allows brief appearances |
| `min_vote_ratio` | 0.5 | Stricter identity assignment | More lenient |
| `min_consistency` | 0.4 | Rejects inconsistent tracks | Allows more variation |
| `min_best_confidence` | 70.0 | Rejects low-confidence matches | Allows weaker matches |

### Recommended settings by video type:

**Talk shows (stable, same people):**
- `iou_threshold`: 0.3
- `min_track_length`: 5
- `min_vote_ratio`: 0.6
- `min_consistency`: 0.5

**Sports (dynamic, fast motion):**
- `iou_threshold`: 0.2 (lower because faces move faster)
- `min_track_length`: 3
- `min_vote_ratio`: 0.5
- `min_consistency`: 0.4

**News collages (scene cuts):**
- `iou_threshold`: 0.3
- `min_track_length`: 2 (shorter appearances)
- `min_vote_ratio`: 0.5
- `min_consistency`: 0.3

---

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `app/services/face_tracker.py` | CREATE | Face tracking with IoU |
| `app/services/track_identity_resolver.py` | CREATE | Per-track identity voting |
| `app/services/video_service.py` | MODIFY | Add `recognize_frames_with_tracking()` |
| `app/services/modal_service.py` | VERIFY | Ensure bbox returned |
| `test_face_tracking.py` | CREATE | Unit tests |

---

## Rollback Plan

If issues occur, you can disable tracking by setting `use_tracking=False`:

```python
result = self.recognize_frames_with_tracking(
    ...
    use_tracking=False,  # Disable tracking, use legacy method
    ...
)
```

Or revert `process_video_gpu` to call `recognize_frames_gpu` directly.

---

## Success Criteria

1. **False positive reduction**: Vladan_Petrov (3 frames) should be filtered out when Rasim_Ljajic (44 frames) is the primary person
2. **True positive retention**: Legitimate brief appearances should still be detected if they form consistent tracks
3. **Processing time**: Should not increase by more than 20%
4. **No regressions**: Existing functionality should still work
