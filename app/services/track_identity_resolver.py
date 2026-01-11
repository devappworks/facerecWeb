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
        embedding_threshold: float = 0.40,  # IMPROVED: 0.40 = 60% confidence threshold
        min_vote_ratio: float = 0.5,
        min_consistency: float = 0.4,
        min_best_confidence: float = 75.0  # IMPROVED: Raised from 70% to 75%
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
