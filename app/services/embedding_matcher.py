"""
Embedding Matcher Service - Matches face embeddings against local database on CPU.
This is the server-side component of the hybrid GPU/CPU architecture:
- GPU (Modal): Face detection + embedding extraction
- CPU (Server): Embedding comparison against local database
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingMatcher:
    """Matches face embeddings against precomputed database embeddings"""

    # Singleton pattern for caching loaded databases
    _instances: Dict[str, 'EmbeddingMatcher'] = {}
    _pkl_cache: Dict[str, Dict] = {}

    STORAGE_BASE = "/root/facerecognition-backend/storage/recognized_faces_prod"
    DEFAULT_THRESHOLD = 0.40  # Cosine distance threshold for ArcFace (IMPROVED: 0.40 distance = 60% confidence minimum)

    def __new__(cls, domain: str):
        if domain not in cls._instances:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[domain] = instance
        return cls._instances[domain]

    def __init__(self, domain: str):
        if self._initialized:
            return

        self.domain = domain
        self.db_path = os.path.join(self.STORAGE_BASE, domain)
        self.embeddings: List[np.ndarray] = []
        self.identities: List[str] = []
        self.person_names: List[str] = []
        self._initialized = True
        self._loaded = False

    def load_database(self, force_reload: bool = False) -> bool:
        """
        Load embeddings from pkl file.

        Returns False if no pkl file exists - caller should use CPU fallback.
        Does NOT auto-generate pkl as this would take hours for large databases.
        Use scripts/generate_embeddings.py to pre-generate pkl files.
        """
        if self._loaded and not force_reload:
            return True

        pkl_path = self._find_pkl_file()

        if pkl_path and os.path.exists(pkl_path):
            return self._load_from_pkl(pkl_path)
        else:
            logger.warning(f"No pkl file found for {self.domain}. "
                          "Use 'python scripts/generate_embeddings.py {self.domain}' to generate. "
                          "Falling back to CPU recognition.")
            return False

    def _find_pkl_file(self) -> Optional[str]:
        """Find the representations pkl file for this domain"""
        # Check for different pkl naming conventions
        pkl_patterns = [
            # DeepFace newer format: ds_model_arcface_detector_...
            os.path.join(self.db_path, "ds_model_arcface_detector_retinaface_aligned_normalization_base_expand_0.pkl"),
            # DeepFace older format: representations_arcface.pkl
            os.path.join(self.db_path, "representations_arcface.pkl"),
            os.path.join(self.db_path, "representations_ArcFace.pkl"),
        ]

        for pattern in pkl_patterns:
            if os.path.exists(pattern):
                return pattern

        # Search for any arcface pkl file
        if os.path.exists(self.db_path):
            for f in os.listdir(self.db_path):
                if "arcface" in f.lower() and f.endswith(".pkl"):
                    return os.path.join(self.db_path, f)
                if f.startswith("representations_") and f.endswith(".pkl"):
                    return os.path.join(self.db_path, f)

        return None

    def _load_from_pkl(self, pkl_path: str) -> bool:
        """Load embeddings from pkl file"""
        try:
            logger.info(f"Loading embeddings from {pkl_path}")

            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            self.embeddings = []
            self.identities = []
            self.person_names = []

            # Handle different pkl formats
            for item in data:
                # New format: list of dicts with 'identity', 'embedding', 'hash'
                if isinstance(item, dict):
                    identity_path = item.get('identity', '')
                    embedding = item.get('embedding', [])
                # Old format: list of [identity_path, embedding_list]
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    identity_path = item[0]
                    embedding = item[1]
                else:
                    continue

                if embedding:
                    self.identities.append(identity_path)
                    self.embeddings.append(np.array(embedding))

                    # Extract person name from path
                    person_name = self._extract_person_name(identity_path)
                    self.person_names.append(person_name)

            logger.info(f"Loaded {len(self.embeddings)} embeddings for {self.domain}")
            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load pkl file {pkl_path}: {e}")
            return False

    def _generate_pkl(self) -> bool:
        """Generate pkl file by computing embeddings for all images"""
        try:
            from deepface import DeepFace

            if not os.path.exists(self.db_path):
                logger.error(f"Database path does not exist: {self.db_path}")
                return False

            logger.info(f"Generating embeddings for {self.db_path}...")

            # Use DeepFace.find with a dummy image to trigger pkl generation
            # Or directly compute embeddings for all images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            image_files = []

            for root, dirs, files in os.walk(self.db_path):
                for f in files:
                    if Path(f).suffix.lower() in image_extensions:
                        image_files.append(os.path.join(root, f))

            logger.info(f"Found {len(image_files)} images to process")

            self.embeddings = []
            self.identities = []
            self.person_names = []
            pkl_data = []

            for i, img_path in enumerate(image_files):
                try:
                    result = DeepFace.represent(
                        img_path=img_path,
                        model_name="ArcFace",
                        detector_backend="retinaface",
                        enforce_detection=False
                    )

                    if result and len(result) > 0:
                        embedding = result[0].get('embedding', [])
                        self.identities.append(img_path)
                        self.embeddings.append(np.array(embedding))
                        self.person_names.append(self._extract_person_name(img_path))
                        pkl_data.append([img_path, embedding])

                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed {i + 1}/{len(image_files)} images")

                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")

            # Save pkl file
            pkl_path = os.path.join(self.db_path, "representations_arcface.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(pkl_data, f)

            logger.info(f"Saved {len(pkl_data)} embeddings to {pkl_path}")
            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to generate pkl: {e}")
            return False

    def _extract_person_name(self, path: str) -> str:
        """Extract person name from file path"""
        filename = os.path.basename(path)
        name_parts = filename.split('_')
        person_name = []

        for part in name_parts:
            # Stop at timestamp-like parts (8+ digits starting with year)
            if len(part) >= 8 and part[0:4].isdigit():
                break
            person_name.append(part)

        return '_'.join(person_name) if person_name else filename

    def find_matches(
        self,
        query_embedding: List[float],
        threshold: float = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find matches for a query embedding.

        Args:
            query_embedding: Face embedding vector (512 dims for ArcFace)
            threshold: Cosine distance threshold (lower = stricter)
            top_k: Maximum number of matches to return

        Returns:
            List of matches with person, distance, confidence
        """
        if not self._loaded:
            if not self.load_database():
                return []

        if not self.embeddings:
            return []

        threshold = threshold or self.DEFAULT_THRESHOLD
        query = np.array(query_embedding)

        # Compute cosine distances to all database embeddings
        distances = []
        for i, db_emb in enumerate(self.embeddings):
            dist = self._cosine_distance(query, db_emb)
            distances.append((i, dist))

        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])

        # Filter by threshold and return top_k
        matches = []
        for idx, dist in distances[:top_k]:
            if dist <= threshold:
                matches.append({
                    "person": self.person_names[idx],
                    "identity": self.identities[idx],
                    "distance": float(dist),
                    "confidence": round((1 - dist) * 100, 2)
                })

        return matches

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between two vectors"""
        # Cosine similarity = dot(a, b) / (norm(a) * norm(b))
        # Cosine distance = 1 - cosine_similarity
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 1.0

        similarity = dot / (norm_a * norm_b)
        return 1 - similarity

    def match_frame_embeddings(
        self,
        faces: List[Dict],
        threshold: float = None
    ) -> Dict[str, Any]:
        """
        Match all face embeddings from a frame.

        Args:
            faces: List of face dicts with 'embedding' key
            threshold: Distance threshold

        Returns:
            Dict with best_match and all_matches
        """
        all_matches = []
        best_match = None

        for face in faces:
            embedding = face.get('embedding', [])
            if not embedding:
                continue

            matches = self.find_matches(embedding, threshold)

            if matches:
                all_matches.extend(matches)
                if best_match is None or matches[0]['distance'] < best_match['distance']:
                    best_match = matches[0]

        return {
            "recognized": best_match is not None,
            "person": best_match['person'] if best_match else None,
            "confidence": best_match['confidence'] if best_match else None,
            "best_match": best_match,
            "all_matches": all_matches
        }

    @classmethod
    def get_matcher(cls, domain: str) -> 'EmbeddingMatcher':
        """Get or create matcher for domain"""
        matcher = cls(domain)
        matcher.load_database()
        return matcher

    @classmethod
    def preload_all(cls, domains: List[str] = None):
        """Preload embeddings for specified domains"""
        if domains is None:
            domains = ['serbia', 'slovenia']

        for domain in domains:
            try:
                matcher = cls.get_matcher(domain)
                logger.info(f"Preloaded {len(matcher.embeddings)} embeddings for {domain}")
            except Exception as e:
                logger.error(f"Failed to preload {domain}: {e}")
