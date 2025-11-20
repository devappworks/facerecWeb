"""
Face recognition configuration profiles for A/B testing
"""


class RecognitionProfile:
    """Base class for recognition configuration"""

    def __init__(self, name, description):
        self.name = name
        self.description = description

    def get_config(self):
        raise NotImplementedError


class CurrentSystemProfile(RecognitionProfile):
    """
    Current production configuration (Pipeline A)
    """

    def __init__(self):
        super().__init__(
            name="current_system",
            description="Current VGG-Face based system"
        )

    def get_config(self):
        return {
            # Model configuration
            "model_name": "VGG-Face",
            "detector_backend": "retinaface",
            "distance_metric": "cosine",

            # Recognition thresholds
            "recognition_threshold": 0.35,
            "detection_confidence_threshold": 0.995,  # 99.5%

            # Quality validation
            "blur_threshold": 100,
            "contrast_threshold": 25,
            "brightness_min": 30,
            "brightness_max": 220,
            "edge_density_threshold": 15,

            # Processing options
            "enforce_detection": False,
            "normalize_face": True,
            "align": True,
            "batched": True,

            # Metadata
            "profile_version": "1.0",
            "created": "2025-01-13",
            "is_production": True
        }


class ArcFaceSystemProfile(RecognitionProfile):
    """
    State-of-the-art ArcFace configuration (Pipeline B)
    - Replaces Facenet512 (too slow: 10-40x slower than VGG-Face)
    - ArcFace: 99.8% LFW accuracy, 17ms inference time
    - Production-ready on CPU, no timeouts
    """

    def __init__(self):
        super().__init__(
            name="arcface_system",
            description="State-of-the-art ArcFace model (2019)"
        )

    def get_config(self):
        return {
            # Model configuration
            "model_name": "ArcFace",  # State-of-the-art (99.8% LFW)
            "detector_backend": "retinaface",
            "distance_metric": "cosine",

            # Recognition thresholds
            "recognition_threshold": 0.50,  # ArcFace uses higher threshold
            "detection_confidence_threshold": 0.995,  # Same as production

            # Quality validation (same as production)
            "blur_threshold": 100,
            "contrast_threshold": 25,
            "brightness_min": 30,
            "brightness_max": 220,
            "edge_density_threshold": 15,

            # Processing options
            "enforce_detection": False,
            "normalize_face": True,
            "align": True,
            "batched": True,

            # Metadata
            "profile_version": "2.0",
            "created": "2025-11-20",
            "replaced": "Facenet512 (too slow)",
            "is_production": False,
            "is_test": True
        }


class EnsembleSystemProfile(RecognitionProfile):
    """
    Multi-model ensemble configuration (Pipeline C - Future)
    """

    def __init__(self):
        super().__init__(
            name="ensemble_system",
            description="Multi-model ensemble for maximum accuracy"
        )

    def get_config(self):
        return {
            # Model configuration
            "models": ["Facenet512", "ArcFace"],  # Multiple models
            "detector_backend": "retinaface",
            "distance_metric": "cosine",

            # Recognition thresholds (per model)
            "model_thresholds": {
                "Facenet512": 0.40,
                "ArcFace": 0.50
            },
            "detection_confidence_threshold": 0.98,

            # Quality validation
            "blur_threshold": 100,
            "contrast_threshold": 25,
            "brightness_min": 30,
            "brightness_max": 220,
            "edge_density_threshold": 15,

            # Processing options
            "enforce_detection": False,
            "normalize_face": True,
            "align": True,
            "batched": True,

            # Ensemble options
            "voting_strategy": "weighted",  # weighted, majority, confidence
            "min_model_agreement": 2,  # Minimum models that must agree

            # Metadata
            "profile_version": "1.0",
            "created": "2025-01-13",
            "is_production": False,
            "is_test": True
        }


class ProfileManager:
    """
    Manages recognition profiles
    """

    _profiles = {
        "current": CurrentSystemProfile(),       # VGG-Face (production)
        "arcface": ArcFaceSystemProfile(),       # ArcFace (A/B testing - NEW)
        "ensemble": EnsembleSystemProfile(),     # Future multi-model
        # "improved": Removed Facenet512 (too slow, not production-viable)
    }

    @classmethod
    def get_profile(cls, name: str) -> RecognitionProfile:
        """Get profile by name"""
        if name not in cls._profiles:
            raise ValueError(f"Unknown profile: {name}. Available: {list(cls._profiles.keys())}")

        return cls._profiles[name]

    @classmethod
    def get_config(cls, name: str) -> dict:
        """Get configuration for profile"""
        profile = cls.get_profile(name)
        return profile.get_config()

    @classmethod
    def list_profiles(cls) -> list:
        """List all available profiles"""
        return [
            {
                "name": profile.name,
                "description": profile.description,
                "is_production": profile.get_config().get("is_production", False)
            }
            for profile in cls._profiles.values()
        ]
