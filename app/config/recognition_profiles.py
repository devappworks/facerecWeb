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


class ImprovedSystemProfile(RecognitionProfile):
    """
    Improved configuration based on research (Pipeline B)
    """

    def __init__(self):
        super().__init__(
            name="improved_system",
            description="Improved Facenet512 based system"
        )

    def get_config(self):
        return {
            # Model configuration
            "model_name": "Facenet512",  # CHANGED
            "detector_backend": "retinaface",
            "distance_metric": "cosine",

            # Recognition thresholds
            "recognition_threshold": 0.40,  # CHANGED (was 0.35)
            "detection_confidence_threshold": 0.98,  # CHANGED (was 0.995)

            # Quality validation (same)
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
        "current": CurrentSystemProfile(),
        "improved": ImprovedSystemProfile(),
        "ensemble": EnsembleSystemProfile()
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
