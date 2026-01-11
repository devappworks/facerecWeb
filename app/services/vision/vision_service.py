"""
Vision Service - unified interface for image analysis.
Handles provider selection, face recognition integration, and sequential processing.
"""

import os
import base64
import logging
from typing import Optional, List, Dict, Any
from io import BytesIO

from PIL import Image

# Register AVIF and HEIF image format support
try:
    import pillow_avif
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # AVIF/HEIF support not installed

from dotenv import load_dotenv

from app.services.vision.base import (
    VisionProvider, VisionConfig, VisionProviderType, ImageMetadata, RecognizedPerson, PhotoMetadata
)
from app.services.vision.openai_provider import OpenAIVisionProvider
from app.services.vision.gemini_provider import GeminiVisionProvider
from app.services.vision.metadata_extractor import get_metadata_extractor

logger = logging.getLogger(__name__)


class VisionService:
    """
    Unified vision service for image analysis.

    Features:
    - Multi-provider support (OpenAI, Gemini)
    - Easy model switching
    - Face recognition integration
    - Bilingual output
    """

    # Default configuration
    DEFAULT_PROVIDER = "openai"
    DEFAULT_MODEL = "gpt-4.1-mini"

    def __init__(
        self,
        provider: str = None,
        model: str = None,
        local_language: str = None
    ):
        """
        Initialize vision service.

        Args:
            provider: Provider name ("openai" or "gemini")
            model: Model name (provider-specific)
            local_language: Local language for bilingual output (e.g., "serbian", "slovenian")
        """
        load_dotenv()

        self.provider_name = provider or os.getenv('VISION_PROVIDER', self.DEFAULT_PROVIDER)
        self.model_name = model or os.getenv('VISION_MODEL', self.DEFAULT_MODEL)
        self.local_language = local_language or os.getenv('VISION_LOCAL_LANGUAGE')

        # Create config
        self.config = VisionConfig(
            provider=VisionProviderType(self.provider_name.lower()),
            model=self.model_name,
            local_language=self.local_language
        )

        # Initialize provider
        self.provider = self._create_provider()

        logger.info(
            f"VisionService initialized: provider={self.provider_name}, "
            f"model={self.model_name}, local_language={self.local_language}"
        )

    def _create_provider(self) -> VisionProvider:
        """Create the appropriate vision provider."""
        if self.config.provider == VisionProviderType.OPENAI:
            return OpenAIVisionProvider(self.config)
        elif self.config.provider == VisionProviderType.GEMINI:
            return GeminiVisionProvider(self.config)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def analyze_image(
        self,
        image_data: bytes,
        recognized_persons: Optional[List[Dict[str, Any]]] = None,
        resize_max: int = 1200,
        extract_photo_metadata: bool = True
    ) -> ImageMetadata:
        """
        Analyze an image and return structured metadata.

        Args:
            image_data: Image bytes
            recognized_persons: Optional list of recognized persons from face recognition
                Each dict should have: name, role (optional), face_coordinates (optional)
            resize_max: Maximum dimension for resizing (default 1200px)
            extract_photo_metadata: Whether to extract EXIF/IPTC metadata from the photo

        Returns:
            ImageMetadata with complete analysis
        """
        logger.info("Starting image analysis")

        # Extract photo metadata (EXIF, IPTC, GPS) from original image before resizing
        photo_metadata = None
        if extract_photo_metadata:
            try:
                extractor = get_metadata_extractor()
                raw_metadata = extractor.extract_all(image_data)
                if raw_metadata:
                    photo_metadata = PhotoMetadata(
                        gps=raw_metadata.get("gps"),
                        datetime=raw_metadata.get("datetime"),
                        camera=raw_metadata.get("camera"),
                        image=raw_metadata.get("image"),
                        iptc=raw_metadata.get("iptc")
                    )
                    logger.info(f"Extracted photo metadata: GPS={photo_metadata.gps is not None}, datetime={photo_metadata.datetime is not None}")
            except Exception as e:
                logger.warning(f"Failed to extract photo metadata: {e}")

        # Resize image for efficiency
        resized_image = self._resize_image(image_data, resize_max)

        # Encode to base64
        image_base64 = base64.b64encode(resized_image).decode('utf-8')

        # Convert recognized persons to RecognizedPerson objects
        persons = None
        if recognized_persons:
            persons = [
                RecognizedPerson(
                    name=p.get('name', ''),
                    role=p.get('role'),
                    face_coordinates=p.get('face_coordinates')
                )
                for p in recognized_persons
            ]
            logger.info(f"Including {len(persons)} recognized persons in analysis")

        # Analyze with provider
        metadata = self.provider.analyze_image(image_base64, persons)

        # Attach photo metadata to result
        if photo_metadata:
            metadata.photo_metadata = photo_metadata

        return metadata

    def analyze_image_with_face_recognition(
        self,
        image_data: bytes,
        domain: str,
        resize_max: int = 1200
    ) -> ImageMetadata:
        """
        Analyze an image with automatic face recognition (sequential processing).

        This method:
        1. First runs face recognition to identify people
        2. Then runs vision analysis with the recognized faces as context

        Args:
            image_data: Image bytes
            domain: Domain for face recognition database
            resize_max: Maximum dimension for resizing

        Returns:
            ImageMetadata with complete analysis including recognized persons
        """
        logger.info(f"Starting sequential analysis (face recognition + vision) for domain: {domain}")

        # Step 1: Run face recognition
        recognized_persons = []
        face_recognition_result = None  # Store full result for logging
        try:
            # Use RecognitionController instead of RecognitionService directly
            # This enables dual-mode pgvector testing when VECTOR_DB_ENABLED=true
            from app.controllers.recognition_controller import RecognitionController

            result = RecognitionController.recognize_face(image_data, domain)
            face_recognition_result = result  # Store full result

            if result.get('status') == 'success' and result.get('recognized_persons'):
                # Get confidence data from all_detected_matches
                all_matches = {m['person_name']: m['metrics'] for m in result.get('all_detected_matches', [])}

                for person in result['recognized_persons']:
                    person_name = person.get('name', '')
                    # Try to find matching metrics (name might be formatted differently)
                    metrics = None
                    for match_name, match_metrics in all_matches.items():
                        if match_name.lower().replace('_', ' ') == person_name.lower().replace('_', ' '):
                            metrics = match_metrics
                            break

                    recognized_persons.append({
                        'name': person_name,
                        'face_coordinates': person.get('face_coordinates'),
                        'confidence': metrics.get('confidence_percentage') if metrics else None,
                        'occurrences': metrics.get('occurrences') if metrics else None,
                        'min_distance': metrics.get('min_distance') if metrics else None,
                        'weighted_score': metrics.get('weighted_score') if metrics else None,
                        'role': None  # Could be enriched from a database
                    })
                logger.info(f"Face recognition found {len(recognized_persons)} persons: {[p['name'] for p in recognized_persons]}")
            else:
                logger.info("No faces recognized in image")

        except Exception as e:
            logger.warning(f"Face recognition failed, proceeding without it: {e}")

        # Step 2: Run vision analysis with recognized persons
        metadata = self.analyze_image(image_data, recognized_persons, resize_max)

        # Attach full face recognition result to metadata for logging
        metadata.face_recognition_result = face_recognition_result

        return metadata

    def _resize_image(self, image_data: bytes, max_size: int) -> bytes:
        """Resize image while maintaining aspect ratio."""
        try:
            image = Image.open(BytesIO(image_data))

            # Only resize if larger than max_size
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.LANCZOS)
                logger.info(f"Image resized to {image.size}")

            # Convert back to bytes
            output = BytesIO()

            # Handle different image modes
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')

            image.save(output, format='JPEG', quality=85)
            return output.getvalue()

        except Exception as e:
            logger.warning(f"Error resizing image: {e}, using original")
            return image_data

    def switch_provider(self, provider: str, model: str = None):
        """
        Switch to a different provider/model.

        Args:
            provider: New provider name ("openai" or "gemini")
            model: New model name (optional, uses default for provider if not specified)
        """
        self.provider_name = provider
        self.config.provider = VisionProviderType(provider.lower())

        if model:
            self.model_name = model
            self.config.model = model
        else:
            # Use default model for provider
            defaults = {
                "openai": "gpt-4.1-mini",
                "gemini": "gemini-2.0-flash"
            }
            self.model_name = defaults.get(provider, "gpt-4.1-mini")
            self.config.model = self.model_name

        self.provider = self._create_provider()
        logger.info(f"Switched to provider: {self.provider_name}, model: {self.model_name}")

    def set_local_language(self, language: str):
        """
        Set the local language for bilingual output.

        Args:
            language: Language name (e.g., "serbian", "slovenian")
        """
        self.local_language = language
        self.config.local_language = language
        # Recreate provider with new config
        self.provider = self._create_provider()
        logger.info(f"Local language set to: {language}")

    @staticmethod
    def list_providers() -> List[str]:
        """List available vision providers."""
        return ["openai", "gemini"]

    @staticmethod
    def list_models(provider: str = None) -> Dict[str, List[str]]:
        """
        List available models.

        Args:
            provider: Optional provider name to filter by

        Returns:
            Dict mapping provider names to list of available models
        """
        models = {
            "openai": OpenAIVisionProvider.AVAILABLE_MODELS,
            "gemini": GeminiVisionProvider.AVAILABLE_MODELS
        }

        if provider:
            return {provider: models.get(provider, [])}
        return models


# Convenience function for quick analysis
def analyze_image(
    image_data: bytes,
    provider: str = "openai",
    model: str = "gpt-4.1-mini",
    local_language: str = None,
    recognized_persons: List[Dict[str, Any]] = None
) -> ImageMetadata:
    """
    Quick function to analyze an image.

    Args:
        image_data: Image bytes
        provider: Provider name ("openai" or "gemini")
        model: Model name
        local_language: Local language for bilingual output
        recognized_persons: Optional list of recognized persons

    Returns:
        ImageMetadata with analysis results
    """
    service = VisionService(provider=provider, model=model, local_language=local_language)
    return service.analyze_image(image_data, recognized_persons)
