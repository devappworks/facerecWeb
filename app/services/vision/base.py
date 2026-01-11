"""
Base classes and interfaces for vision providers.
Supports multiple AI providers (OpenAI, Gemini, etc.) with a unified interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VisionProviderType(Enum):
    """Supported vision provider types."""
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class VisionConfig:
    """Configuration for vision analysis."""
    provider: VisionProviderType = VisionProviderType.OPENAI
    model: str = "gpt-4.1-mini"  # Default model
    temperature: float = 0.1
    max_tokens: int = 8000

    # Language settings
    primary_language: str = "english"  # Always included
    local_language: Optional[str] = None  # e.g., "serbian", "slovenian"

    # Feature flags
    include_scene_analysis: bool = True
    include_event_detection: bool = True
    include_time_context: bool = True
    include_media_analysis: bool = True
    max_tags: int = 12

    # Model presets for easy switching
    MODEL_PRESETS = {
        "openai": {
            "fast": "gpt-4.1-mini",
            "balanced": "gpt-4.1",
            "quality": "gpt-4o",
        },
        "gemini": {
            "fast": "gemini-2.0-flash",
            "balanced": "gemini-2.5-pro",
            "quality": "gemini-2.5-pro",
        }
    }

    @classmethod
    def for_provider(cls, provider: str, preset: str = "fast", local_language: Optional[str] = None) -> "VisionConfig":
        """Create config for a specific provider and preset."""
        provider_type = VisionProviderType(provider.lower())
        model = cls.MODEL_PRESETS.get(provider, {}).get(preset, "gpt-4.1-mini")
        return cls(
            provider=provider_type,
            model=model,
            local_language=local_language
        )


@dataclass
class LocalizedText:
    """Text content in multiple languages."""
    english: str
    local: Optional[str] = None  # Serbian, Slovenian, etc.


@dataclass
class SceneAnalysis:
    """Scene and setting classification."""
    setting: str  # indoor, outdoor, studio, etc.
    location_type: Optional[str] = None  # stadium, office, red carpet, etc.
    atmosphere: Optional[str] = None  # formal, casual, celebratory, etc.
    time_of_day: Optional[str] = None  # day, night, evening, etc.
    season: Optional[str] = None  # if apparent


@dataclass
class EventAnalysis:
    """Event type detection for media content."""
    event_type: Optional[str] = None  # press conference, sports match, concert, etc.
    event_name: Optional[str] = None  # specific event if identifiable
    activity: Optional[str] = None  # speaking, performing, competing, etc.


@dataclass
class MediaAnalysis:
    """Media-specific analysis for news/photo agencies."""
    composition: str  # portrait, group shot, candid, action shot, etc.
    subject_count: int  # number of main subjects
    attire: Optional[str] = None  # formal wear, sports uniform, casual, etc.
    notable_items: List[str] = field(default_factory=list)  # trophies, microphones, etc.


@dataclass
class RecognizedPerson:
    """Information about a recognized person in the image (from face recognition)."""
    name: str
    role: Optional[str] = None  # e.g., "tennis player", "actor", "politician"
    face_coordinates: Optional[Dict[str, Any]] = None


@dataclass
class IdentifiedPerson:
    """Information about a person identified by AI vision model."""
    name: str
    confidence: str = "medium"  # high, medium, low
    role: Optional[str] = None  # e.g., "tennis player", "actor", "politician"
    description: Optional[str] = None  # Brief description of the person in the image


@dataclass
class UsageInfo:
    """Token usage and cost information for vision API calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    # Pricing per 1M tokens (updated December 2025)
    # See: https://ai.google.dev/pricing, https://openai.com/pricing
    PRICING = {
        # OpenAI models
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5": {"input": 1.25, "output": 10.00},
        # Google Gemini models
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
        "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-3-flash-preview": {"input": 0.15, "output": 0.60},
        "gemini-3-pro-preview": {"input": 1.25, "output": 5.00},
    }

    @classmethod
    def calculate_cost(cls, model: str, prompt_tokens: int, completion_tokens: int) -> "UsageInfo":
        """Calculate cost based on model and token counts."""
        pricing = cls.PRICING.get(model, {"input": 0.0, "output": 0.0})
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=round(total_cost, 6)
        )


@dataclass
class PhotoMetadata:
    """Metadata extracted from the photo file itself (EXIF, IPTC, etc.)."""
    # GPS coordinates
    gps: Optional[Dict[str, Any]] = None  # latitude, longitude, altitude

    # Date/time info
    datetime: Optional[Dict[str, str]] = None  # taken, digitized

    # Camera info
    camera: Optional[Dict[str, Any]] = None  # make, model, lens, focal_length, aperture, iso, shutter_speed

    # Image dimensions and format
    image: Optional[Dict[str, Any]] = None  # width, height, format, orientation

    # IPTC/copyright info
    iptc: Optional[Dict[str, str]] = None  # caption, author, copyright, keywords


@dataclass
class ImageMetadata:
    """Complete metadata for an analyzed image."""
    # Core descriptions (bilingual)
    description: LocalizedText
    alt_text: LocalizedText

    # Structured analysis
    scene: Optional[SceneAnalysis] = None
    event: Optional[EventAnalysis] = None
    media: Optional[MediaAnalysis] = None

    # Tags and objects
    objects: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)  # Specific, useful tags

    # Recognized people (from face recognition)
    recognized_persons: List[RecognizedPerson] = field(default_factory=list)

    # AI-identified people (from Gemini 3 vision models)
    identified_persons: List[IdentifiedPerson] = field(default_factory=list)

    # Photo file metadata (EXIF, IPTC, GPS)
    photo_metadata: Optional[PhotoMetadata] = None

    # Provider metadata
    provider: str = ""
    model: str = ""

    # Usage/cost tracking (only shown to admin users)
    usage: Optional[UsageInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "description": {
                "english": self.description.english,
                "local": self.description.local
            },
            "alt_text": {
                "english": self.alt_text.english,
                "local": self.alt_text.local
            },
            "scene": {
                "setting": self.scene.setting,
                "location_type": self.scene.location_type,
                "atmosphere": self.scene.atmosphere,
                "time_of_day": self.scene.time_of_day,
                "season": self.scene.season
            } if self.scene else None,
            "event": {
                "event_type": self.event.event_type,
                "event_name": self.event.event_name,
                "activity": self.event.activity
            } if self.event else None,
            "media": {
                "composition": self.media.composition,
                "subject_count": self.media.subject_count,
                "attire": self.media.attire,
                "notable_items": self.media.notable_items
            } if self.media else None,
            "objects": self.objects,
            "tags": self.tags,
            "recognized_persons": [
                {
                    "name": p.name,
                    "role": p.role,
                    "face_coordinates": p.face_coordinates
                } for p in self.recognized_persons
            ],
            "identified_persons": [
                {
                    "name": p.name,
                    "confidence": p.confidence,
                    "role": p.role,
                    "description": p.description
                } for p in self.identified_persons
            ] if self.identified_persons else [],
            "provider": self.provider,
            "model": self.model
        }

        # Add photo metadata if available
        if self.photo_metadata:
            photo_meta = {}
            if self.photo_metadata.gps:
                photo_meta["gps"] = self.photo_metadata.gps
            if self.photo_metadata.datetime:
                photo_meta["datetime"] = self.photo_metadata.datetime
            if self.photo_metadata.camera:
                photo_meta["camera"] = self.photo_metadata.camera
            if self.photo_metadata.image:
                photo_meta["image"] = self.photo_metadata.image
            if self.photo_metadata.iptc:
                photo_meta["iptc"] = self.photo_metadata.iptc
            if photo_meta:
                result["photo_metadata"] = photo_meta

        # Add usage info if available (for admin users)
        if self.usage:
            result["usage"] = {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
                "cost_usd": self.usage.cost_usd
            }

        return result

    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy format for backward compatibility."""
        return {
            "description": self.description.english,
            "alt": self.alt_text.english,
            "objects": self.objects,
            "metatags": self.tags[:4]  # Legacy limit
        }


class VisionProvider(ABC):
    """Abstract base class for vision providers."""

    def __init__(self, config: VisionConfig):
        self.config = config

    @abstractmethod
    def analyze_image(
        self,
        image_base64: str,
        recognized_persons: Optional[List[RecognizedPerson]] = None
    ) -> ImageMetadata:
        """
        Analyze an image and return structured metadata.

        Args:
            image_base64: Base64-encoded image data
            recognized_persons: Optional list of recognized persons from face recognition

        Returns:
            ImageMetadata with complete analysis
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of this provider."""
        pass

    def _build_system_prompt(self, recognized_persons: Optional[List[RecognizedPerson]] = None) -> str:
        """Build the system prompt for image analysis."""

        # Person context if faces were recognized
        person_context = ""
        if recognized_persons:
            names = [f"{p.name}" + (f" ({p.role})" if p.role else "") for p in recognized_persons]
            person_context = f"""
IMPORTANT CONTEXT: Face recognition has identified the following people in this image:
{', '.join(names)}

Use this information to provide more accurate and contextual descriptions. Reference these people by name in your descriptions.
"""

        # Language instruction
        local_lang = self.config.local_language
        language_instruction = f"""
OUTPUT LANGUAGES:
- All text fields must be provided in BOTH English AND {local_lang.title()}.
- The English version is for archive/search purposes.
- The {local_lang.title()} version is for local SEO and display.
""" if local_lang else """
OUTPUT LANGUAGE: English only.
"""

        prompt = f"""You are an expert image analyst for a media/news photo agency. Your task is to analyze images and extract rich, useful metadata for archive search and SEO purposes.
{person_context}
{language_instruction}
ANALYSIS REQUIREMENTS:

1. DESCRIPTION: Write a descriptive summary of the image (2-3 sentences).
   - If people are recognized, mention them by name and describe what they are doing.
   - Include context about the setting and any notable elements.

2. ALT TEXT: Write concise alt text for accessibility (1 sentence, max 125 characters).
   - Include key subjects and action.

3. SCENE ANALYSIS:
   - setting: The general setting (indoor, outdoor, studio, stage, etc.)
   - location_type: Specific location type (stadium, office, red carpet, press room, concert hall, etc.)
   - atmosphere: The mood/tone (formal, casual, celebratory, tense, professional, etc.)
   - time_of_day: If apparent (day, night, evening, morning)
   - season: If apparent (spring, summer, autumn, winter)

4. EVENT ANALYSIS (for news/media context):
   - event_type: Type of event (press conference, sports match, concert, award ceremony, interview, etc.)
   - event_name: Specific event name if identifiable
   - activity: What subjects are doing (speaking, performing, competing, posing, walking, etc.)

5. MEDIA ANALYSIS:
   - composition: Photo type (portrait, group shot, candid, action shot, posed, wide shot, close-up)
   - subject_count: Number of main subjects in focus
   - attire: Clothing description (formal wear, sports uniform, casual, costume, etc.)
   - notable_items: List specific notable items visible (trophy, microphone, musical instrument, etc.)

6. OBJECTS: List key objects/entities visible in the image.
   - Be specific (not "ball" but "basketball", not "building" but "stadium")

7. TAGS: Generate 8-12 specific, useful tags for search.
   DO NOT include generic useless tags like: person, man, woman, people, building, sky, background, indoor, outdoor
   DO include specific tags like: tennis, press conference, championship trophy, red carpet, formal attire, sports interview

Return your analysis in the exact JSON structure specified."""

        return prompt

    def _get_json_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for structured output."""
        local_lang = self.config.local_language

        # Build localized text schema
        localized_text_schema = {
            "type": "object",
            "properties": {
                "english": {"type": "string", "description": "Text in English"},
            },
            "required": ["english"]
        }

        if local_lang:
            localized_text_schema["properties"]["local"] = {
                "type": "string",
                "description": f"Text in {local_lang.title()}"
            }
            localized_text_schema["required"].append("local")

        return {
            "name": "analyze_image",
            "description": "Analyze image and extract structured metadata",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": localized_text_schema,
                    "alt_text": localized_text_schema,
                    "scene": {
                        "type": "object",
                        "properties": {
                            "setting": {"type": "string"},
                            "location_type": {"type": "string"},
                            "atmosphere": {"type": "string"},
                            "time_of_day": {"type": "string"},
                            "season": {"type": "string"}
                        },
                        "required": ["setting"]
                    },
                    "event": {
                        "type": "object",
                        "properties": {
                            "event_type": {"type": "string"},
                            "event_name": {"type": "string"},
                            "activity": {"type": "string"}
                        }
                    },
                    "media": {
                        "type": "object",
                        "properties": {
                            "composition": {"type": "string"},
                            "subject_count": {"type": "integer"},
                            "attire": {"type": "string"},
                            "notable_items": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["composition", "subject_count"]
                    },
                    "objects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific objects visible in the image"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "8-12 specific, useful tags for search (no generic tags)"
                    }
                },
                "required": ["description", "alt_text", "scene", "media", "objects", "tags"]
            }
        }
