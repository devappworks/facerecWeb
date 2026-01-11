"""
Google Gemini Vision Provider implementation.
Supports Gemini models for image analysis.
"""

import os
import json
import time
import logging
import traceback
import base64
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

from app.services.vision.base import (
    VisionProvider, VisionConfig, ImageMetadata, UsageInfo,
    LocalizedText, SceneAnalysis, EventAnalysis, MediaAnalysis,
    RecognizedPerson, IdentifiedPerson
)

logger = logging.getLogger(__name__)


class GeminiVisionProvider(VisionProvider):
    """Google Gemini Vision API provider for image analysis."""

    # Available models (updated December 2025)
    # See: https://ai.google.dev/gemini-api/docs/models
    AVAILABLE_MODELS = [
        "gemini-2.0-flash",           # Fast, stable (recommended)
        "gemini-2.5-flash",           # 2.5 Flash stable
        "gemini-2.5-pro",             # 2.5 Pro stable (best quality)
        "gemini-3-flash-preview",     # 3 Flash preview
        "gemini-3-pro-preview",       # 3 Pro preview (latest, best)
    ]

    def __init__(self, config: VisionConfig):
        super().__init__(config)
        load_dotenv()
        self.api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set")

        # Import Google Generative AI library
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            self.model = genai.GenerativeModel(self.config.model)
        except ImportError:
            raise ImportError(
                "google-generativeai package is required for Gemini provider. "
                "Install it with: pip install google-generativeai"
            )

    def get_provider_name(self) -> str:
        return "gemini"

    def analyze_image(
        self,
        image_base64: str,
        recognized_persons: Optional[List[RecognizedPerson]] = None
    ) -> ImageMetadata:
        """
        Analyze an image using Google Gemini Vision API.

        Args:
            image_base64: Base64-encoded image data
            recognized_persons: Optional list of recognized persons from face recognition

        Returns:
            ImageMetadata with complete analysis
        """
        start_time = time.time()
        logger.info(f"Starting Gemini image analysis with model: {self.config.model}")

        # Build prompt
        system_prompt = self._build_system_prompt(recognized_persons)
        json_schema = self._get_json_schema()

        # Check if this is a Gemini 3 model (supports person recognition)
        is_gemini_3 = 'gemini-3' in self.config.model

        # Create the full prompt with JSON instruction
        # Note: We show an example structure, not the schema itself, to avoid model confusion
        example_structure = {
            "description": {"english": "A detailed description of the image content"},
            "alt_text": {"english": "Concise accessibility text"},
            "scene": {"setting": "indoor/outdoor", "location_type": "specific place type"},
            "event": {"event_type": "type of event if applicable", "activity": "what is happening"},
            "media": {"composition": "photo/portrait/group", "subject_count": 1},
            "objects": ["list", "of", "detected", "objects"],
            "tags": ["specific", "useful", "searchable", "tags"]
        }

        # Add person recognition field for Gemini 3 models
        if is_gemini_3:
            example_structure["identified_persons"] = [
                {
                    "name": "Full name of the person if recognized (or 'Unknown' if not recognizable)",
                    "confidence": "high/medium/low",
                    "role": "Their profession or why they are notable (e.g., 'Tennis player', 'Actor', 'Politician')",
                    "description": "Brief description of the person in this image"
                }
            ]

        person_recognition_prompt = ""
        if is_gemini_3:
            person_recognition_prompt = """

PERSON RECOGNITION: If there are people in this image, try to identify them. For each person:
- If you recognize them as a public figure, celebrity, athlete, politician, or notable person, provide their full name
- Indicate your confidence level (high/medium/low)
- Include their role/profession and a brief description
- If you cannot identify someone, use "Unknown" as the name but still describe them
- Only include people who are clearly visible in the image"""

        full_prompt = f"""{system_prompt}{person_recognition_prompt}

IMPORTANT: Analyze the image and respond with ONLY a valid JSON object. Do not include any explanatory text, markdown formatting, or code blocks. Output raw JSON only.

Your response must follow this structure (with your actual analysis, not these example values):
{json.dumps(example_structure, indent=2)}

Now analyze this image:"""

        # Decode base64 image for Gemini
        image_data = base64.b64decode(image_base64)

        # Make API request with retry logic
        response = self._safe_request(full_prompt, image_data)

        # Parse response
        metadata = self._parse_response(response, recognized_persons)

        elapsed = time.time() - start_time
        logger.info(f"Gemini analysis completed in {elapsed:.2f}s")

        return metadata

    def _safe_request(self, prompt: str, image_data: bytes) -> Any:
        """Make Gemini API request with retry logic."""
        max_retries = 5
        backoff_factor = 2

        for attempt in range(max_retries):
            try:
                logger.info(f"Gemini API request attempt {attempt + 1}/{max_retries}")

                # Create image part
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": image_data
                }

                # Generate content
                response = self.model.generate_content(
                    [prompt, image_part],
                    generation_config=self.genai.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_tokens,
                    )
                )

                return response

            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(x in error_str for x in ['rate', 'quota', 'overloaded', 'timeout', '503', '429'])

                if is_retryable and attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Gemini API error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Gemini API error after {max_retries} attempts: {e}")
                    traceback.print_exc()
                    raise

    def _fix_truncated_json(self, json_text: str) -> str:
        """Attempt to fix truncated JSON by completing brackets and quotes."""
        if not json_text:
            return "{}"

        # Try parsing as-is first
        try:
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            pass

        # Count brackets and try to fix
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        open_brackets = json_text.count('[')
        close_brackets = json_text.count(']')

        # Check for unterminated strings - find last quote and try to close it
        fixed = json_text.rstrip()

        # If ends with incomplete key-value, try to complete
        if fixed.endswith(','):
            fixed = fixed[:-1]

        # Try to close any open strings
        quote_count = fixed.count('"')
        if quote_count % 2 != 0:
            # Find last field and try to terminate it
            fixed += '"'

        # Close any remaining brackets
        fixed += ']' * (open_brackets - close_brackets)
        fixed += '}' * (open_braces - close_braces)

        # Validate and return
        try:
            json.loads(fixed)
            logger.warning(f"Fixed truncated JSON response")
            return fixed
        except json.JSONDecodeError:
            # If still invalid, return original (will fail with proper error)
            return json_text

    def _parse_response(
        self,
        response: Any,
        recognized_persons: Optional[List[RecognizedPerson]]
    ) -> ImageMetadata:
        """Parse Gemini response into ImageMetadata."""
        try:
            # Extract text from response
            response_text = response.text

            # Find JSON in response (handle markdown code blocks)
            json_text = response_text.strip()
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end > start:
                    json_text = response_text[start:end].strip()
                else:
                    # No closing ```, take everything after ```json
                    json_text = response_text[start:].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                if end > start:
                    json_text = response_text[start:end].strip()
                else:
                    # No closing ```, take everything after ```
                    json_text = response_text[start:].strip()

            # If json_text doesn't start with {, try to find the first {
            if json_text and not json_text.startswith('{'):
                brace_pos = json_text.find('{')
                if brace_pos >= 0:
                    json_text = json_text[brace_pos:]

            # Try to fix truncated JSON by completing brackets
            json_text = self._fix_truncated_json(json_text)

            # Parse JSON
            data = json.loads(json_text)

            # Parse localized text
            desc_data = data.get("description", {})
            if isinstance(desc_data, str):
                description = LocalizedText(english=desc_data)
            else:
                description = LocalizedText(
                    english=desc_data.get("english", ""),
                    local=desc_data.get("local")
                )

            alt_data = data.get("alt_text", {})
            if isinstance(alt_data, str):
                alt_text = LocalizedText(english=alt_data)
            else:
                alt_text = LocalizedText(
                    english=alt_data.get("english", ""),
                    local=alt_data.get("local")
                )

            # Parse scene analysis
            scene_data = data.get("scene", {})
            scene = SceneAnalysis(
                setting=scene_data.get("setting", "unknown"),
                location_type=scene_data.get("location_type"),
                atmosphere=scene_data.get("atmosphere"),
                time_of_day=scene_data.get("time_of_day"),
                season=scene_data.get("season")
            ) if scene_data else None

            # Parse event analysis
            event_data = data.get("event", {})
            event = EventAnalysis(
                event_type=event_data.get("event_type"),
                event_name=event_data.get("event_name"),
                activity=event_data.get("activity")
            ) if event_data else None

            # Parse media analysis
            media_data = data.get("media", {})
            # Ensure notable_items is always a list
            notable_items = media_data.get("notable_items", [])
            if not isinstance(notable_items, list):
                notable_items = [notable_items] if notable_items and notable_items != "none" else []
            media = MediaAnalysis(
                composition=media_data.get("composition", "unknown"),
                subject_count=media_data.get("subject_count", 1),
                attire=media_data.get("attire"),
                notable_items=notable_items
            ) if media_data else None

            # Parse identified persons (Gemini 3 models only)
            identified_persons = []
            identified_data = data.get("identified_persons", [])
            if isinstance(identified_data, list):
                for person_data in identified_data:
                    if isinstance(person_data, dict) and person_data.get("name"):
                        identified_persons.append(IdentifiedPerson(
                            name=person_data.get("name", "Unknown"),
                            confidence=person_data.get("confidence", "medium"),
                            role=person_data.get("role"),
                            description=person_data.get("description")
                        ))
                if identified_persons:
                    logger.info(f"Identified {len(identified_persons)} person(s) via Gemini vision")

            # Calculate token usage and cost
            usage_info = None
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, 'prompt_token_count', 0)
                completion_tokens = getattr(usage, 'candidates_token_count', 0)
                total = getattr(usage, 'total_token_count', 0)
                logger.info(f"Token usage - total: {total}, prompt: {prompt_tokens}, completion: {completion_tokens}")
                usage_info = UsageInfo.calculate_cost(
                    model=self.config.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
                logger.info(f"Estimated cost: ${usage_info.cost_usd:.6f}")

            return ImageMetadata(
                description=description,
                alt_text=alt_text,
                scene=scene,
                event=event,
                media=media,
                objects=data.get("objects", []),
                tags=data.get("tags", [])[:self.config.max_tags],
                recognized_persons=recognized_persons or [],
                identified_persons=identified_persons,
                provider=self.get_provider_name(),
                model=self.config.model,
                usage=usage_info
            )

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Gemini JSON response: {e}")
            logger.error(f"Response text: {response.text[:500]}...")
            traceback.print_exc()
            return ImageMetadata(
                description=LocalizedText(english="Error parsing response"),
                alt_text=LocalizedText(english="Image"),
                provider=self.get_provider_name(),
                model=self.config.model
            )

        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            traceback.print_exc()
            return ImageMetadata(
                description=LocalizedText(english="Error analyzing image"),
                alt_text=LocalizedText(english="Image"),
                provider=self.get_provider_name(),
                model=self.config.model
            )

    @classmethod
    def list_models(cls) -> List[str]:
        """List available Gemini vision models."""
        return cls.AVAILABLE_MODELS
