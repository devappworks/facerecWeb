"""
OpenAI Vision Provider implementation.
Supports GPT-4 Vision models for image analysis.
"""

import os
import json
import time
import logging
import traceback
from typing import List, Optional, Dict, Any

import openai
from openai import OpenAI
from dotenv import load_dotenv

from app.services.vision.base import (
    VisionProvider, VisionConfig, ImageMetadata, UsageInfo,
    LocalizedText, SceneAnalysis, EventAnalysis, MediaAnalysis, RecognizedPerson
)

logger = logging.getLogger(__name__)


class OpenAIVisionProvider(VisionProvider):
    """OpenAI Vision API provider for image analysis."""

    # Available models (updated December 2025)
    AVAILABLE_MODELS = [
        "gpt-4.1-mini",    # Fast, cost-effective (default)
        "gpt-4.1",         # Balanced
        "gpt-4o",          # Best quality (previous gen)
        "gpt-4o-mini",     # Alternative fast option
        "gpt-5-mini",      # GPT-5 fast option
        "gpt-5",           # GPT-5 best quality
    ]

    def __init__(self, config: VisionConfig):
        super().__init__(config)
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=self.api_key)

    def get_provider_name(self) -> str:
        return "openai"

    def analyze_image(
        self,
        image_base64: str,
        recognized_persons: Optional[List[RecognizedPerson]] = None
    ) -> ImageMetadata:
        """
        Analyze an image using OpenAI Vision API.

        Args:
            image_base64: Base64-encoded image data
            recognized_persons: Optional list of recognized persons from face recognition

        Returns:
            ImageMetadata with complete analysis
        """
        start_time = time.time()
        logger.info(f"Starting OpenAI image analysis with model: {self.config.model}")

        # Build messages
        system_prompt = self._build_system_prompt(recognized_persons)
        schema = self._get_json_schema()

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image and provide comprehensive metadata."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

        # Determine which parameters to use based on model
        # Newer models (GPT-5, o1, etc.) use max_completion_tokens instead of max_tokens
        # and don't support custom temperature values
        is_new_model = any(m in self.config.model for m in ['gpt-5', 'o1-', 'o3-'])

        request_params = {
            "model": self.config.model,
            "messages": messages,
            "functions": [schema],
            "function_call": {"name": "analyze_image"}
        }

        # GPT-5 and o1/o3 models don't support custom temperature - only default (1)
        if not is_new_model:
            request_params["temperature"] = self.config.temperature

        if is_new_model:
            request_params["max_completion_tokens"] = self.config.max_tokens
        else:
            request_params["max_tokens"] = self.config.max_tokens

        # Make API request with retry logic
        response = self._safe_request(**request_params)

        # Parse response
        metadata = self._parse_response(response, recognized_persons)

        elapsed = time.time() - start_time
        logger.info(f"OpenAI analysis completed in {elapsed:.2f}s")

        return metadata

    def _safe_request(self, *args, **kwargs) -> Any:
        """Make OpenAI API request with retry logic."""
        max_retries = 5
        backoff_factor = 2

        for attempt in range(max_retries):
            try:
                logger.info(f"OpenAI API request attempt {attempt + 1}/{max_retries}")
                response = self.client.chat.completions.create(*args, **kwargs)
                return response

            except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"OpenAI API error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"OpenAI API error after {max_retries} attempts: {e}")
                    traceback.print_exc()
                    raise

            except Exception as e:
                logger.error(f"Unexpected error in OpenAI request: {e}")
                traceback.print_exc()
                raise

    def _parse_response(
        self,
        response: Any,
        recognized_persons: Optional[List[RecognizedPerson]]
    ) -> ImageMetadata:
        """Parse OpenAI response into ImageMetadata."""
        try:
            # Extract function call arguments
            if not response.choices or not response.choices[0].message.function_call:
                raise ValueError("No function call in response")

            function_call = response.choices[0].message.function_call
            data = json.loads(function_call.arguments)

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
            media = MediaAnalysis(
                composition=media_data.get("composition", "unknown"),
                subject_count=media_data.get("subject_count", 1),
                attire=media_data.get("attire"),
                notable_items=media_data.get("notable_items", [])
            ) if media_data else None

            # Calculate token usage and cost
            usage_info = None
            if hasattr(response, "usage") and response.usage:
                logger.info(f"Token usage - total: {response.usage.total_tokens}, prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}")
                usage_info = UsageInfo.calculate_cost(
                    model=self.config.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
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
                provider=self.get_provider_name(),
                model=self.config.model,
                usage=usage_info
            )

        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {e}")
            traceback.print_exc()
            # Return minimal metadata on error
            return ImageMetadata(
                description=LocalizedText(english="Error analyzing image"),
                alt_text=LocalizedText(english="Image"),
                provider=self.get_provider_name(),
                model=self.config.model
            )

    @classmethod
    def list_models(cls) -> List[str]:
        """List available OpenAI vision models."""
        return cls.AVAILABLE_MODELS
