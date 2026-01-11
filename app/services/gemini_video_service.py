"""
Gemini 3 Flash Video Analysis Service
Provides fast video analysis to complement face recognition.

Architecture:
- Gemini analysis runs fast (~20-30s) and returns initial insights
- Face recognition runs in parallel (slower) and enriches results later
- Results are saved progressively so users see Gemini results first
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from typing import Dict, Optional, List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Video type detection prompt
DETECT_TYPE_PROMPT = """What type of video is this? Respond with ONLY one of these categories:
- press_conference
- news_broadcast
- drone_aerial
- interview
- event_ceremony
- generic

Just the category name, nothing else."""

# Specialized prompts for each video type
PROMPTS = {
    "press_conference": """Analyze this press conference video and provide a detailed analysis in JSON format:

{
  "video_type": "press_conference",
  "event_context": {
    "setting": "venue description",
    "organization": "if identifiable from logos/banners",
    "estimated_date": "if visible indicators",
    "language": "primary language spoken"
  },
  "speakers": [
    {
      "person_id": "S1",
      "identified_name": "Full name if recognized (public figure, celebrity, politician) or null if unknown",
      "name_source": "how name was determined: on-screen_text/recognized_public_figure/introduced_verbally/unknown",
      "role": "their official role or position",
      "physical_description": {
        "gender": "male/female",
        "age_range": "estimate",
        "clothing": "formal/casual, colors, notable items",
        "distinguishing_features": "glasses, beard, hair color, etc"
      },
      "position": "podium/seated/standing",
      "demeanor": {
        "overall_tone": "confident/nervous/passionate/calm/angry/defensive",
        "energy_level": "high/moderate/low",
        "speaking_style": "reading_notes/teleprompter/speaking_freely/mixed"
      },
      "body_language": [
        {"timestamp": "MM:SS", "gesture": "hand gestures/pointing/emphasis/etc", "interpretation": "what it conveys"}
      ],
      "speaking_segments": [
        {"start": "MM:SS", "end": "MM:SS", "topic_hint": "brief description"}
      ]
    }
  ],
  "audience": {
    "estimated_count": "number or range",
    "composition": "journalists/officials/public",
    "reactions": [
      {"timestamp": "MM:SS", "type": "applause/laughter/murmur/silence/heckling", "intensity": "mild/moderate/strong", "context": "what triggered it"}
    ]
  },
  "content_summary": {
    "brief_summary": "2-3 sentence summary of what was discussed",
    "main_topics": ["list of main topics covered"],
    "notable_quotes": [
      {"timestamp": "MM:SS", "quote": "exact or paraphrased notable statement", "context": "why this is significant"}
    ],
    "tone_of_message": "informative/celebratory/warning/confrontational/diplomatic/etc"
  },
  "visible_text": [
    {"text": "content", "type": "banner/nameplate/screen/logo", "timestamp": "MM:SS"}
  ],
  "key_moments": [
    {"timestamp": "MM:SS", "description": "notable event or statement", "significance": "why this moment matters"}
  ]
}

Return ONLY valid JSON, no markdown formatting or extra text.""",

    "news_broadcast": """Analyze this news broadcast video and provide a detailed analysis in JSON format:

{
  "video_type": "news_broadcast",
  "broadcast_info": {
    "network": "if identifiable",
    "program_type": "news/talk show/interview",
    "language": "primary language"
  },
  "on_screen_people": [
    {
      "person_id": "P1",
      "identified_name": "Full name if recognized or shown on screen, null if unknown",
      "name_source": "lower_third/recognized_journalist/introduced/unknown",
      "role": "anchor/reporter/guest/interviewee",
      "physical_description": {
        "gender": "male/female",
        "age_range": "estimate",
        "clothing": "description",
        "distinguishing_features": "notable features"
      },
      "location": "studio/field/remote",
      "demeanor": {
        "overall_tone": "professional/urgent/casual/serious/emotional",
        "delivery_style": "scripted/conversational/live_reporting"
      },
      "appearances": [
        {"start": "MM:SS", "end": "MM:SS", "context": "presenting/interviewing/etc"}
      ]
    }
  ],
  "content_summary": {
    "brief_summary": "2-3 sentence summary of the broadcast content",
    "main_stories": ["list of news stories covered"],
    "notable_quotes": [
      {"timestamp": "MM:SS", "speaker": "person_id or role", "quote": "notable statement", "context": "significance"}
    ],
    "overall_tone": "breaking_news/routine/investigative/human_interest/etc"
  },
  "graphics_and_text": [
    {"type": "lower_third/headline/chyron/ticker", "text": "content", "timestamp": "MM:SS"}
  ],
  "segments": [
    {"start": "MM:SS", "end": "MM:SS", "type": "intro/story/interview/outro", "topic": "brief description"}
  ],
  "b_roll_footage": [
    {"timestamp": "MM:SS", "description": "what's shown in supplementary footage"}
  ]
}

Return ONLY valid JSON, no markdown formatting or extra text.""",

    "drone_aerial": """Analyze this drone/aerial video and provide a detailed analysis in JSON format:

{
  "video_type": "drone_aerial",
  "location_info": {
    "setting": "urban/rural/coastal/industrial/etc",
    "identifiable_landmarks": ["list of recognizable places"],
    "estimated_location": "if determinable",
    "time_of_day": "morning/afternoon/evening/night"
  },
  "people_detected": [
    {
      "group_id": "G1",
      "type": "individual/crowd/group",
      "estimated_count": "number or range",
      "location_in_frame": "description",
      "activity": "walking/gathering/working/etc",
      "timestamps": [{"start": "MM:SS", "end": "MM:SS"}]
    }
  ],
  "vehicles": [
    {"type": "car/truck/boat/etc", "count": "number", "timestamp": "MM:SS"}
  ],
  "points_of_interest": [
    {"timestamp": "MM:SS", "description": "notable structure, event, or activity"}
  ],
  "flight_pattern": {
    "altitude_changes": "stable/varying",
    "movement": "stationary/panning/tracking/circling",
    "coverage_area": "description of what's surveyed"
  }
}

Return ONLY valid JSON, no markdown formatting or extra text.""",

    "interview": """Analyze this interview/conversation video and provide a detailed analysis in JSON format:

{
  "video_type": "interview",
  "format": "one_on_one/panel/group_discussion",
  "setting": {
    "location": "studio/office/outdoor/etc",
    "formality": "formal/casual/professional"
  },
  "participants": [
    {
      "person_id": "P1",
      "identified_name": "Full name if recognized or introduced, null if unknown",
      "name_source": "introduced/recognized_public_figure/on_screen_text/unknown",
      "role": "interviewer/interviewee/panelist/host",
      "physical_description": {
        "gender": "male/female",
        "age_range": "estimate",
        "clothing": "description",
        "distinguishing_features": "notable features"
      },
      "seating_position": "left/right/center",
      "speaking_time_estimate": "percentage",
      "demeanor": {
        "overall_tone": "confident/nervous/animated/calm/defensive/evasive/enthusiastic",
        "engagement_level": "highly_engaged/moderate/disinterested",
        "emotional_moments": [
          {"timestamp": "MM:SS", "emotion": "frustration/joy/surprise/anger/etc", "trigger": "what caused it"}
        ]
      },
      "body_language": [
        {"timestamp": "MM:SS", "observation": "leaning in/crossing arms/nodding/avoiding eye contact/etc", "interpretation": "what it suggests"}
      ]
    }
  ],
  "content_summary": {
    "brief_summary": "2-3 sentence summary of the interview",
    "main_topics": ["list of topics discussed"],
    "notable_quotes": [
      {"timestamp": "MM:SS", "speaker": "person_id", "quote": "memorable statement", "context": "why significant"}
    ],
    "interview_dynamics": "confrontational/friendly/formal/adversarial/collaborative"
  },
  "conversation_flow": [
    {"timestamp": "MM:SS", "speaker": "person_id", "topic_hint": "brief description", "tone": "questioning/defensive/explanatory/etc"}
  ],
  "notable_moments": [
    {"timestamp": "MM:SS", "type": "interruption/revelation/emotional_moment/tension/humor", "description": "what happened", "reaction": "how others reacted"}
  ]
}

Return ONLY valid JSON, no markdown formatting or extra text.""",

    "event_ceremony": """Analyze this event video and provide a detailed analysis in JSON format:

{
  "video_type": "event",
  "event_info": {
    "type": "wedding/graduation/award_ceremony/rally/diplomatic_conference/etc",
    "setting": "indoor/outdoor venue description",
    "estimated_attendance": "number or range",
    "formality_level": "highly_formal/formal/semi_formal/casual"
  },
  "key_people": [
    {
      "person_id": "P1",
      "identified_name": "Full name if recognized or announced, null if unknown",
      "name_source": "announced/recognized_public_figure/on_screen_text/nameplate/unknown",
      "role": "main subject/speaker/performer/official/honoree",
      "physical_description": {
        "gender": "male/female",
        "age_range": "estimate",
        "attire": "formal wear/uniform/costume/etc",
        "distinguishing_features": "notable features"
      },
      "demeanor": {
        "overall_tone": "celebratory/solemn/nervous/confident/emotional",
        "notable_emotions": [
          {"timestamp": "MM:SS", "emotion": "joy/pride/tears/nervousness/etc", "context": "what triggered it"}
        ]
      },
      "appearances": [
        {"timestamp": "MM:SS", "action": "speaking/performing/receiving/walking/etc", "significance": "why notable"}
      ]
    }
  ],
  "content_summary": {
    "brief_summary": "2-3 sentence summary of the event",
    "purpose": "what the event is celebrating or commemorating",
    "notable_quotes": [
      {"timestamp": "MM:SS", "speaker": "person_id or role", "quote": "memorable statement", "context": "significance"}
    ],
    "overall_atmosphere": "joyful/solemn/tense/celebratory/formal/etc"
  },
  "crowd_analysis": {
    "composition": "families/students/officials/diplomats/mixed",
    "energy_level": "enthusiastic/respectful/subdued/mixed",
    "reactions": [
      {"timestamp": "MM:SS", "type": "applause/standing_ovation/cheering/silence/laughter", "intensity": "mild/moderate/strong", "trigger": "what caused it"}
    ]
  },
  "ceremony_segments": [
    {"start": "MM:SS", "end": "MM:SS", "type": "opening/speech/performance/award/closing", "description": "what's happening"}
  ]
}

Return ONLY valid JSON, no markdown formatting or extra text.""",

    "generic": """Analyze this video and provide a detailed analysis in JSON format:

{
  "video_type": "generic",
  "overview": {
    "setting": "description of location/environment",
    "duration_estimate": "length",
    "primary_content": "what the video is mainly about",
    "suggested_category": "what type of video this most closely resembles"
  },
  "people": [
    {
      "person_id": "P1",
      "identified_name": "Full name if recognized (public figure, celebrity) or null if unknown",
      "name_source": "recognized_public_figure/on_screen_text/introduced/unknown",
      "role": "their apparent role",
      "physical_description": {
        "gender": "male/female",
        "age_range": "estimate",
        "clothing": "description",
        "distinguishing_features": "notable features"
      },
      "demeanor": {
        "overall_tone": "confident/nervous/calm/excited/serious/etc",
        "notable_emotions": [
          {"timestamp": "MM:SS", "emotion": "description", "context": "what caused it"}
        ]
      },
      "body_language": [
        {"timestamp": "MM:SS", "observation": "gesture or posture", "interpretation": "what it suggests"}
      ],
      "screen_time": "prominent/occasional/brief",
      "appearances": [
        {"timestamp": "MM:SS", "action": "what they're doing"}
      ]
    }
  ],
  "content_summary": {
    "brief_summary": "2-3 sentence summary of the video content",
    "main_topics": ["list of main topics or activities"],
    "notable_quotes": [
      {"timestamp": "MM:SS", "speaker": "person_id or description", "quote": "notable statement", "context": "significance"}
    ],
    "overall_tone": "informative/entertaining/serious/casual/etc"
  },
  "visible_text": [
    {"text": "content", "location": "where visible", "timestamp": "MM:SS"}
  ],
  "key_moments": [
    {"timestamp": "MM:SS", "description": "notable event", "significance": "why it matters"}
  ],
  "audience_reactions": [
    {"timestamp": "MM:SS", "type": "applause/laughter/silence/etc", "intensity": "mild/moderate/strong"}
  ],
  "technical_notes": {
    "video_quality": "good/moderate/poor",
    "audio_quality": "good/moderate/poor/none",
    "camera_work": "stable/shaky/professional"
  }
}

Return ONLY valid JSON, no markdown formatting or extra text."""
}


class GeminiVideoService:
    """
    Service for analyzing videos using Gemini 3 Flash.

    Designed to run in parallel with face recognition:
    - Gemini analysis: Fast (~20-30s), provides initial insights
    - Face recognition: Slower, enriches with person identification
    """

    MODEL = "gemini-3-flash-preview"

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.client = None

        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set - Gemini video analysis disabled")
        else:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Gemini video service initialized")
            except ImportError:
                logger.warning("google-genai package not installed - Gemini video analysis disabled")

    def is_available(self) -> bool:
        """Check if Gemini video analysis is available"""
        return self.client is not None

    def analyze_video(
        self,
        video_path: str,
        video_type: Optional[str] = None,
        timeout: int = 180
    ) -> Dict:
        """
        Analyze a video file using Gemini 3 Flash.

        Args:
            video_path: Path to video file
            video_type: Optional video type override (auto-detect if None)
            timeout: Max time to wait for video processing (seconds)

        Returns:
            Dictionary with analysis results
        """
        import time as time_module

        if not self.is_available():
            return {
                "success": False,
                "error": "Gemini video service not available",
                "reason": "API key not configured or google-genai not installed"
            }

        start_time = datetime.now()

        try:
            # Step 1: Upload video
            logger.info(f"Uploading video to Gemini: {video_path}")
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

            video_file = self.client.files.upload(file=video_path)
            logger.info(f"Video uploaded: {video_file.name} ({file_size_mb:.2f} MB)")

            # Step 2: Wait for video to become active
            logger.info("Waiting for video processing...")
            max_wait = timeout
            wait_interval = 5
            elapsed = 0

            while elapsed < max_wait:
                file_info = self.client.files.get(name=video_file.name)
                state = file_info.state.name if hasattr(file_info.state, 'name') else str(file_info.state)

                if state == "ACTIVE":
                    logger.info(f"Video ready (state: {state})")
                    break
                elif state == "FAILED":
                    return {
                        "success": False,
                        "error": "Video processing failed on Gemini side",
                        "state": state
                    }
                else:
                    logger.debug(f"Video state: {state}, waiting...")
                    time_module.sleep(wait_interval)
                    elapsed += wait_interval
            else:
                return {
                    "success": False,
                    "error": f"Timeout waiting for video processing ({max_wait}s)"
                }

            # Step 3: Auto-detect video type if not specified
            if not video_type:
                logger.info("Detecting video type...")
                detect_response = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=[video_file, DETECT_TYPE_PROMPT]
                )
                video_type = detect_response.text.strip().lower().replace(" ", "_")

                if video_type not in PROMPTS:
                    logger.warning(f"Unknown video type '{video_type}', using 'generic'")
                    video_type = "generic"

                logger.info(f"Detected video type: {video_type}")

            # Step 4: Run analysis with appropriate prompt
            prompt = PROMPTS.get(video_type, PROMPTS["generic"])

            logger.info(f"Running {video_type} analysis with {self.MODEL}...")
            analysis_start = datetime.now()

            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=[video_file, prompt]
            )

            analysis_time = (datetime.now() - analysis_start).total_seconds()
            logger.info(f"Analysis complete in {analysis_time:.1f}s")

            # Step 5: Parse response
            response_text = response.text
            parsed_json = None

            try:
                # Handle markdown-wrapped JSON
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    if json_end > json_start:
                        response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    if json_end > json_start:
                        response_text = response_text[json_start:json_end].strip()

                parsed_json = json.loads(response_text)
                logger.info("Response parsed as valid JSON")
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse response as JSON: {e}")

            # Step 6: Extract token usage
            usage_info = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, 'prompt_token_count', 0)
                output_tokens = getattr(usage, 'candidates_token_count', 0)
                total_tokens = getattr(usage, 'total_token_count', 0)

                # Gemini 3 Flash pricing
                input_cost = (prompt_tokens / 1_000_000) * 0.10
                output_cost = (output_tokens / 1_000_000) * 0.40
                total_cost = input_cost + output_cost

                usage_info = {
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": round(total_cost, 6)
                }

                logger.info(f"Token usage: {total_tokens:,} tokens, cost: ${total_cost:.4f}")

            # Step 7: Cleanup - delete uploaded file
            try:
                self.client.files.delete(name=video_file.name)
                logger.debug(f"Cleaned up uploaded file: {video_file.name}")
            except Exception as e:
                logger.warning(f"Could not delete uploaded file: {e}")

            total_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": True,
                "video_type": video_type,
                "analysis": parsed_json,
                "analysis_raw": response.text if not parsed_json else None,
                "model": self.MODEL,
                "processing_time_seconds": round(total_time, 2),
                "analysis_time_seconds": round(analysis_time, 2),
                "usage": usage_info,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Gemini video analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def analyze_video_bytes(
        self,
        video_bytes: bytes,
        filename: str,
        video_type: Optional[str] = None
    ) -> Dict:
        """
        Analyze video from bytes (for integration with video upload flow).

        Args:
            video_bytes: Video file bytes
            filename: Original filename (for temp file extension)
            video_type: Optional video type override

        Returns:
            Dictionary with analysis results
        """
        # Get file extension
        ext = os.path.splitext(filename)[1] or '.mp4'

        # Write to temp file (Gemini API requires file path)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            result = self.analyze_video(tmp_path, video_type)
            return result
        finally:
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    @staticmethod
    def get_available_video_types() -> List[str]:
        """Get list of supported video types"""
        return list(PROMPTS.keys())

    @staticmethod
    def merge_with_face_recognition(
        gemini_result: Dict,
        face_recognition_result: Dict
    ) -> Dict:
        """
        Merge Gemini analysis with face recognition results.

        This creates a unified result where:
        - Gemini provides visual analysis, quotes, body language, etc.
        - Face recognition provides verified person identifications

        Args:
            gemini_result: Result from Gemini video analysis
            face_recognition_result: Result from face recognition pipeline

        Returns:
            Merged result with both analyses
        """
        merged = {
            "gemini_analysis": gemini_result.get("analysis") if gemini_result.get("success") else None,
            "gemini_metadata": {
                "success": gemini_result.get("success", False),
                "video_type": gemini_result.get("video_type"),
                "model": gemini_result.get("model"),
                "processing_time_seconds": gemini_result.get("processing_time_seconds"),
                "usage": gemini_result.get("usage"),
                "error": gemini_result.get("error")
            },
            "face_recognition": face_recognition_result,
            "merged_at": datetime.now().isoformat()
        }

        # Cross-reference Gemini-identified people with face recognition
        if gemini_result.get("success") and face_recognition_result.get("success"):
            gemini_analysis = gemini_result.get("analysis", {})

            # Get people from Gemini analysis (field name varies by video type)
            gemini_people = []
            for field in ["speakers", "on_screen_people", "participants", "key_people", "people"]:
                if field in gemini_analysis:
                    gemini_people.extend(gemini_analysis[field])

            # Get confirmed persons from face recognition
            face_rec_persons = set()
            tracking_results = face_recognition_result.get("tracking_results", {})
            identity_results = tracking_results.get("identity_results", {})
            confirmed = identity_results.get("confirmed_persons", {})
            face_rec_persons = set(confirmed.keys())

            # Also check multi_frame_voting for legacy format
            if not face_rec_persons:
                multi_frame = face_recognition_result.get("multi_frame_voting", {})
                confirmed = multi_frame.get("confirmed_persons", {})
                face_rec_persons = set(confirmed.keys())

            # Create cross-reference
            merged["person_correlation"] = {
                "gemini_identified": [
                    {
                        "person_id": p.get("person_id"),
                        "identified_name": p.get("identified_name"),
                        "name_source": p.get("name_source")
                    }
                    for p in gemini_people
                    if p.get("identified_name")
                ],
                "face_recognition_confirmed": list(face_rec_persons),
                "analysis_note": "Compare Gemini visual identification with face recognition database matches"
            }

        return merged


# Singleton instance for easy import
_gemini_service = None

def get_gemini_service() -> GeminiVideoService:
    """Get or create singleton Gemini video service instance"""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiVideoService()
    return _gemini_service
