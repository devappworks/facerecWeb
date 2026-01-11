#!/usr/bin/env python3
"""
Gemini 3 Flash Video Analysis POC
Supports multiple video types with specialized prompts

Usage:
    export GEMINI_API_KEY=your-key-here
    python gemini_video_poc.py [video_type]

Video types: press_conference, news_broadcast, drone_aerial, interview, event_ceremony, generic
If no type specified, auto-detection will be used.
"""
import os
import json
import sys
from datetime import datetime
from google import genai

# Configuration
API_KEY = os.environ.get('GEMINI_API_KEY')
VIDEO_PATH = "/root/photoanalytics/vucic3.mp4"
OUTPUT_DIR = "/root/facerecognition-backend/gemini_results"
MODEL = "gemini-3-flash-preview"

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


class GeminiVideoAnalyzer:
    """Gemini 3 Flash video analyzer with video type detection"""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.video_file = None

    def upload_video(self, video_path: str):
        """Upload video once, reuse for multiple prompts"""
        import time as time_module

        print(f"📤 Uploading video: {video_path}")
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"   File size: {file_size_mb:.2f} MB")

        self.video_file = self.client.files.upload(file=video_path)
        print(f"✅ Upload complete. File ID: {self.video_file.name}")

        # Wait for file to become active (processing on Google's side)
        print("⏳ Waiting for video processing...")
        max_wait = 120  # Max 2 minutes
        wait_interval = 5
        elapsed = 0

        while elapsed < max_wait:
            file_info = self.client.files.get(name=self.video_file.name)
            state = file_info.state.name if hasattr(file_info.state, 'name') else str(file_info.state)

            if state == "ACTIVE":
                print(f"✅ Video is ready (state: {state})")
                break
            elif state == "FAILED":
                raise Exception(f"Video processing failed: {file_info}")
            else:
                print(f"   State: {state}, waiting {wait_interval}s...")
                time_module.sleep(wait_interval)
                elapsed += wait_interval
        else:
            raise Exception(f"Timeout waiting for video to become active after {max_wait}s")

        return self.video_file

    def detect_video_type(self) -> str:
        """Auto-detect video type using Gemini"""
        print("🔍 Detecting video type...")
        response = self.client.models.generate_content(
            model=MODEL,
            contents=[self.video_file, DETECT_TYPE_PROMPT]
        )
        video_type = response.text.strip().lower().replace(" ", "_")

        # Validate response
        valid_types = list(PROMPTS.keys())
        if video_type not in valid_types:
            print(f"⚠️  Unknown type '{video_type}', using 'generic'")
            video_type = "generic"

        print(f"✅ Detected type: {video_type}")
        return video_type

    def analyze(self, video_type: str = None) -> dict:
        """Run analysis with appropriate prompt"""
        if not self.video_file:
            raise ValueError("No video uploaded. Call upload_video() first.")

        # Auto-detect if not specified
        if not video_type:
            video_type = self.detect_video_type()

        prompt = PROMPTS.get(video_type, PROMPTS["generic"])

        print(f"🎬 Running {video_type} analysis...")
        print(f"   Using model: {MODEL}")

        start_time = datetime.now()
        response = self.client.models.generate_content(
            model=MODEL,
            contents=[self.video_file, prompt]
        )
        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"✅ Analysis complete in {elapsed:.1f} seconds")

        # Extract token usage if available
        usage_info = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            prompt_tokens = getattr(usage, 'prompt_token_count', 0)
            output_tokens = getattr(usage, 'candidates_token_count', 0)
            total_tokens = getattr(usage, 'total_token_count', 0)

            # Gemini 3 Flash pricing (as of Dec 2025)
            # Input: $0.10 per 1M tokens, Output: $0.40 per 1M tokens
            input_cost = (prompt_tokens / 1_000_000) * 0.10
            output_cost = (output_tokens / 1_000_000) * 0.40
            total_cost = input_cost + output_cost

            usage_info = {
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(total_cost, 6)
            }

            print(f"📊 Token usage: {prompt_tokens:,} input + {output_tokens:,} output = {total_tokens:,} total")
            print(f"💰 Estimated cost: ${total_cost:.4f}")

        # Try to parse JSON from response
        response_text = response.text
        parsed_json = None

        try:
            # Try to extract JSON from response (may be wrapped in markdown)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            parsed_json = json.loads(response_text)
            print("✅ Response parsed as valid JSON")
        except json.JSONDecodeError as e:
            print(f"⚠️  Could not parse response as JSON: {e}")
            parsed_json = None

        return {
            "video_type": video_type,
            "analysis_raw": response.text,
            "analysis_parsed": parsed_json,
            "model": MODEL,
            "processing_time_seconds": elapsed,
            "usage": usage_info,
            "timestamp": datetime.now().isoformat()
        }

    def save_results(self, results: dict, output_dir: str, video_path: str) -> str:
        """Save results to JSON file"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = os.path.basename(video_path).split('.')[0]
        output_file = f"{output_dir}/{video_name}_{results['video_type']}_{timestamp}.json"

        full_results = {
            "video_path": video_path,
            "video_size_mb": os.path.getsize(video_path) / (1024 * 1024),
            **results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved to: {output_file}")
        return output_file


def main():
    """Main entry point"""
    print("=" * 60)
    print("Gemini 3 Flash Video Analysis POC")
    print("=" * 60)

    if not API_KEY:
        print("\n❌ Error: GEMINI_API_KEY environment variable not set")
        print("   Set it with: export GEMINI_API_KEY=your-key-here")
        sys.exit(1)

    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"\n❌ Error: Video not found: {VIDEO_PATH}")
        sys.exit(1)

    # Allow video type override from command line
    video_type_override = None
    if len(sys.argv) > 1:
        video_type_override = sys.argv[1].lower().replace("-", "_")
        valid_types = list(PROMPTS.keys())
        if video_type_override not in valid_types:
            print(f"\n⚠️  Invalid video type: {video_type_override}")
            print(f"   Valid types: {', '.join(valid_types)}")
            print("   Using auto-detection instead.\n")
            video_type_override = None
        else:
            print(f"\n📋 Using specified video type: {video_type_override}\n")

    # Run analysis
    analyzer = GeminiVideoAnalyzer(API_KEY)
    analyzer.upload_video(VIDEO_PATH)

    results = analyzer.analyze(video_type=video_type_override)
    output_file = analyzer.save_results(results, OUTPUT_DIR, VIDEO_PATH)

    # Print summary
    print("\n" + "=" * 60)
    print(f"VIDEO TYPE: {results['video_type']}")
    print(f"PROCESSING TIME: {results['processing_time_seconds']:.1f}s")
    if results.get('usage'):
        usage = results['usage']
        print(f"TOKEN USAGE: {usage['prompt_tokens']:,} input + {usage['output_tokens']:,} output = {usage['total_tokens']:,} total")
        print(f"ESTIMATED COST: ${usage['estimated_cost_usd']:.4f}")
    print("=" * 60)

    if results['analysis_parsed']:
        print("\n📊 PARSED ANALYSIS:")
        print(json.dumps(results['analysis_parsed'], indent=2, ensure_ascii=False))
    else:
        print("\n📝 RAW RESPONSE:")
        print(results['analysis_raw'])

    print("\n" + "=" * 60)
    print(f"Full results saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
