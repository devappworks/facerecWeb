"""
Configuration for face recognition service

This module contains configuration constants and settings used across the application.
"""

# Historical persons blacklist for video recognition
# These persons should NOT appear in video recognition results as they are:
# - Historical figures who died before modern video era
# - Low-quality historical photos/paintings that cause false positives
# - Should be kept in database for photo recognition but excluded from video processing
HISTORICAL_PERSONS_BLACKLIST = [
    "Stevan_Stojanovic_Mokranjac",  # Died 1914 - composer, historical photos cause false matches
    "Nikola_Tesla",                  # Died 1943 - inventor, historical photos/paintings
    # Add more historical figures here as needed
    # Format: "Firstname_Lastname" (matching the person name in database)
]

# Minimum confidence thresholds
VIDEO_MIN_CONFIDENCE = 45.0  # Minimum confidence for single frame in video (%)
VIDEO_BEST_CONFIDENCE_MIN = 60.0  # At least one frame must exceed this for confirmed person (%)
PHOTO_MIN_CONFIDENCE = 50.0  # Minimum confidence for photo recognition (%)

# Aggregation settings
MIN_FRAME_OCCURRENCE_MIN = 2  # Minimum frames required (for short videos)
MIN_FRAME_OCCURRENCE_MAX = 10  # Maximum frames required (for long videos/collages)
MIN_FRAME_OCCURRENCE_RATIO = 0.05  # 5% of total frames

# Confirmation thresholds for multi-person detection
RELATIVE_THRESHOLD_RATIO = 0.10  # Person must have >= 10% of primary person's weighted score
ABSOLUTE_MINIMUM_SCORE = 50.0  # Or have weighted score >= this value
