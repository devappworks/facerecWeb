"""
Transliteration Service - Uses Claude Haiku to convert Serbian transliterated names
back to their international (English) spelling for better image search results.

Examples:
- Karlos Alkaraz -> Carlos Alcaraz
- Tjeri Anri -> Thierry Henry
- Tejlor Fric -> Taylor Fritz
"""

import os
import json
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class TransliterationService:
    """Service for transliterating Serbian names to international spelling using Claude Haiku."""

    def __init__(self):
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')
        self.client = None
        self.model = "claude-haiku-4-5"  # Claude Haiku 4.5 - fastest and most cost-effective
        self.cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'storage', 'transliteration_cache.json'
        )
        self._cache = self._load_cache()

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self.client is None:
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        return self.client

    def _load_cache(self) -> Dict[str, str]:
        """Load cached transliterations from file."""
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading transliteration cache: {e}")
        return {}

    def _save_cache(self):
        """Save cache to file."""
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error saving transliteration cache: {e}")

    def is_likely_transliterated(self, name: str) -> bool:
        """
        Check if a name looks like it might be Serbian transliteration.
        Returns True if the name contains patterns common in Serbian transliteration.
        """
        # Check for Serbian Cyrillic characters (definitely needs transliteration)
        serbian_chars = set('žšćčđŽŠĆČĐ')
        if any(c in name for c in serbian_chars):
            return True

        name_lower = name.lower()

        # Common endings that suggest actual Serbian names (not transliterations)
        serbian_name_endings = ['ic', 'ovic', 'evic', 'vic']
        if any(name_lower.endswith(end) for end in serbian_name_endings):
            # These are Serbian names - keep as-is
            return False

        # Known transliteration patterns from Serbian phonetic spelling
        # These are combinations that appear in Serbian transliterations of foreign names
        transliteration_patterns = [
            # Consonant combinations
            'tj',   # for "ti" sound (Tjeri -> Thierry)
            'dj',   # for "j" sound
            'nj',   # for "ñ" sound
            'lj',   # for "ll" sound
            'ks',   # for "x"
            'kv',   # for "qu"
            # Vowel sounds typically changed in Serbian
            'aj',   # for "eye" sound
            'ej',   # for "ay" sound
            'oj',   # for "oy" sound
            'ij',   # for "ee" sound
            # Common Serbian phonetic substitutions
            'ae',   # uncommon in Serbian, suggests foreign name
        ]

        # Check if name contains transliteration patterns
        if any(pattern in name_lower for pattern in transliteration_patterns):
            return True

        # Check for specific letter substitutions common in Serbian transliteration
        # c -> k (Carlos -> Karlos), th -> t, w -> v, y -> j
        suspicious_starts = ['karlos', 'tejlor', 'dejvid', 'majkl', 'dzordan', 'dzejms',
                            'dzoni', 'dzef', 'dzek', 'endru', 'kristofer', 'dzastin',
                            'anri', 'derik', 'bred', 'pjer', 'filipe']

        first_name = name_lower.split()[0] if ' ' in name_lower else name_lower
        first_name = first_name.replace('_', '')  # Handle underscored names
        if any(first_name.startswith(s) for s in suspicious_starts):
            return True

        # Check for names that look like phonetic Serbian spelling
        # These patterns are unusual in English but common in Serbian transliteration
        # Note: Must be careful not to match international names
        phonetic_patterns = [
            'alkaraz',  # Alkaraz (Alcaraz) - more specific to avoid matching "Alcaraz"
            'fric',     # Fric (Fritz)
            'vajt',     # Vajt (White)
            'smit',     # Smit (Smith)
            'dzons',    # Dzons (Jones)
        ]

        if any(pattern in name_lower for pattern in phonetic_patterns):
            return True

        return False

    def transliterate_single(self, serbian_name: str, force: bool = False) -> Tuple[str, bool]:
        """
        Transliterate a single Serbian name to international spelling.

        Args:
            serbian_name: The name in Serbian transliteration
            force: If True, always call API even if name doesn't look transliterated

        Returns:
            Tuple of (international_name, was_changed)
        """
        # Clean the name
        clean_name = serbian_name.replace('_', ' ').strip()

        # Check cache first
        cache_key = clean_name.lower()
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return cached, cached.lower() != clean_name.lower()

        # Check if it looks like it needs transliteration
        if not force and not self.is_likely_transliterated(clean_name):
            # Cache as-is to avoid repeated checks
            self._cache[cache_key] = clean_name
            self._save_cache()
            return clean_name, False

        try:
            client = self._get_client()

            response = client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Convert this Serbian transliterated name to its original international (English) spelling.
If it's already in correct international spelling, return it unchanged.
Only return the name, nothing else.

Name: {clean_name}

International spelling:"""
                    }
                ]
            )

            international_name = response.content[0].text.strip()

            # Validate response - should be similar length and structure
            if len(international_name) > len(clean_name) * 2 or len(international_name) < len(clean_name) * 0.5:
                logger.warning(f"Suspicious transliteration result: {clean_name} -> {international_name}")
                international_name = clean_name

            # Cache the result
            self._cache[cache_key] = international_name
            self._save_cache()

            was_changed = international_name.lower() != clean_name.lower()
            if was_changed:
                logger.info(f"Transliterated: {clean_name} -> {international_name}")

            return international_name, was_changed

        except Exception as e:
            logger.error(f"Error transliterating {clean_name}: {e}")
            return clean_name, False

    def transliterate_batch(self, names: List[str]) -> Dict[str, str]:
        """
        Transliterate a batch of names efficiently.
        Uses a single API call for multiple names to reduce costs.

        Args:
            names: List of Serbian names to transliterate

        Returns:
            Dict mapping original names to international spellings
        """
        results = {}
        names_to_translate = []

        # Check cache first
        for name in names:
            clean_name = name.replace('_', ' ').strip()
            cache_key = clean_name.lower()

            if cache_key in self._cache:
                results[name] = self._cache[cache_key]
            elif self.is_likely_transliterated(clean_name):
                names_to_translate.append((name, clean_name))
            else:
                results[name] = clean_name
                self._cache[cache_key] = clean_name

        if not names_to_translate:
            self._save_cache()
            return results

        # Batch translate remaining names
        try:
            client = self._get_client()

            # Build the prompt for batch translation
            names_list = "\n".join([f"{i+1}. {clean}" for i, (orig, clean) in enumerate(names_to_translate)])

            response = client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Convert these Serbian transliterated names to their original international (English) spellings.
Return ONLY a numbered list with the international spelling for each name.
If a name is already correct, return it unchanged.

Names:
{names_list}

International spellings (numbered list):"""
                    }
                ]
            )

            # Parse response
            response_text = response.content[0].text.strip()
            lines = response_text.split('\n')

            for i, (orig_name, clean_name) in enumerate(names_to_translate):
                if i < len(lines):
                    # Extract name from numbered line (e.g., "1. Carlos Alcaraz")
                    line = lines[i].strip()
                    # Remove number prefix
                    if '. ' in line:
                        international = line.split('. ', 1)[1].strip()
                    else:
                        international = line.strip()

                    # Validate
                    if len(international) > 0 and len(international) < len(clean_name) * 2:
                        results[orig_name] = international
                        self._cache[clean_name.lower()] = international
                        if international.lower() != clean_name.lower():
                            logger.info(f"Batch transliterated: {clean_name} -> {international}")
                    else:
                        results[orig_name] = clean_name
                        self._cache[clean_name.lower()] = clean_name
                else:
                    results[orig_name] = clean_name
                    self._cache[clean_name.lower()] = clean_name

            self._save_cache()

        except Exception as e:
            logger.error(f"Error in batch transliteration: {e}")
            # Fall back to original names
            for orig_name, clean_name in names_to_translate:
                results[orig_name] = clean_name

        return results

    def get_search_name(self, display_name: str) -> str:
        """
        Get the international name to use for image searches.
        This is the main method to use when searching for images.

        Args:
            display_name: The name as stored in the database (may be Serbian)

        Returns:
            The international spelling suitable for Google/SERP searches
        """
        international, _ = self.transliterate_single(display_name)
        return international

    def process_training_queue(self, queue_entries: List[Dict]) -> List[Dict]:
        """
        Process a list of training queue entries, adding search_name field.

        Args:
            queue_entries: List of queue entry dicts with 'person_name' field

        Returns:
            Same list with 'search_name' field added to each entry
        """
        names = [entry.get('person_name', '') for entry in queue_entries]
        translations = self.transliterate_batch(names)

        for entry in queue_entries:
            orig_name = entry.get('person_name', '')
            entry['search_name'] = translations.get(orig_name, orig_name)

        return queue_entries


# Singleton instance
_service_instance = None

def get_transliteration_service() -> TransliterationService:
    """Get singleton instance of transliteration service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = TransliterationService()
    return _service_instance
