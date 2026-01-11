"""
Celebrity Discovery Service - Finds trending/hot celebrities to train.

Multi-source celebrity discovery:
1. Google Trends for trending searches
2. Wikidata for celebrities by country/occupation
3. News APIs for people in the news
4. Social media trends (future)

The goal is to find celebrities who are currently "hot" and likely to appear
in photos that users will want to recognize.
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CelebrityDiscoveryService:
    """Service for discovering trending celebrities."""

    # Country to Wikidata mapping
    COUNTRIES = {
        'serbia': 'Q403',
        'croatia': 'Q224',
        'slovenia': 'Q215',
        'bosnia': 'Q225',
        'montenegro': 'Q236',
        'north_macedonia': 'Q221',
        'greece': 'Q41',
        'usa': 'Q30',
        'uk': 'Q145',
    }

    # Occupation categories to discover
    OCCUPATIONS = [
        'actor',
        'singer',
        'athlete',
        'politician',
        'television_presenter',
        'model',
        'musician',
        'businessperson',
    ]

    def __init__(self):
        """Initialize discovery service."""
        self.cache_path = 'storage/discovery_cache'
        os.makedirs(self.cache_path, exist_ok=True)

        # API keys
        self.rapidapi_key = os.getenv(
            'RAPIDAPI_KEY',
            'c3e8343ca0mshe1b719bea5326dbp11db14jsnf52a7fb8ab17'
        )

    def discover_trending_celebrities(
        self,
        country: str = 'serbia',
        max_results: int = 20,
        occupations: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Discover trending celebrities for a country.

        Combines multiple sources:
        1. Wikidata celebrities by occupation
        2. Google Trends (if available)
        3. News mentions

        Args:
            country: Country code
            max_results: Maximum celebrities to return
            occupations: Specific occupations to search (or use defaults)

        Returns:
            List of celebrity dicts with name, wikidata_id, source, etc.
        """
        celebrities = []
        seen_names = set()

        occupations = occupations or self.OCCUPATIONS

        # Source 1: Wikidata by occupation (main source)
        logger.info(f"Discovering celebrities from Wikidata for {country}...")
        for occupation in occupations:
            try:
                wikidata_celebs = self._get_wikidata_celebrities(
                    country=country,
                    occupation=occupation,
                    limit=max_results // len(occupations) + 1
                )

                for celeb in wikidata_celebs:
                    name = celeb.get('name', '').lower()
                    if name and name not in seen_names:
                        seen_names.add(name)
                        celeb['source'] = 'wikidata'
                        celeb['occupation'] = occupation
                        celebrities.append(celeb)

            except Exception as e:
                logger.warning(f"Error getting {occupation}s from Wikidata: {str(e)}")

        # Source 2: Google Trends (trending people)
        logger.info(f"Checking Google Trends for {country}...")
        try:
            trending = self._get_google_trends(country=country, limit=10)
            for person in trending:
                name = person.get('name', '').lower()
                if name and name not in seen_names:
                    seen_names.add(name)
                    person['source'] = 'google_trends'
                    celebrities.append(person)
        except Exception as e:
            logger.warning(f"Error getting Google Trends: {str(e)}")

        # Source 3: News API (people in the news)
        logger.info(f"Checking news for celebrities from {country}...")
        try:
            news_people = self._get_news_celebrities(country=country, limit=10)
            for person in news_people:
                name = person.get('name', '').lower()
                if name and name not in seen_names:
                    seen_names.add(name)
                    person['source'] = 'news'
                    celebrities.append(person)
        except Exception as e:
            logger.warning(f"Error getting news celebrities: {str(e)}")

        # Score and sort celebrities by relevance
        celebrities = self._score_celebrities(celebrities)
        celebrities.sort(key=lambda x: x.get('score', 0), reverse=True)

        logger.info(f"Discovered {len(celebrities)} celebrities for {country}")
        return celebrities[:max_results]

    def _get_wikidata_celebrities(
        self,
        country: str,
        occupation: str,
        limit: int = 20
    ) -> List[Dict]:
        """Get celebrities from Wikidata by country and occupation."""
        from app.services.wikidata_service import WikidataService

        try:
            celebrities = WikidataService.query_celebrities(
                country=country,
                occupation=occupation,
                limit=limit
            )

            return [{
                'name': c.get('full_name'),
                'wikidata_id': c.get('wikidata_id'),
                'image_url': c.get('image_url'),
                'description': c.get('description')
            } for c in celebrities]

        except Exception as e:
            logger.error(f"Wikidata query error: {str(e)}")
            return []

    def _get_google_trends(self, country: str, limit: int = 10) -> List[Dict]:
        """
        Get trending searches from Google Trends.

        Uses RapidAPI Google Trends service.
        """
        trending_people = []

        try:
            # Map country to Google Trends geo code
            geo_codes = {
                'serbia': 'RS',
                'croatia': 'HR',
                'slovenia': 'SI',
                'bosnia': 'BA',
                'montenegro': 'ME',
                'greece': 'GR',
                'usa': 'US',
                'uk': 'GB',
            }

            geo = geo_codes.get(country.lower(), 'RS')

            url = "https://google-trends8.p.rapidapi.com/trendings"
            querystring = {"region": geo, "hl": "en-US"}
            headers = {
                "x-rapidapi-key": self.rapidapi_key,
                "x-rapidapi-host": "google-trends8.p.rapidapi.com"
            }

            response = requests.get(url, headers=headers, params=querystring, timeout=15)

            if response.status_code == 200:
                data = response.json()
                trends = data.get('data', [])

                for trend in trends[:limit * 2]:  # Get more, filter to people
                    title = trend.get('title', '')
                    # Simple heuristic: if it has 2+ words, might be a person name
                    if len(title.split()) >= 2:
                        trending_people.append({
                            'name': title,
                            'trending_rank': trends.index(trend) + 1
                        })

        except Exception as e:
            logger.warning(f"Google Trends error: {str(e)}")

        return trending_people[:limit]

    def _get_news_celebrities(self, country: str, limit: int = 10) -> List[Dict]:
        """
        Get celebrities mentioned in recent news.

        Uses a simple approach: search news for country + celebrity categories.
        """
        news_people = []

        try:
            # NewsAPI or Google News via RapidAPI
            url = "https://google-news13.p.rapidapi.com/search"

            # Search for different celebrity categories in the country
            search_terms = [
                f"{country} actor",
                f"{country} singer",
                f"{country} celebrity",
                f"{country} sports star"
            ]

            for term in search_terms[:2]:  # Limit API calls
                querystring = {"keyword": term, "lr": "en-US"}
                headers = {
                    "x-rapidapi-key": self.rapidapi_key,
                    "x-rapidapi-host": "google-news13.p.rapidapi.com"
                }

                try:
                    response = requests.get(url, headers=headers, params=querystring, timeout=10)

                    if response.status_code == 200:
                        data = response.json()
                        items = data.get('items', [])

                        for item in items[:5]:
                            title = item.get('title', '')
                            # Extract potential person names from titles
                            # This is a simplified approach
                            names = self._extract_names_from_text(title)
                            for name in names:
                                if name and len(name) > 5:
                                    news_people.append({
                                        'name': name,
                                        'news_source': item.get('source', {}).get('name'),
                                        'news_date': item.get('publishedAt')
                                    })
                except:
                    continue

        except Exception as e:
            logger.warning(f"News API error: {str(e)}")

        # Deduplicate
        seen = set()
        unique_people = []
        for person in news_people:
            name = person.get('name', '').lower()
            if name not in seen:
                seen.add(name)
                unique_people.append(person)

        return unique_people[:limit]

    def _extract_names_from_text(self, text: str) -> List[str]:
        """
        Simple name extraction from text.

        Looks for capitalized word pairs that might be names.
        """
        names = []
        words = text.split()

        for i in range(len(words) - 1):
            word1 = words[i].strip('.,!?:;"\'-')
            word2 = words[i + 1].strip('.,!?:;"\'-')

            # Check if both words are capitalized (potential name)
            if (word1 and word2 and
                word1[0].isupper() and word2[0].isupper() and
                len(word1) > 1 and len(word2) > 1 and
                word1.isalpha() and word2.isalpha()):

                potential_name = f"{word1} {word2}"
                # Filter out common non-name pairs
                skip_words = {'The', 'This', 'That', 'How', 'What', 'When', 'Where', 'Why'}
                if word1 not in skip_words:
                    names.append(potential_name)

        return names

    def _score_celebrities(self, celebrities: List[Dict]) -> List[Dict]:
        """
        Score celebrities by relevance/hotness.

        Factors:
        - Has Wikidata entry (reliable)
        - Has P18 image (trainable)
        - In trending/news (current)
        - Occupation relevance
        """
        for celeb in celebrities:
            score = 50  # Base score

            # Has Wikidata ID (+20)
            if celeb.get('wikidata_id'):
                score += 20

            # Has image URL (+15)
            if celeb.get('image_url'):
                score += 15

            # Source bonuses
            source = celeb.get('source', '')
            if source == 'google_trends':
                score += 25  # Trending now
            elif source == 'news':
                score += 15  # In the news
            elif source == 'wikidata':
                score += 10  # Established celebrity

            # Trending rank bonus
            if celeb.get('trending_rank'):
                rank_bonus = max(0, 20 - celeb['trending_rank'] * 2)
                score += rank_bonus

            # Occupation bonus (actors/singers more likely to be photographed)
            high_photo_occupations = ['actor', 'singer', 'model', 'television_presenter']
            if celeb.get('occupation') in high_photo_occupations:
                score += 10

            celeb['score'] = score

        return celebrities

    def get_country_top_celebrities(
        self,
        country: str,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get top celebrities for a country (cached).

        This uses Wikidata to get established celebrities, useful for
        initial training setup.

        Args:
            country: Country code
            limit: Max celebrities to return

        Returns:
            List of celebrity dicts
        """
        cache_file = os.path.join(self.cache_path, f'{country}_top_celebrities.json')

        # Check cache (valid for 7 days)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)

                cache_date = datetime.fromisoformat(cache.get('cached_at', '2000-01-01'))
                if (datetime.now() - cache_date).days < 7:
                    return cache.get('celebrities', [])[:limit]

            except:
                pass

        # Fetch from Wikidata
        all_celebrities = []

        for occupation in self.OCCUPATIONS:
            try:
                celebs = self._get_wikidata_celebrities(
                    country=country,
                    occupation=occupation,
                    limit=limit // len(self.OCCUPATIONS) + 5
                )
                all_celebrities.extend(celebs)
            except:
                continue

        # Score and sort
        all_celebrities = self._score_celebrities(all_celebrities)
        all_celebrities.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Deduplicate
        seen = set()
        unique_celebrities = []
        for celeb in all_celebrities:
            name = celeb.get('name', '').lower()
            if name not in seen:
                seen.add(name)
                unique_celebrities.append(celeb)

        # Cache results
        try:
            cache = {
                'cached_at': datetime.now().isoformat(),
                'country': country,
                'celebrities': unique_celebrities[:limit]
            }
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
        except:
            pass

        return unique_celebrities[:limit]

    def search_celebrity(
        self,
        query: str,
        country: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for a specific celebrity.

        Args:
            query: Search query (name)
            country: Optional country filter

        Returns:
            List of matching celebrities
        """
        from app.services.wikidata_service import WikidataService

        try:
            results = WikidataService.search_person(query, limit=10)

            celebrities = [{
                'name': r.get('full_name'),
                'wikidata_id': r.get('wikidata_id'),
                'image_url': r.get('image_url'),
                'description': r.get('description'),
                'source': 'wikidata_search'
            } for r in results]

            return celebrities

        except Exception as e:
            logger.error(f"Celebrity search error: {str(e)}")
            return []
