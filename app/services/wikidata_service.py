"""
Wikidata Service for querying celebrities by country and occupation.
Provides comprehensive celebrity data from Wikipedia's structured database.
"""

import requests
import logging
from typing import List, Dict, Optional
from flask import current_app

logger = logging.getLogger(__name__)


class WikidataService:
    """Service for querying Wikidata for celebrity information"""

    SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

    # Wikidata entity IDs for countries
    COUNTRIES = {
        "serbia": "Q403",
        "usa": "Q30",
        "uk": "Q145",
        "france": "Q142",
        "germany": "Q183",
        "spain": "Q29",
        "italy": "Q38",
        "canada": "Q16",
        "australia": "Q408",
        "brazil": "Q155",
        "argentina": "Q414",
        "croatia": "Q224",
        "bosnia": "Q225",
        "montenegro": "Q236",
        "north_macedonia": "Q221",
        "slovenia": "Q215",
        "greece": "Q41",
        "turkey": "Q43",
        "russia": "Q159",
        "china": "Q148",
        "india": "Q668",
        "japan": "Q17"
    }

    # Wikidata entity IDs for occupations
    OCCUPATIONS = {
        "actor": "Q33999",
        "politician": "Q82955",
        "tennis_player": "Q10833314",
        "football_player": "Q937857",
        "basketball_player": "Q3665646",
        "musician": "Q639669",
        "singer": "Q177220",
        "writer": "Q36180",
        "director": "Q2526255",
        "scientist": "Q901",
        "athlete": "Q2066131",
        "model": "Q4610556",
        "journalist": "Q1930187",
        "entrepreneur": "Q131524",
        "painter": "Q1028181",
        "photographer": "Q33231",
        "poet": "Q49757",
        "university_teacher": "Q1622272",
        "volleyball_player": "Q15117302",
        "handball_player": "Q12840545",
        "historian": "Q201788",
        "screenwriter": "Q28389",
        "composer": "Q36834",
        "translator": "Q333634",
        "film_actor": "Q10800557",
        "football_coach": "Q628099"
    }

    @classmethod
    def get_available_countries(cls) -> List[Dict[str, str]]:
        """Get list of available countries"""
        return [
            {"id": key, "name": key.replace("_", " ").title()}
            for key in sorted(cls.COUNTRIES.keys())
        ]

    @classmethod
    def get_available_occupations(cls) -> List[Dict[str, str]]:
        """Get list of available occupations"""
        return [
            {"id": key, "name": key.replace("_", " ").title()}
            for key in sorted(cls.OCCUPATIONS.keys())
        ]

    @classmethod
    def query_celebrities(cls, country: str, occupation: str, limit: int = 500) -> List[Dict]:
        """
        Query Wikidata for celebrities matching country and occupation.

        Args:
            country: Country code (e.g., "serbia", "usa")
            occupation: Occupation code (e.g., "actor", "politician")
            limit: Maximum number of results (default 500)

        Returns:
            List of celebrity dictionaries with name, description, image, etc.
        """
        try:
            # Get Wikidata IDs
            country_id = cls.COUNTRIES.get(country.lower())
            occupation_id = cls.OCCUPATIONS.get(occupation.lower())

            if not country_id:
                logger.error(f"Unknown country: {country}")
                return []

            if not occupation_id:
                logger.error(f"Unknown occupation: {occupation}")
                return []

            # Build SPARQL query
            query = f"""
            SELECT DISTINCT ?person ?personLabel ?personDescription ?image ?birthDate ?deathDate ?sitelinks
            WHERE {{
              ?person wdt:P31 wd:Q5 .                    # is a human
              ?person wdt:P27 wd:{country_id} .          # citizenship
              ?person wdt:P106 wd:{occupation_id} .      # occupation
              ?person wdt:P18 ?image .                   # image (REQUIRED for face training)
              ?person wikibase:sitelinks ?sitelinks .    # Wikipedia article count (notability metric)
              OPTIONAL {{ ?person wdt:P569 ?birthDate . }}
              OPTIONAL {{ ?person wdt:P570 ?deathDate . }}

              FILTER(?sitelinks >= 1)                    # At least 1 Wikipedia article (image required already filters quality)

              SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en,sr" .
              }}
            }}
            ORDER BY DESC(?sitelinks)                    # Most famous/notable first
            LIMIT {limit}
            """

            logger.info(f"Querying Wikidata for {country} {occupation}s (limit: {limit})")

            # Execute query
            response = requests.get(
                cls.SPARQL_ENDPOINT,
                params={
                    'query': query,
                    'format': 'json'
                },
                headers={
                    'User-Agent': 'FaceRecognitionTrainingApp/1.0'
                },
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"Wikidata query failed with status {response.status_code}")
                return []

            data = response.json()
            results = data.get('results', {}).get('bindings', [])

            logger.info(f"Found {len(results)} {country} {occupation}s in Wikidata")

            # Parse results
            celebrities = []
            for result in results:
                try:
                    person_name = result.get('personLabel', {}).get('value', '')

                    # Skip if no proper name
                    if not person_name or person_name.startswith('Q'):
                        continue

                    # Split name into first and last
                    name_parts = person_name.strip().split(' ', 1)
                    first_name = name_parts[0] if len(name_parts) > 0 else ''
                    last_name = name_parts[1] if len(name_parts) > 1 else ''

                    celebrity = {
                        'full_name': person_name,
                        'name': first_name,
                        'last_name': last_name,
                        'occupation': occupation,
                        'country': country,
                        'description': result.get('personDescription', {}).get('value', ''),
                        'wikidata_id': result.get('person', {}).get('value', '').split('/')[-1],
                        'image_url': result.get('image', {}).get('value', ''),
                        'birth_date': result.get('birthDate', {}).get('value', ''),
                        'death_date': result.get('deathDate', {}).get('value', ''),
                        'has_image': 'image' in result,
                        'sitelinks': int(result.get('sitelinks', {}).get('value', 0))
                    }

                    celebrities.append(celebrity)

                except Exception as parse_error:
                    logger.warning(f"Error parsing result: {str(parse_error)}")
                    continue

            return celebrities

        except requests.exceptions.Timeout:
            logger.error("Wikidata query timed out")
            return []
        except Exception as e:
            logger.error(f"Error querying Wikidata: {str(e)}")
            return []

    @classmethod
    def search_person(cls, query: str, limit: int = 20) -> List[Dict]:
        """
        Search for a specific person by name.
        Used for autocomplete/search functionality.
        Uses Wikidata Search API instead of SPARQL for better performance.

        Args:
            query: Search query (person name)
            limit: Maximum number of results

        Returns:
            List of matching people with their info
        """
        try:
            # Use Wikidata Search API for fast autocomplete
            search_response = requests.get(
                'https://www.wikidata.org/w/api.php',
                params={
                    'action': 'wbsearchentities',
                    'search': query,
                    'language': 'en',
                    'type': 'item',
                    'limit': limit,
                    'format': 'json'
                },
                headers={'User-Agent': 'FaceRecognitionTrainingApp/1.0'},
                timeout=10
            )

            if search_response.status_code != 200:
                return []

            search_data = search_response.json()
            search_results = search_data.get('search', [])

            # Filter to only humans (P31:Q5) and get additional info
            people = []
            for item in search_results[:limit]:
                entity_id = item.get('id')
                if not entity_id:
                    continue

                # Get entity details
                entity_response = requests.get(
                    'https://www.wikidata.org/w/api.php',
                    params={
                        'action': 'wbgetentities',
                        'ids': entity_id,
                        'props': 'claims|labels|descriptions',
                        'languages': 'en',
                        'format': 'json'
                    },
                    headers={'User-Agent': 'FaceRecognitionTrainingApp/1.0'},
                    timeout=5
                )

                if entity_response.status_code != 200:
                    continue

                entity_data = entity_response.json()
                entity = entity_data.get('entities', {}).get(entity_id, {})
                claims = entity.get('claims', {})

                # Check if it's a human (P31:Q5)
                instance_of = claims.get('P31', [])
                is_human = any(
                    claim.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id') == 'Q5'
                    for claim in instance_of
                )

                if not is_human:
                    continue

                # Get ALL images if available (P18) - supports multiple references
                import hashlib
                image_urls = []
                images = claims.get('P18', [])
                for img_claim in images:
                    image_filename = img_claim.get('mainsnak', {}).get('datavalue', {}).get('value', '')
                    if image_filename:
                        # Convert filename to Commons URL
                        md5 = hashlib.md5(image_filename.replace(' ', '_').encode()).hexdigest()
                        image_url = f"https://upload.wikimedia.org/wikipedia/commons/{md5[0]}/{md5[0:2]}/{image_filename.replace(' ', '_')}"
                        image_urls.append(image_url)

                person_name = entity.get('labels', {}).get('en', {}).get('value', item.get('label', ''))
                name_parts = person_name.strip().split(' ', 1)

                people.append({
                    'full_name': person_name,
                    'name': name_parts[0] if len(name_parts) > 0 else '',
                    'last_name': name_parts[1] if len(name_parts) > 1 else '',
                    'description': entity.get('descriptions', {}).get('en', {}).get('value', item.get('description', '')),
                    'occupation': '',
                    'country': '',
                    'image_url': image_urls[0] if image_urls else '',  # Backward compatibility
                    'image_urls': image_urls,  # All P18 images for multi-reference support
                    'wikidata_id': entity_id
                })

                if len(people) >= limit:
                    break

            return people

        except Exception as e:
            logger.error(f"Error searching Wikidata: {str(e)}")
            return []
