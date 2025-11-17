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
        "photographer": "Q33231"
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

              FILTER(?sitelinks > 5)                     # Only notable people (5+ Wikipedia articles)

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

        Args:
            query: Search query (person name)
            limit: Maximum number of results

        Returns:
            List of matching people with their info
        """
        try:
            sparql_query = f"""
            SELECT DISTINCT ?person ?personLabel ?personDescription ?occupation ?occupationLabel ?country ?countryLabel ?image
            WHERE {{
              ?person wdt:P31 wd:Q5 .                           # is a human
              ?person rdfs:label ?label .
              FILTER(CONTAINS(LCASE(?label), LCASE("{query}")))
              OPTIONAL {{
                ?person wdt:P106 ?occupation .
                ?occupation rdfs:label ?occupationLabel .
                FILTER(LANG(?occupationLabel) = "en")
              }}
              OPTIONAL {{
                ?person wdt:P27 ?country .
                ?country rdfs:label ?countryLabel .
                FILTER(LANG(?countryLabel) = "en")
              }}
              OPTIONAL {{ ?person wdt:P18 ?image . }}
              SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en" .
              }}
            }}
            LIMIT {limit}
            """

            response = requests.get(
                cls.SPARQL_ENDPOINT,
                params={
                    'query': sparql_query,
                    'format': 'json'
                },
                headers={
                    'User-Agent': 'FaceRecognitionTrainingApp/1.0'
                },
                timeout=15
            )

            if response.status_code != 200:
                return []

            data = response.json()
            results = data.get('results', {}).get('bindings', [])

            people = []
            for result in results:
                person_name = result.get('personLabel', {}).get('value', '')
                if not person_name or person_name.startswith('Q'):
                    continue

                name_parts = person_name.strip().split(' ', 1)

                people.append({
                    'full_name': person_name,
                    'name': name_parts[0] if len(name_parts) > 0 else '',
                    'last_name': name_parts[1] if len(name_parts) > 1 else '',
                    'description': result.get('personDescription', {}).get('value', ''),
                    'occupation': result.get('occupationLabel', {}).get('value', ''),
                    'country': result.get('countryLabel', {}).get('value', ''),
                    'image_url': result.get('image', {}).get('value', ''),
                    'wikidata_id': result.get('person', {}).get('value', '').split('/')[-1]
                })

            return people

        except Exception as e:
            logger.error(f"Error searching Wikidata: {str(e)}")
            return []
