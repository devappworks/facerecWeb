"""
Merge Candidates Service

Automatically detects potential duplicate or similar person entries in the database
that may need to be merged, renamed, or cleaned up.

Detection categories:
1. TYPO - Names starting with lowercase or missing first letters
2. DUPLICATE - Very similar names (>90% match)
3. SPELLING_VARIANT - Serbian/English spelling variations
4. NICKNAME - Shortened name versions
"""

import os
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class MergeCandidatesService:
    """Service to detect and manage potential duplicate persons."""

    STORAGE_PATH = 'storage/merge_candidates'

    # Common Serbian-English name mappings
    SERBIAN_ENGLISH_MAPPINGS = {
        'Dj': ['Đ', 'Dz'],
        'C': ['Č', 'Ć'],
        'S': ['Š'],
        'Z': ['Ž'],
        'dzejms': 'james',
        'dzordan': 'jordan',
        'dzons': 'johns',
        'dzeksn': 'jackson',
        'majkl': 'michael',
        'majk': 'mike',
        'vilijams': 'williams',
        'tompson': 'thompson',
        'kris': 'chris',
        'kristofer': 'christopher',
        'stefn': 'stephen',
        'stiven': 'steven',
        'entoni': 'anthony',
        'rodrriguez': 'rodriguez',
        'mesi': 'messi',
        'ronaldo': 'ronaldo',
        'tramp': 'trump',
        'bajden': 'biden',
        'mask': 'musk',
        'durent': 'durant',
        'brajant': 'bryant',
        'alkaraz': 'alcaraz',
        'alkaras': 'alcaraz',
        'zverev': 'zverev',
        'siner': 'sinner',
    }

    # Common nickname patterns
    NICKNAME_PATTERNS = [
        ('vasa', 'vasilije'),
        ('dule', 'dusko'),
        ('pera', 'petar'),
        ('mika', 'milan'),
        ('zika', 'zivko'),
        ('bata', 'bratislav'),
        ('ceca', 'svetlana'),
        ('seka', 'jelisaveta'),
    ]

    def __init__(self, domain: str = 'serbia'):
        self.domain = domain
        os.makedirs(self.STORAGE_PATH, exist_ok=True)
        os.makedirs(os.path.join(self.STORAGE_PATH, domain), exist_ok=True)

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity ratio between two names (0-100)."""
        name1_lower = name1.lower().strip()
        name2_lower = name2.lower().strip()

        if name1_lower == name2_lower:
            return 100.0

        # Use SequenceMatcher for similarity
        ratio = SequenceMatcher(None, name1_lower, name2_lower).ratio()
        return ratio * 100

    def normalize_name(self, name: str) -> str:
        """Normalize a name for comparison."""
        # Convert to lowercase
        normalized = name.lower().strip()

        # Remove common suffixes/prefixes
        normalized = re.sub(r'\s+(jr|sr|ii|iii|iv)\.?$', '', normalized)

        # Replace Serbian characters
        replacements = {
            'đ': 'dj', 'ž': 'z', 'č': 'c', 'ć': 'c', 'š': 's',
            'Đ': 'Dj', 'Ž': 'Z', 'Č': 'C', 'Ć': 'C', 'Š': 'S'
        }
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)

        return normalized

    def detect_typo(self, name: str) -> Optional[Dict]:
        """Detect if a name is likely a typo (starts with lowercase, missing letters)."""
        if not name:
            return None

        # Check if starts with lowercase
        if name[0].islower():
            return {
                'type': 'TYPO',
                'reason': 'Name starts with lowercase letter',
                'suggestion': f"Missing first letter(s) - likely '{name[0].upper()}{name[1:]}' or similar"
            }

        # Check for very short first names (likely truncated)
        parts = name.split()
        if parts and len(parts[0]) <= 2 and parts[0][0].islower():
            return {
                'type': 'TYPO',
                'reason': 'First name appears truncated',
                'suggestion': 'Check for missing letters at the beginning'
            }

        return None

    def find_spelling_variant(self, name: str, all_names: List[str]) -> Optional[Dict]:
        """Find if name has a spelling variant (Serbian/English)."""
        name_lower = name.lower()

        for other_name in all_names:
            if other_name == name:
                continue

            other_lower = other_name.lower()

            # Check for Serbian-English mappings
            for serbian, english in self.SERBIAN_ENGLISH_MAPPINGS.items():
                if isinstance(english, str):
                    if serbian in name_lower and english in other_lower:
                        return {
                            'type': 'SPELLING_VARIANT',
                            'target': other_name,
                            'reason': f'Serbian "{serbian}" vs English "{english}"'
                        }
                    if english in name_lower and serbian in other_lower:
                        return {
                            'type': 'SPELLING_VARIANT',
                            'target': other_name,
                            'reason': f'English "{english}" vs Serbian "{serbian}"'
                        }

        return None

    def find_nickname_match(self, name: str, all_names: List[str]) -> Optional[Dict]:
        """Find if name is a nickname of another person."""
        name_parts = name.lower().split()

        for other_name in all_names:
            if other_name == name:
                continue

            other_parts = other_name.lower().split()

            # Check nickname patterns
            for nickname, full_name in self.NICKNAME_PATTERNS:
                # Check if first name matches nickname pattern
                if name_parts and other_parts:
                    if name_parts[0] == nickname and other_parts[0] == full_name:
                        # Check if last names match
                        if len(name_parts) > 1 and len(other_parts) > 1:
                            if self.name_similarity(name_parts[-1], other_parts[-1]) > 80:
                                return {
                                    'type': 'NICKNAME',
                                    'target': other_name,
                                    'reason': f'"{nickname}" is nickname for "{full_name}"'
                                }

        return None

    def find_similar_names(self, name: str, all_names: List[str], threshold: float = 85) -> List[Dict]:
        """Find names that are very similar (potential duplicates)."""
        similar = []
        name_normalized = self.normalize_name(name)

        for other_name in all_names:
            if other_name == name:
                continue

            other_normalized = self.normalize_name(other_name)
            similarity = self.name_similarity(name_normalized, other_normalized)

            if similarity >= threshold:
                similar.append({
                    'type': 'DUPLICATE',
                    'target': other_name,
                    'similarity': round(similarity, 1),
                    'reason': f'{similarity:.1f}% name similarity'
                })

        return similar

    def get_persons_from_db(self) -> List[Dict]:
        """Get all persons from the database."""
        import psycopg2
        from urllib.parse import unquote

        db_url = os.getenv(
            'VECTOR_DB_URL',
            'postgresql://facerecadmin:1Un5J20uOLstxUxJYzvb76GCdAVOjZ5eN17Cq%2FQ%2FbrU%3D@localhost:5432/facerecognition'
        )

        # Parse URL
        if '@' in db_url:
            parts = db_url.replace('postgresql://', '').split('@')
            user_pass = parts[0]
            host_db = parts[1]
            user, password = user_pass.split(':')
            password = unquote(password)
            host_port_db = host_db.split('/')
            host_port = host_port_db[0]
            database = host_port_db[1]
            host, port = host_port.split(':') if ':' in host_port else (host_port, '5432')

        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor()

        cursor.execute("""
            SELECT p.id, p.name, COUNT(fe.id) as embedding_count
            FROM persons p
            LEFT JOIN face_embeddings fe ON p.id = fe.person_id
            WHERE p.domain = %s
            GROUP BY p.id, p.name
            ORDER BY p.name
        """, (self.domain,))

        persons = []
        for row in cursor.fetchall():
            persons.append({
                'id': row[0],
                'name': row[1],
                'embedding_count': row[2]
            })

        cursor.close()
        conn.close()

        return persons

    def scan_for_candidates(self) -> Dict:
        """Scan the database for merge candidates."""
        logger.info(f"Scanning for merge candidates in domain: {self.domain}")

        persons = self.get_persons_from_db()
        all_names = [p['name'] for p in persons]
        person_map = {p['name']: p for p in persons}

        candidates = []
        processed_pairs = set()  # Avoid duplicate pairs

        for person in persons:
            name = person['name']

            # 1. Check for typos
            typo = self.detect_typo(name)
            if typo:
                # Try to find the correct name
                potential_matches = self.find_similar_names(name, all_names, threshold=70)
                target = potential_matches[0]['target'] if potential_matches else None

                candidates.append({
                    'source_id': person['id'],
                    'source_name': name,
                    'source_embeddings': person['embedding_count'],
                    'target_id': person_map[target]['id'] if target else None,
                    'target_name': target,
                    'target_embeddings': person_map[target]['embedding_count'] if target else None,
                    'type': 'TYPO',
                    'reason': typo['reason'],
                    'suggestion': typo.get('suggestion', ''),
                    'action': 'RENAME' if not target else 'MERGE'
                })
                continue

            # 2. Check for spelling variants
            variant = self.find_spelling_variant(name, all_names)
            if variant:
                pair_key = tuple(sorted([name, variant['target']]))
                if pair_key not in processed_pairs:
                    processed_pairs.add(pair_key)
                    target = variant['target']

                    candidates.append({
                        'source_id': person['id'],
                        'source_name': name,
                        'source_embeddings': person['embedding_count'],
                        'target_id': person_map[target]['id'],
                        'target_name': target,
                        'target_embeddings': person_map[target]['embedding_count'],
                        'type': 'SPELLING_VARIANT',
                        'reason': variant['reason'],
                        'suggestion': f'Merge "{name}" into "{target}"',
                        'action': 'MERGE'
                    })
                continue

            # 3. Check for nicknames
            nickname = self.find_nickname_match(name, all_names)
            if nickname:
                pair_key = tuple(sorted([name, nickname['target']]))
                if pair_key not in processed_pairs:
                    processed_pairs.add(pair_key)
                    target = nickname['target']

                    candidates.append({
                        'source_id': person['id'],
                        'source_name': name,
                        'source_embeddings': person['embedding_count'],
                        'target_id': person_map[target]['id'],
                        'target_name': target,
                        'target_embeddings': person_map[target]['embedding_count'],
                        'type': 'NICKNAME',
                        'reason': nickname['reason'],
                        'suggestion': f'Merge nickname into full name',
                        'action': 'MERGE'
                    })
                continue

            # 4. Check for duplicates (very high similarity)
            duplicates = self.find_similar_names(name, all_names, threshold=90)
            for dup in duplicates:
                pair_key = tuple(sorted([name, dup['target']]))
                if pair_key not in processed_pairs:
                    processed_pairs.add(pair_key)
                    target = dup['target']

                    candidates.append({
                        'source_id': person['id'],
                        'source_name': name,
                        'source_embeddings': person['embedding_count'],
                        'target_id': person_map[target]['id'],
                        'target_name': target,
                        'target_embeddings': person_map[target]['embedding_count'],
                        'type': 'DUPLICATE',
                        'reason': dup['reason'],
                        'suggestion': f'Merge duplicate entries',
                        'action': 'MERGE',
                        'similarity': dup['similarity']
                    })

        # Sort by type priority: TYPO > DUPLICATE > SPELLING_VARIANT > NICKNAME
        type_order = {'TYPO': 0, 'DUPLICATE': 1, 'SPELLING_VARIANT': 2, 'NICKNAME': 3}
        candidates.sort(key=lambda x: (type_order.get(x['type'], 99), x['source_name']))

        result = {
            'domain': self.domain,
            'scanned_at': datetime.now().isoformat(),
            'total_persons': len(persons),
            'total_candidates': len(candidates),
            'summary': {
                'typos': len([c for c in candidates if c['type'] == 'TYPO']),
                'duplicates': len([c for c in candidates if c['type'] == 'DUPLICATE']),
                'spelling_variants': len([c for c in candidates if c['type'] == 'SPELLING_VARIANT']),
                'nicknames': len([c for c in candidates if c['type'] == 'NICKNAME']),
            },
            'candidates': candidates
        }

        # Save to file
        self._save_candidates(result)

        return result

    def _save_candidates(self, result: Dict):
        """Save candidates to file."""
        filepath = os.path.join(
            self.STORAGE_PATH,
            self.domain,
            f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved merge candidates to: {filepath}")

        # Also save as "latest"
        latest_path = os.path.join(self.STORAGE_PATH, self.domain, 'latest.json')
        with open(latest_path, 'w') as f:
            json.dump(result, f, indent=2)

    def get_latest_candidates(self) -> Optional[Dict]:
        """Get the most recent candidates scan."""
        latest_path = os.path.join(self.STORAGE_PATH, self.domain, 'latest.json')
        if os.path.exists(latest_path):
            with open(latest_path, 'r') as f:
                return json.load(f)
        return None

    def execute_action(self, candidate_id: int, action: str, new_name: str = None) -> Dict:
        """
        Execute an action on a candidate.

        Actions:
        - MERGE: Merge source into target
        - RENAME: Rename source to new_name
        - DELETE: Delete source person
        - SKIP: Mark as reviewed/skipped
        """
        import psycopg2
        from urllib.parse import unquote

        db_url = os.getenv(
            'VECTOR_DB_URL',
            'postgresql://facerecadmin:1Un5J20uOLstxUxJYzvb76GCdAVOjZ5eN17Cq%2FQ%2FbrU%3D@localhost:5432/facerecognition'
        )

        # Parse URL
        if '@' in db_url:
            parts = db_url.replace('postgresql://', '').split('@')
            user_pass = parts[0]
            host_db = parts[1]
            user, password = user_pass.split(':')
            password = unquote(password)
            host_port_db = host_db.split('/')
            host_port = host_port_db[0]
            database = host_port_db[1]
            host, port = host_port.split(':') if ':' in host_port else (host_port, '5432')

        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor()

        # Load latest candidates
        candidates = self.get_latest_candidates()
        if not candidates:
            return {'success': False, 'error': 'No candidates found'}

        # Find the candidate
        candidate = None
        for i, c in enumerate(candidates['candidates']):
            if i == candidate_id:
                candidate = c
                break

        if not candidate:
            return {'success': False, 'error': f'Candidate {candidate_id} not found'}

        try:
            if action == 'MERGE':
                if not candidate['target_id']:
                    return {'success': False, 'error': 'No target to merge into'}

                # Move embeddings from source to target
                cursor.execute("""
                    UPDATE face_embeddings
                    SET person_id = %s
                    WHERE person_id = %s
                """, (candidate['target_id'], candidate['source_id']))
                moved = cursor.rowcount

                # Delete source person
                cursor.execute("DELETE FROM persons WHERE id = %s", (candidate['source_id'],))

                conn.commit()
                return {
                    'success': True,
                    'action': 'MERGE',
                    'message': f"Merged '{candidate['source_name']}' into '{candidate['target_name']}' ({moved} embeddings moved)"
                }

            elif action == 'RENAME':
                if not new_name:
                    return {'success': False, 'error': 'New name required for rename'}

                cursor.execute("""
                    UPDATE persons SET name = %s WHERE id = %s
                """, (new_name, candidate['source_id']))

                conn.commit()
                return {
                    'success': True,
                    'action': 'RENAME',
                    'message': f"Renamed '{candidate['source_name']}' to '{new_name}'"
                }

            elif action == 'DELETE':
                # Delete embeddings first
                cursor.execute("DELETE FROM face_embeddings WHERE person_id = %s", (candidate['source_id'],))
                deleted_emb = cursor.rowcount

                # Delete person
                cursor.execute("DELETE FROM persons WHERE id = %s", (candidate['source_id'],))

                conn.commit()
                return {
                    'success': True,
                    'action': 'DELETE',
                    'message': f"Deleted '{candidate['source_name']}' ({deleted_emb} embeddings removed)"
                }

            elif action == 'SKIP':
                # Just mark as skipped (handled in frontend)
                return {
                    'success': True,
                    'action': 'SKIP',
                    'message': f"Skipped '{candidate['source_name']}'"
                }

            else:
                return {'success': False, 'error': f'Unknown action: {action}'}

        except Exception as e:
            conn.rollback()
            logger.error(f"Error executing action: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()
            conn.close()


# Convenience function for CLI usage
def scan_merge_candidates(domain: str = 'serbia') -> Dict:
    """Scan for merge candidates in a domain."""
    service = MergeCandidatesService(domain=domain)
    return service.scan_for_candidates()
