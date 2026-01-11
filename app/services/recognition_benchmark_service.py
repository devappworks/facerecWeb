"""
Recognition Benchmark Service - Tests recognition quality for individuals.

This service:
1. Takes SERP images for a person
2. Runs them through recognition
3. Calculates a "recognition score" (0-100%)
4. Helps identify who needs training
"""

import os
import json
import logging
import requests
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from io import BytesIO

logger = logging.getLogger(__name__)

# Check if pgvector is enabled
VECTOR_DB_ENABLED = os.getenv('VECTOR_DB_ENABLED', 'false').lower() == 'true'


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

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


def name_similarity(name1: str, name2: str) -> float:
    """
    Calculate name similarity (0-1, higher is more similar).
    Returns 1.0 for exact match, 0.0 for completely different.
    """
    # Normalize names
    n1 = name1.lower().replace('_', ' ').strip()
    n2 = name2.lower().replace('_', ' ').strip()

    if n1 == n2:
        return 1.0

    # Calculate Levenshtein distance
    distance = levenshtein_distance(n1, n2)
    max_len = max(len(n1), len(n2))

    if max_len == 0:
        return 1.0

    return 1.0 - (distance / max_len)


class RecognitionBenchmarkService:
    """Service for benchmarking recognition quality."""

    STORAGE_BASE = 'storage'

    # Domain to region mapping (same as training service)
    REGION_MAP = {
        'serbia': 'rs',
        'croatia': 'hr',
        'bosnia': 'ba',
        'montenegro': 'me',
        'slovenia': 'si',
        'macedonia': 'mk',
        'greece': 'gr',
        'bulgaria': 'bg',
        'romania': 'ro',
        'hungary': 'hu',
    }

    def __init__(self, domain: str = 'serbia'):
        """
        Initialize benchmark service for a domain.

        Args:
            domain: Domain code (e.g., 'serbia')
        """
        self.domain = domain
        self.region = self.REGION_MAP.get(domain.lower(), 'us')
        self.benchmark_path = os.path.join(self.STORAGE_BASE, 'benchmarks', domain)
        os.makedirs(self.benchmark_path, exist_ok=True)

        # API configuration
        self.rapidapi_key = os.getenv(
            'RAPIDAPI_KEY',
            'c3e8343ca0mshe1b719bea5326dbp11db14jsnf52a7fb8ab17'
        )

    def benchmark_person(
        self,
        person_name: str,
        num_images: int = 20,
        save_results: bool = True,
        save_images_to: str = None
    ) -> Dict:
        """
        Benchmark recognition for a specific person.

        Downloads SERP images and tests how many are recognized.

        Args:
            person_name: Full name of the person to benchmark
            num_images: Number of test images to download (default 20)
            save_results: Whether to save benchmark results
            save_images_to: Optional directory to save test images (for review)

        Returns:
            Dict with benchmark results including recognition_score
        """
        # Import appropriate recognition service based on configuration
        if VECTOR_DB_ENABLED:
            from app.services.recognition_service_pgvector import PgVectorRecognitionService as RecognitionService
            logger.info("[Benchmark] Using pgvector recognition service")
        else:
            from app.services.recognition_service import RecognitionService
            logger.info("[Benchmark] Using PKL recognition service")

        benchmark_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"[Benchmark {benchmark_id}] Starting benchmark for: {person_name}")

        result = {
            'benchmark_id': benchmark_id,
            'person_name': person_name,
            'domain': self.domain,
            'timestamp': datetime.now().isoformat(),
            'num_images_requested': num_images,
            'num_images_downloaded': 0,
            'num_recognized': 0,
            'num_recognized_correctly': 0,
            'num_recognized_wrong': 0,
            'num_not_recognized': 0,
            'recognition_score': 0.0,
            'needs_training': False,
            'training_priority': 'none',
            'details': [],
            'suspicious_matches': []  # Names that are similar but different (possible data issues)
        }

        # Download test images
        test_images = self._download_test_images(person_name, num_images, benchmark_id)
        result['num_images_downloaded'] = len(test_images)

        if not test_images:
            logger.warning(f"[Benchmark {benchmark_id}] No images downloaded for {person_name}")
            result['error'] = 'No test images could be downloaded'
            return result

        # Test each image
        for img_info in test_images:
            img_result = self._test_single_image(
                image_path=img_info['path'],
                expected_name=person_name,
                benchmark_id=benchmark_id
            )
            result['details'].append(img_result)

            if img_result['recognized']:
                result['num_recognized'] += 1
                if img_result['recognized_correctly']:
                    result['num_recognized_correctly'] += 1
                else:
                    result['num_recognized_wrong'] += 1

                    # Check for suspicious matches (similar but different names)
                    recognized_as = img_result.get('recognized_as', '')
                    if recognized_as:
                        similarity = name_similarity(person_name, recognized_as)
                        # Flag as suspicious if name is 60-95% similar
                        # (too similar to be coincidence, but not an exact match)
                        if 0.6 <= similarity < 0.95:
                            suspicious_entry = {
                                'expected_name': person_name,
                                'recognized_as': recognized_as,
                                'similarity': round(similarity * 100, 1),
                                'confidence': img_result.get('confidence'),
                                'image': img_result.get('image_path')
                            }
                            # Avoid duplicates
                            if not any(s['recognized_as'] == recognized_as for s in result['suspicious_matches']):
                                result['suspicious_matches'].append(suspicious_entry)
                                logger.warning(
                                    f"[Benchmark {benchmark_id}] Suspicious match: "
                                    f"'{recognized_as}' vs expected '{person_name}' "
                                    f"(similarity: {similarity*100:.1f}%)"
                                )
            else:
                result['num_not_recognized'] += 1

        # Calculate recognition score
        if result['num_images_downloaded'] > 0:
            # Score based on correct recognitions only
            result['recognition_score'] = round(
                (result['num_recognized_correctly'] / result['num_images_downloaded']) * 100, 2
            )

        # Determine training priority
        result['needs_training'], result['training_priority'] = self._calculate_training_priority(result)

        # Save images if requested (for manual review)
        if save_images_to:
            import shutil
            os.makedirs(save_images_to, exist_ok=True)
            for img_info in test_images:
                try:
                    if os.path.exists(img_info['path']):
                        filename = os.path.basename(img_info['path'])
                        dest_path = os.path.join(save_images_to, filename)
                        shutil.copy2(img_info['path'], dest_path)
                except Exception as e:
                    logger.warning(f"[Benchmark {benchmark_id}] Failed to save image: {str(e)}")
            logger.info(f"[Benchmark {benchmark_id}] Saved {len(test_images)} images to {save_images_to}")

        # Cleanup temp images
        for img_info in test_images:
            try:
                if os.path.exists(img_info['path']):
                    os.remove(img_info['path'])
            except:
                pass

        # Save results
        if save_results:
            self._save_benchmark_result(result)

        logger.info(f"[Benchmark {benchmark_id}] Complete: {person_name} - Score: {result['recognition_score']}%")

        return result

    def batch_benchmark(
        self,
        people: List[str],
        num_images_per_person: int = 20
    ) -> List[Dict]:
        """
        Benchmark multiple people.

        Args:
            people: List of person names to benchmark
            num_images_per_person: Images to test per person

        Returns:
            List of benchmark results
        """
        results = []
        for person in people:
            try:
                result = self.benchmark_person(person, num_images_per_person)
                results.append(result)
            except Exception as e:
                logger.error(f"Error benchmarking {person}: {str(e)}")
                results.append({
                    'person_name': person,
                    'error': str(e),
                    'recognition_score': 0.0,
                    'needs_training': True,
                    'training_priority': 'unknown'
                })

        return results

    def get_training_candidates(self, min_score: float = 80.0) -> List[Dict]:
        """
        Get list of people who need training based on recent benchmarks.

        Args:
            min_score: Minimum acceptable recognition score

        Returns:
            List of people needing training, sorted by priority
        """
        candidates = []

        # Load recent benchmarks
        benchmarks = self._load_recent_benchmarks()

        for benchmark in benchmarks:
            if benchmark.get('recognition_score', 100) < min_score:
                candidates.append({
                    'person_name': benchmark['person_name'],
                    'recognition_score': benchmark.get('recognition_score', 0),
                    'training_priority': benchmark.get('training_priority', 'medium'),
                    'last_benchmark': benchmark.get('timestamp'),
                    'benchmark_id': benchmark.get('benchmark_id')
                })

        # Sort by priority (high > medium > low) and then by score (lower first)
        priority_order = {'high': 0, 'medium': 1, 'low': 2, 'unknown': 1, 'none': 3}
        candidates.sort(key=lambda x: (
            priority_order.get(x['training_priority'], 1),
            x['recognition_score']
        ))

        return candidates

    def _download_test_images(
        self,
        person_name: str,
        num_images: int,
        benchmark_id: str
    ) -> List[Dict]:
        """
        Download test images from SERP with face validation.

        Uses same settings as training service:
        - Domain-specific region (rs for Serbia, etc.)
        - type=photo (not face - our detection is better)
        - Face count validation (exactly 1 face required)
        """
        downloaded = []
        rejected_faces = 0

        try:
            # Query SERP API - match training service settings
            url = "https://real-time-image-search.p.rapidapi.com/search"
            querystring = {
                "query": f"{person_name}",
                "limit": str(min(num_images * 5, 200)),  # Get 5x to account for face filtering
                "size": "large",
                "type": "photo",  # Use 'photo' like training, not 'face'
                "region": self.region  # Domain-specific region
            }
            headers = {
                "x-rapidapi-key": self.rapidapi_key,
                "x-rapidapi-host": "real-time-image-search.p.rapidapi.com"
            }

            response = requests.get(url, headers=headers, params=querystring, timeout=30)

            if response.status_code != 200:
                logger.error(f"[Benchmark {benchmark_id}] SERP API error: {response.status_code}")
                return downloaded

            data = response.json()
            results = data.get('data', [])

            logger.info(f"[Benchmark {benchmark_id}] SERP returned {len(results)} results (region={self.region})")

            # Create temp directory for this benchmark
            temp_dir = os.path.join(tempfile.gettempdir(), f'benchmark_{benchmark_id}')
            os.makedirs(temp_dir, exist_ok=True)

            # Download images with face validation
            for i, item in enumerate(results):
                if len(downloaded) >= num_images:
                    break

                try:
                    image_url = item.get('thumbnail_url') or item.get('url')
                    if not image_url:
                        continue

                    img_response = requests.get(image_url, timeout=10, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })

                    if img_response.status_code != 200:
                        continue

                    # Validate image content
                    content_type = img_response.headers.get('content-type', '')
                    if 'image' not in content_type:
                        continue

                    # Save temp image
                    ext = '.jpg'
                    if 'png' in content_type:
                        ext = '.png'
                    elif 'webp' in content_type:
                        ext = '.webp'

                    filepath = os.path.join(temp_dir, f'test_{i:03d}{ext}')
                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)

                    # Validate face count - only keep images with exactly 1 face
                    face_count = self._count_faces_in_image(filepath, benchmark_id)

                    if face_count != 1:
                        # Remove file - wrong number of faces
                        try:
                            os.remove(filepath)
                        except:
                            pass
                        rejected_faces += 1
                        continue

                    downloaded.append({
                        'path': filepath,
                        'source_url': image_url,
                        'index': i
                    })

                except Exception as e:
                    logger.warning(f"[Benchmark {benchmark_id}] Failed to download image {i}: {str(e)}")
                    continue

            logger.info(f"[Benchmark {benchmark_id}] Downloaded {len(downloaded)} valid images "
                       f"(rejected {rejected_faces} with wrong face count)")

        except Exception as e:
            logger.error(f"[Benchmark {benchmark_id}] Error downloading images: {str(e)}")

        return downloaded

    def _count_faces_in_image(self, image_path: str, benchmark_id: str) -> int:
        """
        Count faces in an image for filtering multi-person photos.

        Returns:
            Number of faces, or:
            -1 on error
            0 if no faces
        """
        try:
            from deepface import DeepFace

            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='retinaface',
                enforce_detection=False
            )

            if not faces:
                return 0

            MIN_FACE_SIZE = 50  # Minimum face size for quality
            MIN_DETECT_SIZE = 20  # Minimum to count as real face

            real_faces = 0
            for face in faces:
                confidence = face.get('confidence', 0)
                facial_area = face.get('facial_area', {})
                left_eye = facial_area.get('left_eye')
                right_eye = facial_area.get('right_eye')
                width = facial_area.get('w', 0)
                height = facial_area.get('h', 0)

                # Skip fake detections
                if confidence <= 0 or left_eye is None or right_eye is None:
                    continue
                if width < MIN_DETECT_SIZE or height < MIN_DETECT_SIZE:
                    continue

                real_faces += 1

            return real_faces

        except Exception as e:
            logger.debug(f"[Benchmark {benchmark_id}] Face count error: {str(e)}")
            return -1

    def _test_single_image(
        self,
        image_path: str,
        expected_name: str,
        benchmark_id: str
    ) -> Dict:
        """
        Test recognition on a single image.

        Returns:
            Dict with recognition result
        """
        # Import appropriate recognition service based on configuration
        if VECTOR_DB_ENABLED:
            from app.services.recognition_service_pgvector import PgVectorRecognitionService as RecognitionService
        else:
            from app.services.recognition_service import RecognitionService

        result = {
            'image_path': os.path.basename(image_path),
            'recognized': False,
            'recognized_correctly': False,
            'recognized_as': None,
            'confidence': None,
            'error': None
        }

        try:
            # Read image bytes
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            # Run recognition
            recognition_result = RecognitionService.recognize_face(
                BytesIO(image_bytes),
                self.domain
            )

            if recognition_result.get('status') == 'success':
                # Get person name - handle both PKL and pgvector response formats
                person_name = recognition_result.get('person')  # PKL format
                if not person_name:
                    # pgvector format: get from recognized_persons or best_match
                    recognized_persons = recognition_result.get('recognized_persons', [])
                    if recognized_persons:
                        person_name = recognized_persons[0].get('name')
                    elif recognition_result.get('best_match'):
                        person_name = recognition_result['best_match'].get('person_name')

                if person_name:
                    result['recognized'] = True
                    result['recognized_as'] = person_name

                    # Get confidence from best match - handle both formats
                    best_match = recognition_result.get('best_match', {})
                    if best_match:
                        # PKL format uses confidence_metrics
                        confidence_metrics = best_match.get('confidence_metrics', {})
                        result['confidence'] = confidence_metrics.get('confidence_percentage')
                        # pgvector format uses direct confidence field
                        if not result['confidence'] and 'confidence' in best_match:
                            result['confidence'] = best_match.get('confidence')
                else:
                    # No person found in the result
                    result['error'] = 'No faces recognized'

                # Check if recognized correctly
                # Compare normalized names (lowercase, no underscores)
                recognized_name = (result['recognized_as'] or '').lower().replace('_', ' ').strip()
                expected_normalized = expected_name.lower().replace('_', ' ').strip()

                # Check for partial match (handles cases like "John Smith" matching "John_Smith")
                if recognized_name == expected_normalized:
                    result['recognized_correctly'] = True
                elif recognized_name in expected_normalized or expected_normalized in recognized_name:
                    result['recognized_correctly'] = True

            elif recognition_result.get('status') == 'no_faces':
                result['error'] = 'No faces detected'
            else:
                result['error'] = recognition_result.get('message', 'Recognition failed')

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"[Benchmark {benchmark_id}] Error testing image: {str(e)}")

        return result

    def _calculate_training_priority(self, result: Dict) -> Tuple[bool, str]:
        """
        Calculate training priority based on benchmark results.

        Returns:
            Tuple of (needs_training: bool, priority: str)
        """
        score = result.get('recognition_score', 0)
        num_wrong = result.get('num_recognized_wrong', 0)

        # High priority: very low recognition or many wrong recognitions
        if score < 30 or num_wrong >= 5:
            return True, 'high'

        # Medium priority: moderate recognition issues
        if score < 50 or num_wrong >= 3:
            return True, 'medium'

        # Low priority: slight improvement needed
        if score < 80:
            return True, 'low'

        # No training needed
        return False, 'none'

    def _save_benchmark_result(self, result: Dict):
        """Save benchmark result to file."""
        try:
            # Save individual result
            person_folder = self._safe_folder_name(result['person_name'])
            person_path = os.path.join(self.benchmark_path, person_folder)
            os.makedirs(person_path, exist_ok=True)

            result_path = os.path.join(
                person_path,
                f"benchmark_{result['benchmark_id']}.json"
            )

            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)

            # Also update summary file
            self._update_summary(result)

        except Exception as e:
            logger.error(f"Error saving benchmark result: {str(e)}")

    def _update_summary(self, result: Dict):
        """Update benchmark summary file."""
        summary_path = os.path.join(self.benchmark_path, 'summary.json')

        try:
            # Load existing summary
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
            else:
                summary = {'people': {}, 'last_updated': None}

            # Update person entry
            person_name = result['person_name']
            summary['people'][person_name] = {
                'last_benchmark': result['timestamp'],
                'recognition_score': result['recognition_score'],
                'needs_training': result['needs_training'],
                'training_priority': result['training_priority'],
                'benchmark_id': result['benchmark_id']
            }

            summary['last_updated'] = datetime.now().isoformat()

            # Save summary
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

        except Exception as e:
            logger.error(f"Error updating summary: {str(e)}")

    def _load_recent_benchmarks(self, days: int = 7) -> List[Dict]:
        """Load recent benchmark results."""
        benchmarks = []

        summary_path = os.path.join(self.benchmark_path, 'summary.json')
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)

                for person_name, data in summary.get('people', {}).items():
                    benchmarks.append({
                        'person_name': person_name,
                        **data
                    })

            except Exception as e:
                logger.error(f"Error loading benchmarks: {str(e)}")

        return benchmarks

    def _safe_folder_name(self, name: str) -> str:
        """Convert name to safe folder name."""
        import unicodedata
        import re

        normalized = unicodedata.normalize('NFKD', name)
        ascii_name = ''.join([c for c in normalized if not unicodedata.combining(c)])
        ascii_name = ascii_name.encode('ascii', 'ignore').decode('ascii')
        safe = re.sub(r'[^\w_-]', '', ascii_name.replace(' ', '_'))

        return safe.lower()
