"""
Smart Training Service - Self-improving face recognition training system.

This service implements an intelligent training loop:
1. Discovers celebrities who need training (via benchmark or new discovery)
2. Automatically triggers training for people with low recognition scores
3. Tracks training history and prevents redundant training
4. Runs on a schedule (nightly) or on-demand

Key features:
- Recognition benchmarking to identify training gaps
- Automatic SERP image collection and face validation
- Integration with existing training pipeline
- Training queue management with priorities
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from threading import Thread

logger = logging.getLogger(__name__)


class SmartTrainingService:
    """Intelligent self-improving training service."""

    STORAGE_BASE = 'storage'

    # Training priority thresholds
    HIGH_PRIORITY_THRESHOLD = 30    # Recognition score < 30% = high priority
    MEDIUM_PRIORITY_THRESHOLD = 50  # Score < 50% = medium priority
    LOW_PRIORITY_THRESHOLD = 80     # Score < 80% = low priority

    # Minimum days between re-training same person
    MIN_RETRAIN_DAYS = 7

    def __init__(self, domain: str = 'serbia'):
        """
        Initialize smart training service.

        Args:
            domain: Domain code for training
        """
        self.domain = domain

        # Paths
        self.queue_path = os.path.join(self.STORAGE_BASE, 'training_queue', domain)
        self.history_path = os.path.join(self.STORAGE_BASE, 'training_history', domain)
        self.runs_path = os.path.join(self.STORAGE_BASE, 'smart_training_runs', domain)

        for path in [self.queue_path, self.history_path, self.runs_path]:
            os.makedirs(path, exist_ok=True)

    def run_smart_training_cycle(
        self,
        discover_new: bool = True,
        benchmark_existing: bool = True,
        max_new_discoveries: int = 10,
        max_training_per_run: int = 5,
        min_images_per_person: int = 20,
        app_context=None
    ) -> Dict:
        """
        Run a complete smart training cycle.

        This is the main entry point for scheduled/nightly runs.

        Steps:
        1. Discover new trending celebrities (if enabled)
        2. Benchmark existing trained people (if enabled)
        3. Build training queue based on priorities
        4. Execute training for top priority people

        Args:
            discover_new: Whether to discover new celebrities
            benchmark_existing: Whether to re-benchmark existing people
            max_new_discoveries: Max new people to discover
            max_training_per_run: Max people to train in this run
            min_images_per_person: Target images per person during training
            app_context: Flask app context for background processing

        Returns:
            Dict with run results
        """
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"[SmartTraining {run_id}] Starting smart training cycle for {self.domain}")

        run_result = {
            'run_id': run_id,
            'domain': self.domain,
            'started_at': datetime.now().isoformat(),
            'completed_at': None,
            'status': 'running',
            'discovery': None,
            'benchmark': None,
            'training': None,
            'error': None
        }

        try:
            # Step 1: Discover new celebrities
            if discover_new:
                logger.info(f"[SmartTraining {run_id}] Discovering new celebrities...")
                run_result['discovery'] = self._discover_new_celebrities(
                    run_id=run_id,
                    max_discoveries=max_new_discoveries
                )

            # Step 2: Benchmark existing trained people
            if benchmark_existing:
                logger.info(f"[SmartTraining {run_id}] Benchmarking existing people...")
                run_result['benchmark'] = self._benchmark_existing_people(run_id=run_id)

            # Step 3: Build/update training queue
            logger.info(f"[SmartTraining {run_id}] Building training queue...")
            queue = self._build_training_queue()

            # Step 4: Execute training for top priority people
            logger.info(f"[SmartTraining {run_id}] Executing training...")
            run_result['training'] = self._execute_training_queue(
                run_id=run_id,
                queue=queue,
                max_people=max_training_per_run,
                images_per_person=min_images_per_person,
                app_context=app_context
            )

            run_result['status'] = 'completed'
            run_result['completed_at'] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"[SmartTraining {run_id}] Error in training cycle: {str(e)}")
            run_result['status'] = 'failed'
            run_result['error'] = str(e)
            run_result['completed_at'] = datetime.now().isoformat()

        # Save run result
        self._save_run_result(run_result)

        return run_result

    def add_to_training_queue(
        self,
        person_name: str,
        wikidata_id: Optional[str] = None,
        priority: str = 'medium',
        source: str = 'manual',
        recognition_score: Optional[float] = None
    ) -> Dict:
        """
        Add a person to the training queue.

        Args:
            person_name: Full name of person
            wikidata_id: Optional Wikidata ID
            priority: Training priority (high, medium, low)
            source: Source of the request (manual, benchmark, discovery)
            recognition_score: Current recognition score if known

        Returns:
            Dict with queue entry info
        """
        queue_entry = {
            'person_name': person_name,
            'wikidata_id': wikidata_id,
            'priority': priority,
            'source': source,
            'recognition_score': recognition_score,
            'added_at': datetime.now().isoformat(),
            'status': 'pending'
        }

        # Check if already in queue
        safe_name = self._safe_folder_name(person_name)
        queue_file = os.path.join(self.queue_path, f'{safe_name}.json')

        if os.path.exists(queue_file):
            # Update existing entry if new priority is higher
            with open(queue_file, 'r') as f:
                existing = json.load(f)

            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            if priority_order.get(priority, 1) < priority_order.get(existing.get('priority'), 1):
                existing['priority'] = priority
                existing['updated_at'] = datetime.now().isoformat()
                with open(queue_file, 'w') as f:
                    json.dump(existing, f, indent=2)
                return existing

            return existing

        # Save new queue entry
        with open(queue_file, 'w') as f:
            json.dump(queue_entry, f, indent=2)

        logger.info(f"Added {person_name} to training queue with priority: {priority}")
        return queue_entry

    def remove_from_queue(self, person_name: str) -> Dict:
        """
        Remove a person from the training queue.

        Args:
            person_name: Full name of person to remove

        Returns:
            Dict with removal status
        """
        safe_name = self._safe_folder_name(person_name)
        queue_file = os.path.join(self.queue_path, f'{safe_name}.json')

        if not os.path.exists(queue_file):
            return {
                'success': False,
                'error': f'{person_name} not found in queue'
            }

        try:
            os.remove(queue_file)
            logger.info(f"Removed {person_name} from training queue")
            return {
                'success': True,
                'message': f'Removed {person_name} from queue'
            }
        except Exception as e:
            logger.error(f"Error removing {person_name} from queue: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def update_queue_priority(self, person_name: str, priority: str, move_to_top: bool = False) -> Dict:
        """
        Update a person's priority in the training queue.

        Args:
            person_name: Full name of person
            priority: New priority (high, medium, low)
            move_to_top: If True, set order=0 to move to absolute top of queue

        Returns:
            Dict with update status
        """
        if priority not in ['high', 'medium', 'low']:
            return {
                'success': False,
                'error': f'Invalid priority: {priority}. Must be high, medium, or low'
            }

        safe_name = self._safe_folder_name(person_name)
        queue_file = os.path.join(self.queue_path, f'{safe_name}.json')

        if not os.path.exists(queue_file):
            return {
                'success': False,
                'error': f'{person_name} not found in queue'
            }

        try:
            with open(queue_file, 'r') as f:
                entry = json.load(f)

            old_priority = entry.get('priority')
            entry['priority'] = priority
            entry['updated_at'] = datetime.now().isoformat()

            # If moving to top, set order to 0 (highest priority)
            if move_to_top:
                entry['order'] = 0

            with open(queue_file, 'w') as f:
                json.dump(entry, f, indent=2)

            logger.info(f"Updated {person_name} priority from {old_priority} to {priority}" +
                       (" (moved to top)" if move_to_top else ""))
            return {
                'success': True,
                'message': f'Updated {person_name} priority to {priority}' +
                          (' and moved to top' if move_to_top else ''),
                'entry': entry
            }
        except Exception as e:
            logger.error(f"Error updating priority for {person_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_training_queue(self) -> List[Dict]:
        """Get current training queue sorted by priority and manual order."""
        queue = []

        for filename in os.listdir(self.queue_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.queue_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        entry = json.load(f)
                    if entry.get('status') == 'pending':
                        queue.append(entry)
                except Exception as e:
                    logger.warning(f"Error reading queue file {filename}: {str(e)}")

        # Sort by priority, manual order (if set), then added_at
        # Lower 'order' values = higher priority within same priority tier
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        queue.sort(key=lambda x: (
            priority_order.get(x.get('priority'), 1),
            x.get('order', 999999),  # Default to high number if not set
            x.get('added_at', '')
        ))

        return queue

    def benchmark_and_queue_person(
        self,
        person_name: str,
        wikidata_id: Optional[str] = None,
        num_test_images: int = 20
    ) -> Dict:
        """
        Benchmark a person and add to queue if needed.

        Args:
            person_name: Person to benchmark
            wikidata_id: Optional Wikidata ID
            num_test_images: Number of test images

        Returns:
            Dict with benchmark result and queue status
        """
        from app.services.recognition_benchmark_service import RecognitionBenchmarkService

        benchmark_service = RecognitionBenchmarkService(self.domain)
        result = benchmark_service.benchmark_person(person_name, num_test_images)

        response = {
            'person_name': person_name,
            'benchmark_result': result,
            'queued': False,
            'queue_entry': None
        }

        if result.get('needs_training'):
            queue_entry = self.add_to_training_queue(
                person_name=person_name,
                wikidata_id=wikidata_id,
                priority=result.get('training_priority', 'medium'),
                source='benchmark',
                recognition_score=result.get('recognition_score')
            )
            response['queued'] = True
            response['queue_entry'] = queue_entry

        return response

    def _discover_new_celebrities(self, run_id: str, max_discoveries: int) -> Dict:
        """Discover new celebrities to potentially train."""
        from app.services.celebrity_discovery_service import CelebrityDiscoveryService

        discovery_result = {
            'discovered': 0,
            'queued': 0,
            'skipped': 0,
            'people': []
        }

        try:
            discovery_service = CelebrityDiscoveryService()

            # Get trending/hot celebrities
            celebrities = discovery_service.discover_trending_celebrities(
                country=self.domain,
                max_results=max_discoveries
            )

            discovery_result['discovered'] = len(celebrities)

            for celeb in celebrities:
                person_name = celeb.get('name')
                wikidata_id = celeb.get('wikidata_id')

                # Check if already trained recently
                if self._was_recently_trained(person_name):
                    discovery_result['skipped'] += 1
                    continue

                # Benchmark the person
                benchmark_result = self.benchmark_and_queue_person(
                    person_name=person_name,
                    wikidata_id=wikidata_id,
                    num_test_images=10  # Fewer images for initial discovery benchmark
                )

                # Safely extract values with null checks
                br = benchmark_result.get('benchmark_result') or {}
                qe = benchmark_result.get('queue_entry') or {}

                discovery_result['people'].append({
                    'name': person_name,
                    'recognition_score': br.get('recognition_score'),
                    'queued': benchmark_result.get('queued', False),
                    'priority': qe.get('priority')
                })

                if benchmark_result.get('queued'):
                    discovery_result['queued'] += 1

        except Exception as e:
            logger.error(f"[SmartTraining {run_id}] Discovery error: {str(e)}")
            discovery_result['error'] = str(e)

        return discovery_result

    def _benchmark_existing_people(self, run_id: str) -> Dict:
        """Benchmark existing trained people to check for drift."""
        benchmark_result = {
            'benchmarked': 0,
            'degraded': 0,
            'queued_for_retraining': 0,
            'people': []
        }

        try:
            # Get list of trained people from production folder
            prod_path = os.path.join(self.STORAGE_BASE, 'recognized_faces_prod', self.domain)

            if not os.path.exists(prod_path):
                return benchmark_result

            # Get unique person names from trained images
            trained_people = set()
            for filename in os.listdir(prod_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    # Extract name from filename (before date)
                    parts = filename.split('_')
                    name_parts = []
                    for part in parts:
                        if len(part) >= 8 and part[:4].isdigit():
                            break
                        name_parts.append(part)
                    if name_parts:
                        trained_people.add('_'.join(name_parts))

            logger.info(f"[SmartTraining {run_id}] Found {len(trained_people)} trained people to benchmark")

            # Sample a subset for benchmarking (don't do all every night)
            import random
            sample_size = min(100, len(trained_people))
            sample = random.sample(list(trained_people), sample_size)

            for person_name in sample:
                # Skip if recently benchmarked
                if self._was_recently_benchmarked(person_name):
                    continue

                result = self.benchmark_and_queue_person(
                    person_name=person_name.replace('_', ' '),
                    num_test_images=10
                )

                benchmark_result['benchmarked'] += 1
                benchmark_result['people'].append({
                    'name': person_name,
                    'recognition_score': result['benchmark_result'].get('recognition_score'),
                    'queued': result['queued']
                })

                if result['queued']:
                    benchmark_result['queued_for_retraining'] += 1

                # Check for degradation (score dropped)
                previous_score = self._get_previous_benchmark_score(person_name)
                if previous_score and result['benchmark_result'].get('recognition_score', 100) < previous_score - 10:
                    benchmark_result['degraded'] += 1

        except Exception as e:
            logger.error(f"[SmartTraining {run_id}] Benchmark error: {str(e)}")
            benchmark_result['error'] = str(e)

        return benchmark_result

    def _build_training_queue(self) -> List[Dict]:
        """Build prioritized training queue."""
        queue = self.get_training_queue()

        # Filter out people who were trained too recently
        filtered_queue = []
        for entry in queue:
            if not self._was_recently_trained(entry['person_name']):
                filtered_queue.append(entry)

        return filtered_queue

    def _execute_training_queue(
        self,
        run_id: str,
        queue: List[Dict],
        max_people: int,
        images_per_person: int,
        app_context=None
    ) -> Dict:
        """Execute training for people in the queue."""
        import gc
        from app.services.automated_training_service import AutomatedTrainingService

        training_result = {
            'attempted': 0,
            'successful': 0,
            'failed': 0,
            'people': []
        }

        training_service = AutomatedTrainingService(domain=self.domain)

        for entry in queue[:max_people]:
            person_name = entry['person_name']
            wikidata_id = entry.get('wikidata_id')

            logger.info(f"[SmartTraining {run_id}] Training: {person_name}")

            try:
                # Mark as in progress
                self._update_queue_status(person_name, 'in_progress')

                # Execute training
                # We need to use the existing training service
                # For now, we'll create a mini-batch for this person
                result = self._train_single_person(
                    person_name=person_name,
                    wikidata_id=wikidata_id,
                    images_per_person=images_per_person,
                    run_id=run_id
                )

                training_result['attempted'] += 1

                if result.get('success'):
                    training_result['successful'] += 1
                    self._update_queue_status(person_name, 'completed')
                    self._record_training_history(person_name, result)
                else:
                    training_result['failed'] += 1
                    self._update_queue_status(person_name, 'failed')

                training_result['people'].append({
                    'name': person_name,
                    'success': result.get('success', False),
                    'images_accepted': result.get('images_accepted', 0),
                    'error': result.get('error')
                })

            except Exception as e:
                logger.error(f"[SmartTraining {run_id}] Training failed for {person_name}: {str(e)}")
                training_result['failed'] += 1
                self._update_queue_status(person_name, 'failed')
                training_result['people'].append({
                    'name': person_name,
                    'success': False,
                    'error': str(e)
                })

            # Force garbage collection after each person to prevent OOM
            gc.collect()
            logger.info(f"[SmartTraining {run_id}] Memory cleanup after {person_name}")

        return training_result

    def _train_single_person(
        self,
        person_name: str,
        wikidata_id: Optional[str],
        images_per_person: int,
        run_id: str,
        search_name: Optional[str] = None
    ) -> Dict:
        """
        Train a single person using SERP images.

        This is a simplified training flow that:
        1. Gets P18 reference from Wikidata (if wikidata_id available)
        2. Downloads SERP images (using international name for better results)
        3. Validates against reference
        4. Adds valid faces to training set

        Args:
            person_name: Name as stored in database (may be Serbian transliteration)
            wikidata_id: Wikidata entity ID (optional)
            images_per_person: Target number of training images
            run_id: Training run identifier
            search_name: International name for SERP search (if different from person_name)
        """
        from app.services.automated_training_service import AutomatedTrainingService
        from app.services.wikidata_service import WikidataService

        result = {
            'success': False,
            'person_name': person_name,
            'search_name': search_name,
            'images_found': 0,
            'images_accepted': 0,
            'error': None
        }

        try:
            service = AutomatedTrainingService(domain=self.domain)

            # Get international name for SERP search if not provided
            if not search_name:
                try:
                    from app.services.transliteration_service import get_transliteration_service
                    trans_service = get_transliteration_service()
                    search_name, was_changed = trans_service.transliterate_single(person_name)
                    if was_changed:
                        logger.info(f"Transliterated for search: {person_name} -> {search_name}")
                except Exception as e:
                    logger.warning(f"Transliteration failed for {person_name}: {e}")
                    search_name = person_name

            # Get P18 URL(s) from Wikidata - supports multiple references
            p18_url = None
            p18_urls = []
            if wikidata_id:
                try:
                    p18_url = WikidataService.get_p18_image_url(wikidata_id)
                except:
                    pass

            if not p18_url:
                # Try to search by international name first, then original name
                for name_to_search in [search_name, person_name]:
                    try:
                        search_results = WikidataService.search_person(name_to_search, limit=1)
                        if search_results:
                            wikidata_id = search_results[0].get('wikidata_id')
                            p18_url = search_results[0].get('image_url')
                            p18_urls = search_results[0].get('image_urls', [])  # Get all P18 references
                            break
                    except:
                        pass

            # Note: p18_url may be None - that's OK, _process_person will try SERP consensus first
            # Only log if we found P18 for fallback
            if p18_url:
                logger.info(f"[SmartTraining {run_id}] {person_name}: Found {len(p18_urls) if p18_urls else 1} P18 reference(s)")
            else:
                logger.info(f"[SmartTraining {run_id}] {person_name}: No P18 found, will try SERP consensus only")

            # Use the existing _process_person method with search_name for SERP
            # _process_person tries SERP consensus first, then falls back to P18
            batch_id = f"smart_{run_id}"

            process_result = service._process_person(
                batch_id=batch_id,
                person_name=person_name,
                wikidata_id=wikidata_id or '',
                p18_url=p18_url,
                images_per_person=images_per_person,
                search_name=search_name,  # Pass international name for SERP search
                p18_urls=p18_urls if p18_urls else None  # Pass all P18 references
            )

            result['images_found'] = process_result.get('images_found', 0)
            result['images_accepted'] = process_result.get('images_accepted', 0)
            result['gallery_url'] = process_result.get('gallery_url')

            # Mark as success if we have at least 7 images
            # Lowered from 10 to improve success rate for harder cases
            min_images_for_training = 7
            result['success'] = result['images_accepted'] >= min_images_for_training

            logger.info(f"[SmartTraining {run_id}] {person_name}: Found {result['images_found']} images, "
                       f"Accepted {result['images_accepted']}/{images_per_person}, "
                       f"Success: {result['success']} (min required: {min_images_for_training})")

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Training error for {person_name}: {str(e)}")

        return result

    def _was_recently_trained(self, person_name: str) -> bool:
        """Check if person was trained within MIN_RETRAIN_DAYS."""
        safe_name = self._safe_folder_name(person_name)
        history_file = os.path.join(self.history_path, f'{safe_name}.json')

        if not os.path.exists(history_file):
            return False

        try:
            with open(history_file, 'r') as f:
                history = json.load(f)

            last_trained = history.get('last_trained')
            if last_trained:
                last_date = datetime.fromisoformat(last_trained)
                days_ago = (datetime.now() - last_date).days
                return days_ago < self.MIN_RETRAIN_DAYS

        except Exception as e:
            logger.warning(f"Error checking training history: {str(e)}")

        return False

    def _was_recently_benchmarked(self, person_name: str) -> bool:
        """Check if person was benchmarked recently (within 1 day)."""
        from app.services.recognition_benchmark_service import RecognitionBenchmarkService

        benchmark_service = RecognitionBenchmarkService(self.domain)
        summary_path = os.path.join(benchmark_service.benchmark_path, 'summary.json')

        if not os.path.exists(summary_path):
            return False

        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            person_data = summary.get('people', {}).get(person_name)
            if person_data:
                last_benchmark = person_data.get('last_benchmark')
                if last_benchmark:
                    last_date = datetime.fromisoformat(last_benchmark)
                    hours_ago = (datetime.now() - last_date).total_seconds() / 3600
                    return hours_ago < 24

        except:
            pass

        return False

    def _get_previous_benchmark_score(self, person_name: str) -> Optional[float]:
        """Get the previous benchmark score for comparison."""
        from app.services.recognition_benchmark_service import RecognitionBenchmarkService

        benchmark_service = RecognitionBenchmarkService(self.domain)
        summary_path = os.path.join(benchmark_service.benchmark_path, 'summary.json')

        try:
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                return summary.get('people', {}).get(person_name, {}).get('recognition_score')
        except:
            pass

        return None

    def _update_queue_status(self, person_name: str, status: str):
        """Update queue entry status."""
        safe_name = self._safe_folder_name(person_name)
        queue_file = os.path.join(self.queue_path, f'{safe_name}.json')

        try:
            if os.path.exists(queue_file):
                with open(queue_file, 'r') as f:
                    entry = json.load(f)

                entry['status'] = status
                entry['updated_at'] = datetime.now().isoformat()

                with open(queue_file, 'w') as f:
                    json.dump(entry, f, indent=2)

                # Remove from queue if completed
                if status in ['completed', 'failed']:
                    os.remove(queue_file)

        except Exception as e:
            logger.error(f"Error updating queue status: {str(e)}")

    def _record_training_history(self, person_name: str, result: Dict):
        """Record training in history."""
        safe_name = self._safe_folder_name(person_name)
        history_file = os.path.join(self.history_path, f'{safe_name}.json')

        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = {'person_name': person_name, 'training_runs': []}

            history['last_trained'] = datetime.now().isoformat()
            history['training_runs'].append({
                'timestamp': datetime.now().isoformat(),
                'images_accepted': result.get('images_accepted', 0),
                'success': result.get('success', False)
            })

            # Keep only last 10 runs
            history['training_runs'] = history['training_runs'][-10:]

            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Error recording training history: {str(e)}")

    def _save_run_result(self, result: Dict):
        """Save smart training run result."""
        try:
            run_file = os.path.join(self.runs_path, f"run_{result['run_id']}.json")
            with open(run_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving run result: {str(e)}")

    def _safe_folder_name(self, name: str) -> str:
        """Convert name to safe folder name with proper Serbian transliteration."""
        import unicodedata
        import re

        # Only đ/Đ needs manual transliteration - other Serbian chars decompose with NFKD
        transliteration_map = {
            'đ': 'dj', 'Đ': 'Dj',
            'ß': 'ss',
            'ø': 'o', 'Ø': 'O',
            'æ': 'ae', 'Æ': 'AE',
            'œ': 'oe', 'Œ': 'OE',
        }

        result = name
        for char, replacement in transliteration_map.items():
            result = result.replace(char, replacement)

        normalized = unicodedata.normalize('NFKD', result)
        ascii_name = ''.join([c for c in normalized if not unicodedata.combining(c)])
        ascii_name = ascii_name.encode('ascii', 'ignore').decode('ascii')
        safe = re.sub(r'[^\w_-]', '', ascii_name.replace(' ', '_'))

        return safe.lower()
