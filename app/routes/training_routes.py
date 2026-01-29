"""
Training Routes - API endpoints for automated face training.

Provides endpoints to:
- Start a new training batch (from Wikidata celebrities)
- Check batch status
- List all batches
- Get available countries/occupations
- Smart training (self-improving recognition)
- Benchmark recognition quality
- Discover trending celebrities

All endpoints require authentication via Authorization header.
"""

from flask import Blueprint, request, jsonify, current_app
import logging
from threading import Thread

from app.services.automated_training_service import AutomatedTrainingService
from app.services.wikidata_service import WikidataService
from app.services.smart_training_service import SmartTrainingService
from app.services.recognition_benchmark_service import RecognitionBenchmarkService
from app.services.celebrity_discovery_service import CelebrityDiscoveryService
from app.services.merge_candidates_service import MergeCandidatesService
from app.decorators.auth import require_auth

logger = logging.getLogger(__name__)

training_bp = Blueprint('training', __name__, url_prefix='/api/training')


@training_bp.route('/countries', methods=['GET'])
@require_auth
def get_countries():
    """Get list of available countries for training."""
    try:
        countries = WikidataService.get_available_countries()
        return jsonify({
            'success': True,
            'countries': countries
        })
    except Exception as e:
        logger.error(f"Error getting countries: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/occupations', methods=['GET'])
@require_auth
def get_occupations():
    """Get list of available occupations for training."""
    try:
        occupations = WikidataService.get_available_occupations()
        return jsonify({
            'success': True,
            'occupations': occupations
        })
    except Exception as e:
        logger.error(f"Error getting occupations: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/start', methods=['POST'])
@require_auth
def start_training():
    """
    Start a new training batch.

    Request body:
    {
        "country": "serbia",
        "occupation": "actor",
        "domain": "serbia",  # optional, defaults to country
        "limit": 10,         # optional, max celebrities
        "images_per_person": 50  # optional, target images per person
    }
    """
    try:
        data = request.get_json() or {}

        country = data.get('country')
        occupation = data.get('occupation')
        domain = data.get('domain', country)
        limit = data.get('limit', 10)
        images_per_person = data.get('images_per_person', 50)

        if not country:
            return jsonify({'success': False, 'error': 'Country is required'}), 400

        if not occupation:
            return jsonify({'success': False, 'error': 'Occupation is required'}), 400

        # Validate country
        if country.lower() not in WikidataService.COUNTRIES:
            return jsonify({
                'success': False,
                'error': f'Unknown country: {country}',
                'available': list(WikidataService.COUNTRIES.keys())
            }), 400

        # Validate occupation
        if occupation.lower() not in WikidataService.OCCUPATIONS:
            return jsonify({
                'success': False,
                'error': f'Unknown occupation: {occupation}',
                'available': list(WikidataService.OCCUPATIONS.keys())
            }), 400

        # Start training
        service = AutomatedTrainingService(domain=domain)
        result = service.start_training_batch(
            country=country,
            occupation=occupation,
            limit=limit,
            images_per_person=images_per_person
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/batch/<batch_id>/status', methods=['GET'])
@require_auth
def get_batch_status(batch_id):
    """Get status of a training batch."""
    try:
        # Try to determine domain from batch status
        # Default to serbia if not specified
        domain = request.args.get('domain', 'serbia')

        service = AutomatedTrainingService(domain=domain)
        status = service.get_batch_status(batch_id)

        if not status:
            return jsonify({
                'success': False,
                'error': 'Batch not found'
            }), 404

        return jsonify({
            'success': True,
            'batch': status
        })

    except Exception as e:
        logger.error(f"Error getting batch status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/batches', methods=['GET'])
@require_auth
def list_batches():
    """List all training batches."""
    try:
        domain = request.args.get('domain', 'serbia')

        service = AutomatedTrainingService(domain=domain)
        batches = service.list_batches()

        return jsonify({
            'success': True,
            'batches': batches
        })

    except Exception as e:
        logger.error(f"Error listing batches: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/queue-status', methods=['GET'])
@require_auth
def get_queue_status():
    """Get queue statistics for the dashboard."""
    try:
        domain = request.args.get('domain', 'serbia')

        # Get batch statistics
        service = AutomatedTrainingService(domain=domain)
        batches = service.list_batches()

        # Calculate stats from batches
        total_in_queue = 0
        processed_today = 0
        failed_today = 0
        remaining = 0

        from datetime import datetime, timedelta
        today = datetime.now().date()

        for batch in batches:
            status = batch.get('status', 'unknown')
            created = batch.get('created_at', '')

            # Check if batch was created today
            try:
                batch_date = datetime.fromisoformat(created.replace('Z', '+00:00')).date()
                is_today = batch_date == today
            except:
                is_today = False

            if status == 'pending':
                count = batch.get('total_people', 0)
                total_in_queue += count
                remaining += count
            elif status == 'completed' and is_today:
                processed_today += batch.get('successful', 0)
            elif status == 'failed' and is_today:
                failed_today += batch.get('total_people', 0)

        return jsonify({
            'success': True,
            'data': {
                'queue': {
                    'total_in_queue': total_in_queue,
                    'processed_today': processed_today,
                    'remaining': remaining,
                    'failed_today': failed_today
                }
            }
        })

    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/queue-list', methods=['GET'])
@require_auth
def get_queue_list():
    """Get list of people in the training queue."""
    try:
        domain = request.args.get('domain', 'serbia')

        # Get batches with pending status
        service = AutomatedTrainingService(domain=domain)
        batches = service.list_batches()

        # Extract people from pending batches
        queue_items = []
        for batch in batches:
            if batch.get('status') == 'pending':
                people = batch.get('people', [])
                for person in people:
                    queue_items.append({
                        'name': person.get('name', person.get('label', 'Unknown')),
                        'batch_id': batch.get('batch_id'),
                        'created_at': batch.get('created_at'),
                        'status': 'pending'
                    })

        return jsonify({
            'success': True,
            'data': {
                'queue': queue_items
            }
        })

    except Exception as e:
        logger.error(f"Error getting queue list: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/queue-failed', methods=['GET'])
@require_auth
def get_failed_queue_entries():
    """Get list of failed queue entries from individual JSON files."""
    try:
        import json
        from pathlib import Path

        domain = request.args.get('domain', 'serbia')
        queue_dir = Path(f'storage/training_queue/{domain}')

        if not queue_dir.exists():
            return jsonify({
                'success': True,
                'data': {'failed': [], 'total': 0}
            })

        failed_entries = []
        for f in queue_dir.glob('*.json'):
            try:
                with open(f) as file:
                    data = json.load(file)
                    if data.get('status') == 'failed':
                        failed_entries.append({
                            'person_name': data.get('person_name'),
                            'error': data.get('error'),
                            'processed_at': data.get('processed_at'),
                            'priority': data.get('priority'),
                            'source': data.get('source'),
                            'file_name': f.name
                        })
            except Exception as e:
                logger.warning(f"Error reading queue file {f}: {e}")

        # Sort by processed_at descending (most recent first)
        failed_entries.sort(key=lambda x: x.get('processed_at', ''), reverse=True)

        return jsonify({
            'success': True,
            'data': {
                'failed': failed_entries,
                'total': len(failed_entries)
            }
        })

    except Exception as e:
        logger.error(f"Error getting failed queue entries: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/queue-retry', methods=['POST'])
@require_auth
def retry_failed_queue_entry():
    """Reset a failed queue entry back to pending status."""
    try:
        import json
        from pathlib import Path

        data = request.get_json() or {}
        domain = data.get('domain', 'serbia')
        file_name = data.get('file_name')

        if not file_name:
            return jsonify({'success': False, 'error': 'file_name is required'}), 400

        queue_file = Path(f'storage/training_queue/{domain}/{file_name}')

        if not queue_file.exists():
            return jsonify({'success': False, 'error': 'Queue entry not found'}), 404

        with open(queue_file, 'r') as f:
            entry_data = json.load(f)

        # Reset status to pending
        entry_data['status'] = 'pending'
        entry_data.pop('error', None)
        entry_data.pop('processed_at', None)

        with open(queue_file, 'w') as f:
            json.dump(entry_data, f, indent=2)

        return jsonify({
            'success': True,
            'message': f"Queue entry {entry_data.get('person_name')} reset to pending"
        })

    except Exception as e:
        logger.error(f"Error retrying failed queue entry: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/preview', methods=['POST'])
@require_auth
def preview_celebrities():
    """
    Preview celebrities that would be trained (without starting training).

    Request body:
    {
        "country": "serbia",
        "occupation": "actor",
        "limit": 10
    }
    """
    try:
        data = request.get_json() or {}

        country = data.get('country')
        occupation = data.get('occupation')
        limit = data.get('limit', 10)

        if not country or not occupation:
            return jsonify({
                'success': False,
                'error': 'Country and occupation are required'
            }), 400

        # Query Wikidata
        celebrities = WikidataService.query_celebrities(country, occupation, limit)

        return jsonify({
            'success': True,
            'count': len(celebrities),
            'celebrities': celebrities
        })

    except Exception as e:
        logger.error(f"Error previewing celebrities: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/generate-candidates', methods=['POST'])
@require_auth
def generate_candidates():
    """
    Generate candidate celebrities for training (alias for preview).
    Used by frontend UI.

    Request body:
    {
        "country": "serbia",
        "occupation": "actor",
        "limit": 10
    }
    """
    try:
        data = request.get_json() or {}

        country = data.get('country')
        occupation = data.get('occupation')
        limit = data.get('limit', 10)

        if not country or not occupation:
            return jsonify({
                'success': False,
                'error': 'Country and occupation are required'
            }), 400

        # Query Wikidata
        celebrities = WikidataService.query_celebrities(country, occupation, limit)

        return jsonify({
            'success': True,
            'candidates': celebrities,  # Frontend expects 'candidates' key
            'count': len(celebrities)
        })

    except Exception as e:
        logger.error(f"Error generating candidates: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/search', methods=['GET'])
@require_auth
def search_person():
    """
    Search for a specific person by name.

    Query params:
    - q: Search query
    - limit: Max results (default 20)
    """
    try:
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 20))

        if not query or len(query) < 2:
            return jsonify({
                'success': False,
                'error': 'Query must be at least 2 characters'
            }), 400

        results = WikidataService.search_person(query, limit)

        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })

    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# SMART TRAINING ENDPOINTS - Self-improving recognition system
# ============================================================

@training_bp.route('/smart/run', methods=['POST'])
@require_auth
def run_smart_training():
    """
    Run a smart training cycle.

    This is the main endpoint for the self-improving system.
    It can be triggered manually or via cron/scheduler.

    Request body:
    {
        "domain": "serbia",            # Target domain
        "discover_new": true,          # Discover new trending celebrities
        "benchmark_existing": true,    # Re-test existing trained people
        "max_new_discoveries": 10,     # Max new people to discover
        "max_training_per_run": 5,     # Max people to train this run
        "images_per_person": 20        # Target images when training
    }

    Returns:
        Run ID and initial status. Use /smart/run/<run_id>/status for progress.
    """
    try:
        data = request.get_json() or {}

        domain = data.get('domain', 'serbia')
        discover_new = data.get('discover_new', True)
        benchmark_existing = data.get('benchmark_existing', True)
        max_new_discoveries = data.get('max_new_discoveries', 10)
        max_training_per_run = data.get('max_training_per_run', 5)
        images_per_person = data.get('images_per_person', 20)

        service = SmartTrainingService(domain=domain)

        # Get app context for background thread
        app_context = current_app._get_current_object().app_context()

        # Run in background
        def run_in_background():
            with app_context:
                service.run_smart_training_cycle(
                    discover_new=discover_new,
                    benchmark_existing=benchmark_existing,
                    max_new_discoveries=max_new_discoveries,
                    max_training_per_run=max_training_per_run,
                    min_images_per_person=images_per_person,
                    app_context=app_context
                )

        thread = Thread(target=run_in_background)
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Smart training cycle started',
            'domain': domain,
            'config': {
                'discover_new': discover_new,
                'benchmark_existing': benchmark_existing,
                'max_new_discoveries': max_new_discoveries,
                'max_training_per_run': max_training_per_run,
                'images_per_person': images_per_person
            }
        })

    except Exception as e:
        logger.error(f"Error starting smart training: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/smart/queue', methods=['GET'])
@require_auth
def get_training_queue():
    """
    Get current training queue.

    Shows people waiting to be trained, sorted by priority.

    Query params:
    - domain: Domain code (default 'serbia')
    """
    try:
        domain = request.args.get('domain', 'serbia')

        service = SmartTrainingService(domain=domain)
        queue = service.get_training_queue()

        return jsonify({
            'success': True,
            'domain': domain,
            'queue_size': len(queue),
            'queue': queue
        })

    except Exception as e:
        logger.error(f"Error getting training queue: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/smart/runs', methods=['GET'])
@require_auth
def get_smart_training_runs():
    """
    Get history of smart training runs.

    Query params:
    - domain: Domain code (default 'serbia')
    - limit: Maximum number of runs to return (default 50)
    """
    try:
        domain = request.args.get('domain', 'serbia')
        limit = int(request.args.get('limit', 50))

        service = SmartTrainingService(domain=domain)

        # Read run files from directory
        import os
        import json
        runs_path = os.path.join(service.runs_path)

        runs = []
        if os.path.exists(runs_path):
            files = sorted(os.listdir(runs_path), reverse=True)[:limit]
            for filename in files:
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(runs_path, filename), 'r') as f:
                            run_data = json.load(f)
                            runs.append(run_data)
                    except Exception as e:
                        logger.warning(f"Error reading run file {filename}: {str(e)}")
                        continue

        return jsonify({
            'success': True,
            'domain': domain,
            'count': len(runs),
            'runs': runs
        })

    except Exception as e:
        logger.error(f"Error getting training runs: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/smart/queue/add', methods=['POST'])
@require_auth
def add_to_queue():
    """
    Manually add a person to the training queue.

    Request body:
    {
        "person_name": "John Smith",
        "domain": "serbia",
        "wikidata_id": "Q12345",    # Optional
        "priority": "medium",       # high, medium, low
        "recognition_score": 45.5   # Optional, current score
    }
    """
    try:
        data = request.get_json() or {}

        person_name = data.get('person_name')
        if not person_name:
            return jsonify({
                'success': False,
                'error': 'person_name is required'
            }), 400

        domain = data.get('domain', 'serbia')
        wikidata_id = data.get('wikidata_id')
        priority = data.get('priority', 'medium')
        recognition_score = data.get('recognition_score')

        service = SmartTrainingService(domain=domain)
        entry = service.add_to_training_queue(
            person_name=person_name,
            wikidata_id=wikidata_id,
            priority=priority,
            source='manual',
            recognition_score=recognition_score
        )

        return jsonify({
            'success': True,
            'message': f'Added {person_name} to queue',
            'entry': entry
        })

    except Exception as e:
        logger.error(f"Error adding to queue: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/smart/queue/remove', methods=['POST'])
@require_auth
def remove_from_queue():
    """
    Remove a person from the training queue.

    Request body:
    {
        "person_name": "John Smith",
        "domain": "serbia"
    }
    """
    try:
        data = request.get_json() or {}

        person_name = data.get('person_name')
        if not person_name:
            return jsonify({
                'success': False,
                'error': 'person_name is required'
            }), 400

        domain = data.get('domain', 'serbia')

        service = SmartTrainingService(domain=domain)
        result = service.remove_from_queue(person_name)

        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 404

    except Exception as e:
        logger.error(f"Error removing from queue: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/smart/queue/priority', methods=['POST'])
@require_auth
def update_queue_priority():
    """
    Update a person's priority in the training queue.

    Request body:
    {
        "person_name": "John Smith",
        "domain": "serbia",
        "priority": "high",        # high, medium, or low
        "move_to_top": true        # Optional: move to absolute top of queue
    }
    """
    try:
        data = request.get_json() or {}

        person_name = data.get('person_name')
        if not person_name:
            return jsonify({
                'success': False,
                'error': 'person_name is required'
            }), 400

        priority = data.get('priority')
        if not priority:
            return jsonify({
                'success': False,
                'error': 'priority is required'
            }), 400

        domain = data.get('domain', 'serbia')
        move_to_top = data.get('move_to_top', False)

        service = SmartTrainingService(domain=domain)
        result = service.update_queue_priority(person_name, priority, move_to_top)

        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"Error updating queue priority: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# BENCHMARK ENDPOINTS - Test recognition quality
# ============================================================

@training_bp.route('/benchmark/person', methods=['POST'])
@require_auth
def benchmark_person():
    """
    Benchmark recognition for a specific person.

    Downloads test images and checks recognition accuracy.

    Request body:
    {
        "person_name": "John Smith",
        "domain": "serbia",
        "num_images": 20          # Test images to download
    }
    """
    try:
        data = request.get_json() or {}

        person_name = data.get('person_name')
        if not person_name:
            return jsonify({
                'success': False,
                'error': 'person_name is required'
            }), 400

        domain = data.get('domain', 'serbia')
        num_images = data.get('num_images', 20)

        service = RecognitionBenchmarkService(domain=domain)
        result = service.benchmark_person(
            person_name=person_name,
            num_images=num_images
        )

        return jsonify({
            'success': True,
            'benchmark': result
        })

    except Exception as e:
        logger.error(f"Error benchmarking person: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/benchmark/batch', methods=['POST'])
@require_auth
def benchmark_batch():
    """
    Benchmark multiple people.

    Request body:
    {
        "people": ["John Smith", "Jane Doe"],
        "domain": "serbia",
        "num_images_per_person": 10
    }
    """
    try:
        data = request.get_json() or {}

        people = data.get('people', [])
        if not people:
            return jsonify({
                'success': False,
                'error': 'people list is required'
            }), 400

        domain = data.get('domain', 'serbia')
        num_images = data.get('num_images_per_person', 10)

        service = RecognitionBenchmarkService(domain=domain)
        results = service.batch_benchmark(
            people=people,
            num_images_per_person=num_images
        )

        return jsonify({
            'success': True,
            'benchmarks': results
        })

    except Exception as e:
        logger.error(f"Error in batch benchmark: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/benchmark/and-queue', methods=['POST'])
@require_auth
def benchmark_and_queue():
    """
    Benchmark a person and add to training queue if needed.

    Combines benchmarking with automatic queue addition.

    Request body:
    {
        "person_name": "John Smith",
        "domain": "serbia",
        "wikidata_id": "Q12345",    # Optional
        "num_images": 20
    }
    """
    try:
        data = request.get_json() or {}

        person_name = data.get('person_name')
        if not person_name:
            return jsonify({
                'success': False,
                'error': 'person_name is required'
            }), 400

        domain = data.get('domain', 'serbia')
        wikidata_id = data.get('wikidata_id')
        num_images = data.get('num_images', 20)

        service = SmartTrainingService(domain=domain)
        result = service.benchmark_and_queue_person(
            person_name=person_name,
            wikidata_id=wikidata_id,
            num_test_images=num_images
        )

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        logger.error(f"Error in benchmark and queue: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/benchmark/candidates', methods=['GET'])
@require_auth
def get_training_candidates():
    """
    Get list of people who need training based on benchmarks.

    Query params:
    - domain: Domain code
    - min_score: Minimum acceptable score (default 80)
    """
    try:
        domain = request.args.get('domain', 'serbia')
        min_score = float(request.args.get('min_score', 80))

        service = RecognitionBenchmarkService(domain=domain)
        candidates = service.get_training_candidates(min_score=min_score)

        return jsonify({
            'success': True,
            'domain': domain,
            'min_score': min_score,
            'candidates_count': len(candidates),
            'candidates': candidates
        })

    except Exception as e:
        logger.error(f"Error getting training candidates: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# DISCOVERY ENDPOINTS - Find new celebrities
# ============================================================

@training_bp.route('/discover/trending', methods=['GET'])
@require_auth
def discover_trending():
    """
    Discover trending celebrities for a country.

    Query params:
    - country: Country code (default 'serbia')
    - max_results: Max celebrities to return (default 20)
    """
    try:
        country = request.args.get('country', 'serbia')
        max_results = int(request.args.get('max_results', 20))

        service = CelebrityDiscoveryService()
        celebrities = service.discover_trending_celebrities(
            country=country,
            max_results=max_results
        )

        return jsonify({
            'success': True,
            'country': country,
            'count': len(celebrities),
            'celebrities': celebrities
        })

    except Exception as e:
        logger.error(f"Error discovering celebrities: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/discover/top', methods=['GET'])
@require_auth
def get_top_celebrities():
    """
    Get top celebrities for a country (cached Wikidata query).

    Query params:
    - country: Country code
    - limit: Max results (default 50)
    """
    try:
        country = request.args.get('country', 'serbia')
        limit = int(request.args.get('limit', 50))

        service = CelebrityDiscoveryService()
        celebrities = service.get_country_top_celebrities(
            country=country,
            limit=limit
        )

        return jsonify({
            'success': True,
            'country': country,
            'count': len(celebrities),
            'celebrities': celebrities
        })

    except Exception as e:
        logger.error(f"Error getting top celebrities: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/discover/search', methods=['GET'])
@require_auth
def search_celebrity():
    """
    Search for a specific celebrity.

    Query params:
    - q: Search query (name)
    - country: Optional country filter
    """
    try:
        query = request.args.get('q', '')
        country = request.args.get('country')

        if not query or len(query) < 2:
            return jsonify({
                'success': False,
                'error': 'Query must be at least 2 characters'
            }), 400

        service = CelebrityDiscoveryService()
        results = service.search_celebrity(
            query=query,
            country=country
        )

        return jsonify({
            'success': True,
            'query': query,
            'count': len(results),
            'results': results
        })

    except Exception as e:
        logger.error(f"Error searching celebrity: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# REJECTED FACES ENDPOINTS - View and manage rejected faces
# ============================================================

@training_bp.route('/rejected-faces', methods=['GET'])
@require_auth
def get_rejected_faces():
    """
    Get summary of rejected faces for a date.

    Query params:
    - date: Date string (YYYY-MM-DD), defaults to today
    - person_name: Optional filter by person name
    """
    try:
        from app.services.image_rejection_logger import ImageRejectionLogger

        date = request.args.get('date')
        person_name = request.args.get('person_name')

        summary = ImageRejectionLogger.get_rejection_summary(date, person_name)

        return jsonify({
            'success': True,
            'summary': summary
        })

    except Exception as e:
        logger.error(f"Error getting rejected faces: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/rejected-faces/dates', methods=['GET'])
@require_auth
def get_rejection_dates():
    """
    Get list of dates that have rejection logs.
    """
    try:
        import os
        from app.services.image_rejection_logger import ImageRejectionLogger

        dates = []
        log_path = ImageRejectionLogger.REJECTION_LOG_PATH

        if os.path.exists(log_path):
            for folder_name in sorted(os.listdir(log_path), reverse=True):
                # Check if it's a valid date folder
                try:
                    from datetime import datetime
                    datetime.strptime(folder_name, '%Y-%m-%d')
                    dates.append(folder_name)
                except ValueError:
                    continue

        return jsonify({
            'success': True,
            'dates': dates[:30]  # Last 30 days max
        })

    except Exception as e:
        logger.error(f"Error getting rejection dates: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/rejected-faces/<date>/<person_name>/images', methods=['GET'])
@require_auth
def get_rejected_images_for_person(date, person_name):
    """
    Get rejected images for a specific person on a specific date.

    Returns base64-encoded images for display in the frontend.
    """
    try:
        import os
        import base64
        from app.services.image_rejection_logger import ImageRejectionLogger

        images = ImageRejectionLogger.get_rejected_images_for_person(person_name, date)

        result = []
        for image_path in images:
            if os.path.exists(image_path):
                try:
                    with open(image_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')

                    # Extract reason from filename (format: reason_originalfilename.jpg)
                    filename = os.path.basename(image_path)
                    reason = filename.split('_')[0] if '_' in filename else 'unknown'

                    result.append({
                        'filename': filename,
                        'reason': reason,
                        'image_base64': image_data,
                        'path': image_path
                    })
                except Exception as e:
                    logger.warning(f"Error reading image {image_path}: {e}")
                    continue

        return jsonify({
            'success': True,
            'date': date,
            'person_name': person_name,
            'count': len(result),
            'images': result
        })

    except Exception as e:
        logger.error(f"Error getting rejected images: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/rejected-faces/cleanup', methods=['POST'])
@require_auth
def cleanup_rejected_faces():
    """
    Clean up old rejected faces (older than retention period).

    Request body (optional):
    {
        "days": 7  # Number of days to retain (default: 7)
    }
    """
    try:
        from app.services.image_rejection_logger import ImageRejectionLogger

        data = request.get_json() or {}
        days = data.get('days')

        removed_count = ImageRejectionLogger.cleanup_old_rejections(days)

        return jsonify({
            'success': True,
            'message': f'Cleaned up {removed_count} old rejection folders',
            'removed_count': removed_count
        })

    except Exception as e:
        logger.error(f"Error cleaning up rejected faces: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/progress', methods=['GET'])
@require_auth
def get_training_progress():
    """
    Get list of persons/embeddings from the database with image counts.

    Query params:
    - domain: serbia|slovenia (required)
    - view: production|staging (used for hide_approved filtering)
    - page: page number (default: 1)
    - limit: items per page (default: 50)
    - search: optional search query
    - hide_approved: true|false - filter out approved persons (default: false)
    """
    from pathlib import Path
    import subprocess
    import json
    import os

    try:
        domain = request.args.get('domain', 'serbia')
        view = request.args.get('view', 'production')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))
        search = request.args.get('search', None)  # Optional search query
        hide_approved = request.args.get('hide_approved', 'false').lower() == 'true'

        # Get persons from database via subprocess (avoid TensorFlow/psycopg2 conflict)
        worker_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'scripts', 'pgvector_list_persons.py'
        )

        # Include fs_count when searching for specific people (few results = fast)
        # But not for bulk listing (500+ people = too slow)
        include_fs_count = bool(search)  # Only when filtering by search term
        request_data = {'domain': domain, 'include_fs_count': include_fs_count}
        if search:
            request_data['search'] = search

        # Pass environment variables to subprocess
        env = os.environ.copy()

        result = subprocess.run(
            ['/root/facerecognition-backend/venv/bin/python', worker_path],
            input=json.dumps(request_data),
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )

        if result.returncode != 0:
            logger.error(f"Worker failed: {result.stderr}")
            return jsonify({'success': False, 'error': 'Database query failed'}), 500

        worker_response = json.loads(result.stdout)

        if worker_response['status'] != 'success':
            logger.error(f"Worker error: {worker_response.get('error')}")
            return jsonify({'success': False, 'error': worker_response.get('error')}), 500

        all_persons = worker_response['persons']

        # Filter out approved persons if requested
        if hide_approved:
            approvals_file = Path('/root/facerecognition-backend/storage/approvals') / f'{domain}_{view}.json'
            approvals = {}
            if approvals_file.exists():
                with open(approvals_file, 'r') as f:
                    approvals = json.load(f)
            all_persons = [p for p in all_persons if p['name'] not in approvals]

        # Use embedding count as image count (they're 1:1 in the database)
        # This avoids iterating through 30k+ files for each person
        folders = []
        total_images = 0
        ready_count = 0
        insufficient_count = 0

        for person in all_persons:
            person_name = person['name']
            embedding_count = person['embedding_count']
            # Use filesystem count if available (accurate), otherwise fall back to embedding count
            fs_count = person.get('fs_count', embedding_count)

            # Use filesystem count as image_count for consistency with gallery view
            image_count = fs_count

            # Skip persons with no embeddings and no filesystem images
            # These are orphaned entries from deleted/rejected training data
            if embedding_count == 0 and image_count == 0:
                continue

            folders.append({
                'name': person_name,
                'display_name': person_name.replace('_', ' '),
                'image_count': image_count,
                'embedding_count': embedding_count,
                'fs_count': fs_count
            })

            total_images += image_count

            if image_count >= 50:
                ready_count += 1
            elif image_count < 20 and image_count > 0:
                insufficient_count += 1

        # Pagination
        total_items = len(folders)
        total_pages = (total_items + limit - 1) // limit
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit

        paginated_folders = folders[start_idx:end_idx]

        return jsonify({
            'success': True,
            'data': {
                'folders': paginated_folders,
                'pagination': {
                    'page': page,
                    'limit': limit,
                    'total_items': total_items,
                    'total_pages': total_pages
                },
                'summary': {
                    'total_images': total_images,
                    'ready_for_training': ready_count,
                    'insufficient_images': insufficient_count
                }
            }
        })

    except Exception as e:
        logger.error(f"Error getting training progress: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/folders/<folder_name>', methods=['GET'])
@require_auth
def get_folder_images(folder_name):
    """
    Get images for a specific person from the database.

    Query params:
    - domain: serbia|slovenia (required)
    - view: production|staging (ignored - always uses database)
    - page: page number (default: 1)
    - limit: items per page (default: 50)
    """
    from pathlib import Path
    import glob

    try:
        domain = request.args.get('domain', 'serbia')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))

        # Get all images for this person from flat file structure
        # Use absolute path to storage directory
        storage_dir = Path(current_app.config.get('STORAGE_DIR', '/root/facerecognition-backend/storage'))
        image_dir = storage_dir / 'recognized_faces_prod' / domain

        if not image_dir.exists():
            return jsonify({'success': False, 'error': 'Image directory not found'}), 404

        # Use glob for faster file matching (instead of iterating all 30k+ files)
        # Convert spaces to underscores to match filename convention
        file_prefix = folder_name.replace(' ', '_')

        images = []
        for ext in ['jpg', 'jpeg', 'png', 'webp']:
            pattern = str(image_dir / f"{file_prefix}_*.{ext}")
            for img_path in sorted(glob.glob(pattern)):
                img_file = Path(img_path)
                images.append({
                    'name': img_file.name,
                    'path': f'recognized_faces_prod/{domain}/{img_file.name}'
                })

        # Pagination
        total_items = len(images)
        total_pages = (total_items + limit - 1) // limit
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit

        paginated_images = images[start_idx:end_idx]

        return jsonify({
            'success': True,
            'data': {
                'folder': folder_name,
                'images': paginated_images,
                'pagination': {
                    'page': page,
                    'limit': limit,
                    'total_items': total_items,
                    'total_pages': total_pages
                }
            }
        })

    except Exception as e:
        logger.error(f"Error getting folder images: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/image/<view>/<domain>/<path:image_path>', methods=['GET'])
def serve_training_image(view, domain, image_path):
    """
    Serve a training image file from the database.
    No authentication required - images are public training data.

    Path params:
    - view: production|staging (ignored - always uses production database)
    - domain: serbia|slovenia
    - image_path: filename from recognized_faces_prod
    """
    from flask import send_file
    from pathlib import Path

    try:
        # Always use production database images
        # Use absolute path to storage directory
        storage_dir = Path(current_app.config.get('STORAGE_DIR', '/root/facerecognition-backend/storage'))
        base_path = storage_dir / 'recognized_faces_prod' / domain
        full_path = base_path / image_path

        # Security: ensure path is within allowed directory
        if not str(full_path.resolve()).startswith(str(base_path.resolve())):
            return jsonify({'success': False, 'error': 'Invalid path'}), 403

        if not full_path.exists() or not full_path.is_file():
            return jsonify({'success': False, 'error': 'Image not found'}), 404

        # Detect MIME type based on file extension
        ext = full_path.suffix.lower()
        mimetype_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp'
        }
        mimetype = mimetype_map.get(ext, 'image/jpeg')

        response = send_file(full_path, mimetype=mimetype)
        # Cache images for 1 hour in browser, 1 day in CDN/proxy
        response.headers['Cache-Control'] = 'public, max-age=3600, s-maxage=86400'
        return response

    except Exception as e:
        logger.error(f"Error serving training image: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# MERGE CANDIDATES ENDPOINTS
# ============================================================================

@training_bp.route('/merge-candidates/scan', methods=['POST'])
@require_auth
def scan_merge_candidates():
    """
    Scan the database for potential merge candidates.

    Detects:
    - Typos (names starting with lowercase or missing letters)
    - Duplicates (very similar names >90% match)
    - Spelling variants (Serbian/English variations)
    - Nicknames (shortened name versions)

    Body:
    {
        "domain": "serbia"
    }

    Returns list of candidates with suggested actions.
    """
    try:
        data = request.get_json() or {}
        domain = data.get('domain', 'serbia')

        service = MergeCandidatesService(domain=domain)
        result = service.scan_for_candidates()

        return jsonify({
            'success': True,
            'data': result
        })

    except Exception as e:
        logger.error(f"Error scanning for merge candidates: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/merge-candidates', methods=['GET'])
@require_auth
def get_merge_candidates():
    """
    Get the latest merge candidates scan results.

    Query params:
    - domain: Domain code (default 'serbia')

    Returns cached results from the last scan.
    """
    try:
        domain = request.args.get('domain', 'serbia')

        service = MergeCandidatesService(domain=domain)
        result = service.get_latest_candidates()

        if not result:
            return jsonify({
                'success': True,
                'data': None,
                'message': 'No scan results available. Run a scan first.'
            })

        return jsonify({
            'success': True,
            'data': result
        })

    except Exception as e:
        logger.error(f"Error getting merge candidates: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/merge-candidates/<int:candidate_id>/action', methods=['POST'])
@require_auth
def execute_merge_action(candidate_id):
    """
    Execute an action on a merge candidate.

    Actions:
    - MERGE: Merge source person into target person
    - RENAME: Rename source person to new name
    - DELETE: Delete source person and all embeddings
    - SKIP: Mark as reviewed (no database changes)

    Body:
    {
        "domain": "serbia",
        "action": "MERGE" | "RENAME" | "DELETE" | "SKIP",
        "new_name": "New Name",  // Required for RENAME action
        "swap_direction": false  // If true, swap source and target for MERGE
    }
    """
    import subprocess
    import json as json_module

    try:
        data = request.get_json() or {}
        domain = data.get('domain', 'serbia')
        action = data.get('action')
        new_name = data.get('new_name')
        swap_direction = data.get('swap_direction', False)

        if not action:
            return jsonify({'success': False, 'error': 'Action is required'}), 400

        if action not in ['MERGE', 'RENAME', 'DELETE', 'SKIP']:
            return jsonify({'success': False, 'error': f'Invalid action: {action}'}), 400

        if action == 'RENAME' and not new_name:
            return jsonify({'success': False, 'error': 'new_name is required for RENAME action'}), 400

        # Build command to execute the action in a subprocess
        # This avoids crashes in the gunicorn worker from psycopg2 issues
        cmd = [
            '/root/facerecognition-backend/venv/bin/python',
            '/root/facerecognition-backend/scripts/execute_merge_action.py',
            '--domain', domain,
            '--candidate-id', str(candidate_id),
            '--action', action
        ]

        if new_name:
            cmd.extend(['--new-name', new_name])

        if swap_direction:
            cmd.append('--swap-direction')

        # Run subprocess with minimal environment
        env = {
            'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            'VECTOR_DB_URL': 'postgresql://facerecadmin:1Un5J20uOLstxUxJYzvb76GCdAVOjZ5eN17Cq%2FQ%2FbrU%3D@localhost:5432/facerecognition'
        }

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd='/root/facerecognition-backend'
        )

        # Parse output
        try:
            output = json_module.loads(result.stdout)
            if output.get('success'):
                return jsonify(output)
            else:
                return jsonify(output), 400
        except json_module.JSONDecodeError:
            error_msg = result.stderr.strip() if result.stderr else result.stdout.strip() if result.stdout else 'Unknown error'
            logger.error(f"Subprocess error: returncode={result.returncode}, stdout={result.stdout}, stderr={result.stderr}")
            return jsonify({
                'success': False,
                'error': f'Script error: {error_msg}'
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Action timed out'}), 500
    except Exception as e:
        logger.error(f"Error executing merge action: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# STAGING ENDPOINTS - View and manage staging (trainingPass) data
# ============================================================================

@training_bp.route('/staging-list', methods=['GET'])
@require_auth
def get_staging_list():
    """
    Get list of people in staging (trainingPass) ready for review/deployment.

    Query params:
    - domain: serbia|slovenia (default: serbia)
    - include_images: if 'true', include image URLs for each person (default: true)

    Returns list of people with their image counts and image URLs from trainingPass folder.
    """
    import os
    from pathlib import Path

    try:
        domain = request.args.get('domain', 'serbia')
        include_images = request.args.get('include_images', 'true').lower() == 'true'

        # Path to staging folder
        staging_path = Path(f'/root/facerecognition-backend/storage/trainingPass/{domain}')

        if not staging_path.exists():
            return jsonify({
                'success': True,
                'data': {
                    'people': [],
                    'total': 0
                }
            })

        people = []
        for person_dir in sorted(staging_path.iterdir()):
            if person_dir.is_dir() and not person_dir.name.startswith('_'):
                # Collect images (excluding reference folders)
                images = []
                for f in sorted(person_dir.iterdir()):
                    if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                        images.append({
                            'filename': f.name,
                            'url': f'/api/training/staging/image/{domain}/{person_dir.name}/{f.name}'
                        })

                if len(images) > 0:
                    person_data = {
                        'folder_name': person_dir.name,
                        'full_name': person_dir.name.replace('_', ' '),
                        'display_name': person_dir.name.replace('_', ' '),
                        'image_count': len(images),
                        'status': 'ready' if len(images) >= 5 else 'insufficient'
                    }

                    # Include image URLs if requested
                    if include_images:
                        person_data['images'] = images  # All images for review

                    people.append(person_data)

        # Mark people as ready_for_production based on image count
        for p in people:
            p['ready_for_production'] = p['image_count'] >= 5

        return jsonify({
            'success': True,
            'people': people,
            'total': len(people),
            'ready_count': sum(1 for p in people if p['status'] == 'ready'),
            'insufficient_count': sum(1 for p in people if p['status'] == 'insufficient')
        })

    except Exception as e:
        logger.error(f"Error getting staging list: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/staging/serp-batches', methods=['GET'])
@require_auth
def get_serp_batches():
    """
    Get list of SERP training batches (automated training runs).

    Query params:
    - domain: serbia|slovenia (default: serbia)

    Returns list of batch metadata from storage/batches folder.
    Format matches frontend expectations with running_batches and completed_batches.
    """
    import os
    import json
    from pathlib import Path

    try:
        domain = request.args.get('domain', 'serbia')

        # Path to batches folder
        batches_path = Path(f'/root/facerecognition-backend/storage/batches/{domain}')

        running_batches = []
        completed_batches = []

        if batches_path.exists():
            for batch_file in sorted(batches_path.glob('*.json'), reverse=True):
                try:
                    with open(batch_file, 'r') as f:
                        batch_data = json.load(f)
                        batch_info = {
                            'batch_id': batch_file.stem,
                            'created_at': batch_data.get('created_at'),
                            'status': batch_data.get('status', 'unknown'),
                            'total_people': batch_data.get('total_celebrities', 0),
                            'processed': batch_data.get('processed', 0),
                            'successful': batch_data.get('successful', 0),
                            'failed': batch_data.get('failed', 0),
                            'people': batch_data.get('celebrities', [])[:10]  # First 10 for preview
                        }

                        # Categorize by status
                        if batch_info['status'] in ['pending', 'processing', 'running']:
                            running_batches.append(batch_info)
                        else:
                            completed_batches.append(batch_info)
                except Exception as e:
                    logger.warning(f"Error reading batch file {batch_file}: {e}")
                    continue

        return jsonify({
            'success': True,
            'running_batches': running_batches[:20],
            'completed_batches': completed_batches[:50],
            'running_count': len(running_batches),
            'completed_count': len(completed_batches)
        })

    except Exception as e:
        logger.error(f"Error getting SERP batches: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/staging/training-batches', methods=['GET'])
@require_auth
def get_training_batches():
    """
    Get list of training batches with their status.
    Alias for serp-batches for UI compatibility.
    """
    return get_serp_batches()


@training_bp.route('/staging/<folder_name>/images', methods=['GET'])
@require_auth
def get_staging_images(folder_name):
    """
    Get images for a specific person in staging.

    Path params:
    - folder_name: Person folder name (e.g., "Marko_Jaric")

    Query params:
    - domain: serbia|slovenia (default: serbia)
    - page: page number (default: 1)
    - limit: items per page (default: 50)
    """
    from pathlib import Path

    try:
        domain = request.args.get('domain', 'serbia')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 50))

        # Path to person's staging folder
        person_path = Path(f'/root/facerecognition-backend/storage/trainingPass/{domain}/{folder_name}')

        if not person_path.exists():
            return jsonify({'success': False, 'error': 'Person not found in staging'}), 404

        # Get all images
        images = []
        for f in sorted(person_path.iterdir()):
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                images.append({
                    'filename': f.name,
                    'path': f'trainingPass/{domain}/{folder_name}/{f.name}'
                })

        # Pagination
        total_items = len(images)
        total_pages = (total_items + limit - 1) // limit
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit

        paginated_images = images[start_idx:end_idx]

        return jsonify({
            'success': True,
            'data': {
                'folder_name': folder_name,
                'images': paginated_images,
                'pagination': {
                    'page': page,
                    'limit': limit,
                    'total_items': total_items,
                    'total_pages': total_pages
                }
            }
        })

    except Exception as e:
        logger.error(f"Error getting staging images: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/staging/image/<domain>/<path:image_path>', methods=['GET'])
def serve_staging_image(domain, image_path):
    """
    Serve an image from staging (trainingPass).
    No authentication required for image serving.
    """
    from flask import send_file
    from pathlib import Path

    try:
        base_path = Path(f'/root/facerecognition-backend/storage/trainingPass/{domain}')
        full_path = base_path / image_path

        # Security: ensure path is within allowed directory
        if not str(full_path.resolve()).startswith(str(base_path.resolve())):
            return jsonify({'success': False, 'error': 'Invalid path'}), 403

        if not full_path.exists() or not full_path.is_file():
            return jsonify({'success': False, 'error': 'Image not found'}), 404

        # Detect MIME type
        ext = full_path.suffix.lower()
        mimetype_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp'
        }
        mimetype = mimetype_map.get(ext, 'image/jpeg')

        return send_file(full_path, mimetype=mimetype)

    except Exception as e:
        logger.error(f"Error serving staging image: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/staging/image/<domain>/<path:image_path>', methods=['DELETE'])
@require_auth
def delete_staging_image(domain, image_path):
    """
    Delete a single image from staging (trainingPass).

    Path params:
    - domain: serbia|slovenia
    - image_path: folder_name/filename.jpg
    """
    from pathlib import Path

    try:
        base_path = Path(f'/root/facerecognition-backend/storage/trainingPass/{domain}')
        full_path = base_path / image_path

        # Security: ensure path is within allowed directory
        if not str(full_path.resolve()).startswith(str(base_path.resolve())):
            return jsonify({'success': False, 'error': 'Invalid path'}), 403

        if not full_path.exists() or not full_path.is_file():
            return jsonify({'success': False, 'error': 'Image not found'}), 404

        # Check it's an image file
        if full_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
            return jsonify({'success': False, 'error': 'Not an image file'}), 400

        # Delete the file
        full_path.unlink()

        # Get remaining image count for the person
        person_dir = full_path.parent
        remaining_images = sum(1 for f in person_dir.iterdir()
                              if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'])

        logger.info(f"Deleted staging image: {domain}/{image_path}, {remaining_images} images remaining")

        return jsonify({
            'success': True,
            'message': f'Image deleted successfully',
            'remaining_images': remaining_images
        })

    except Exception as e:
        logger.error(f"Error deleting staging image: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/deploy', methods=['POST'])
@require_auth
def deploy_to_production():
    """
    Deploy people from staging (trainingPass) to production (recognized_faces_prod).

    Request body:
    {
        "people": ["folder_name1", "folder_name2"],  # List of folder names to deploy
        "domain": "serbia"  # Target domain
    }

    Returns:
        - deployed: List of successfully deployed people with image counts
        - skipped: List of people skipped (not enough images)
        - errors: List of people with errors
    """
    import shutil
    from pathlib import Path
    from datetime import datetime
    import hashlib

    try:
        data = request.get_json() or {}
        people = data.get('people', [])
        domain = data.get('domain', 'serbia')

        if not people:
            return jsonify({'success': False, 'error': 'No people specified'}), 400

        staging_base = Path(f'/root/facerecognition-backend/storage/trainingPass/{domain}')
        prod_base = Path(f'/root/facerecognition-backend/storage/recognized_faces_prod/{domain}')

        # Ensure production directory exists
        prod_base.mkdir(parents=True, exist_ok=True)

        deployed = []
        skipped = []
        errors = []

        today = datetime.now().strftime('%Y-%m-%d')

        for folder_name in people:
            staging_dir = staging_base / folder_name

            if not staging_dir.exists() or not staging_dir.is_dir():
                errors.append({
                    'folder': folder_name,
                    'error': 'Folder not found in staging'
                })
                continue

            # Get all images in staging folder
            images = [f for f in staging_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]

            if len(images) < 5:
                skipped.append({
                    'folder': folder_name,
                    'reason': f'Too few images ({len(images)}), minimum is 5'
                })
                continue

            # Get existing images for this person in production to avoid duplicates
            person_name = folder_name  # e.g., "Antonela_Riha"
            existing_in_prod = list(prod_base.glob(f'{person_name}_*'))
            existing_hashes = set()

            # Build hash set of existing production images
            for existing_file in existing_in_prod:
                try:
                    with open(existing_file, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        existing_hashes.add(file_hash)
                except Exception:
                    pass

            # Copy images to production
            copied_count = 0
            duplicates_skipped = 0

            for idx, img_path in enumerate(images, start=1):
                try:
                    # Check for duplicate by content hash
                    with open(img_path, 'rb') as f:
                        img_hash = hashlib.md5(f.read()).hexdigest()

                    if img_hash in existing_hashes:
                        duplicates_skipped += 1
                        continue

                    # Generate production filename: PersonName_date_uniqueid.ext
                    # Use microseconds + idx for uniqueness
                    unique_id = f"{datetime.now().strftime('%H%M%S')}{idx:02d}"
                    new_filename = f"{person_name}_{today}_{unique_id}{img_path.suffix.lower()}"
                    dest_path = prod_base / new_filename

                    # Handle potential name collision
                    collision_counter = 0
                    while dest_path.exists():
                        collision_counter += 1
                        new_filename = f"{person_name}_{today}_{unique_id}_{collision_counter}{img_path.suffix.lower()}"
                        dest_path = prod_base / new_filename

                    # Copy file
                    shutil.copy2(img_path, dest_path)
                    existing_hashes.add(img_hash)
                    copied_count += 1

                except Exception as e:
                    logger.error(f"Error copying image {img_path}: {str(e)}")

            if copied_count > 0:
                deployed.append({
                    'folder': folder_name,
                    'image_count': copied_count,
                    'duplicates_skipped': duplicates_skipped
                })

                # Remove staging folder after successful deploy
                try:
                    shutil.rmtree(staging_dir)
                    logger.info(f"Deployed {folder_name}: {copied_count} images to production, removed staging folder")
                except Exception as e:
                    logger.warning(f"Could not remove staging folder {staging_dir}: {str(e)}")
            else:
                if duplicates_skipped > 0:
                    skipped.append({
                        'folder': folder_name,
                        'reason': f'All {duplicates_skipped} images already exist in production'
                    })
                else:
                    errors.append({
                        'folder': folder_name,
                        'error': 'No images could be copied'
                    })

        return jsonify({
            'success': True,
            'message': f'Deployed {len(deployed)} people to production',
            'deployed': deployed,
            'skipped': skipped,
            'errors': errors,
            'statistics': {
                'deployed_count': len(deployed),
                'skipped_count': len(skipped),
                'error_count': len(errors)
            }
        })

    except Exception as e:
        logger.error(f"Error deploying to production: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/merge-persons', methods=['POST'])
@require_auth
def manual_merge_persons():
    """
    Manually merge two persons by name.

    This endpoint allows direct person-to-person merging without
    going through the auto-detected candidates system.

    Request body:
    {
        "source_person": "Person A",   # Will be deleted
        "target_person": "Person B",   # Will receive all images/embeddings
        "domain": "serbia"
    }
    """
    import subprocess
    import json as json_module
    import glob
    from pathlib import Path

    try:
        data = request.get_json() or {}
        source_person = data.get('source_person')
        target_person = data.get('target_person')
        domain = data.get('domain', 'serbia')

        if not source_person or not target_person:
            return jsonify({
                'success': False,
                'error': 'Both source_person and target_person are required'
            }), 400

        if source_person == target_person:
            return jsonify({
                'success': False,
                'error': 'Source and target must be different persons'
            }), 400

        # Run the merge in a subprocess to avoid psycopg2 issues
        env = {
            'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            'VECTOR_DB_URL': 'postgresql://facerecadmin:1Un5J20uOLstxUxJYzvb76GCdAVOjZ5eN17Cq%2FQ%2FbrU%3D@localhost:5432/facerecognition'
        }

        script = f'''
import psycopg2
import json
import os
import glob
from pathlib import Path
from urllib.parse import unquote

# Parse DB URL
db_url = os.getenv('VECTOR_DB_URL')
parts = db_url.replace('postgresql://', '').split('@')
user_pass = parts[0]
host_db = parts[1]
user, password = user_pass.split(':')
password = unquote(password)
host_port_db = host_db.split('/')
host_port = host_port_db[0]
database = host_port_db[1]
host, port = host_port.split(':') if ':' in host_port else (host_port, '5432')

conn = psycopg2.connect(host=host, port=port, database=database, user=user, password=password)
cursor = conn.cursor()

domain = {json_module.dumps(domain)}
source_person = {json_module.dumps(source_person)}
target_person = {json_module.dumps(target_person)}

# Get source person ID
cursor.execute("SELECT id FROM persons WHERE name = %s AND domain = %s", (source_person, domain))
source_row = cursor.fetchone()
if not source_row:
    print(json.dumps({{"success": False, "error": f"Source person '{{source_person}}' not found"}}))
    exit(1)
source_id = source_row[0]

# Get target person ID
cursor.execute("SELECT id FROM persons WHERE name = %s AND domain = %s", (target_person, domain))
target_row = cursor.fetchone()
if not target_row:
    print(json.dumps({{"success": False, "error": f"Target person '{{target_person}}' not found"}}))
    exit(1)
target_id = target_row[0]

# Move embeddings from source to target
cursor.execute("UPDATE face_embeddings SET person_id = %s WHERE person_id = %s", (target_id, source_id))
moved_embeddings = cursor.rowcount

# Delete source person
cursor.execute("DELETE FROM persons WHERE id = %s", (source_id,))

conn.commit()
cursor.close()
conn.close()

# Also rename image files on filesystem
storage_dir = Path('/root/facerecognition-backend/storage/recognized_faces_prod') / domain
source_prefix = source_person.replace(' ', '_')
target_prefix = target_person.replace(' ', '_')

renamed_files = 0
if storage_dir.exists():
    for ext in ['jpg', 'jpeg', 'png', 'webp']:
        for old_path in glob.glob(str(storage_dir / f"{{source_prefix}}_*.{{ext}}")):
            old_file = Path(old_path)
            # Extract timestamp from old filename
            old_name = old_file.name
            # Replace source prefix with target prefix
            new_name = old_name.replace(source_prefix + '_', target_prefix + '_', 1)
            new_path = storage_dir / new_name
            if not new_path.exists():
                old_file.rename(new_path)
                renamed_files += 1

print(json.dumps({{
    "success": True,
    "message": f"Merged '{{source_person}}' into '{{target_person}}'",
    "embeddings_moved": moved_embeddings,
    "files_renamed": renamed_files
}}))
'''

        result = subprocess.run(
            ['/root/facerecognition-backend/venv/bin/python', '-c', script],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd='/root/facerecognition-backend'
        )

        try:
            output = json_module.loads(result.stdout)
            if output.get('success'):
                return jsonify(output)
            else:
                return jsonify(output), 400
        except json_module.JSONDecodeError:
            error_msg = result.stderr.strip() if result.stderr else result.stdout.strip() if result.stdout else 'Unknown error'
            logger.error(f"Merge subprocess error: {error_msg}")
            return jsonify({
                'success': False,
                'error': f'Script error: {error_msg}'
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Merge operation timed out'}), 500
    except Exception as e:
        logger.error(f"Error merging persons: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/person-preview', methods=['GET'])
@require_auth
def get_person_preview():
    """
    Get preview information for a person including sample images.

    Query params:
    - name: Person name
    - domain: serbia|slovenia
    """
    import glob
    from pathlib import Path

    try:
        person_name = request.args.get('name')
        domain = request.args.get('domain', 'serbia')

        if not person_name:
            return jsonify({'success': False, 'error': 'name parameter is required'}), 400

        # Get images from filesystem
        storage_dir = Path('/root/facerecognition-backend/storage/recognized_faces_prod') / domain
        file_prefix = person_name.replace(' ', '_')

        images = []
        for ext in ['jpg', 'jpeg', 'png', 'webp']:
            pattern = str(storage_dir / f"{file_prefix}_*.{ext}")
            for img_path in sorted(glob.glob(pattern))[:6]:  # Return up to 6 preview images
                img_file = Path(img_path)
                images.append({
                    'name': img_file.name,
                    'url': f'/api/training/image/production/{domain}/{img_file.name}'
                })

        return jsonify({
            'success': True,
            'data': {
                'person_name': person_name,
                'image_count': len(glob.glob(str(storage_dir / f"{file_prefix}_*.*"))),
                'preview_images': images
            }
        })

    except Exception as e:
        logger.error(f"Error getting person preview: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# Person Approval Endpoints (server-side storage)
# ============================================================

@training_bp.route('/approvals', methods=['GET'])
@require_auth
def get_approvals():
    """
    Get all approved persons for a domain/view.

    Query params:
    - domain: serbia|slovenia
    - view: production|staging
    """
    import json as json_module
    from pathlib import Path

    try:
        domain = request.args.get('domain', 'serbia')
        view = request.args.get('view', 'production')

        approvals_file = Path('/root/facerecognition-backend/storage/approvals') / f'{domain}_{view}.json'

        if approvals_file.exists():
            with open(approvals_file, 'r') as f:
                approvals = json_module.load(f)
        else:
            approvals = {}

        return jsonify({
            'success': True,
            'data': {
                'domain': domain,
                'view': view,
                'approvals': approvals
            }
        })

    except Exception as e:
        logger.error(f"Error getting approvals: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/approvals/<person_name>', methods=['POST'])
@require_auth
def approve_person(person_name):
    """
    Approve a person.

    Request body:
    {
        "domain": "serbia",
        "view": "production",
        "image_count": 29
    }
    """
    import json as json_module
    from pathlib import Path
    from datetime import datetime

    try:
        data = request.get_json() or {}
        domain = data.get('domain', 'serbia')
        view = data.get('view', 'production')
        image_count = data.get('image_count', 0)

        approvals_dir = Path('/root/facerecognition-backend/storage/approvals')
        approvals_dir.mkdir(parents=True, exist_ok=True)

        approvals_file = approvals_dir / f'{domain}_{view}.json'

        if approvals_file.exists():
            with open(approvals_file, 'r') as f:
                approvals = json_module.load(f)
        else:
            approvals = {}

        approvals[person_name] = {
            'approved_at': datetime.now().isoformat(),
            'image_count': image_count,
            'approved_by': request.headers.get('X-User-Email', 'unknown')
        }

        with open(approvals_file, 'w') as f:
            json_module.dump(approvals, f, indent=2)

        return jsonify({
            'success': True,
            'message': f"Approved '{person_name}'"
        })

    except Exception as e:
        logger.error(f"Error approving person: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/approvals/<person_name>', methods=['DELETE'])
@require_auth
def unapprove_person(person_name):
    """
    Remove approval for a person.

    Query params:
    - domain: serbia|slovenia
    - view: production|staging
    """
    import json as json_module
    from pathlib import Path

    try:
        domain = request.args.get('domain', 'serbia')
        view = request.args.get('view', 'production')

        approvals_file = Path('/root/facerecognition-backend/storage/approvals') / f'{domain}_{view}.json'

        if approvals_file.exists():
            with open(approvals_file, 'r') as f:
                approvals = json_module.load(f)

            if person_name in approvals:
                del approvals[person_name]

                with open(approvals_file, 'w') as f:
                    json_module.dump(approvals, f, indent=2)

        return jsonify({
            'success': True,
            'message': f"Removed approval for '{person_name}'"
        })

    except Exception as e:
        logger.error(f"Error removing approval: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@training_bp.route('/delete-person', methods=['POST'])
@require_auth
def delete_person():
    """
    Delete a person entry from the database.

    Useful for removing persons with 0 images/embeddings.

    Body:
    {
        "person_name": "Person_Name",
        "domain": "serbia"
    }
    """
    try:
        data = request.get_json()
        person_name = data.get('person_name')
        domain = data.get('domain', 'serbia')

        if not person_name:
            return jsonify({'success': False, 'error': 'person_name is required'}), 400

        import os
        import psycopg2
        from urllib.parse import unquote

        db_url = os.getenv(
            'VECTOR_DB_URL',
            'postgresql://facerecadmin:1Un5J20uOLstxUxJYzvb76GCdAVOjZ5eN17Cq%2FQ%2FbrU%3D@localhost:5432/facerecognition'
        )

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

        # First check if person exists and get embedding count
        cursor.execute(
            "SELECT id FROM persons WHERE name = %s AND domain = %s",
            (person_name, domain)
        )
        result = cursor.fetchone()

        if not result:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'error': f"Person '{person_name}' not found in domain '{domain}'"}), 404

        person_id = result[0]

        # Get embedding count
        cursor.execute(
            "SELECT COUNT(*) FROM face_embeddings WHERE person_id = %s",
            (person_id,)
        )
        embedding_count = cursor.fetchone()[0]

        # Delete embeddings first (if any)
        cursor.execute("DELETE FROM face_embeddings WHERE person_id = %s", (person_id,))
        deleted_embeddings = cursor.rowcount

        # Delete the person
        cursor.execute("DELETE FROM persons WHERE id = %s", (person_id,))

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"Deleted person '{person_name}' from domain '{domain}' ({deleted_embeddings} embeddings removed)")

        return jsonify({
            'success': True,
            'message': f"Deleted '{person_name}' from database",
            'embeddings_deleted': deleted_embeddings
        })

    except Exception as e:
        logger.error(f"Error deleting person: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
