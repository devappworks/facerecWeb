"""
Training Routes for the new Wikidata-based celebrity training workflow.
Provides endpoints for generating candidates, batch training, and deployment.
"""

from flask import Blueprint, jsonify, request, current_app
from app.services.wikidata_service import WikidataService
from app.services.training_batch_service import TrainingBatchService

training_bp = Blueprint('training', __name__, url_prefix='/api/training')


@training_bp.route('/countries', methods=['GET'])
def get_countries():
    """
    Get list of available countries.

    Returns:
        JSON: List of countries with ID and name
    """
    try:
        countries = WikidataService.get_available_countries()
        return jsonify({
            "success": True,
            "countries": countries
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error getting countries: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@training_bp.route('/occupations', methods=['GET'])
def get_occupations():
    """
    Get list of available occupations.

    Returns:
        JSON: List of occupations with ID and name
    """
    try:
        occupations = WikidataService.get_available_occupations()
        return jsonify({
            "success": True,
            "occupations": occupations
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error getting occupations: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@training_bp.route('/generate-candidates', methods=['POST'])
def generate_candidates():
    """
    Generate celebrity candidates from Wikidata.

    Request Body:
        {
            "country": "serbia",
            "occupation": "actor",
            "domain": "serbia"  // optional, defaults to "serbia"
        }

    Returns:
        JSON: List of candidates with existing DB status
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "Request body is required"
            }), 400

        country = data.get('country')
        occupation = data.get('occupation')
        domain = data.get('domain', 'serbia')

        if not country:
            return jsonify({
                "success": False,
                "message": "Parameter 'country' is required"
            }), 400

        if not occupation:
            return jsonify({
                "success": False,
                "message": "Parameter 'occupation' is required"
            }), 400

        current_app.logger.info(f"Generating candidates: {country} {occupation}s")

        result = TrainingBatchService.generate_candidates(country, occupation, domain)

        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code

    except Exception as e:
        current_app.logger.error(f"Error generating candidates: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@training_bp.route('/search-person', methods=['GET'])
def search_person():
    """
    Search for a specific person by name (for autocomplete).

    Query Parameters:
        query: Search term (person name)
        limit: Maximum results (default 20)

    Returns:
        JSON: List of matching people
    """
    try:
        query = request.args.get('query')
        limit = request.args.get('limit', 20, type=int)

        if not query:
            return jsonify({
                "success": False,
                "message": "Parameter 'query' is required"
            }), 400

        if len(query) < 2:
            return jsonify({
                "success": False,
                "message": "Query must be at least 2 characters"
            }), 400

        people = WikidataService.search_person(query, limit)

        return jsonify({
            "success": True,
            "results": people,
            "count": len(people)
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error searching person: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@training_bp.route('/start-batch', methods=['POST'])
def start_batch_training():
    """
    Start batch training for selected candidates.

    Request Body:
        {
            "candidates": [
                {
                    "full_name": "Novak Djokovic",
                    "name": "Novak",
                    "last_name": "Djokovic",
                    "occupation": "tennis_player",
                    ...
                },
                ...
            ],
            "domain": "serbia"  // optional
        }

    Returns:
        JSON: Batch information with batch_id for tracking
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "Request body is required"
            }), 400

        candidates = data.get('candidates', [])
        domain = data.get('domain', 'serbia')

        if not candidates:
            return jsonify({
                "success": False,
                "message": "At least one candidate is required"
            }), 400

        if not isinstance(candidates, list):
            return jsonify({
                "success": False,
                "message": "Candidates must be an array"
            }), 400

        current_app.logger.info(f"Starting batch training for {len(candidates)} people")

        result = TrainingBatchService.start_batch_training(candidates, domain)

        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code

    except Exception as e:
        current_app.logger.error(f"Error starting batch training: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@training_bp.route('/batch/<batch_id>/status', methods=['GET'])
def get_batch_status(batch_id):
    """
    Get status of a training batch.

    Path Parameters:
        batch_id: Batch identifier

    Returns:
        JSON: Batch status with progress information
    """
    try:
        result = TrainingBatchService.get_batch_status(batch_id)

        status_code = 200 if result.get('success') else 404
        return jsonify(result), status_code

    except Exception as e:
        current_app.logger.error(f"Error getting batch status: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@training_bp.route('/deploy', methods=['POST'])
def deploy_to_production():
    """
    Deploy trained people from staging to production.

    Request Body:
        {
            "people": ["novak_djokovic", "ana_ivanovic", ...],
            "domain": "serbia"  // optional
        }

    Returns:
        JSON: Deployment result with statistics
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "Request body is required"
            }), 400

        people = data.get('people', [])
        domain = data.get('domain', 'serbia')

        if not people:
            return jsonify({
                "success": False,
                "message": "At least one person is required"
            }), 400

        if not isinstance(people, list):
            return jsonify({
                "success": False,
                "message": "People must be an array"
            }), 400

        current_app.logger.info(f"Deploying {len(people)} people to production")

        result = TrainingBatchService.deploy_to_production(people, domain)

        return jsonify(result), 200

    except Exception as e:
        current_app.logger.error(f"Error deploying to production: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@training_bp.route('/staging-list', methods=['GET'])
def get_staging_list():
    """
    Get list of people in staging (trainingPass) ready for deployment.

    Query Parameters:
        domain: Domain name (default "serbia")

    Returns:
        JSON: List of folders with image counts
    """
    try:
        import os

        domain = request.args.get('domain', 'serbia')
        staging_path = "storage/trainingPassSerbia"

        if not os.path.exists(staging_path):
            return jsonify({
                "success": True,
                "people": [],
                "count": 0
            }), 200

        people = []
        for folder in os.listdir(staging_path):
            folder_path = os.path.join(staging_path, folder)

            if not os.path.isdir(folder_path):
                continue

            # Count images
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
            ]

            people.append({
                "folder_name": folder,
                "image_count": len(image_files),
                "ready_for_production": len(image_files) >= 5
            })

        # Sort by image count descending
        people.sort(key=lambda x: x['image_count'], reverse=True)

        return jsonify({
            "success": True,
            "people": people,
            "count": len(people),
            "ready_count": len([p for p in people if p['ready_for_production']])
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error getting staging list: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@training_bp.route('/queue-list', methods=['GET'])
def get_queue_list():
    """
    Get list of names currently in the training queue (data.xlsx).

    Returns:
        JSON: List of people waiting to be processed
    """
    try:
        import os
        import pandas as pd

        queue_file = 'storage/excel/data.xlsx'

        if not os.path.exists(queue_file):
            return jsonify({
                "success": False,
                "message": "Queue file not found or empty"
            }), 404

        # Read Excel file
        df = pd.read_excel(queue_file)

        if df.empty:
            return jsonify({
                "success": True,
                "data": {
                    "queue": [],
                    "total": 0,
                    "processed": 0,
                    "remaining": 0
                }
            }), 200

        # Convert to list of dicts
        queue = []
        for idx, row in df.iterrows():
            queue.append({
                "id": idx + 1,
                "name": row.get('name', ''),
                "last_name": row.get('last_name', ''),
                "occupation": row.get('occupation', ''),
                "country": row.get('country', 'Serbia')
            })

        return jsonify({
            "success": True,
            "data": {
                "queue": queue,
                "total": len(queue),
                "processed": 0,  # TODO: Track processed count
                "remaining": len(queue)
            }
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error getting queue list: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@training_bp.route('/queue', methods=['DELETE'])
def remove_from_queue():
    """
    Remove a specific person from the training queue.

    Request Body:
        {
            "id": 1,
            "name": "Dragan",
            "last_name": "Bjelogrlic"
        }

    Returns:
        JSON: Success status and remaining count
    """
    try:
        import os
        import pandas as pd

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "Request body is required"
            }), 400

        queue_file = 'storage/excel/data.xlsx'

        if not os.path.exists(queue_file):
            return jsonify({
                "success": False,
                "message": "Queue file not found"
            }), 404

        # Read Excel file
        df = pd.read_excel(queue_file)

        if df.empty:
            return jsonify({
                "success": False,
                "message": "Queue is empty"
            }), 400

        # Find and remove the entry
        entry_id = data.get('id')
        name = data.get('name')
        last_name = data.get('last_name')

        if entry_id is not None:
            # Remove by ID (row index)
            idx = entry_id - 1
            if 0 <= idx < len(df):
                df = df.drop(df.index[idx])
                df.reset_index(drop=True, inplace=True)
            else:
                return jsonify({
                    "success": False,
                    "message": f"Entry with ID {entry_id} not found"
                }), 404
        elif name and last_name:
            # Remove by name match
            mask = (df['name'] == name) & (df['last_name'] == last_name)
            if mask.any():
                df = df[~mask]
                df.reset_index(drop=True, inplace=True)
            else:
                return jsonify({
                    "success": False,
                    "message": f"Entry '{name} {last_name}' not found in queue"
                }), 404
        else:
            return jsonify({
                "success": False,
                "message": "Either 'id' or both 'name' and 'last_name' are required"
            }), 400

        # Save updated Excel
        df.to_excel(queue_file, index=False)

        current_app.logger.info(f"Removed entry from queue. Remaining: {len(df)}")

        return jsonify({
            "success": True,
            "message": "Entry removed from queue",
            "remaining_count": len(df)
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error removing from queue: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@training_bp.route('/queue-status', methods=['GET'])
def get_queue_status():
    """
    Get current status of training queue and processing.

    Returns:
        JSON: Queue statistics and processing status
    """
    try:
        import os
        import pandas as pd
        from datetime import datetime

        queue_file = 'storage/excel/data.xlsx'
        staging_path = 'storage/trainingPassSerbia'

        # Queue stats
        total_in_queue = 0
        if os.path.exists(queue_file):
            df = pd.read_excel(queue_file)
            total_in_queue = len(df)

        # Staging stats (processed today - approximate)
        processed_today = 0
        if os.path.exists(staging_path):
            today = datetime.now().date()
            for folder in os.listdir(staging_path):
                folder_path = os.path.join(staging_path, folder)
                if os.path.isdir(folder_path):
                    # Check if modified today
                    mtime = os.path.getmtime(folder_path)
                    if datetime.fromtimestamp(mtime).date() == today:
                        processed_today += 1

        # Check if processing thread is active (simplified check)
        is_processing = False
        current_person = None

        # TODO: Implement proper thread tracking
        # For now, we'll check if there are recent changes in staging

        return jsonify({
            "success": True,
            "data": {
                "queue": {
                    "total_in_queue": total_in_queue,
                    "processed_today": processed_today,
                    "failed_today": 0,  # TODO: Track failures
                    "remaining": total_in_queue
                },
                "processing": {
                    "is_processing": is_processing,
                    "current_person": current_person
                },
                "generation": {
                    "is_generating": False,  # TODO: Track GPT generation state
                    "last_generated": None,
                    "last_generated_count": 0
                }
            }
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error getting queue status: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@training_bp.route('/progress', methods=['GET'])
def get_training_progress():
    """
    Get detailed status of all training folders with image counts.

    Query Parameters:
        domain: Domain name (default "serbia")

    Returns:
        JSON: Detailed folder status with image counts and classifications
    """
    try:
        import os
        from datetime import datetime

        domain = request.args.get('domain', 'serbia')
        staging_path = 'storage/trainingPassSerbia'

        if not os.path.exists(staging_path):
            return jsonify({
                "success": True,
                "data": {
                    "folders": [],
                    "summary": {
                        "total_people": 0,
                        "total_images": 0,
                        "ready_for_training": 0,
                        "insufficient_images": 0,
                        "empty_folders": 0
                    }
                }
            }), 200

        folders = []
        total_images = 0
        ready_count = 0
        insufficient_count = 0
        empty_count = 0

        # Load name mappings for display names
        name_mapping = {}
        mapping_file = 'storage/name_mapping.json'
        if os.path.exists(mapping_file):
            import json
            with open(mapping_file, 'r', encoding='utf-8') as f:
                name_mapping = json.load(f)

        for folder in sorted(os.listdir(staging_path)):
            folder_path = os.path.join(staging_path, folder)

            if not os.path.isdir(folder_path):
                continue

            # Count images
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))
            ]

            image_count = len(image_files)
            total_images += image_count

            # Determine status
            if image_count == 0:
                status = 'empty'
                empty_count += 1
            elif image_count < 20:
                status = 'insufficient'
                insufficient_count += 1
            elif image_count < 40:
                status = 'adequate'
            else:
                status = 'ready'
                ready_count += 1

            # Get display name from mapping
            display_name = name_mapping.get(folder.lower(), folder.replace('_', ' ').title())

            # Get last modified time
            mtime = os.path.getmtime(folder_path)
            last_modified = datetime.fromtimestamp(mtime).isoformat()

            folders.append({
                "name": folder,
                "display_name": display_name,
                "occupation": "",  # TODO: Track occupation in metadata
                "image_count": image_count,
                "status": status,
                "folder_path": folder_path,
                "last_modified": last_modified
            })

        return jsonify({
            "success": True,
            "data": {
                "folders": folders,
                "summary": {
                    "total_people": len(folders),
                    "total_images": total_images,
                    "ready_for_training": ready_count,
                    "insufficient_images": insufficient_count,
                    "empty_folders": empty_count
                }
            }
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error getting training progress: {str(e)}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500
