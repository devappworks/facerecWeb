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
