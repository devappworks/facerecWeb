from flask import Blueprint, jsonify, request
from app.controllers.image_controller import ImageController
from app.controllers.sync_controller import SyncController
from app.services.validation_service import ValidationService
from datetime import datetime
from app.controllers.recognition_controller import RecognitionController
from app.controllers.object_detection_controller import ObjectDetectionController
import logging
import os

image_routes = Blueprint('image', __name__)

logger = logging.getLogger(__name__)

@image_routes.route('/upload-with-domain', methods=['POST'])
def upload_with_domain():
    auth_token = request.headers.get('Authorization')
    validation_service = ValidationService()

    if not auth_token:
        return jsonify({'message': 'Unauthorized'}), 401
    
    if not validation_service.validate_auth_token(auth_token):
        return jsonify({'message': 'Unauthorized'}), 401
    
    if 'image' not in request.files:
        return jsonify({"error": "Nema slike u zahtevu"}), 400
    
    if 'person' not in request.form:
        return jsonify({"error": "Nedostaje parametar 'person'"}), 400
        
    if 'created_date' not in request.form:
        return jsonify({"error": "Nedostaje parametar 'created_date'"}), 400
    
    try:
        created_date = datetime.strptime(request.form['created_date'], '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Neispravan format datuma. Koristite YYYY-MM-DD"}), 400
    
    image_file = request.files['image']
    person = request.form['person']
    domain = validation_service.get_domain()

    result = ImageController.handle_image_upload(
        image_file=image_file,
        person=person,
        created_date=created_date,
        domain=domain
    )
    return jsonify(result), 202  # 202 Accepted status kod 

@image_routes.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        auth_token = request.headers.get('Authorization')
        validation_service = ValidationService()
        
        if not auth_token:
            return jsonify({'message': 'Unauthorized'}), 401
        
        if not validation_service.validate_auth_token(auth_token):
            return jsonify({'message': 'Unauthorized'}), 401

        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'No selected file'}), 400

        # Validate file type - only accept supported image formats
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.avif'}
        file_ext = os.path.splitext(image_file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Unsupported image format: {file_ext}. Supported formats: JPG, PNG, GIF, BMP, TIFF, WEBP, AVIF'
            }), 400

        domain = validation_service.get_domain()
        
        # Čitaj sliku kao bytes
        image_bytes = image_file.read()
        
        # Pozovi kontroler
        result = RecognitionController.recognize_face(image_bytes, domain)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in recognize_face endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500 

@image_routes.route('/sync-faces', methods=['POST', 'GET'])
def sync_faces():
    try:
        # Dobavi parametre iz zahteva
        source_dir = 'storage/recognized_faces'
        target_dir = 'storage/recognized_faces_prod'
        
        # Ako je JSON zahtev, pokušaj dobiti parametre iz njega
        if request.is_json:
            source_dir = request.json.get('source_dir', source_dir)
            target_dir = request.json.get('target_dir', target_dir)
        # Ako su parametri poslati kao form-data
        elif request.form:
            source_dir = request.form.get('source_dir', source_dir)
            target_dir = request.form.get('target_dir', target_dir)
        # Ako su parametri poslati kao query string
        elif request.args:
            source_dir = request.args.get('source_dir', source_dir)
            target_dir = request.args.get('target_dir', target_dir)
            
        logger.info(f"Sinhronizacija pokrenuta sa parametrima: source_dir={source_dir}, target_dir={target_dir}")
        
        # Putanja do test slike
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        test_image_path = os.path.join(script_dir, 'scripts', 'test_face.JPG')
        
        # Proveri da li test slika postoji
        if not os.path.exists(test_image_path):
            logger.warning(f"Test slika ne postoji na putanji: {test_image_path}")
            return jsonify({"warning": "Test slika ne postoji, prepoznavanje lica neće biti izvršeno", 
                           "path_checked": test_image_path}), 200
        
        # Pozovi kontroler za pozadinsku sinhronizaciju
        result = SyncController.sync_faces_background(source_dir, target_dir, test_image_path)
        
        return jsonify(result), 202  # 202 Accepted
        
    except Exception as e:
        logger.error(f"Greška u sync_faces endpoint-u: {str(e)}")
        return jsonify({'error': str(e)}), 500         

@image_routes.route('/sync-kylo', methods=['POST'])
def sync_kylo():
    """
    Endpoint za sinhronizaciju slika sa Kylo sistema.
    Preuzima slike sa Kylo API-ja, obrađuje ih i čuva prepoznata lica.
    """
    try:
        auth_token = request.headers.get('Authorization')
       
        validation_service = ValidationService()
        
        if not auth_token:
            return jsonify({'message': 'Unauthorized'}), 401

        if not validation_service.validate_auth_token(auth_token):
            return jsonify({'message': 'Unauthorized'}), 401

        domain = validation_service.get_domain()
        result = SyncController.sync_images_from_kylo(domain)
        return jsonify(True), 202  # 202 Accepted
    except Exception as e:
        logger.error(f"Greška u sync_kylo endpoint-u: {str(e)}")
        return jsonify({'error': str(e)}), 500         

@image_routes.route('/transfer-images', methods=['POST'])
def transfer_images():
    """
    Endpoint za transfer slika iz storage/transfer_images u storage/recognized_faces_prod/media24
    """
    try:
        # Parametri za transfer
        source_dir = 'storage/transfer_images'
        target_domain = 'media24'
        batch_size = 30
        
        # Pokretanje transfera u pozadini
        result = SyncController.transfer_images_background(
            source_dir=source_dir,
            target_domain=target_domain,
            batch_size=batch_size
        )
        
        return jsonify(result), 202  # 202 Accepted
        
    except Exception as e:
        logger.error(f"Greška u transfer_images endpoint-u: {str(e)}")
        return jsonify({'error': str(e)}), 500         

@image_routes.route('/upload-for-detection', methods=['POST'])
def upload_for_detection():

    """
    Endpoint for uploading images for object detection.
    Images are resized and stored in storage/objectDetection.
    """
    try:
        auth_token = request.headers.get('Authorization')
        validation_service = ValidationService()

        if not auth_token:
            return jsonify({'message': 'Unauthorized'}), 401

        if not validation_service.validate_auth_token(auth_token):
            return jsonify({'message': 'Unauthorized'}), 401

        if 'image' not in request.files:
            return jsonify({"error": "No image in request"}), 400

        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({"error": "No selected file"}), 400

        # Validate file type - only accept supported image formats
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.avif'}
        file_ext = os.path.splitext(image_file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Unsupported image format: {file_ext}. Supported formats: JPG, PNG, GIF, BMP, TIFF, WEBP, AVIF'
            }), 400

        # Call the controller to handle the image

        tracking_token = ObjectDetectionController.generate_tracking_token()
        result = ObjectDetectionController.handle_detection_image(image_file, tracking_token)

        return jsonify(result), 202  # 202 Accepted

    except Exception as e:
        logger.error(f"Error in upload_for_detection endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


def is_admin_user(email: str) -> bool:
    """Check if the user is an admin (has access to model selection and usage info)."""
    ADMIN_EMAILS = ['nikola1jankovic@gmail.com']
    return email.lower() in [e.lower() for e in ADMIN_EMAILS] if email else False


@image_routes.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Enhanced image analysis endpoint with face recognition integration.

    This endpoint performs sequential processing:
    1. Face recognition to identify known persons
    2. Vision analysis with face context for richer descriptions

    Headers:
        - Authorization: Auth token (required)
        - X-User-Email: User's email address (optional, used for admin check)

    Query parameters:
        - provider: Vision provider ("openai" or "gemini", default: "openai")
        - model: Model name (default: "gpt-4.1-mini")
          Admin users can select from: gpt-4.1-mini, gpt-5-mini, gpt-5, gemini-2.5-flash, gemini-3-flash, gemini-3-pro
        - language: Local language for bilingual output (e.g., "serbian", "slovenian")
        - face_recognition: Whether to use face recognition ("true" or "false", default: "true")

    Returns:
        Rich metadata including:
        - Bilingual descriptions (English + local)
        - Scene analysis (setting, location, atmosphere)
        - Event detection (type, activity)
        - Media analysis (composition, subjects)
        - Recognized persons with face coordinates
        - Specific, useful tags
        - Usage/cost info (admin users only)
    """
    try:
        auth_token = request.headers.get('Authorization')
        validation_service = ValidationService()

        if not auth_token:
            return jsonify({'message': 'Unauthorized'}), 401

        if not validation_service.validate_auth_token(auth_token):
            return jsonify({'message': 'Unauthorized'}), 401

        if 'image' not in request.files:
            return jsonify({"error": "No image in request"}), 400

        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({"error": "No selected file"}), 400

        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.avif'}
        file_ext = os.path.splitext(image_file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Unsupported image format: {file_ext}. Supported formats: JPG, PNG, GIF, BMP, TIFF, WEBP, AVIF'
            }), 400

        # Get user email from header and check admin status
        # Frontend sends this from localStorage where it's stored after login
        user_email = request.headers.get('X-User-Email', '')
        is_admin = is_admin_user(user_email)

        # Get configuration from query params
        provider = request.args.get('provider', 'openai')
        model = request.args.get('model', 'gpt-4.1-mini')

        # Validate model selection - only admin can use other models
        ALLOWED_ADMIN_MODELS = [
            'gpt-4.1-mini', 'gpt-4.1', 'gpt-4o', 'gpt-5-mini', 'gpt-5',
            'gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro',
            'gemini-3-flash-preview', 'gemini-3-pro-preview'
        ]
        if not is_admin and model != 'gpt-4.1-mini':
            model = 'gpt-4.1-mini'  # Force default for non-admin users
        elif is_admin and model not in ALLOWED_ADMIN_MODELS:
            return jsonify({
                'error': f'Invalid model: {model}. Allowed models: {", ".join(ALLOWED_ADMIN_MODELS)}'
            }), 400

        local_language = request.args.get('language', None)
        use_face_recognition = request.args.get('face_recognition', 'true').lower() == 'true'

        # Get domain for face recognition
        domain = validation_service.get_domain()

        # Read image data
        image_data = image_file.read()

        # Import and use vision service
        from app.services.vision import VisionService

        vision_service = VisionService(
            provider=provider,
            model=model,
            local_language=local_language
        )

        # Run analysis
        if use_face_recognition:
            metadata = vision_service.analyze_image_with_face_recognition(
                image_data=image_data,
                domain=domain
            )
        else:
            metadata = vision_service.analyze_image(image_data=image_data)

        # Return result
        result = metadata.to_dict()
        result["legacy"] = metadata.to_legacy_format()

        # Log the recognition result if batch_id is provided (for batch processing)
        batch_id = request.args.get('batch_id', None)
        if batch_id and use_face_recognition:
            try:
                from app.services.batch_photo_logging_service import BatchPhotoLoggingService
                logging_service = BatchPhotoLoggingService()

                # Extract face recognition result from full metadata
                # The metadata object contains recognized_persons with full details
                recognized_persons = result.get('recognized_persons', [])

                # Get full face recognition result if available (includes all_detected_matches)
                face_rec_full = getattr(metadata, 'face_recognition_result', None) or {}

                face_rec_result = {
                    'recognized': len(recognized_persons) > 0,
                    'person': recognized_persons[0].get('name') if recognized_persons else 'Unknown',
                    'confidence': recognized_persons[0].get('confidence') if recognized_persons else None,
                    'recognized_persons': recognized_persons,
                    'face_count': len(recognized_persons),
                    # Include all detected matches for top 3 analysis
                    'all_detected_matches': face_rec_full.get('all_detected_matches', []),
                    'best_match': face_rec_full.get('best_match', {})
                }

                logging_service.log_recognition_result(
                    filename=image_file.filename,
                    domain=domain,
                    recognition_result=face_rec_result,
                    batch_id=batch_id
                )
            except Exception as log_error:
                logger.warning(f"Failed to log recognition result: {str(log_error)}")

        # Remove usage info for non-admin users
        if not is_admin and "usage" in result:
            del result["usage"]

        return jsonify({
            "success": True,
            "metadata": result,
            "is_admin": is_admin
        }), 200

    except Exception as e:
        logger.error(f"Error in analyze_image endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@image_routes.route('/vision/models', methods=['GET'])
def list_vision_models():
    """
    List available vision models.

    Query parameters:
        - provider: Filter by provider ("openai" or "gemini")

    Returns:
        Dict of provider -> list of available models
    """
    try:
        from app.services.vision import VisionService

        provider = request.args.get('provider', None)
        models = VisionService.list_models(provider)

        return jsonify({
            "success": True,
            "providers": VisionService.list_providers(),
            "models": models
        }), 200

    except Exception as e:
        logger.error(f"Error listing vision models: {str(e)}")
        return jsonify({'error': str(e)}), 500         

@image_routes.route('/batch-logs/<batch_id>', methods=['GET'])
def get_batch_log(batch_id: str):
    """
    GET /batch-logs/<batch_id>

    Retrieve detailed log for a batch with top 3 matches for each photo.
    """
    try:
        auth_token = request.headers.get('Authorization')
        validation_service = ValidationService()

        if not auth_token:
            return jsonify({'message': 'Unauthorized'}), 401

        if not validation_service.validate_auth_token(auth_token):
            return jsonify({'message': 'Unauthorized'}), 401

        from app.services.batch_photo_logging_service import BatchPhotoLoggingService
        logging_service = BatchPhotoLoggingService()

        batch_log = logging_service.get_batch_log(batch_id)

        if not batch_log:
            return jsonify({
                "success": False,
                "error": f"Batch {batch_id} not found"
            }), 404

        return jsonify({
            "success": True,
            **batch_log
        }), 200

    except Exception as e:
        logger.error(f"Error retrieving batch log: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@image_routes.route('/batch-logs/<batch_id>/export', methods=['GET'])
def export_batch_csv(batch_id: str):
    """
    GET /batch-logs/<batch_id>/export

    Export batch log to CSV format with top 3 matches.
    """
    try:
        auth_token = request.headers.get('Authorization')
        validation_service = ValidationService()

        if not auth_token:
            return jsonify({'message': 'Unauthorized'}), 401

        if not validation_service.validate_auth_token(auth_token):
            return jsonify({'message': 'Unauthorized'}), 401

        from app.services.batch_photo_logging_service import BatchPhotoLoggingService
        logging_service = BatchPhotoLoggingService()

        csv_file = logging_service.export_batch_to_csv(batch_id)

        if not csv_file or not os.path.exists(csv_file):
            return jsonify({
                "success": False,
                "error": f"Could not export batch {batch_id} to CSV"
            }), 404

        from flask import send_file
        return send_file(
            csv_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"batch_{batch_id}_results.csv"
        )

    except Exception as e:
        logger.error(f"Error exporting batch to CSV: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@image_routes.route('/batch-logs', methods=['GET'])
def list_batch_logs():
    """
    GET /batch-logs

    List all batch logs, sorted by date (newest first).
    """
    try:
        auth_token = request.headers.get('Authorization')
        validation_service = ValidationService()

        if not auth_token:
            return jsonify({'message': 'Unauthorized'}), 401

        if not validation_service.validate_auth_token(auth_token):
            return jsonify({'message': 'Unauthorized'}), 401

        from app.services.batch_photo_logging_service import BatchPhotoLoggingService
        logging_service = BatchPhotoLoggingService()

        limit = int(request.args.get('limit', 50))
        batches = logging_service.list_batches(limit=limit)

        return jsonify({
            "success": True,
            "total": len(batches),
            "batches": batches
        }), 200

    except Exception as e:
        logger.error(f"Error listing batch logs: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@image_routes.route('/manage-image', methods=['POST'])
def manage_image():
    """
    Endpoint for managing images (edit or delete)
    """
    try:
        auth_token = request.headers.get('Authorization')
        validation_service = ValidationService()

        if not auth_token:
            return jsonify({'message': 'Unauthorized'}), 401
        
        if not validation_service.validate_auth_token(auth_token):
            return jsonify({'message': 'Unauthorized'}), 401
        
        # Get JSON data from request, handling potential Content-Type issues
        try:
            data = request.get_json(force=True)  # force=True ignores Content-Type header
        except Exception as e:
            return jsonify({"error": f"Invalid JSON data: {str(e)}"}), 400
            
        if not data:
            return jsonify({"error": "No JSON data in request"}), 400
            
        # Validate required fields
        if 'filename' not in data:
            return jsonify({"error": "Missing required field: filename"}), 400
            
        if 'action' not in data:
            return jsonify({"error": "Missing required field: action"}), 400
            
        # Get domain from request body or fall back to validation service
        # This allows gallery to delete images from any domain the user is viewing
        domain = data.get('domain') or validation_service.get_domain()

        # Process based on action
        if data['action'] == 'delete':
            # Call controller to handle image deletion
            from app.controllers.image_management_controller import ImageManagementController
            result = ImageManagementController.handle_image_deletion(
                filename=data['filename'],
                domain=domain
            )
            return jsonify(result), 200
            
        elif data['action'] == 'edit':
            # Validate person field for edit action
            if 'person' not in data:
                return jsonify({"error": "Missing required field for edit action: person"}), 400
                
            # Call controller to handle image editing (placeholder)
            from app.controllers.image_management_controller import ImageManagementController
            result = ImageManagementController.handle_image_editing(
                filename=data['filename'],
                person=data['person'],
                domain=domain
            )
            return jsonify(result), 200
            
        else:
            return jsonify({"error": f"Invalid action: {data['action']}"}), 400
            
    except Exception as e:
        logger.error(f"Error in manage_image endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500         