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
            
        # Call the controller to handle the image

        tracking_token = ObjectDetectionController.generate_tracking_token()
        result = ObjectDetectionController.handle_detection_image(image_file, tracking_token)
        
        return jsonify(result), 202  # 202 Accepted
        
    except Exception as e:
        logger.error(f"Error in upload_for_detection endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500         

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
            
        # Get domain from validation service
        domain = validation_service.get_domain()
        
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