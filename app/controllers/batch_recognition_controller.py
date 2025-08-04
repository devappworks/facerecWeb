import logging
import time
from flask import request, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO

# Import postojećih servisa (ne menjam ih!)
from app.services.batch_recognition_service import BatchRecognitionService
from app.services.batch_management_service import BatchManagementService

logger = logging.getLogger(__name__)

class BatchRecognitionController:
    """
    Controller za batch face recognition endpoint.
    Koristi BatchRecognitionService za paralelno prepoznavanje kroz batch-eve.
    """
    
    # Dozvoljene ekstenzije slika (isti kao postojeći sistem)
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    @staticmethod
    def allowed_file(filename: str) -> bool:
        """Proverava da li je fajl ekstenzija dozvoljena"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in BatchRecognitionController.ALLOWED_EXTENSIONS
    
    @staticmethod
    def validate_image_file(file_data) -> tuple:
        """
        Validira upload-ovani fajl
        
        Returns:
            Tuple (is_valid: bool, error_message: str or None)
        """
        if not file_data:
            return False, "No file provided"
        
        # Proveri ime fajla ako postoji
        if hasattr(file_data, 'filename') and file_data.filename:
            if not BatchRecognitionController.allowed_file(file_data.filename):
                return False, f"File type not allowed. Allowed types: {', '.join(BatchRecognitionController.ALLOWED_EXTENSIONS)}"
        
        # Proveri veličinu fajla
        if hasattr(file_data, 'content_length') and file_data.content_length:
            if file_data.content_length > BatchRecognitionController.MAX_FILE_SIZE:
                return False, f"File too large. Maximum size: {BatchRecognitionController.MAX_FILE_SIZE / (1024*1024):.1f}MB"
        
        return True, None
    
    @staticmethod
    def recognize_face_batch():
        """
        API endpoint za batch face recognition
        
        Expects:
            - file: Image file (multipart/form-data)
            - domain: Domain string (form data)
            - max_threads: Optional max threads (form data)
            
        Returns:
            JSON response sa rezultatima prepoznavanja
        """
        try:
            start_time = time.time()
            logger.info("Batch face recognition API called")
            
            # Validacija request-a
            if 'file' not in request.files:
                return jsonify({
                    "status": "error",
                    "message": "No file part in request"
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    "status": "error", 
                    "message": "No file selected"
                }), 400
            
            # Validacija domain-a
            domain = request.form.get('domain')
            if not domain:
                return jsonify({
                    "status": "error",
                    "message": "Domain parameter is required"
                }), 400
            
            # Validacija fajla
            is_valid, error_message = BatchRecognitionController.validate_image_file(file)
            if not is_valid:
                return jsonify({
                    "status": "error",
                    "message": error_message
                }), 400
            
            # Opcioni max_threads parametar
            max_threads = request.form.get('max_threads')
            if max_threads:
                try:
                    max_threads = int(max_threads)
                    if max_threads < 1 or max_threads > 10:
                        return jsonify({
                            "status": "error",
                            "message": "max_threads must be between 1 and 10"
                        }), 400
                except ValueError:
                    return jsonify({
                        "status": "error",
                        "message": "max_threads must be a valid integer"
                    }), 400
            
            logger.info(f"Processing batch recognition for domain: {domain}")
            logger.info(f"File: {file.filename}, Max threads: {max_threads}")
            
            # Čitaj file bytes
            file.seek(0)  # Reset file pointer
            image_bytes = file.read()
            
            if len(image_bytes) == 0:
                return jsonify({
                    "status": "error",
                    "message": "Empty file provided"
                }), 400
            
            # Pozovi batch recognition service
            result = BatchRecognitionService.recognize_face_batch(
                image_bytes=BytesIO(image_bytes),
                domain=domain,
                max_threads=max_threads
            )
            
            # Debug log rezultata
            logger.info(f"Batch recognition result status: {result.get('status')}")
            logger.info(f"Batch recognition result message: {result.get('message')}")
            
            # Dodaj API specifične informacije
            api_info = {
                "api_version": "batch_v1",
                "request_processing_time": time.time() - start_time,
                "file_info": {
                    "filename": secure_filename(file.filename) if file.filename else "unknown",
                    "size_bytes": len(image_bytes)
                },
                "request_params": {
                    "domain": domain,
                    "max_threads": max_threads
                }
            }
            
            # Dodaj API info u rezultat
            if isinstance(result, dict):
                result["api_info"] = api_info
            
            # Odredi HTTP status kod na osnovu rezultata
            if result.get("status") == "success":
                http_status = 200
            elif result.get("status") == "error":
                # "No matches" nije prava greška - treba 200
                if "No matches" in result.get("message", ""):
                    http_status = 200
                    logger.info(f"No matches found - returning 200: {result.get('message')}")
                else:
                    http_status = 422  # Unprocessable Entity samo za prave greške
                    logger.warning(f"Returning 422 due to error: {result.get('message')}")
            else:
                http_status = 200  # Default
            
            logger.info(f"Batch recognition completed in {time.time() - start_time:.2f}s")
            return jsonify(result), http_status
            
        except Exception as e:
            error_msg = f"Unexpected error in batch recognition API: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return jsonify({
                "status": "error",
                "message": "Internal server error occurred during batch recognition",
                "error_details": str(e) if logger.isEnabledFor(logging.DEBUG) else None
            }), 500
    
    @staticmethod
    def get_batch_stats():
        """
        API endpoint za dobijanje statistika o batch strukturi
        
        Expects:
            - domain: Domain string (query parameter)
            
        Returns:
            JSON response sa batch statistikama
        """
        try:
            logger.info("Batch stats API called")
            
            # Validacija domain-a
            domain = request.args.get('domain')
            if not domain:
                return jsonify({
                    "status": "error",
                    "message": "Domain parameter is required"
                }), 400
            
            logger.info(f"Getting batch stats for domain: {domain}")
            
            # Pozovi service za statistike
            stats = BatchRecognitionService.get_batch_recognition_stats(domain)
            
            # Dodaj API informacije
            stats["api_info"] = {
                "api_version": "batch_v1",
                "endpoint": "batch_stats",
                "domain_requested": domain
            }
            
            # Odredi HTTP status
            if stats.get("status") == "error":
                http_status = 422
            elif stats.get("status") == "not_available":
                http_status = 404
            else:
                http_status = 200
            
            return jsonify(stats), http_status
            
        except Exception as e:
            error_msg = f"Error getting batch stats: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return jsonify({
                "status": "error",
                "message": "Error retrieving batch statistics",
                "error_details": str(e) if logger.isEnabledFor(logging.DEBUG) else None
            }), 500
    
    @staticmethod
    def list_batch_domains():
        """
        API endpoint za listing svih domain-a sa batch strukturom
        
        Returns:
            JSON response sa listom domain-a
        """
        try:
            logger.info("List batch domains API called")
            
            # Pozovi service za listing
            domains = BatchManagementService.list_batch_domains()
            
            result = {
                "status": "success",
                "message": f"Found {len(domains)} domains with batch structure",
                "total_domains": len(domains),
                "domains": domains,
                "api_info": {
                    "api_version": "batch_v1",
                    "endpoint": "list_batch_domains"
                }
            }
            
            return jsonify(result), 200
            
        except Exception as e:
            error_msg = f"Error listing batch domains: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return jsonify({
                "status": "error",
                "message": "Error listing batch domains",
                "error_details": str(e) if logger.isEnabledFor(logging.DEBUG) else None
            }), 500
    
    @staticmethod
    def get_batch_info():
        """
        API endpoint za detaljne informacije o batch strukturi za domain
        
        Expects:
            - domain: Domain string (query parameter)
            
        Returns:
            JSON response sa detaljnim batch informacijama
        """
        try:
            logger.info("Batch info API called")
            
            # Validacija domain-a
            domain = request.args.get('domain')
            if not domain:
                return jsonify({
                    "status": "error",
                    "message": "Domain parameter is required"
                }), 400
            
            logger.info(f"Getting batch info for domain: {domain}")
            
            # Pozovi service za informacije
            info = BatchManagementService.get_batch_info(domain)
            
            # Dodaj API informacije
            info["api_info"] = {
                "api_version": "batch_v1",
                "endpoint": "batch_info",
                "domain_requested": domain
            }
            
            # Odredi HTTP status
            if info.get("status") == "not_found":
                http_status = 404
            elif info.get("status") == "error":
                http_status = 422
            else:
                http_status = 200
            
            return jsonify(info), http_status
            
        except Exception as e:
            error_msg = f"Error getting batch info: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return jsonify({
                "status": "error",
                "message": "Error retrieving batch information",
                "error_details": str(e) if logger.isEnabledFor(logging.DEBUG) else None
            }), 500 