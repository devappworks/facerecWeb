from flask import Blueprint
import logging

# Import novog controllera (ne menjam postojeće!)
from app.controllers.batch_recognition_controller import BatchRecognitionController

logger = logging.getLogger(__name__)

# Kreiraj novi Blueprint za batch recognition
batch_recognition_bp = Blueprint('batch_recognition', __name__, url_prefix='/api/batch')

@batch_recognition_bp.route('/recognize', methods=['POST'])
def recognize_face_batch():
    """
    POST /api/batch/recognize
    
    Batch face recognition endpoint koji paralelno pretražuje kroz batch-eve.
    
    Form Data:
        - file: Image file (required)
        - domain: Domain string (required)
        - max_threads: Optional max threads (1-10, default 3)
    
    Returns:
        JSON response sa rezultatima prepoznavanja ili greškom
        
    Example curl:
        curl -X POST \
             -F "file=@test_image.jpg" \
             -F "domain=example.com" \
             -F "max_threads=3" \
             http://localhost:5000/api/batch/recognize
    """
    logger.info("Batch recognition endpoint called")
    return BatchRecognitionController.recognize_face_batch()

@batch_recognition_bp.route('/stats', methods=['GET'])
def get_batch_stats():
    """
    GET /api/batch/stats?domain=example.com
    
    Vraća statistike o batch strukturi za dati domain.
    
    Query Parameters:
        - domain: Domain string (required)
    
    Returns:
        JSON response sa batch statistikama
        
    Example curl:
        curl "http://localhost:5000/api/batch/stats?domain=example.com"
    """
    logger.info("Batch stats endpoint called")
    return BatchRecognitionController.get_batch_stats()

@batch_recognition_bp.route('/domains', methods=['GET'])
def list_batch_domains():
    """
    GET /api/batch/domains
    
    Lista svih domain-a koji imaju batch strukturu.
    
    Returns:
        JSON response sa listom domain-a
        
    Example curl:
        curl "http://localhost:5000/api/batch/domains"
    """
    logger.info("List batch domains endpoint called")
    return BatchRecognitionController.list_batch_domains()

@batch_recognition_bp.route('/info', methods=['GET'])
def get_batch_info():
    """
    GET /api/batch/info?domain=example.com
    
    Detaljne informacije o batch strukturi za dati domain.
    
    Query Parameters:
        - domain: Domain string (required)
    
    Returns:
        JSON response sa detaljnim batch informacijama
        
    Example curl:
        curl "http://localhost:5000/api/batch/info?domain=example.com"
    """
    logger.info("Batch info endpoint called")
    return BatchRecognitionController.get_batch_info()

@batch_recognition_bp.route('/health', methods=['GET'])
def batch_health_check():
    """
    GET /api/batch/health
    
    Health check endpoint za batch recognition API.
    
    Returns:
        JSON response sa statusom servisa
        
    Example curl:
        curl "http://localhost:5000/api/batch/health"
    """
    from app.services.batch_management_service import BatchManagementService
    
    try:
        # Jednostavna provera da li batch base path postoji
        import os
        batch_base_exists = os.path.exists(BatchManagementService.BATCH_BASE_PATH)
        
        # Prebrojimo domain-e sa batch strukturom
        available_domains = BatchManagementService.list_batch_domains()
        
        return {
            "status": "healthy",
            "service": "batch_recognition_api",
            "version": "v1",
            "batch_base_path_exists": batch_base_exists,
            "available_batch_domains": len(available_domains),
            "max_batch_threads": 3,
            "endpoints": [
                "POST /api/batch/recognize",
                "GET /api/batch/stats?domain=<domain>",
                "GET /api/batch/domains", 
                "GET /api/batch/info?domain=<domain>",
                "GET /api/batch/health"
            ]
        }, 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "batch_recognition_api", 
            "error": str(e)
        }, 500

# Error handlers za batch recognition blueprint
@batch_recognition_bp.errorhandler(404)
def batch_not_found(error):
    """Custom 404 handler za batch routes"""
    return {
        "status": "error",
        "message": "Batch recognition endpoint not found",
        "available_endpoints": [
            "POST /api/batch/recognize",
            "GET /api/batch/stats?domain=<domain>",
            "GET /api/batch/domains",
            "GET /api/batch/info?domain=<domain>",
            "GET /api/batch/health"
        ]
    }, 404

@batch_recognition_bp.errorhandler(405)
def batch_method_not_allowed(error):
    """Custom 405 handler za batch routes"""
    return {
        "status": "error",
        "message": "Method not allowed for this batch recognition endpoint",
        "help": "Check the API documentation for allowed methods"
    }, 405

@batch_recognition_bp.errorhandler(413)
def batch_file_too_large(error):
    """Custom 413 handler za prevelike fajlove"""
    return {
        "status": "error",
        "message": "File too large for batch recognition",
        "max_file_size": "10MB"
    }, 413 