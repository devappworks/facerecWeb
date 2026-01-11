from flask import Blueprint
from app.controllers.auth_controller import AuthController
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint for authentication routes
auth_routes = Blueprint('auth_routes', __name__)

# Initialize controller
try:
    auth_controller = AuthController()
except ValueError as e:
    logger.error(f"Failed to initialize AuthController: {str(e)}")
    auth_controller = None


@auth_routes.route('/api/auth/token-by-email', methods=['POST'])
def get_token_by_email():
    """
    Endpoint to get authentication token(s) by email address.
    Supports multiple domains - returns single format for one result, array for multiple.
    
    Method: POST
    Content-Type: application/json
    
    Request body:
    {
        "email": "rts@gmail.com"
    }
    
    Responses:
    
    200 OK - Single token found and returned (backwards compatible)
    {
        "success": true,
        "data": {
            "token": "dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD",
            "email": "rts@gmail.com"
        }
    }
    
    200 OK - Multiple tokens found for different domains
    {
        "success": true,
        "data": [
            {
                "token": "dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD",
                "email": "rts@gmail.com",
                "domain": "rts"
            },
            {
                "token": "anotherTokenForDifferentDomain123456789",
                "email": "rts@gmail.com", 
                "domain": "rts_domain2"
            }
        ]
    }
    
    404 Not Found - Email not found in authorized users
    {
        "success": false,
        "error": "Email 'unknown@gmail.com' not found in authorized users"
    }
    
    500 Internal Server Error - Token mapping issue
    {
        "success": false,
        "error": "Token not found for key(s) 'rts, rts_domain2'. Please contact administrator"
    }
    
    400 Bad Request - Invalid request format
    {
        "success": false,
        "error": "Email field is required"
    }
    """
    if auth_controller is None:
        return {
            'success': False,
            'error': 'Authentication service is not available'
        }, 503
    
    return auth_controller.get_token_by_email()


@auth_routes.route('/api/auth/validate-email', methods=['POST'])
def validate_email_access():
    """
    Endpoint to validate if email has access without returning token.
    
    Method: POST
    Content-Type: application/json
    
    Request body:
    {
        "email": "rts@gmail.com"
    }
    
    Response:
    200 OK
    {
        "success": true,
        "data": {
            "email": "rts@gmail.com",
            "has_access": true,
            "key": "rts"
        }
    }
    """
    if auth_controller is None:
        return {
            'success': False,
            'error': 'Authentication service is not available'
        }, 503
    
    return auth_controller.validate_email_access()


@auth_routes.route('/api/auth/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for authentication service.
    
    Method: GET
    
    Response:
    200 OK
    {
        "success": true,
        "message": "Authentication service is running",
        "service_available": true
    }
    """
    return {
        'success': True,
        'message': 'Authentication service is running',
        'service_available': auth_controller is not None
    }, 200 