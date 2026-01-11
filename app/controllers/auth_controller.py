from flask import request, jsonify
from app.services.email_token_service import EmailTokenService
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuthController:
    """Controller for handling authentication-related requests."""
    
    def __init__(self):
        """Initialize the controller with required services."""
        try:
            self.email_token_service = EmailTokenService()
        except ValueError as e:
            logger.error(f"Failed to initialize EmailTokenService: {str(e)}")
            raise
    
    def get_token_by_email(self) -> Dict[str, Any]:
        """
        Handle POST request to get token by email address.
        Supports multiple domains - returns single format for one result, array for multiple.
        
        Expected JSON payload:
        {
            "email": "rts@gmail.com"
        }
        
        Returns:
            JSON response with token(s) or error message
        """
        try:
            # Validate request content type
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Content-Type must be application/json'
                }), 400
            
            # Get JSON data from request
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided in request body'
                }), 400
            
            # Extract email from request
            email = data.get('email')
            
            if not email:
                return jsonify({
                    'success': False,
                    'error': 'Email field is required'
                }), 400
            
            # Get all tokens using the new multi-domain service method
            tokens_data, error_message = self.email_token_service.get_tokens_by_email(email)
            
            if error_message:
                # Log the attempt for security monitoring
                logger.warning(f"Failed token request for email: {email} - {error_message}")
                
                return jsonify({
                    'success': False,
                    'error': error_message
                }), 404 if 'not found' in error_message.lower() else 500
            
            # Log successful token retrieval
            logger.info(f"Token(s) successfully retrieved for email: {email} - {len(tokens_data)} domain(s)")
            
            # If only one result, return single format (backwards compatible)
            if len(tokens_data) == 1:
                return jsonify({
                    'success': True,
                    'data': {
                        'token': tokens_data[0]['token'],
                        'email': email.strip().lower()
                    }
                }), 200
            
            # If multiple results, return array format
            else:
                return jsonify({
                    'success': True,
                    'data': [
                        {
                            'token': token_data['token'],
                            'email': email.strip().lower(),
                            'domain': token_data['domain']
                        }
                        for token_data in tokens_data
                    ]
                }), 200
            
        except Exception as e:
            logger.error(f"Unexpected error in get_token_by_email: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Internal server error occurred'
            }), 500
    
    def validate_email_access(self) -> Dict[str, Any]:
        """
        Handle POST request to validate if email has access.
        
        Expected JSON payload:
        {
            "email": "rts@gmail.com"
        }
        
        Returns:
            JSON response indicating if email is authorized
        """
        try:
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Content-Type must be application/json'
                }), 400
            
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided in request body'
                }), 400
            
            email = data.get('email')
            
            if not email:
                return jsonify({
                    'success': False,
                    'error': 'Email field is required'
                }), 400
            
            # Validate email access
            has_access = self.email_token_service.validate_email_exists(email)
            key = self.email_token_service.get_key_by_email(email)
            
            return jsonify({
                'success': True,
                'data': {
                    'email': email.strip().lower(),
                    'has_access': has_access,
                    'key': key if has_access else None
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Unexpected error in validate_email_access: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Internal server error occurred'
            }), 500 