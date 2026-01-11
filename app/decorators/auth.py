"""
Authentication decorators for protecting API routes.
"""

from functools import wraps
from flask import request, jsonify
from app.services.validation_service import ValidationService


def require_auth(f):
    """
    Decorator that requires valid authentication token.

    Token can be provided in:
    - Authorization header (preferred)
    - X-Auth-Token header

    The validated domain is stored in request.auth_domain for use by the route.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_token = request.headers.get('Authorization') or request.headers.get('X-Auth-Token')

        if not auth_token:
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'message': 'Please provide a valid token in the Authorization header'
            }), 401

        validation_service = ValidationService()

        if not validation_service.validate_auth_token(auth_token):
            return jsonify({
                'success': False,
                'error': 'Invalid or expired token',
                'message': 'The provided authentication token is not valid'
            }), 401

        # Store the validated domain for use in the route
        request.auth_domain = validation_service.get_domain()

        return f(*args, **kwargs)

    return decorated_function


def require_auth_optional_domain(f):
    """
    Decorator that requires auth but allows domain override.

    Uses the authenticated domain by default, but allows request to specify
    a different domain if the token has access to it.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_token = request.headers.get('Authorization') or request.headers.get('X-Auth-Token')

        if not auth_token:
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'message': 'Please provide a valid token in the Authorization header'
            }), 401

        validation_service = ValidationService()

        if not validation_service.validate_auth_token(auth_token):
            return jsonify({
                'success': False,
                'error': 'Invalid or expired token',
                'message': 'The provided authentication token is not valid'
            }), 401

        # Get domain from token
        token_domain = validation_service.get_domain()

        # Allow domain override from request (for multi-domain tokens in future)
        # For now, we use the token's domain
        request.auth_domain = token_domain

        return f(*args, **kwargs)

    return decorated_function
