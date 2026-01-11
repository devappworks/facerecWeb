"""
Batch Test Routes
API endpoints for managing batch testing sessions and results.

All endpoints require authentication via Authorization header.
"""

from flask import Blueprint, request, jsonify, current_app
from app.database import db
from app.models.batch_test import BatchTestSession, BatchTestResult
from app.decorators.auth import require_auth
from datetime import datetime
import uuid
import json

batch_test_bp = Blueprint('batch_test', __name__, url_prefix='/api/batch-tests')


@batch_test_bp.route('/sessions', methods=['POST'])
@require_auth
def create_session():
    """
    Create a new batch testing session.

    Request Body:
        {
            "name": string (optional) - User-defined name for the session
            "model": string (optional) - AI model being used
        }

    Response:
        JSON with session details including session_id
    """
    try:
        data = request.get_json() or {}
        user_email = request.headers.get('X-User-Email', 'unknown')
        domain = getattr(request, 'auth_domain', 'default')

        session = BatchTestSession(
            session_id=str(uuid.uuid4()),
            user_email=user_email,
            domain=domain,
            name=data.get('name'),
            model_used=data.get('model'),
            total_images=0,
            successful_count=0,
            failed_count=0
        )

        db.session.add(session)
        db.session.commit()

        return jsonify({
            'success': True,
            'session': session.to_dict()
        }), 201

    except Exception as e:
        current_app.logger.error(f"Error creating batch session: {str(e)}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


@batch_test_bp.route('/sessions/<session_id>/results', methods=['POST'])
@require_auth
def add_result(session_id):
    """
    Add a result to a batch testing session.

    Path Parameters:
        session_id: Session UUID

    Request Body:
        {
            "filename": string (required)
            "image_thumbnail": string (optional) - Base64 thumbnail
            "model_used": string (optional)
            "status": string (required) - 'success' or 'error'
            "error_message": string (optional)
            "recognized_persons": array (optional) - [{name, confidence}]
            "identified_persons": array (optional) - [{name, confidence}]
            "description": string (optional)
            "full_metadata": object (optional) - Complete API response
        }

    Response:
        JSON with result details
    """
    try:
        session = BatchTestSession.query.filter_by(session_id=session_id).first()

        if not session:
            return jsonify({
                'success': False,
                'message': 'Session not found'
            }), 404

        data = request.get_json()

        if not data or 'filename' not in data or 'status' not in data:
            return jsonify({
                'success': False,
                'message': 'Missing required fields: filename, status'
            }), 400

        result = BatchTestResult(
            session_id=session.id,
            filename=data['filename'],
            image_thumbnail=data.get('image_thumbnail'),
            model_used=data.get('model_used'),
            status=data['status'],
            error_message=data.get('error_message'),
            recognized_persons=json.dumps(data.get('recognized_persons', [])),
            identified_persons=json.dumps(data.get('identified_persons', [])),
            description=data.get('description'),
            full_metadata=json.dumps(data.get('full_metadata')) if data.get('full_metadata') else None
        )

        db.session.add(result)

        # Update session counts
        session.total_images += 1
        if data['status'] == 'success':
            session.successful_count += 1
        else:
            session.failed_count += 1

        db.session.commit()

        return jsonify({
            'success': True,
            'result': result.to_dict()
        }), 201

    except Exception as e:
        current_app.logger.error(f"Error adding batch result: {str(e)}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


@batch_test_bp.route('/sessions/<session_id>/complete', methods=['POST'])
@require_auth
def complete_session(session_id):
    """
    Mark a batch testing session as complete.

    Path Parameters:
        session_id: Session UUID

    Response:
        JSON with updated session details
    """
    try:
        session = BatchTestSession.query.filter_by(session_id=session_id).first()

        if not session:
            return jsonify({
                'success': False,
                'message': 'Session not found'
            }), 404

        session.completed_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            'success': True,
            'session': session.to_dict()
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error completing batch session: {str(e)}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


@batch_test_bp.route('/sessions', methods=['GET'])
@require_auth
def list_sessions():
    """
    List batch testing sessions for the authenticated user.

    Query Parameters:
        limit: int (optional, default 20) - Number of sessions to return
        offset: int (optional, default 0) - Pagination offset

    Response:
        JSON with list of sessions (without results)
    """
    try:
        user_email = request.headers.get('X-User-Email', 'unknown')
        domain = getattr(request, 'auth_domain', 'default')

        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)

        # Limit max results
        limit = min(limit, 100)

        sessions = BatchTestSession.query.filter_by(
            user_email=user_email,
            domain=domain
        ).order_by(
            BatchTestSession.created_at.desc()
        ).offset(offset).limit(limit).all()

        total = BatchTestSession.query.filter_by(
            user_email=user_email,
            domain=domain
        ).count()

        return jsonify({
            'success': True,
            'sessions': [s.to_dict() for s in sessions],
            'total': total,
            'limit': limit,
            'offset': offset
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error listing batch sessions: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


@batch_test_bp.route('/sessions/<session_id>', methods=['GET'])
@require_auth
def get_session(session_id):
    """
    Get a specific batch testing session with all results.

    Path Parameters:
        session_id: Session UUID

    Response:
        JSON with session details including all results
    """
    try:
        user_email = request.headers.get('X-User-Email', 'unknown')

        session = BatchTestSession.query.filter_by(session_id=session_id).first()

        if not session:
            return jsonify({
                'success': False,
                'message': 'Session not found'
            }), 404

        # Check ownership
        if session.user_email != user_email:
            return jsonify({
                'success': False,
                'message': 'Access denied'
            }), 403

        return jsonify({
            'success': True,
            'session': session.to_dict(include_results=True)
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error getting batch session: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


@batch_test_bp.route('/sessions/<session_id>', methods=['DELETE'])
@require_auth
def delete_session(session_id):
    """
    Delete a batch testing session and all its results.

    Path Parameters:
        session_id: Session UUID

    Response:
        JSON with deletion status
    """
    try:
        user_email = request.headers.get('X-User-Email', 'unknown')

        session = BatchTestSession.query.filter_by(session_id=session_id).first()

        if not session:
            return jsonify({
                'success': False,
                'message': 'Session not found'
            }), 404

        # Check ownership
        if session.user_email != user_email:
            return jsonify({
                'success': False,
                'message': 'Access denied'
            }), 403

        db.session.delete(session)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Session deleted successfully'
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error deleting batch session: {str(e)}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500


@batch_test_bp.route('/sessions/<session_id>/name', methods=['PATCH'])
@require_auth
def update_session_name(session_id):
    """
    Update the name of a batch testing session.

    Path Parameters:
        session_id: Session UUID

    Request Body:
        {
            "name": string (required)
        }

    Response:
        JSON with updated session details
    """
    try:
        user_email = request.headers.get('X-User-Email', 'unknown')

        session = BatchTestSession.query.filter_by(session_id=session_id).first()

        if not session:
            return jsonify({
                'success': False,
                'message': 'Session not found'
            }), 404

        # Check ownership
        if session.user_email != user_email:
            return jsonify({
                'success': False,
                'message': 'Access denied'
            }), 403

        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({
                'success': False,
                'message': 'Missing required field: name'
            }), 400

        session.name = data['name']
        db.session.commit()

        return jsonify({
            'success': True,
            'session': session.to_dict()
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error updating batch session name: {str(e)}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500
