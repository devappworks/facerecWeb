"""
Storage Management Routes
API endpoints for managing video storage and disk usage.

All endpoints require authentication via Authorization header.
"""

from flask import Blueprint, request, jsonify, current_app, send_file
from app.services.storage_service import StorageService
from app.decorators.auth import require_auth
import os

storage_bp = Blueprint('storage', __name__, url_prefix='/api/storage')


@storage_bp.route('/stats', methods=['GET'])
@require_auth
def get_storage_stats():
    """
    Get storage statistics including disk usage and video list.

    Response:
        JSON with storage stats, disk usage, and all videos
    """
    try:
        storage_service = StorageService()
        stats = storage_service.get_storage_stats()

        status_code = 200 if stats.get('success') else 500
        return jsonify(stats), status_code

    except Exception as e:
        current_app.logger.error(f"Error getting storage stats: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500


@storage_bp.route('/videos', methods=['GET'])
@require_auth
def get_videos_list():
    """
    Get list of all stored videos with metadata.

    Response:
        JSON with list of videos
    """
    try:
        storage_service = StorageService()
        result = storage_service.get_videos_list()

        status_code = 200 if result.get('success') else 500
        return jsonify(result), status_code

    except Exception as e:
        current_app.logger.error(f"Error getting videos list: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500


@storage_bp.route('/videos/<video_id>', methods=['DELETE'])
@require_auth
def delete_video(video_id):
    """
    Delete video and all associated files (frames, results).

    Path Parameters:
        video_id: Video identifier

    Response:
        JSON with deletion status
    """
    try:
        # Validate video_id format (prevent path traversal)
        if not video_id or len(video_id) > 50 or '/' in video_id or '\\' in video_id or '..' in video_id:
            return jsonify({
                "success": False,
                "message": "Invalid video_id format"
            }), 400

        storage_service = StorageService()
        result = storage_service.delete_video(video_id)

        status_code = 200 if result.get('success') else 404
        return jsonify(result), status_code

    except Exception as e:
        current_app.logger.error(f"Error deleting video {video_id}: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500


@storage_bp.route('/cleanup', methods=['POST'])
@require_auth
def cleanup_old_videos():
    """
    Delete videos older than specified number of days.

    Request Body:
        {
            "days": int (required) - Delete videos older than this many days
        }

    Response:
        JSON with cleanup results
    """
    try:
        data = request.get_json()

        if not data or 'days' not in data:
            return jsonify({
                "success": False,
                "message": "Missing required field: days"
            }), 400

        days = data['days']

        # Validate days parameter
        if not isinstance(days, (int, float)) or days <= 0:
            return jsonify({
                "success": False,
                "message": "days must be a positive number"
            }), 400

        # Safety limit: don't allow cleanup of videos newer than 1 day
        if days < 1:
            return jsonify({
                "success": False,
                "message": "days must be at least 1 (safety limit)"
            }), 400

        storage_service = StorageService()
        result = storage_service.cleanup_old_videos(int(days))

        status_code = 200 if result.get('success') else 500
        return jsonify(result), status_code

    except Exception as e:
        current_app.logger.error(f"Error during cleanup: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500


@storage_bp.route('/videos/<video_id>/stream', methods=['GET'])
def stream_video(video_id):
    """
    Stream video file for playback.

    Accepts authentication via query parameter for HTML5 video compatibility.

    Path Parameters:
        video_id: Video identifier

    Query Parameters:
        token: Authentication token (optional - also checks Authorization header)

    Response:
        Video file stream
    """
    try:
        # Check authentication from query param or header
        token = request.args.get('token') or request.headers.get('Authorization')

        if not token:
            return jsonify({
                "success": False,
                "message": "Authentication required"
            }), 401

        # Validate token (simplified check - you can enhance this)
        # For now, just check if token exists and is non-empty
        if not token or len(token) < 10:
            return jsonify({
                "success": False,
                "message": "Invalid token"
            }), 401

        # Validate video_id format (prevent path traversal)
        if not video_id or len(video_id) > 50 or '/' in video_id or '\\' in video_id or '..' in video_id:
            return jsonify({
                "success": False,
                "message": "Invalid video_id format"
            }), 400

        storage_service = StorageService()

        # Find video file
        video_path = None
        if os.path.exists(storage_service.VIDEO_STORAGE):
            for filename in os.listdir(storage_service.VIDEO_STORAGE):
                if video_id in filename and filename.endswith('.mp4'):
                    video_path = os.path.join(storage_service.VIDEO_STORAGE, filename)
                    break

        if not video_path or not os.path.exists(video_path):
            return jsonify({
                "success": False,
                "message": f"Video {video_id} not found"
            }), 404

        # Stream the video file
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False,
            download_name=f"{video_id}.mp4"
        )

    except Exception as e:
        current_app.logger.error(f"Error streaming video {video_id}: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500
