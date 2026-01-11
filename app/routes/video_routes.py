"""
Video Face Recognition Routes
API endpoints for processing videos and recognizing faces.

All endpoints require authentication via Authorization header.
"""

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.services.video_service import VideoService
from app.decorators.auth import require_auth
import os

video_bp = Blueprint('video', __name__, url_prefix='/api/video')


def allowed_video_file(filename):
    """Check if file is an allowed video format"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@video_bp.route('/upload', methods=['POST'])
@require_auth
def upload_video():
    """
    Upload and process video file synchronously.

    Request:
        - file: Video file (multipart/form-data)
        - domain: Domain for face recognition (form field)
        - interval_seconds: Frame extraction interval (optional, default 3.0)

    Response:
        JSON with complete processing results
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "message": "No file provided"
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                "success": False,
                "message": "No file selected"
            }), 400

        if not allowed_video_file(file.filename):
            return jsonify({
                "success": False,
                "message": "Invalid file type. Allowed: mp4, avi, mov, mkv, webm, flv, wmv"
            }), 400

        # Get parameters
        domain = request.form.get('domain', 'serbia')
        interval_seconds = float(request.form.get('interval_seconds', 3.0))

        # Validate interval
        if interval_seconds < 0.1 or interval_seconds > 60:
            return jsonify({
                "success": False,
                "message": "interval_seconds must be between 0.1 and 60"
            }), 400

        # Read video bytes
        video_bytes = file.read()

        # Get file size
        file_size_mb = len(video_bytes) / (1024 * 1024)

        current_app.logger.info(
            f"Received video upload: {file.filename}, "
            f"size: {file_size_mb:.2f} MB, "
            f"domain: {domain}, "
            f"interval: {interval_seconds}s"
        )

        # Check file size limit (100MB default)
        max_size_mb = current_app.config.get('MAX_VIDEO_SIZE_MB', 100)
        if file_size_mb > max_size_mb:
            return jsonify({
                "success": False,
                "message": f"Video too large. Maximum size: {max_size_mb} MB"
            }), 413

        # Process video (GPU by default, falls back to CPU if unavailable)
        video_service = VideoService()
        result = video_service.process_video_gpu(
            video_bytes,
            secure_filename(file.filename),
            domain,
            interval_seconds
        )

        status_code = 200 if result.get('success') else 500

        return jsonify(result), status_code

    except Exception as e:
        current_app.logger.error(f"Error processing video upload: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500


@video_bp.route('/upload-async', methods=['POST'])
@require_auth
def upload_video_async():
    """
    Upload video and process asynchronously in background.

    Request:
        - file: Video file (multipart/form-data)
        - domain: Domain for face recognition (form field)
        - interval_seconds: Frame extraction interval (optional, default 3.0)

    Response:
        JSON with video_id for tracking progress
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "message": "No file provided"
            }), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({
                "success": False,
                "message": "No file selected"
            }), 400

        if not allowed_video_file(file.filename):
            return jsonify({
                "success": False,
                "message": "Invalid file type. Allowed: mp4, avi, mov, mkv, webm, flv, wmv"
            }), 400

        # Get parameters
        domain = request.form.get('domain', 'serbia')
        interval_seconds = float(request.form.get('interval_seconds', 3.0))

        # Validate interval
        if interval_seconds < 0.1 or interval_seconds > 60:
            return jsonify({
                "success": False,
                "message": "interval_seconds must be between 0.1 and 60"
            }), 400

        # Read video bytes
        video_bytes = file.read()

        # Get file size
        file_size_mb = len(video_bytes) / (1024 * 1024)

        current_app.logger.info(
            f"Received async video upload: {file.filename}, "
            f"size: {file_size_mb:.2f} MB, "
            f"domain: {domain}, "
            f"interval: {interval_seconds}s"
        )

        # Check file size limit
        max_size_mb = current_app.config.get('MAX_VIDEO_SIZE_MB', 100)
        if file_size_mb > max_size_mb:
            return jsonify({
                "success": False,
                "message": f"Video too large. Maximum size: {max_size_mb} MB"
            }), 413

        # Start async processing
        video_service = VideoService()
        video_id = video_service.process_video_async(
            video_bytes,
            secure_filename(file.filename),
            domain,
            interval_seconds
        )

        return jsonify({
            "success": True,
            "message": "Video uploaded successfully. Processing in background.",
            "video_id": video_id,
            "status_endpoint": f"/api/video/status/{video_id}"
        }), 202  # 202 Accepted

    except Exception as e:
        current_app.logger.error(f"Error processing async video upload: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500


@video_bp.route('/status/<video_id>', methods=['GET'])
@require_auth
def get_video_status(video_id):
    """
    Get processing status and results for a video.

    Path Parameters:
        video_id: Video identifier

    Response:
        JSON with processing status and results.

    Processing Stages:
        - "processing": Video uploaded, no results yet
        - "gemini_complete": Gemini analysis done, face recognition in progress
        - "gemini_failed": Gemini failed, face recognition in progress
        - "complete": Both Gemini and face recognition complete
        - "extraction_failed": Frame extraction failed
        - "face_recognition_failed": Face recognition failed (Gemini may have succeeded)

    The response includes:
        - gemini_analysis: Visual analysis, quotes, body language (available early)
        - face_recognition: Person identification from database (available later)
        - person_correlation: Cross-reference between Gemini and face rec
    """
    try:
        video_service = VideoService()
        result = video_service.get_video_result(video_id)

        if result is None:
            # Check if video exists but not processed yet
            video_file_exists = False
            for filename in os.listdir(video_service.VIDEO_STORAGE):
                if video_id in filename:
                    video_file_exists = True
                    break

            if video_file_exists:
                return jsonify({
                    "success": False,
                    "video_id": video_id,
                    "status": "processing",
                    "processing_stage": "started",
                    "message": "Video is being processed. Gemini analysis will be available shortly..."
                }), 202
            else:
                return jsonify({
                    "success": False,
                    "video_id": video_id,
                    "status": "not_found",
                    "message": "Video not found"
                }), 404

        # Determine status based on processing_stage
        processing_stage = result.get("processing_stage", "complete")

        if processing_stage == "complete":
            status = "completed"
            http_code = 200
        elif processing_stage in ["gemini_complete", "gemini_failed"]:
            # Gemini done, face recognition still in progress
            status = "partial"
            http_code = 202
        elif processing_stage in ["extraction_failed", "face_recognition_failed"]:
            # Processing encountered errors but may have partial results
            status = "partial_error"
            http_code = 200
        else:
            status = "processing"
            http_code = 202

        return jsonify({
            "success": True,
            "video_id": video_id,
            "status": status,
            **result
        }), http_code

    except Exception as e:
        current_app.logger.error(f"Error getting video status: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500


@video_bp.route('/info', methods=['GET'])
@require_auth
def get_video_info():
    """
    Get information about video processing capabilities.

    Response:
        JSON with supported formats, limits, and configuration
    """
    max_size_mb = current_app.config.get('MAX_VIDEO_SIZE_MB', 100)

    return jsonify({
        "success": True,
        "supported_formats": ["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"],
        "max_file_size_mb": max_size_mb,
        "default_interval_seconds": 3.0,
        "min_interval_seconds": 0.1,
        "max_interval_seconds": 60.0,
        "endpoints": {
            "upload": "/api/video/upload",
            "upload_async": "/api/video/upload-async",
            "status": "/api/video/status/{video_id}",
            "info": "/api/video/info"
        }
    }), 200


@video_bp.route('/upload-gpu', methods=['POST'])
@require_auth
def upload_video_gpu():
    """
    Upload and process video using GPU acceleration.
    Falls back to CPU if GPU unavailable.

    Request:
        - file: Video file (multipart/form-data)
        - domain: Domain for face recognition (form field)
        - interval_seconds: Frame extraction interval (optional, default 3.0)

    Response:
        JSON with complete processing results
    """
    try:
        # Validation (same as upload())
        if 'file' not in request.files:
            return jsonify({"success": False, "message": "No file provided"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"success": False, "message": "No file selected"}), 400

        if not allowed_video_file(file.filename):
            return jsonify({
                "success": False,
                "message": "Invalid file type. Allowed: mp4, avi, mov, mkv, webm, flv, wmv"
            }), 400

        domain = request.form.get('domain', 'serbia')
        interval_seconds = float(request.form.get('interval_seconds', 3.0))

        if interval_seconds < 0.1 or interval_seconds > 60:
            return jsonify({
                "success": False,
                "message": "interval_seconds must be between 0.1 and 60"
            }), 400

        video_bytes = file.read()
        file_size_mb = len(video_bytes) / (1024 * 1024)

        max_size_mb = current_app.config.get('MAX_VIDEO_SIZE_MB', 100)
        if file_size_mb > max_size_mb:
            return jsonify({
                "success": False,
                "message": f"Video too large. Maximum: {max_size_mb} MB"
            }), 413

        current_app.logger.info(
            f"GPU video upload: {file.filename}, {file_size_mb:.2f} MB, "
            f"domain: {domain}, interval: {interval_seconds}s"
        )

        # Process with GPU
        video_service = VideoService()
        result = video_service.process_video_gpu(
            video_bytes,
            secure_filename(file.filename),
            domain,
            interval_seconds
        )

        return jsonify(result), 200 if result.get('success') else 500

    except Exception as e:
        current_app.logger.error(f"GPU video upload error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@video_bp.route('/gpu/health', methods=['GET'])
@require_auth
def gpu_health():
    """Check GPU service health"""
    from app.services.modal_service import ModalService

    health = ModalService.check_health()
    status_code = 200 if health.get('status') == 'healthy' else 503

    return jsonify({
        "success": health.get('status') == 'healthy',
        **health
    }), status_code


@video_bp.route('/retry/<video_id>', methods=['POST'])
@require_auth
def retry_video_processing(video_id):
    """
    Retry face recognition for a video stuck in 'gemini_complete' stage.

    This is useful when the background thread was killed (e.g., server restart)
    before face recognition could complete.

    Path Parameters:
        video_id: Video identifier

    Response:
        JSON with retry status
    """
    try:
        video_service = VideoService()
        result = video_service.retry_face_recognition(video_id)

        if result.get('success'):
            return jsonify(result), 202  # Accepted
        else:
            return jsonify(result), 400

    except Exception as e:
        current_app.logger.error(f"Error retrying video {video_id}: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500
