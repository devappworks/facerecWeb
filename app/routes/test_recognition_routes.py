from flask import Blueprint, jsonify, request
from app.services.test_recognition_service import TestRecognitionService
from app.services.validation_service import ValidationService
from app.services.metrics_reporting_service import MetricsReportingService
import logging

test_recognition_routes = Blueprint('test_recognition', __name__)
logger = logging.getLogger(__name__)


@test_recognition_routes.route('/api/test/recognize', methods=['POST'])
def test_recognize():
    """
    Test endpoint that runs both pipelines and compares results
    """
    try:
        # Authentication
        auth_token = request.headers.get('Authorization')
        validation_service = ValidationService()

        if not auth_token:
            return jsonify({'message': 'Unauthorized'}), 401

        if not validation_service.validate_auth_token(auth_token):
            return jsonify({'message': 'Unauthorized'}), 401

        # Check for image
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'No selected file'}), 400

        # Validate file type - only accept supported image formats
        import os
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif'}
        file_ext = os.path.splitext(image_file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Unsupported image format: {file_ext}. Supported formats: JPG, PNG, GIF, BMP, TIFF'
            }), 400

        # Optional parameters
        image_id = request.form.get('image_id')
        ground_truth = request.form.get('ground_truth')  # For testing with known answers

        # Get domain
        domain = validation_service.get_domain()

        # Read image
        image_bytes = image_file.read()

        # Run comparison
        result = TestRecognitionService.recognize_face_comparison(
            image_bytes, domain, image_id, ground_truth
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in test_recognize endpoint: {str(e)}")
        logger.exception("Full traceback:")  # This will log the full stack trace
        return jsonify({'error': str(e)}), 500


@test_recognition_routes.route('/api/test/metrics/daily', methods=['GET'])
def get_daily_metrics():
    """
    Get daily metrics report
    """
    try:
        date = request.args.get('date')  # Optional: YYYY-MM-DD

        report = MetricsReportingService.generate_daily_report(date)

        return jsonify(report)

    except Exception as e:
        logger.error(f"Error in get_daily_metrics endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@test_recognition_routes.route('/api/test/metrics/weekly', methods=['GET'])
def get_weekly_metrics():
    """
    Get weekly metrics report
    """
    try:
        report = MetricsReportingService.generate_weekly_report()

        return jsonify(report)

    except Exception as e:
        logger.error(f"Error in get_weekly_metrics endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@test_recognition_routes.route('/api/test/health', methods=['GET'])
def test_health():
    """
    Health check for test system
    """
    try:
        from app.config.recognition_profiles import ProfileManager

        return jsonify({
            "status": "operational",
            "available_profiles": ProfileManager.list_profiles(),
            "comparison_logging": "enabled",
            "metrics_reporting": "enabled"
        })

    except Exception as e:
        logger.error(f"Error in test_health endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500
