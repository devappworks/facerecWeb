"""
Enhanced test recognition service with detailed logging
"""

import logging
import time
from typing import Dict, Optional
from io import BytesIO

from app.services.recognition_service import RecognitionService
from app.config.recognition_profiles import ProfileManager
from app.services.comparison_service import ComparisonService

# Configure logger with more detail
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure file logging
import os
log_dir = 'storage/logs'
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, 'ab_testing.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Also configure recognition_service logger to write to ab_testing.log
recognition_logger = logging.getLogger('app.services.recognition_service')
recognition_logger.addHandler(file_handler)
recognition_logger.setLevel(logging.DEBUG)


class TestRecognitionService:
    """
    Service to run recognition with different configurations for testing
    """

    @staticmethod
    def recognize_face_with_profile(image_bytes, domain: str, profile_name: str) -> Dict:
        """
        Run recognition with specific profile

        Args:
            image_bytes: Image data
            domain: Domain for database lookup
            profile_name: Profile to use (current, improved, ensemble)

        Returns:
            Recognition result
        """
        try:
            # Get configuration
            config = ProfileManager.get_config(profile_name)

            logger.info(f"===== Running recognition with profile: {profile_name} =====")
            logger.info(f"Model: {config.get('model_name')}, Threshold: {config.get('recognition_threshold')}, Detection Confidence: {config.get('detection_confidence_threshold')}")

            # Run recognition with this configuration
            result = RecognitionService.recognize_face_with_config(
                image_bytes, domain, config
            )

            logger.info(f"===== Profile {profile_name} result: status={result.get('status')} =====")
            logger.info(f"Total faces detected: {result.get('total_faces_detected', 'N/A')}")
            logger.info(f"Valid faces after filtering: {result.get('valid_faces_after_filtering', 'N/A')}")
            logger.info(f"Recognized faces: {len(result.get('recognized_faces', []))}")

            if result.get("status") == "error":
                logger.error(f"ERROR MESSAGE: {result.get('message', 'No message provided')}")
            elif result.get("status") == "no_faces":
                logger.warning(f"No faces detected - message: {result.get('message', 'No message provided')}")

            # Add profile metadata to result
            result["profile_used"] = {
                "name": profile_name,
                "model": config.get("model_name"),
                "threshold": config.get("recognition_threshold"),
                "detection_confidence": config.get("detection_confidence_threshold")
            }

            return result

        except Exception as e:
            logger.error(f"===== ERROR in recognize_face_with_profile ({profile_name}) =====")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception message: {str(e)}")
            logger.exception("Full traceback:")
            return {
                "status": "error",
                "message": str(e),
                "profile_used": profile_name
            }

    @staticmethod
    def recognize_face_comparison(image_bytes, domain: str, image_id: Optional[str] = None,
                                  ground_truth: Optional[str] = None) -> Dict:
        """
        Run recognition with both current and improved profiles and compare

        Args:
            image_bytes: Image data
            domain: Domain for database lookup
            image_id: Unique identifier for this image
            ground_truth: Known correct answer (optional)

        Returns:
            Comparison results
        """
        if image_id is None:
            import uuid
            image_id = str(uuid.uuid4())

        logger.info(f"\n\n========== Starting A/B comparison for image: {image_id} ==========")
        logger.info(f"Domain: {domain}")
        logger.info(f"Ground truth: {ground_truth}")

        # Run Pipeline A (current system)
        logger.info("\n----- Running Pipeline A (current system) -----")
        start_time_a = time.time()
        result_a = TestRecognitionService.recognize_face_with_profile(
            image_bytes, domain, "current"
        )
        time_a = time.time() - start_time_a
        result_a["processing_time"] = time_a
        logger.info(f"Pipeline A completed in {time_a:.2f}s")

        # Run Pipeline B (ArcFace system - state-of-the-art)
        logger.info("\n----- Running Pipeline B (ArcFace system) -----")
        start_time_b = time.time()

        # Need to reset BytesIO pointer if it was read
        if hasattr(image_bytes, 'seek'):
            image_bytes.seek(0)

        result_b = TestRecognitionService.recognize_face_with_profile(
            image_bytes, domain, "arcface"
        )
        time_b = time.time() - start_time_b
        result_b["processing_time"] = time_b
        logger.info(f"Pipeline B completed in {time_b:.2f}s")

        # Compare results
        logger.info("\n----- Comparing results -----")
        comparison = ComparisonService.compare_results(
            result_a, result_b, image_id, ground_truth
        )

        # Combine into single response
        response = {
            "image_id": image_id,
            "ground_truth": ground_truth,
            "pipeline_a_result": result_a,
            "pipeline_b_result": result_b,
            "comparison": comparison,
            "recommendation": TestRecognitionService._get_recommendation(comparison)
        }

        logger.info(f"========== A/B comparison complete for {image_id} ==========\n\n")

        return response

    @staticmethod
    def _get_recommendation(comparison: Dict) -> str:
        """
        Generate recommendation based on comparison
        """
        metrics = comparison["comparison_metrics"]

        if metrics["both_failed"]:
            return "Both pipelines failed to recognize face. Consider image quality."

        if metrics["only_b_succeeded"]:
            return "Pipeline B found a face that Pipeline A missed!"

        if metrics["only_a_succeeded"]:
            return "Pipeline A found a face but Pipeline B didn't. Review B configuration."

        if metrics["both_succeeded"]:
            if metrics["results_match"]:
                conf_diff = metrics.get("confidence_difference", 0)
                if conf_diff > 5:
                    return f"Both agree. Pipeline B has {conf_diff}% higher confidence."
                elif conf_diff < -5:
                    return f"Both agree. Pipeline A has {abs(conf_diff)}% higher confidence."
                else:
                    return "Both pipelines agree with similar confidence."
            else:
                return "Pipelines disagree on result. Manual review recommended."

        return "Unknown comparison state."
