from app.services.recognition_service import RecognitionService
from app.services.recognition_service_pgvector import PgVectorRecognitionService
import logging
import os

logger = logging.getLogger(__name__)

# Configuration flags
VECTOR_DB_ENABLED = os.getenv('VECTOR_DB_ENABLED', 'false').lower() == 'true'

class RecognitionController:
    @staticmethod
    def recognize_face(image_bytes, domain, collect_diagnostics=False):
        """
        Main recognition controller - routes to PKL or pgvector based on config.

        Config options:
        - VECTOR_DB_ENABLED=false: Use PKL only (default, legacy behavior)
        - VECTOR_DB_ENABLED=true: Use pgvector only (new fast database search)
        """
        try:
            # Route to pgvector or PKL based on configuration
            if VECTOR_DB_ENABLED:
                logger.info(f"[PGVECTOR] Using pgvector for recognition, domain={domain}")
                return PgVectorRecognitionService.recognize_face(image_bytes, domain, collect_diagnostics=collect_diagnostics)
            else:
                logger.info(f"[PKL] Using PKL for recognition, domain={domain}")
                return RecognitionService.recognize_face(image_bytes, domain, collect_diagnostics=collect_diagnostics)

        except Exception as e:
            logger.error(f"Error in RecognitionController.recognize_face: {str(e)}", exc_info=True)

            # Fallback to PKL if pgvector fails
            if VECTOR_DB_ENABLED:
                logger.warning("[PGVECTOR] Failed, falling back to PKL")
                try:
                    return RecognitionService.recognize_face(image_bytes, domain, collect_diagnostics=collect_diagnostics)
                except Exception as fallback_error:
                    logger.error(f"PKL fallback also failed: {fallback_error}")
                    raise
            else:
                raise
