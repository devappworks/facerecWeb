"""Modal GPU Service Client - Wrapper for calling Modal from Flask

This is the client-side component of the hybrid GPU/CPU architecture:
- Sends frames to Modal for GPU-accelerated embedding extraction
- Returns embeddings for server-side matching
"""
import modal
import base64
import logging
import time
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ModalService:
    """Client for Modal GPU face embedding extraction"""

    APP_NAME = "facereco-gpu"
    CLASS_NAME = "FaceRecognitionGPU"
    MAX_BATCH_SIZE = 16
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2.0

    _service = None
    _available = None

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if Modal GPU is enabled via config"""
        return os.getenv("MODAL_GPU_ENABLED", "true").lower() == "true"

    @classmethod
    def get_service(cls):
        """Get or create Modal service instance"""
        if cls._service is None:
            try:
                # Use Modal 1.x from_name API and instantiate the class
                ServiceClass = modal.Cls.from_name(cls.APP_NAME, cls.CLASS_NAME)
                cls._service = ServiceClass()  # Instantiate the class
                cls._available = True
                logger.info(f"Modal service connected: {cls.APP_NAME}/{cls.CLASS_NAME}")
            except Exception as e:
                cls._available = False
                logger.warning(f"Modal service unavailable: {e}")
        return cls._service

    @classmethod
    def is_available(cls) -> bool:
        """Check if Modal service is available"""
        if not cls.is_enabled():
            return False
        if cls._available is None:
            cls.get_service()
        return cls._available or False

    @classmethod
    def check_health(cls) -> Dict[str, Any]:
        """Check Modal GPU service health"""
        try:
            service = cls.get_service()
            if service is None:
                return {"status": "unavailable"}
            return service.health_check.remote()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @classmethod
    def extract_embeddings(
        cls,
        frames: List[bytes],
        model_name: str = "ArcFace"
    ) -> Dict[str, Any]:
        """
        Extract face embeddings from frames using Modal GPU.

        Args:
            frames: List of frame bytes (JPEG encoded)
            model_name: Face recognition model to use

        Returns:
            Dict with success status and embeddings per frame
        """
        start_time = time.time()

        if not cls.is_available():
            return {"success": False, "use_fallback": True, "message": "Modal unavailable"}

        try:
            service = cls.get_service()

            # Encode frames to base64
            frames_b64 = [base64.b64encode(f).decode('utf-8') for f in frames]

            logger.info(f"Sending {len(frames_b64)} frames to Modal GPU for embedding extraction")

            # Process in batches
            all_results = []
            for i in range(0, len(frames_b64), cls.MAX_BATCH_SIZE):
                batch = frames_b64[i:i + cls.MAX_BATCH_SIZE]

                for attempt in range(cls.RETRY_ATTEMPTS):
                    try:
                        result = service.extract_embeddings_batch.remote(
                            frames_b64=batch,
                            model_name=model_name
                        )

                        if result.get('success'):
                            for r in result['results']:
                                r['frame_index'] += i  # Adjust for batch offset
                            all_results.extend(result['results'])
                            break
                    except Exception as e:
                        if attempt < cls.RETRY_ATTEMPTS - 1:
                            logger.warning(f"Retry {attempt + 1}: {e}")
                            time.sleep(cls.RETRY_DELAY)
                        else:
                            raise

            total_time = time.time() - start_time
            fps = len(frames) / total_time if total_time > 0 else 0

            return {
                "success": True,
                "frames_processed": len(frames),
                "processing_time": round(total_time, 2),
                "fps": round(fps, 2),
                "results": all_results
            }

        except Exception as e:
            logger.error(f"Modal GPU embedding extraction failed: {e}")
            return {"success": False, "use_fallback": True, "message": str(e)}

    @classmethod
    def reset_connection(cls):
        """Reset connection (for testing)"""
        cls._service = None
        cls._available = None


# Keep old method name for backward compatibility during transition
ModalService.process_frames = ModalService.extract_embeddings
