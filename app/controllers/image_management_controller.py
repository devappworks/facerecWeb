import logging
import os
import threading
from app.services.image_management_service import ImageManagementService

logger = logging.getLogger(__name__)

class ImageManagementController:
    """
    Controller for handling image management operations (edit, delete)
    """
    
    @staticmethod
    def handle_image_deletion(filename, domain):
        """
        Handle the deletion of an image
        
        Args:
            filename: The filename of the image to delete
            domain: The domain identifier
            
        Returns:
            dict: Result of the operation
        """
        try:
            # Create service instance
            management_service = ImageManagementService()
            
            # Delete the image
            result = management_service.delete_image(filename, domain)
            
            # Get path to test image for face recognition
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            test_image_path = os.path.join(script_dir, 'scripts', 'test_face.JPG')
            
            # Check if test image exists
            if not os.path.exists(test_image_path):
                logger.warning(f"Test image does not exist at path: {test_image_path}")
                return {
                    "success": True,
                    "message": f"Image successfully deleted: {filename}, but face recognition update skipped (test image not found)",
                    "data": result,
                    "recognition_started": False
                }
            
            # # Start background thread for face recognition
            # thread = threading.Thread(
            #     target=ImageManagementController._run_face_recognition_in_background,
            #     args=(test_image_path, domain)
            # )
            # thread.daemon = True
            # thread.start()
            
            return {
                "success": True,
                "message": f"Image successfully deleted: {filename}.",
                "data": result,
                "recognition_started": True
            }
            
        except Exception as e:
            logger.error(f"Error deleting image: {str(e)}")
            raise
    
    @staticmethod
    def _run_face_recognition_in_background(test_image_path, domain):
        """
        Run face recognition in a background thread
        
        Args:
            test_image_path: Path to the test image
            domain: The domain identifier
        """
        try:
            logger.info(f"Starting background face recognition for domain: {domain}")
            
            # Import here to avoid circular imports
            from app.controllers.recognition_controller import RecognitionController
            
            # Read test image
            with open(test_image_path, 'rb') as f:
                test_image_bytes = f.read()
            
            # Run face recognition to update the .pkl file
            recognition_result = RecognitionController.recognize_face(test_image_bytes, domain)
            
            logger.info(f"Background face recognition completed for domain: {domain}")
            logger.debug(f"Recognition result: {recognition_result}")
            
        except Exception as e:
            logger.error(f"Error in background face recognition for domain {domain}: {str(e)}")
    
    @staticmethod
    def handle_image_editing(filename, person, domain):
        """
        Handle the editing of an image's person name
        
        Args:
            filename: The filename of the image to edit
            person: The new person name for the image
            domain: The domain identifier
            
        Returns:
            dict: Result of the operation
        """
        try:
            # Create service instance
            management_service = ImageManagementService()
            
            # Edit the image metadata
            result = management_service.edit_image(filename, person, domain)
            
            # If edit was successful, trigger face recognition update in background
            if result.get("edited", False):
                # Get path to test image for face recognition
                script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                test_image_path = os.path.join(script_dir, 'scripts', 'test_face.JPG')
                
                # Check if test image exists
                if os.path.exists(test_image_path):
                    # Start background thread for face recognition
                    # thread = threading.Thread(
                    #     target=ImageManagementController._run_face_recognition_in_background,
                    #     args=(test_image_path, domain)
                    # )
                    # thread.daemon = True
                    # thread.start()
                    recognition_started = True
                else:
                    logger.warning(f"Test image does not exist at path: {test_image_path}")
                    recognition_started = False
            else:
                recognition_started = False
            
            # Create response with new_filename at the top level for Laravel compatibility
            response = {
                "success": result.get("edited", False),
                "message": result.get("message", "Unknown error occurred"),
                "data": result,
                "recognition_started": recognition_started
            }
            
            # Add new_filename at the top level if it exists in the result
            if "new_filename" in result:
                response["new_filename"] = result["new_filename"]
            
            return response
            
        except Exception as e:
            logger.error(f"Error editing image: {str(e)}")
            raise 