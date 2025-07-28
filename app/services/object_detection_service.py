import os
import uuid
import logging
import json
from PIL import Image
from io import BytesIO
import datetime
import base64
import pusher

from app.services.openai_service import OpenAIService

logger = logging.getLogger(__name__)

pusher_client = pusher.Pusher(
  app_id='1999185',
  key='3a3e4e065f86231ecf84',
  secret='acbb3d46dff78d95be40',
  cluster='eu',
  ssl=True
)

class ObjectDetectionService:
    """
    Service for handling object detection image processing and storage
    """
    
    def __init__(self):
        self.storage_path = 'storage/objectDetection'
        # Ensure the storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
    
    def process_and_save_image(self, image_file):
        """
        Process and save the uploaded image
        
        Args:
            image_file: The uploaded image file
            
        Returns:
            dict: Information about the saved image
        """
        try:
            # Generate a unique filename
            original_filename = image_file.filename
            file_extension = os.path.splitext(original_filename)[1].lower()
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            
            # Full path for the resized image
            full_path = os.path.join(self.storage_path, unique_filename)
            
            # Resize and save the image
            image = Image.open(image_file)
            
            # Resize the image while maintaining aspect ratio
            max_size = (1200, 1200)
            image.thumbnail(max_size, Image.LANCZOS)
            
            # Save the resized image
            image.save(full_path)
            
            logger.info(f"Image saved for object detection: {full_path}")
            
            return {
                "filename": unique_filename,
                "path": full_path
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise 

    def _process_image_in_background(image_path, tracking_token):
        """
        Process the image in a background thread
        
        Args:
            image_path: Path to the saved image
        """
        try:
            logger.info(f"Starting background processing for image: {image_path}")
            schema = OpenAIService().get_moderation_schema()
            base64_image = ObjectDetectionService.encode_image(image_path)
            messages = [
                {"role": "system", "content": f"""
                    Your purpose is to analyze images.
                    You must return a:
                    1) description of the image
                    2) alt text for the image
                    3) an array of all the object found in the image
                    4) possible SEO metatags for the image (no more than 5)
                    The description must be a short description of the image. It must be in the English language.
                    The alt text must be a short description of the image shorter than description. It must be in the English language.
                    The array of objects must be an array of strings, each string is the name of an object found in the image.
                    The SEO metatags must be an array of strings, each string is a possible SEO metatag for the image.
                    The output must be in JSON format.
                    You must return the output in the English language.
                    """
                },
                {"role": "user", "content": [
                        { "type": "text", "text": "what's in this image?" },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ]
                }
            ]
            response = OpenAIService().safe_openai_request(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=2500,
                functions=[schema],
                function_call={"name": "generate_metadata"}
            )
            if response.choices and response.choices[0].message.function_call:
                function_call = response.choices[0].message.function_call
                arguments = json.loads(function_call.arguments)
                pusher_client.trigger('my-channel', 'my-event', {'message': f"{json.dumps(arguments,indent=4)}"})
                logger.info(f"Response: {json.dumps(arguments,indent=4)}")

            if hasattr(response, "usage"):
                total_tokens = response.usage.total_tokens
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                logger.info(f"Token usage - total: {total_tokens}, prompt: {prompt_tokens}, completion: {completion_tokens}")
            else:
                logger.warning("Token usage data not found in response.")  

            logger.info(f"Tokennnnnnnnnnn: {tracking_token}")
            logger.info(f"Background processing completed for image: {image_path}")
            
            # Delete the image after successful processing
            ObjectDetectionService.delete_image(image_path)
            
        except Exception as e:
            logger.error(f"Error in background processing for image {image_path}: {str(e)}")
            # Delete the image even if processing fails
            ObjectDetectionService.delete_image(image_path)

    @staticmethod
    def delete_image(image_path):
        """
        Delete an image file from the filesystem
        
        Args:
            image_path: Path to the image file to delete
        """
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                logger.info(f"Successfully deleted image: {image_path}")
            else:
                logger.warning(f"Image not found for deletion: {image_path}")
        except Exception as e:
            logger.error(f"Error deleting image {image_path}: {str(e)}")

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")