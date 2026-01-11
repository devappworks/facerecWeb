from flask import Blueprint, jsonify, request
from app.services.image_service import ImageService
from app.services.face_processing_service import FaceProcessingService
from app.services.text_service import TextService
import logging

image_bp = Blueprint('image', __name__)

logger = logging.getLogger(__name__)

@image_bp.route('/test', methods=['GET'])
def test_route():
    return jsonify({"message": "Test ruta radi!"})

@image_bp.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "Nema slike u zahtevu"}), 400
    
    return jsonify({"message": "Slika primljena"}), 200

@image_bp.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "Nema slike u zahtevu"}), 400
    
    return jsonify({"message": "Slika obrađena"}), 200

class ImageController:
    @staticmethod
    def handle_image_upload(image_file, person, created_date, domain):
        """
        Obrađuje upload slike, ekstrahuje lice i čuva ga
        """
        try:
            # Sačuvaj originalno ime pre normalizacije
            original_person = person
            
            # Normalizuj ime osobe
            normalized_person = TextService.normalize_text(person, save_mapping=True)
            logger.info(f"Normalizovano ime osobe: '{original_person}' -> '{normalized_person}'")
            
            # Ako je normalizacija uklonila sve karaktere, koristi originalno ime
            if not normalized_person:
                logger.warning(f"Normalizacija je uklonila sve karaktere iz imena '{original_person}', koristim originalno ime")
                normalized_person = person
            
            # Pokrenemo asinhronu obradu
            ImageService.process_image_async(
                image_file=image_file,
                person=normalized_person,
                created_date=created_date,
                domain=domain
            )
            
            return {"status": "success", "message": "Slika je poslata na obradu"}
            
        except Exception as e:
            logger.error(f"Error in ImageController.handle_image_upload: {str(e)}")
            return {"error": str(e)} 