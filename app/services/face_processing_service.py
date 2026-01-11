import os
import time
import re
import cv2
import numpy as np
from datetime import datetime
from PIL import Image as PILImage
from deepface import DeepFace
from app.services.kylo_service import KyloService
import logging
from app.services.wasabi_service import WasabiService
from app.services.text_service import TextService

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class FaceProcessingService:
    RECOGNIZED_DATABASE_PATH = 'storage/recognized_faces'
    MAX_TOTAL_IMAGES = 40
    MAX_DAILY_IMAGES = 3
    
    @staticmethod
    def get_domain_folder(domain):
        """Kreira i vraća putanju do foldera za specifični domain"""
        # Čistimo domain string (uklanjamo port ako postoji)
        domain_folder = domain.split(':')[0]
        # Kreiramo punu putanju
        domain_path = os.path.join(FaceProcessingService.RECOGNIZED_DATABASE_PATH, domain_folder)
        # Kreiramo folder ako ne postoji
        if not os.path.exists(domain_path):
            os.makedirs(domain_path, exist_ok=True)
        return domain_path

    @staticmethod
    def is_blurred(image_array, number_of_detected_faces):
        try:
            # Prvo konvertujemo u uint8 ako je potrebno
            if image_array.dtype == 'float64' or image_array.dtype == 'float32':
                image_array = (image_array * 255).astype(np.uint8)
            elif image_array.dtype != np.uint8:
                image_array = image_array.astype(np.uint8)

            logger.info(f"Image array dtype before grayscale: {image_array.dtype}")
            logger.info(f"Image array shape before grayscale: {image_array.shape}")

            # Konverzija u grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            logger.info(f"Grayscale image dtype: {gray.dtype}")
            logger.info(f"Grayscale image shape: {gray.shape}")

            # Povećanje kontrasta pre Laplaciana if there is only one face
            if number_of_detected_faces == 1:
                gray = cv2.equalizeHist(gray)
                logger.info(f"After equalizeHist - Grayscale mean: {gray.mean()}, var: {gray.var()}")

            # Izračunavanje Laplacian varijanse
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = laplacian.var()
            logger.info(f"Laplacian variance after contrast adjustment: {laplacian_var}")
            print(f"Laplacian variance after contrast adjustment: {laplacian_var}")
            
            # Provera mutnoće sa povećanjem praga
            return laplacian_var < 40  # Prag za mutnoću snižen sa 55 na 40 da prihvati više slika
        except Exception as e:
            logger.error(f"Error in is_blurred: {str(e)}")
            logger.info(f"Image array info - dtype: {image_array.dtype}, shape: {image_array.shape}")
            return False

    @staticmethod
    def extract_faces_with_timeout(img_path, image_id, person, timeout_duration=70):
        try:
            logger.info(f"Extracting faces from {img_path}")
            return DeepFace.extract_faces(img_path=img_path, enforce_detection=True, detector_backend='retinaface')
        except Exception as e:
            logger.error(f"Face extraction error: {str(e)}")
            KyloService.send_skipped_info_to_kylo(image_id, person, "No face found.")
            return None

    @staticmethod
    def count_images_for_person_on_date(person: str, date_str: str, domain: str) -> int:
        """
        Broji broj slika za određenu osobu na određeni datum u oba foldera
        """
        count = 0
        
        # Putanje do oba foldera
        recognized_faces_path = os.path.join('storage/recognized_faces', domain)
        recognized_faces_prod_path = os.path.join('storage/recognized_faces_prod', domain)
        
        # Funkcija za proveru da li slika odgovara osobi i datumu
        def matches_person_and_date(filename):
            # Pretpostavljamo format: ime_osobe_YYYYMMDD_*.jpg
            parts = filename.split('_')
            if len(parts) < 3:
                return False
            
            # Proveri da li je ime osobe u nazivu fajla
            if person not in filename:
                return False
            
            # Proveri da li je datum u nazivu fajla
            for part in parts:
                if part == date_str or part.startswith(date_str):
                    return True
            
            return False
        
        # Brojanje u recognized_faces folderu
        if os.path.exists(recognized_faces_path):
            try:
                for filename in os.listdir(recognized_faces_path):
                    if FaceProcessingService.is_image_file(filename) and matches_person_and_date(filename):
                        count += 1
            except Exception as e:
                logger.error(f"Error counting daily images in recognized_faces: {str(e)}")
        
        # Brojanje u recognized_faces_prod folderu
        if os.path.exists(recognized_faces_prod_path):
            try:
                for filename in os.listdir(recognized_faces_prod_path):
                    if FaceProcessingService.is_image_file(filename) and matches_person_and_date(filename):
                        count += 1
            except Exception as e:
                logger.error(f"Error counting daily images in recognized_faces_prod: {str(e)}")
        
        logger.info(f"Daily images for {person} on {date_str} in both folders: {count}")
        return count

    @staticmethod
    def count_total_images_for_person(person: str, domain: str) -> int:
        """
        Broji ukupan broj slika za određenu osobu u oba foldera (recognized_faces i recognized_faces_prod)
        """
        count = 0
        
        # Putanje do oba foldera
        recognized_faces_path = os.path.join('storage/recognized_faces', domain)
        recognized_faces_prod_path = os.path.join('storage/recognized_faces_prod', domain)
        
        # Brojanje u recognized_faces folderu
        if os.path.exists(recognized_faces_path):
            try:
                for filename in os.listdir(recognized_faces_path):
                    if FaceProcessingService.is_image_file(filename) and person in filename:
                        count += 1
            except Exception as e:
                logger.error(f"Error counting images in recognized_faces: {str(e)}")
        
        # Brojanje u recognized_faces_prod folderu
        if os.path.exists(recognized_faces_prod_path):
            try:
                for filename in os.listdir(recognized_faces_prod_path):
                    if FaceProcessingService.is_image_file(filename) and person in filename:
                        count += 1
            except Exception as e:
                logger.error(f"Error counting images in recognized_faces_prod: {str(e)}")
        
        logger.info(f"Total images for {person} in both folders: {count}")
        return count

    @staticmethod
    def cleanup_original_image(image_path):
        """Briše originalnu sliku nakon obrade"""
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                logger.info(f"Successfully deleted original image: {image_path}")
            else:
                logger.warning(f"Original image not found for deletion: {image_path}")
        except Exception as e:
            logger.error(f"Error deleting original image {image_path}: {str(e)}")

    @staticmethod
    def process_face(image_path, person, date_str, domain, image_id=None):
        """
        Procesira sliku, ekstrahuje lice i čuva ga
        
        Args:
            image_path (str): Putanja do slike
            person (str): Ime osobe
            date_str (str): Datum u formatu YYYY-MM-DD
            domain (str): Domen sa kojeg dolazi slika
            image_id (int, optional): ID slike za Kylo API
            
        Returns:
            dict: Rezultat obrade
        """
        try:
            # Convert date string to datetime object
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            date_str = date_obj.strftime("%Y-%m-%d")

            # Provera ukupnog broja slika
            total_images = FaceProcessingService.count_total_images_for_person(person, domain)
            logger.info(f"Current total images for {person} in domain {domain}: {total_images}")
            if total_images >= FaceProcessingService.MAX_TOTAL_IMAGES:
                logger.info(f"Skipping extraction: {person} already has {FaceProcessingService.MAX_TOTAL_IMAGES} saved images.")
                KyloService.send_skipped_info_to_kylo(image_id, person, "Person image limit reached.")
                FaceProcessingService.cleanup_original_image(image_path)
                raise Exception(f"Person image limit reached ({FaceProcessingService.MAX_TOTAL_IMAGES} images).")


            # Provera broja slika za taj dan
            daily_images = FaceProcessingService.count_images_for_person_on_date(person, date_str, domain)
            logger.info(f"Current daily images for {person} on {date_str} in domain {domain}: {daily_images}")
            if daily_images >= FaceProcessingService.MAX_DAILY_IMAGES:
                logger.info(f"Skipping extraction: Already {FaceProcessingService.MAX_DAILY_IMAGES} images for {person} on {date_str}.")
                KyloService.send_skipped_info_to_kylo(image_id, person, "Daily image limit reached.")
                FaceProcessingService.cleanup_original_image(image_path)
                raise Exception(f"Daily image limit reached ({FaceProcessingService.MAX_DAILY_IMAGES} images).")

            # Kreiramo base folder ako ne postoji
            if not os.path.exists(FaceProcessingService.RECOGNIZED_DATABASE_PATH):
                os.makedirs(FaceProcessingService.RECOGNIZED_DATABASE_PATH, exist_ok=True)

            # Kreiramo i dobijamo putanju do domain foldera
            domain_path = FaceProcessingService.get_domain_folder(domain)

            # Provera da li fajl postoji
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")

            # Extract faces
            try:
                face_objs = FaceProcessingService.extract_faces_with_timeout(image_path, image_id, person)
                if face_objs is not None:
                    logger.info(f"Detected {len(face_objs)} faces in the image.")
                else:
                    logger.error("Face extraction returned None")
            except Exception as e:
                logger.error(f"Face extraction failed with error: {str(e)}")
                KyloService.send_skipped_info_to_kylo(image_id, person, "Face extraction failed.")
                raise Exception(f"Face extraction failed: {str(e)}")

            valid_faces = []
            invalid_faces = []
            number_of_detected_faces = len(face_objs)
            
            for i, item in enumerate(face_objs):
                face_image_array = item['face']
                w, h = face_image_array.shape[1], face_image_array.shape[0]

                logger.info(f"Original face array shape: {face_image_array.shape}")
                logger.info(f"Original face array dtype: {face_image_array.dtype}")

                # Check face size
                if w < 70 or h < 70:
                    print(f"Face too smallllllllllllllllll w : {w} h : {h}")
                    invalid_faces.append((face_image_array, i))
                    continue

                # Check for blurriness
                if FaceProcessingService.is_blurred(face_image_array, number_of_detected_faces):
                    invalid_faces.append((face_image_array, i))
                    continue

                valid_faces.append((face_image_array, i))

            # Process results
            if len(valid_faces) > 1:
                logger.error("Multiple faces detected")
                KyloService.send_skipped_info_to_kylo(image_id, person, "Multiple faces detected.")
                print("Multiple faces detected")
                raise Exception("No valid faces found in the image.")
            elif len(valid_faces) == 0:
                if len(invalid_faces) == 1:
                    logger.info(f"Skipping image: blurry.")
                    KyloService.send_skipped_info_to_kylo(image_id, person, "Face blurry.")
                    print(f"Skipping image: blurry.")
                elif len(invalid_faces) > 1:
                    logger.info(f"Skipping image: multiple invalid faces.")
                    KyloService.send_skipped_info_to_kylo(image_id, person, "Multiple invalid faces.")
                    print(f"Skipping image: multiple invalid faces.")
                else:
                    logger.info(f"Skipping image: No face found.")
                    KyloService.send_skipped_info_to_kylo(image_id, person, "No face found.")
                    print(f"Skipping image: No face found.")
                # Dodajemo rano vraćanje iz funkcije ako nema validnih lica
                FaceProcessingService.cleanup_original_image(image_path)
                raise Exception("No valid faces found in the image.")

            # Process the single valid face
            face_image_array, i = valid_faces[0]
            # Uklanjamo navodnike ako postoje i zamenjujemo razmake sa '_'
            sanitized_person = person.strip('"\'').replace(' ', '_')
            unique_id = f"{sanitized_person}_{date_str}_{int(time.time() * 1000)}"

            logger.info(f"Processing face array shape: {face_image_array.shape}")
            logger.info(f"Processing face array dtype: {face_image_array.dtype}")
            
            # Convert to PIL Image
            face_image_pil = PILImage.fromarray((face_image_array * 255).astype(np.uint8))
            
            # Resize nakon cropa
            new_height = 224
            new_width = int(face_image_pil.width * (new_height / face_image_pil.height))
            face_image_pil = face_image_pil.resize((new_width, new_height), PILImage.LANCZOS)
            
            logger.info(f"PIL Image size after resize: {face_image_pil.size}")

            # Get face coordinates
            facial_area = face_objs[i].get('facial_area', {})
            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]
            
            coordinates = {
                "x": round(facial_area.get('x', 0) / img_width, 3),
                "y": round(facial_area.get('y', 0) / img_height, 3),
                "width": round(facial_area.get('w', 0) / img_width, 3),
                "height": round(facial_area.get('h', 0) / img_height, 3)
            }
            logger.info(f"Face coordinates: {coordinates}")
            # Save processed face
            sanitized_filename = re.sub(r'[^\w\-_. ]', '_', f"{unique_id}.jpg")
            face_path = os.path.join(domain_path, sanitized_filename)
            face_image_pil.save(face_path, format="JPEG", quality=85)


            if os.path.isfile(face_path):
                logger.info(f"Face saved at: {face_path}")
                download_url = sanitized_filename
                s3_key = f"recognized_faces/{sanitized_filename}"
                
                try:
                    WasabiService.upload_to_s3(face_path, "facerec", s3_key)
                except Exception as e:
                    logger.warning(f"S3 upload failed, continuing without it: {str(e)}")
                
                # Ako je image_id None, preskočimo slanje u Kylo
                if image_id is not None:
                    # Dobavi originalno ime osobe
                    formatted_person = TextService.get_original_text(person)
                    logger.info(f"Using original person name: {formatted_person} (normalized: {person})")
                    
                    # Proveri da li je originalno ime zaista dobijeno
                    if formatted_person == person and '_' in person:
                        # Ako nije pronađeno mapiranje, a ime sadrži donju crtu, pokušaj da je zameniš razmakom
                        alternative_person = person.replace('_', ' ')
                        logger.info(f"No mapping found, using alternative formatting: {alternative_person}")
                        KyloService.send_info_to_kylo(image_id, download_url, alternative_person, coordinates)
                    else:
                        KyloService.send_info_to_kylo(image_id, download_url, formatted_person, coordinates)
                else:
                    logger.info(f"Skipping Kylo API call because image_id is None")
            else:
                logger.error(f"Failed to save face at: {face_path}")
                
                # Ako je image_id None, preskočimo slanje u Kylo
                if image_id is not None:
                    # Dobavi originalno ime osobe i za skipped info
                    formatted_person = TextService.get_original_text(person)
                    KyloService.send_skipped_info_to_kylo(image_id, formatted_person, "Failed to save face.")
                
                raise Exception("Failed to save face.")

            result = {
                'face_path': face_path,
                'coordinates': coordinates,
                'filename': sanitized_filename
            }
            

            FaceProcessingService.cleanup_original_image(image_path)
            
            return result
            
        except Exception as e:
            # Brisanje originalne slike i u slučaju greške
            FaceProcessingService.cleanup_original_image(image_path)
            logger.error(f"Error in process_face: {str(e)}")
            
            # Ako je image_id None, preskočimo slanje u Kylo
            if image_id is not None:
                # Dobavi originalno ime osobe i za skipped info
                formatted_person = TextService.get_original_text(person)
            raise e

    @staticmethod
    def is_image_file(filename):
        """Proverava da li je fajl slika na osnovu ekstenzije"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'} 