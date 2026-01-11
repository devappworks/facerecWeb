import os
from werkzeug.utils import secure_filename
from datetime import datetime
from threading import Thread
from io import BytesIO
import time
from app.services.face_processing_service import FaceProcessingService
from app.services.image_rejection_logger import ImageRejectionLogger
import logging
from PIL import Image as PILImage
import requests
import json
from flask import current_app
from urllib.parse import quote
import pandas as pd
import shutil
from deepface import DeepFace
import numpy as np
import cv2
import uuid

logger = logging.getLogger(__name__)

class ImageService:
    BASE_UPLOAD_FOLDER = 'storage/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    MAX_IMAGE_SIZE = (1024, 1024)  # Maksimalna veličina slike

    def __init__(self, domain='serbia'):
        """
        Initialize ImageService for a specific domain.

        Args:
            domain: Domain code (e.g., 'serbia', 'greece', 'slovenia')
        """
        self.domain = domain
        self.api_key = os.getenv('SERPAPI_SEARCH_API_KEY', 'af309518c81f312d3abcffb4fc2165e6ae6bd320b0d816911d0d1153ccea88c8')
        self.cx = os.getenv('GOOGLE_SEARCH_CX', '444622b2b520b4d97')

        # Domain-specific paths
        self.storage_path = f'storage/training/{domain}'
        self.training_pass_path = f'storage/trainingPass/{domain}'
        self.originals_archive_path = f'storage/serp_originals/{domain}'

    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ImageService.ALLOWED_EXTENSIONS

    @staticmethod
    def resize_image(image_data):
        """
        Smanjuje veličinu slike održavajući proporcije originalne slike
        
        Args:
            image_data: Bytes ili BytesIO objekat sa slikom
            
        Returns:
            BytesIO: Procesirana slika kao BytesIO objekat
        """
        try:
            # Konvertuj bytes u BytesIO ako je potrebno
            if isinstance(image_data, bytes):
                image_data = BytesIO(image_data)
            
            # Otvori sliku
            with PILImage.open(image_data) as img:
                # Sačuvaj original format
                img_format = img.format or 'JPEG'
                
                # Proveri orijentaciju iz EXIF podataka
                try:
                    exif = img._getexif()
                    if exif and 274 in exif:  # 274 je tag za orijentaciju
                        orientation = exif[274]
                        rotate_values = {
                            3: 180,
                            6: 270,
                            8: 90
                        }
                        if orientation in rotate_values:
                            img = img.rotate(rotate_values[orientation], expand=True)
                except:
                    pass  # Ignoriši ako nema EXIF podataka

                # Uzmi trenutne dimenzije
                width, height = img.size
                
                # Izračunaj nove dimenzije
                if width > height:
                    # Horizontalna slika
                    if width > ImageService.MAX_IMAGE_SIZE[0]:
                        ratio = ImageService.MAX_IMAGE_SIZE[0] / width
                        new_width = ImageService.MAX_IMAGE_SIZE[0]
                        new_height = int(height * ratio)
                    else:
                        image_data.seek(0)  # Reset position before returning
                        return image_data  # Vrati original ako je već manja
                else:
                    # Vertikalna slika
                    if height > ImageService.MAX_IMAGE_SIZE[1]:
                        ratio = ImageService.MAX_IMAGE_SIZE[1] / height
                        new_height = ImageService.MAX_IMAGE_SIZE[1]
                        new_width = int(width * ratio)
                    else:
                        image_data.seek(0)  # Reset position before returning
                        return image_data  # Vrati original ako je već manja

                logger.info(f"Resizing image from {img.size} to {(new_width, new_height)}")
                
                # Resize sliku
                img = img.resize((new_width, new_height), PILImage.LANCZOS)
                
                # Sačuvaj procesiranu sliku u BytesIO
                output = BytesIO()
                img.save(output, format=img_format, quality=85)
                output.seek(0)
                
                return output
                
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            raise

    @staticmethod
    def process_image_async(image_file, person, created_date, domain, image_id=None):
        """
        Asinhrona obrada slike
        
        Args:
            image_file: Fajl slike
            person: Ime osobe
            created_date: Datum kreiranja
            domain: Domen
            image_id: ID slike (opciono, koristi se samo za Kylo API)
        """
        # Prvo smanjimo veličinu slike
        # resized_image = ImageService.resize_image(image_file)
        file_content = image_file.getvalue()
        original_filename = image_file.filename
        
        def background_processing():
            try:
                logger.info(f"Započinje obrada slike za osobu: {person} sa domaina: {domain}")
                
                # Sačuvaj smanjenu sliku
                file_copy = BytesIO(file_content)
                file_copy.filename = original_filename
                saved_path = ImageService.save_image(
                    file_copy, 
                    person=person, 
                    created_date=created_date,
                    domain=domain
                )
                
                # Zatim procesiramo lice
                try:
                    # Ako image_id nije prosleđen, prosledi None
                    result = FaceProcessingService.process_face(
                        saved_path,
                        person,
                        created_date.strftime('%Y-%m-%d'),
                        domain,
                        image_id  # Može biti None
                    )
                    logger.info(f"Uspešno obrađeno lice: {result['filename']}")
                except Exception as e:
                    logger.error(f"Greška pri obradi lica: {str(e)}")

            except Exception as e:
                logger.error(f"Greška prilikom obrade slike: {str(e)}")

        thread = Thread(target=background_processing)
        thread.daemon = True
        thread.start()
        
        return "Processing started"

    @staticmethod
    def save_image(image_file, person, created_date, domain):
        """Čuva sliku u folder specifičan za domain"""
        if image_file and ImageService.allowed_file(image_file.filename):
            # Čistimo domain string za ime foldera (uklanjamo port ako postoji)
            domain_folder = domain.split(':')[0]
            
            # Kreiramo putanju do foldera specifičnog za domain
            domain_path = os.path.join(ImageService.BASE_UPLOAD_FOLDER, domain_folder)
            
            # Kreiranje imena fajla sa person i created_date
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = secure_filename(image_file.filename)
            filename = f"{person}_{created_date.strftime('%Y%m%d')}_{timestamp}_{original_filename}"
            
            # Kreiramo folder za domain ako ne postoji
            if not os.path.exists(domain_path):
                os.makedirs(domain_path)
            
            # Puna putanja do fajla
            file_path = os.path.join(domain_path, filename)
            
            # Čuvamo fajl
            with open(file_path, 'wb') as f:
                if isinstance(image_file, BytesIO):
                    f.write(image_file.getvalue())
                else:
                    image_file.save(f)
            
            return file_path
        return None 

    def fetch_and_save_images(self, name, last_name, occupation, original_name='', original_last_name=''):
        """
        Fetch images from Google Custom Search API and save them locally
        """
        try:
            # Store original name for folder creation
            self.original_name = original_name or name
            self.original_last_name = original_last_name or last_name
            
            # Create the search query - handle None or NaN values
            name = "" if pd.isna(name) else name
            last_name = "" if pd.isna(last_name) else last_name
            occupation = "" if pd.isna(occupation) else occupation
            
            # Create the search query
            search_term = f"{name} {last_name} {occupation}".strip()
            encoded_search_term = quote(search_term)
            encoded_search_term = encoded_search_term.replace("%20", " ")  # Zameni %20 sa razmakom
            
            exact_terms = f"{name} {last_name}".strip()
            encoded_exact_terms = quote(exact_terms)
            encoded_exact_terms = encoded_exact_terms.replace("%20", " ")  # Zameni %20 sa razmakom
            
            # Construct the API URL
            # url = (
            #     f"https://serpapi.com/search"
            #     f"?engine=google_images"
            #     f"&q={encoded_search_term}"
            #     f"&key={self.api_key}"
            #     f"&imgsz=xga"
            #     f"&device=desktop"
            #     f"&google_domain=google.com"
            #     f"&hl=en"
            #     f"&gl=us"
            #     f"&image_type=face"
            # )
            


            url = "https://real-time-image-search.p.rapidapi.com/search"

            # Request 200 images - RapidAPI charges per request, not per image
            querystring = {"query": f"{encoded_search_term}","limit":"200","size":"1024x768_and_more","type":"face","region":"us"}
            current_app.logger.info(querystring)
            headers = {
                "x-rapidapi-key": "c3e8343ca0mshe1b719bea5326dbp11db14jsnf52a7fb8ab17",
                "x-rapidapi-host": "real-time-image-search.p.rapidapi.com"
            }

            response = requests.get(url, headers=headers, params=querystring)



            current_app.logger.info(response.json())

            # Log the full URL for debugging
            current_app.logger.info(f"Making API request to: {url}")
            
            # Make the API request
            # response = requests.get(url)
            
            if response.status_code != 200:
                current_app.logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return {"success": False, "message": f"API request failed with status code {response.status_code}"}
            
            # Parse the response
            data = response.json()
            
            # Check if there are any search results
            if 'data' not in data:
                current_app.logger.warning(f"No images found for search term: {search_term}")
                return {"success": True, "message": "No images found", "count": 0}
            
            # Create the storage directory if it doesn't exist
            os.makedirs(self.storage_path, exist_ok=True)
            current_app.logger.info(f"Storage directory: {self.storage_path}")
            
            # Download and save each image
            saved_images = []
            failed_images = []
            
            # Limit to maximum 150 images for processing (from 200 fetched)
            max_images = 150
            image_results = data['data'][:max_images]
            
            # Log the number of items found and limit
            current_app.logger.info(f"Found {len(data['data'])} images, limiting to {len(image_results)} for search term: {search_term}")
            
            # Get current timestamp for unique filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Process each image in a separate try-except block
            for i, item in enumerate(image_results):
                try:
                    # Try to get the original image URL first, then thumbnail if not available
                    image_url = item.get('thumbnail_url') or item.get('url')
                    if not image_url:
                        current_app.logger.warning(f"No link found for item {i+1}")
                        continue
                    
                    current_app.logger.info(f"Processing image {i+1}/{len(image_results)}: {image_url}")
                    
                    # Get the image file extension
                    file_extension = self._get_file_extension(image_url)
                    
                    # Create a filename based on the search term, index and timestamp
                    sanitized_name = f"{name.replace(' ', '_')}_{last_name.replace(' ', '_')}_{timestamp}_{i+1}{file_extension}"
                    file_path = os.path.join(self.storage_path, sanitized_name)
                    
                    current_app.logger.info(f"Will save to: {file_path}")
                    
                    # Download and save the image with improved error handling
                    if self._download_and_save_image(image_url, file_path):
                        # Verify the file was created and is valid
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 100:  # Minimum size check
                            # Verify the image can be opened
                            try:
                                with PILImage.open(file_path) as img:
                                    # Get image dimensions for logging
                                    width, height = img.size
                                    current_app.logger.info(f"Successfully saved valid image to {file_path} (size: {os.path.getsize(file_path)} bytes, dimensions: {width}x{height})")
                                    
                                    saved_images.append({
                                        "filename": sanitized_name,
                                        "path": file_path,
                                        "source_url": image_url,
                                        "size": os.path.getsize(file_path),
                                        "dimensions": f"{width}x{height}"
                                    })
                            except Exception as img_error:
                                # Not a valid image, delete it
                                current_app.logger.error(f"Downloaded file is not a valid image: {file_path}, error: {str(img_error)}")
                                os.remove(file_path)
                                failed_images.append({
                                    "source_url": image_url,
                                    "error": f"Not a valid image: {str(img_error)}"
                                })
                        else:
                            current_app.logger.error(f"File was not created or is too small: {file_path}")
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            failed_images.append({
                                "source_url": image_url,
                                "error": "File was not created or is too small"
                            })
                    else:
                        failed_images.append({
                            "source_url": image_url,
                            "error": "Failed to download image"
                        })
                
                except Exception as item_error:
                    current_app.logger.error(f"Error processing item {i+1}: {str(item_error)}")
                    failed_images.append({
                        "item_index": i,
                        "error": f"Processing error: {str(item_error)}"
                    })
            
            # Final summary
            current_app.logger.info(f"Download summary: {len(saved_images)} successful, {len(failed_images)} failed out of {len(image_results)} total")
            
            # Start background processing of images
            if saved_images:
                self.process_images_with_deepface(saved_images)
            
            return {
                "success": True,
                "message": f"Successfully downloaded {len(saved_images)} images",
                "count": len(saved_images),
                "images": saved_images,
                "failed": failed_images,
                "total_found": len(data['data']),
                "processed": len(image_results)
            }
            
        except Exception as e:
            current_app.logger.error(f"Error fetching and saving images: {str(e)}")
            raise Exception(f"Failed to fetch and save images: {str(e)}")
    
    def process_images_with_deepface(self, saved_images):
        """
        Process images with DeepFace in a background thread
        """
        # Generate a unique ID for this processing batch
        batch_id = str(uuid.uuid4())[:8]
        
        # Get the app context for the background thread
        app_context = current_app.app_context()
        
        # Start a background thread for processing
        thread = Thread(target=self._process_images_with_deepface_thread, args=(saved_images, app_context, batch_id))
        thread.daemon = True
        thread.start()
        
        return {"success": True, "message": "Started background processing of images", "batch_id": batch_id}

    def _process_images_with_deepface_thread(self, saved_images, app_context, batch_id):
        """
        Background thread to process images with DeepFace

        1. Find the first three valid images and move them to trainingPass/person_name
        2. Compare each remaining image with each of the three valid images
        3. If similar to any of them, move to trainingPass/person_name, otherwise delete
        4. Stop processing when 100 images are reached for a person
        """
        # DEBUG: Write to simple English log file that's easy to understand
        log_file = '/tmp/VALIDATION_PROCESS.log'
        def log_step(message):
            try:
                with open(log_file, 'a') as f:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"{timestamp} | {message}\n")
                    f.flush()  # Make sure it's written immediately
            except Exception as e:
                import sys
                print(f"ERROR writing log: {str(e)}", file=sys.stderr)
                pass

        log_step(f"=== BATCH {batch_id} VALIDATION STARTED ===")
        log_step(f"Total images to validate: {len(saved_images) if saved_images else 0}")

        try:
            with app_context:
                current_app.logger.info(f"[Batch {batch_id}] Starting DeepFace processing in background thread")

                if not saved_images:
                    current_app.logger.warning(f"[Batch {batch_id}] No images to process with DeepFace")
                    log_step(f"ERROR: No images to process!")
                    return

                # Create a list to track files that are being processed by this thread
                processing_files = []

                # Extract person name from the filename of the first image
                # This ensures we use the ACTUAL name from files (with special chars like ć)
                # instead of ASCII-normalized name from self.original_name
                first_image = saved_images[0]
                first_image_path = first_image["path"]
                filename = os.path.basename(first_image_path)
                name_parts = filename.split('_')
                if len(name_parts) >= 2:
                    # Keep original name with special chars for file matching
                    normalized_name = name_parts[0]
                    normalized_last_name = name_parts[1]
                    person_name_for_folder = f"{normalized_name}_{normalized_last_name}"
                    # Normalize only for folder creation (for trainingPass directory)
                    person_name = self._ensure_ascii_path(person_name_for_folder)
                    current_app.logger.info(f"[Batch {batch_id}] Extracted name from filename: normalized_name='{normalized_name}', normalized_last_name='{normalized_last_name}'")
                else:
                    # Fallback if filename doesn't have expected format
                    if hasattr(self, 'original_name') and hasattr(self, 'original_last_name'):
                        normalized_name = self._ensure_ascii_path(self._normalize_for_filename(self.original_name))
                        normalized_last_name = self._ensure_ascii_path(self._normalize_for_filename(self.original_last_name))
                        person_name = f"{normalized_name}_{normalized_last_name}"
                        current_app.logger.info(f"[Batch {batch_id}] Fallback: Using normalized name for folder: {person_name}")
                    else:
                        normalized_name = "unknown"
                        normalized_last_name = ""
                        person_name = "unknown"
                        current_app.logger.warning(f"[Batch {batch_id}] Could not extract person name from filename or attributes")
                
                # Create person-specific directory in trainingPass
                person_dir = os.path.join(self.training_pass_path, person_name)
                os.makedirs(person_dir, exist_ok=True)
                
                current_app.logger.info(f"[Batch {batch_id}] Created person directory: {person_dir}")
                
                # Check if we already have 40 or more images for this person
                existing_images = [f for f in os.listdir(person_dir) if self._is_image_file(os.path.join(person_dir, f))]
                if len(existing_images) >= 40:
                    current_app.logger.info(f"[Batch {batch_id}] Already have {len(existing_images)} images for {person_name}, skipping processing")
                    
                    # Clean up all images in the training directory for this person
                    training_dir = self.storage_path
                    for filename in os.listdir(training_dir):
                        if filename.startswith(f"{normalized_name}_{normalized_last_name}"):
                            file_path = os.path.join(training_dir, filename)
                            if os.path.isfile(file_path) and self._is_image_file(file_path):
                                try:
                                    os.remove(file_path)
                                    current_app.logger.info(f"[Batch {batch_id}] Removed image as we already have 40: {file_path}")
                                except Exception as e:
                                    current_app.logger.error(f"[Batch {batch_id}] Error removing image: {str(e)}")
                    
                    return
                
                current_app.logger.info(f"[Batch {batch_id}] Currently have {len(existing_images)} images for {person_name}")
                
                # Get all image files in the training directory that match this person's name
                training_dir = self.storage_path
                image_files = []

                # DEBUG: Log the exact pattern being searched
                search_pattern = f"{normalized_name}_{normalized_last_name}"
                current_app.logger.info(f"[Batch {batch_id}] Searching for files starting with: '{search_pattern}' (type: {type(search_pattern)})")
                current_app.logger.info(f"[Batch {batch_id}] normalized_name={repr(normalized_name)}, normalized_last_name={repr(normalized_last_name)}")

                for filename in os.listdir(training_dir):
                    if filename.startswith(search_pattern):
                        file_path = os.path.join(training_dir, filename)
                        if os.path.isfile(file_path) and self._is_image_file(file_path):
                            image_files.append(file_path)
                            processing_files.append(file_path)

                current_app.logger.info(f"[Batch {batch_id}] Found {len(image_files)} images matching pattern {normalized_name}_{normalized_last_name}")
                log_step(f"Found {len(image_files)} images to process for {person_name}")

                if not image_files:
                    current_app.logger.warning(f"[Batch {batch_id}] No images found to process in {training_dir}")
                    log_step(f"ERROR: No images found! Stopping.")
                    return
                
                # Sort images by their sequence number in the filename
                def extract_sequence_number(file_path):
                    try:
                        filename = os.path.basename(file_path)
                        # Try to find the sequence number at the end of the filename before the extension
                        # Example: name_lastname_1.jpg -> 1
                        name_without_ext = os.path.splitext(filename)[0]
                        parts = name_without_ext.split('_')
                        if parts and parts[-1].isdigit():
                            return int(parts[-1])
                        return float('inf')  # If no number found, put at the end
                    except:
                        return float('inf')  # If error, put at the end
                
                # Sort the image files by sequence number
                image_files.sort(key=extract_sequence_number)
                current_app.logger.info(f"[Batch {batch_id}] Sorted {len(image_files)} images by sequence number")
                
                current_app.logger.info(f"[Batch {batch_id}] Found {len(image_files)} images to process in {training_dir}")
                
                # NEW APPROACH: Use P18 (Wikidata primary image) as the reference
                # P18 is always downloaded as sequence #1 by wikimedia_image_service.py
                # This is a curated, verified image - much more reliable than blindly accepting first 3

                # Counter for images processed so far
                processed_count = len(existing_images)
                max_images_per_person = 100

                # PHASE 1: Find P18 reference image (sequence #1)
                current_app.logger.info(f"[Batch {batch_id}] Looking for P18 reference image (sequence #1)")
                log_step(f"PHASE 1: Finding P18 reference image from {len(image_files)} candidates")

                p18_reference = None
                p18_image_path = None

                # Look for sequence #1 (P18 from Wikidata)
                for image_path in image_files[:]:
                    seq_num = extract_sequence_number(image_path)
                    if seq_num == 1:
                        p18_image_path = image_path
                        current_app.logger.info(f"[Batch {batch_id}] Found P18 candidate: {image_path}")
                        break

                # Try to extract face from P18
                if p18_image_path and os.path.exists(p18_image_path):
                    image_filename = os.path.basename(p18_image_path)
                    image_dest = os.path.join(person_dir, image_filename)

                    face_extracted = self._extract_and_save_face(p18_image_path, image_dest, batch_id)

                    if face_extracted:
                        p18_reference = {
                            'path': image_dest,
                            'source': 'wikidata_p18',
                            'is_primary': True
                        }
                        current_app.logger.info(f"[Batch {batch_id}] ✓ P18 reference extracted successfully")
                        log_step(f"P18 Reference: Extracted from Wikidata primary image")

                        # Increment processed count
                        processed_count += 1

                        # Archive as reference
                        extraction_result = {
                            'status': 'reference',
                            'face_file': image_filename,
                            'reason': 'P18 primary reference from Wikidata',
                            'is_reference': True
                        }
                        self._archive_original(p18_image_path, person_name, batch_id, extraction_result)

                        # Remove from processing list
                        image_files.remove(p18_image_path)

                        # Remove original
                        try:
                            os.remove(p18_image_path)
                        except Exception as e:
                            current_app.logger.error(f"[Batch {batch_id}] Error removing P18 image: {str(e)}")
                    else:
                        current_app.logger.warning(f"[Batch {batch_id}] P18 face extraction failed, will use fallback")
                        log_step(f"P18 extraction failed - falling back to first valid image")

                        # Archive P18 as rejected
                        extraction_result = {
                            'status': 'rejected',
                            'face_file': None,
                            'reason': 'P18 face extraction failed'
                        }
                        self._archive_original(p18_image_path, person_name, batch_id, extraction_result)

                        try:
                            os.remove(p18_image_path)
                            image_files.remove(p18_image_path)
                        except Exception as e:
                            current_app.logger.error(f"[Batch {batch_id}] Error removing failed P18: {str(e)}")

                # PHASE 2: If P18 failed, fall back to first valid image as reference
                if not p18_reference:
                    current_app.logger.info(f"[Batch {batch_id}] No P18 reference, falling back to first valid image")
                    log_step(f"PHASE 1b: Finding fallback reference from remaining {len(image_files)} images")

                    for image_path in image_files[:]:
                        try:
                            if not os.path.exists(image_path):
                                image_files.remove(image_path)
                                continue

                            image_filename = os.path.basename(image_path)
                            image_dest = os.path.join(person_dir, image_filename)

                            face_extracted = self._extract_and_save_face(image_path, image_dest, batch_id)

                            if face_extracted:
                                p18_reference = {
                                    'path': image_dest,
                                    'source': 'fallback',
                                    'is_primary': True
                                }
                                seq_num = extract_sequence_number(image_path)
                                current_app.logger.info(f"[Batch {batch_id}] ✓ Fallback reference #{seq_num} extracted")
                                log_step(f"Fallback Reference: Using image #{seq_num}")

                                # Increment processed count
                                processed_count += 1

                                # Archive as reference
                                extraction_result = {
                                    'status': 'reference',
                                    'face_file': image_filename,
                                    'reason': 'Fallback reference (P18 unavailable)',
                                    'is_reference': True
                                }
                                self._archive_original(image_path, person_name, batch_id, extraction_result)

                                image_files.remove(image_path)
                                try:
                                    os.remove(image_path)
                                except Exception as e:
                                    current_app.logger.error(f"[Batch {batch_id}] Error removing fallback ref: {str(e)}")
                                break
                            else:
                                # Archive as rejected and continue to next
                                seq_num = extract_sequence_number(image_path)
                                extraction_result = {
                                    'status': 'rejected',
                                    'face_file': None,
                                    'reason': 'Face extraction failed'
                                }
                                self._archive_original(image_path, person_name, batch_id, extraction_result)

                                try:
                                    os.remove(image_path)
                                    image_files.remove(image_path)
                                except Exception as e:
                                    current_app.logger.error(f"[Batch {batch_id}] Error removing: {str(e)}")

                        except Exception as e:
                            current_app.logger.error(f"[Batch {batch_id}] Error with fallback ref {image_path}: {str(e)}")
                            try:
                                if os.path.exists(image_path):
                                    os.remove(image_path)
                                    image_files.remove(image_path)
                            except:
                                pass
                
                # Check if we found a valid reference image
                if not p18_reference:
                    current_app.logger.error(f"[Batch {batch_id}] No valid reference image found, stopping processing")
                    log_step(f"CRITICAL ERROR: Could not find reference image! Validation cannot continue.")

                    # Clean up any remaining images
                    for image_path in image_files:
                        try:
                            if os.path.exists(image_path):
                                os.remove(image_path)
                                current_app.logger.info(f"[Batch {batch_id}] Cleaned up remaining image: {image_path}")
                        except Exception as e:
                            current_app.logger.error(f"[Batch {batch_id}] Error cleaning up image: {str(e)}")

                    return

                ref_source = p18_reference.get('source', 'unknown')
                current_app.logger.info(f"[Batch {batch_id}] Using reference image from: {ref_source}")
                log_step(f"SUCCESS: Reference image ready (source: {ref_source})")
                log_step(f"PHASE 2: Comparing {len(image_files)} remaining images against reference")

                # Keep track of processed image hashes to avoid duplicates
                processed_hashes = set()

                # Calculate hash of reference image
                try:
                    ref_img = cv2.imread(p18_reference['path'])
                    if ref_img is not None:
                        ref_hash = self._calculate_image_hash(ref_img)
                        processed_hashes.add(ref_hash)
                        current_app.logger.info(f"[Batch {batch_id}] Reference image hash calculated")
                except Exception as e:
                    current_app.logger.error(f"[Batch {batch_id}] Error calculating reference hash: {str(e)}")
                
                # Process each remaining image
                for image_path in image_files:
                    # Check if we've reached the maximum number of images
                    if processed_count >= max_images_per_person:
                        current_app.logger.info(f"[Batch {batch_id}] Reached maximum of {max_images_per_person} images for {person_name}, stopping processing")
                        log_step(f"PHASE 2 STOPPED: Reached limit of {max_images_per_person} images")
                        
                        # Clean up any remaining images
                        try:
                            if os.path.exists(image_path):
                                os.remove(image_path)
                                current_app.logger.info(f"[Batch {batch_id}] Removed remaining image after reaching limit: {image_path}")
                        except Exception as e:
                            current_app.logger.error(f"[Batch {batch_id}] Error removing remaining image: {str(e)}")
                        
                        continue
                    
                    try:
                        # Check if file still exists (might have been processed by another thread)
                        if not os.path.exists(image_path):
                            current_app.logger.info(f"[Batch {batch_id}] Image no longer exists, skipping: {image_path}")
                            continue
                        
                        # Check if this image is a duplicate of already processed images
                        try:
                            img = cv2.imread(image_path)
                            if img is None:
                                current_app.logger.error(f"[Batch {batch_id}] Failed to read image: {image_path}")
                                try:
                                    if os.path.exists(image_path):
                                        os.remove(image_path)
                                        current_app.logger.info(f"[Batch {batch_id}] Removed unreadable image: {image_path}")
                                except Exception as remove_error:
                                    current_app.logger.error(f"[Batch {batch_id}] Error removing image: {str(remove_error)}")
                                continue
                            
                            img_hash = self._calculate_image_hash(img)
                            
                            if img_hash in processed_hashes:
                                current_app.logger.info(f"[Batch {batch_id}] Skipping duplicate image: {image_path}")
                                try:
                                    if os.path.exists(image_path):
                                        os.remove(image_path)
                                        current_app.logger.info(f"[Batch {batch_id}] Removed duplicate image: {image_path}")
                                except Exception as remove_error:
                                    current_app.logger.error(f"[Batch {batch_id}] Error removing duplicate image: {str(remove_error)}")
                                continue
                            
                            processed_hashes.add(img_hash)
                        except Exception as hash_error:
                            current_app.logger.error(f"[Batch {batch_id}] Error calculating hash: {str(hash_error)}")
                            continue
                        
                        # First, extract face to validate it (check for multiple faces, etc.)
                        seq_num = extract_sequence_number(image_path)
                        image_filename = os.path.basename(image_path)
                        image_dest = os.path.join(person_dir, image_filename)

                        current_app.logger.info(f"[Batch {batch_id}] Extracting face from candidate #{seq_num} for validation")
                        face_extracted = self._extract_and_save_face(image_path, image_dest, batch_id)

                        if not face_extracted:
                            # Face extraction failed (no face, multiple faces, too small, blurry, etc.)
                            current_app.logger.warning(f"[Batch {batch_id}] Face extraction failed for #{seq_num}")

                            # Archive with extraction failure details
                            extraction_result = {
                                'status': 'rejected',
                                'face_file': None,
                                'reason': 'Face extraction failed (no face, multiple faces, too small, or blurry)'
                            }
                            self._archive_original(image_path, person_name, batch_id, extraction_result)
                            log_step(f"✗ Image #{seq_num} REJECTED (extraction failed)")
                        else:
                            # Face extracted successfully, now verify against P18 reference
                            current_app.logger.info(f"[Batch {batch_id}] Verifying candidate #{seq_num} against P18 reference")

                            try:
                                # Single reference comparison (P18)
                                ref_path = p18_reference['path']
                                is_same_person, distance = self._verify_faces_with_distance(ref_path, image_dest, batch_id)

                                current_app.logger.info(f"[Batch {batch_id}]   → Result: distance={distance:.4f}, matched={is_same_person}")

                                # Build validation result for metadata
                                validation_result = {
                                    'reference_distance': distance,
                                    'threshold': 0.75,
                                    'passed': is_same_person,
                                    'reference_source': p18_reference.get('source', 'unknown')
                                }

                                if not is_same_person:
                                    # Doesn't match reference - remove extracted face
                                    try:
                                        os.remove(image_dest)
                                        current_app.logger.info(f"[Batch {batch_id}] Removed extracted face (rejected): {image_dest}")
                                    except Exception as e:
                                        current_app.logger.error(f"[Batch {batch_id}] Error removing rejected face: {str(e)}")

                                    # Archive as rejected with validation details
                                    extraction_result = {
                                        'status': 'rejected',
                                        'face_file': None,
                                        'reason': f'Distance {distance:.4f} > 0.75 threshold'
                                    }
                                    validation_result['reason'] = f'Distance {distance:.4f} exceeds threshold 0.75'
                                    self._archive_original(image_path, person_name, batch_id, extraction_result, validation_result)

                                    log_step(f"✗ Image #{seq_num} REJECTED (distance={distance:.4f} > 0.75)")
                                    current_app.logger.info(f"[Batch {batch_id}] ✗ Candidate #{seq_num} does NOT match reference (distance={distance:.4f})")
                                else:
                                    # Matches reference - keep extracted face
                                    current_app.logger.info(f"[Batch {batch_id}] ✓ Candidate #{seq_num} MATCHES reference (distance={distance:.4f})")

                                    # Archive as accepted with validation details
                                    extraction_result = {
                                        'status': 'accepted',
                                        'face_file': image_filename,
                                        'reason': None
                                    }
                                    self._archive_original(image_path, person_name, batch_id, extraction_result, validation_result)

                                    log_step(f"✓ Image #{seq_num} ACCEPTED (distance={distance:.4f})")

                                    # Increment processed count
                                    processed_count += 1
                                    current_app.logger.info(f"[Batch {batch_id}] Processed {processed_count}/{max_images_per_person} images for {person_name}")

                            except Exception as e:
                                current_app.logger.error(f"[Batch {batch_id}] Error in face verification for #{seq_num}: {str(e)}")
                                # Archive as rejected due to verification error
                                extraction_result = {
                                    'status': 'rejected',
                                    'face_file': None,
                                    'reason': f'Verification error: {str(e)}'
                                }
                                self._archive_original(image_path, person_name, batch_id, extraction_result)

                        # Delete the original image regardless of match result
                        try:
                            if os.path.exists(image_path):
                                os.remove(image_path)
                                current_app.logger.info(f"[Batch {batch_id}] Removed {image_path}")
                        except Exception as remove_error:
                            current_app.logger.error(f"[Batch {batch_id}] Error removing image: {str(remove_error)}")
                        
                    except Exception as e:
                        current_app.logger.error(f"[Batch {batch_id}] Error processing image {image_path}: {str(e)}")
                        # Delete the image if there was an error
                        try:
                            if os.path.exists(image_path):
                                os.remove(image_path)
                                current_app.logger.info(f"[Batch {batch_id}] Removed image with error: {image_path}")
                        except Exception as remove_error:
                            current_app.logger.error(f"[Batch {batch_id}] Error removing image with error: {str(remove_error)}")
                
                current_app.logger.info(f"[Batch {batch_id}] Completed DeepFace processing with {processed_count} images for {person_name}")
                log_step(f"VALIDATION COMPLETE: Total {processed_count} images validated and accepted")

                # Regenerate ALL galleries for this person with updated counts
                try:
                    gallery_base = f"storage/training-galleries/{self.domain}/{person_name.lower()}"
                    if os.path.exists(gallery_base):
                        for gallery_folder in os.listdir(gallery_base):
                            gallery_folder_path = os.path.join(gallery_base, gallery_folder)
                            if os.path.isdir(gallery_folder_path):
                                # Extract the batch_id from folder name (e.g., "72654963_jovana_jankovic" -> "72654963_jovana_jankovic")
                                current_app.logger.info(f"[Batch {batch_id}] Regenerating gallery: {gallery_folder}")
                                self._generate_simple_gallery(person_name, gallery_folder, self.domain)
                except Exception as gallery_err:
                    current_app.logger.error(f"[Batch {batch_id}] Error regenerating galleries: {str(gallery_err)}")

                # Generate comparison gallery showing originals vs extracted faces
                gallery_path = self._generate_comparison_gallery(person_name, batch_id)
                if gallery_path:
                    current_app.logger.info(f"[Batch {batch_id}] Comparison gallery available at: {gallery_path}")

                    # Add gallery URL to SERP batch status if this was triggered by SERP training
                    try:
                        # Gallery URL format: /training-galleries/{domain}/{person_name}/{batch_id}/index.html
                        gallery_url = f"/training-galleries/{self.domain}/{person_name}/{batch_id}/index.html"

                        # Find and update the SERP batch status file
                        serp_batches_dir = "storage/serp_batches"
                        if os.path.exists(serp_batches_dir):
                            for status_file in os.listdir(serp_batches_dir):
                                if batch_id in status_file or status_file.endswith('.json'):
                                    status_path = os.path.join(serp_batches_dir, status_file)
                                    try:
                                        with open(status_path, 'r') as f:
                                            status_data = json.load(f)

                                        # Update gallery_url for matching person
                                        if 'people' in status_data:
                                            for person in status_data['people']:
                                                if person.get('folder_name') == person_name.replace(' ', '_'):
                                                    person['gallery_url'] = gallery_url
                                                    current_app.logger.info(f"[Batch {batch_id}] Added gallery_url to batch status: {gallery_url}")

                                            with open(status_path, 'w') as f:
                                                json.dump(status_data, f, indent=2)
                                            break
                                    except Exception as e:
                                        current_app.logger.error(f"[Batch {batch_id}] Error updating batch status with gallery_url: {str(e)}")
                    except Exception as e:
                        current_app.logger.error(f"[Batch {batch_id}] Error adding gallery_url to batch status: {str(e)}")

        except Exception as e:
            log_step(f"FATAL ERROR: {str(e)}")
            current_app.logger.error(f"[Batch {batch_id}] FATAL ERROR in validation thread: {str(e)}")

    def _verify_faces(self, img1_path, img2_path, batch_id="unknown"):
        """
        Verify if two images contain the same person using DeepFace

        Returns:
            bool: True if the same person, False otherwise
        """
        try:
            # Check if both images exist
            if not os.path.exists(img1_path):
                current_app.logger.error(f"[Batch {batch_id}] First image not found: {img1_path}")
                return False

            if not os.path.exists(img2_path):
                current_app.logger.error(f"[Batch {batch_id}] Second image not found: {img2_path}")
                return False

            # Set verification threshold
            # Increased from 0.6 to 0.75 to accept more valid variations of the same person
            # (different angles, lighting, age) while still filtering out wrong people
            threshold = 0.75  # Adjust as needed (lower = more strict)

            # Perform verification
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name="VGG-Face",
                distance_metric="cosine",
                detector_backend="retinaface",
                threshold=threshold,
                enforce_detection=False
            )

            # Log the verification result
            current_app.logger.info(f"[Batch {batch_id}] DeepFace verification result: {result}")

            # Check if the faces match
            is_match = result["verified"]
            distance = result["distance"]

            current_app.logger.info(f"[Batch {batch_id}] Match: {is_match}, Distance: {distance}")

            return is_match

        except Exception as e:
            current_app.logger.error(f"[Batch {batch_id}] Error in face verification: {str(e)}")
            return False  # Default to not a match on error

    def _verify_faces_with_distance(self, img1_path, img2_path, batch_id="unknown"):
        """
        Verify if two images contain the same person using DeepFace and return distance

        Returns:
            tuple: (bool, float) - (is_match, distance)
        """
        try:
            # Check if both images exist
            if not os.path.exists(img1_path):
                current_app.logger.error(f"[Batch {batch_id}] First image not found: {img1_path}")
                return False, 999.0

            if not os.path.exists(img2_path):
                current_app.logger.error(f"[Batch {batch_id}] Second image not found: {img2_path}")
                return False, 999.0

            # Set verification threshold
            threshold = 0.75

            # Perform verification
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name="VGG-Face",
                distance_metric="cosine",
                detector_backend="retinaface",
                threshold=threshold,
                enforce_detection=False
            )

            # Check if the faces match
            is_match = result["verified"]
            distance = result["distance"]

            return is_match, distance

        except Exception as e:
            current_app.logger.error(f"[Batch {batch_id}] Error in face verification: {str(e)}")
            return False, 999.0  # Default to not a match on error
    
    def _is_image_file(self, file_path):
        """
        Check if a file is an image based on its extension
        """
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        ext = os.path.splitext(file_path)[1].lower()
        return ext in allowed_extensions
    
    def _get_file_extension(self, url):
        """
        Extract file extension from URL or default to .jpg
        """
        lower_url = url.lower()
        if lower_url.endswith('.jpg') or lower_url.endswith('.jpeg'):
            return '.jpg'
        elif lower_url.endswith('.png'):
            return '.png'
        elif lower_url.endswith('.webp'):
            return '.webp'
        else:
            return '.jpg'  # Default to jpg

    def _download_and_save_image(self, image_url, file_path):
        """
        Download and save an image with multiple fallback methods
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Define common headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/'
        }
        
        # Try multiple methods to download the image
        methods = [
            # Method 1: Standard request with headers
            lambda: self._download_with_requests(image_url, file_path, headers),
            
            # Method 2: Try without referer (some sites block specific referers)
            lambda: self._download_with_requests(image_url, file_path, {k: v for k, v in headers.items() if k != 'Referer'}),
            
            # Method 3: Try with minimal headers
            lambda: self._download_with_requests(image_url, file_path, {'User-Agent': headers['User-Agent']}),
            
            # Method 4: Try with urllib (different library)
            lambda: self._download_with_urllib(image_url, file_path, headers)
        ]
        
        # Try each method until one succeeds
        for i, method in enumerate(methods):
            try:
                current_app.logger.info(f"Trying download method {i+1} for {image_url}")
                if method():
                    return True
            except Exception as e:
                current_app.logger.warning(f"Download method {i+1} failed for {image_url}: {str(e)}")
        
        return False

    def _download_with_requests(self, url, file_path, headers):
        """Download image using requests library"""
        response = requests.get(url, headers=headers, stream=True, timeout=15)
        
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        return False

    def _download_with_urllib(self, url, file_path, headers):
        """Download image using urllib library"""
        import urllib.request
        
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(file_path, 'wb') as f:
                f.write(response.read())
        return True

    def _extract_and_save_face(self, source_path, dest_path, batch_id="unknown"):
        """
        Extract face from image and save to destination path using existing FaceProcessingService
        
        Returns:
            bool: True if face was extracted and saved, False otherwise
        """
        try:
            # Check if source file exists
            if not os.path.exists(source_path):
                current_app.logger.error(f"[Batch {batch_id}] Source file does not exist: {source_path}")
                return False
            
            # Use the existing FaceProcessingService to extract faces
            from app.services.face_processing_service import FaceProcessingService
            
            # Extract person name from the file path for logging purposes
            filename = os.path.basename(source_path)
            # Try to extract person name from filename (usually in format: name_lastname_timestamp_number.jpg)
            name_parts = filename.split('_')
            if len(name_parts) >= 2:
                person_name = f"{name_parts[0]}_{name_parts[1]}"
            else:
                person_name = "unknown_person"
            
            # Extract faces using the existing method
            current_app.logger.info(f"[Batch {batch_id}] Extracting faces from {source_path}")
            face_objs = FaceProcessingService.extract_faces_with_timeout(source_path, None, person_name)
            
            if not face_objs:
                current_app.logger.warning(f"[Batch {batch_id}] No faces detected in {source_path}")
                ImageRejectionLogger.log_rejection(
                    image_path=source_path,
                    person_name=person_name,
                    reason=ImageRejectionLogger.REASON_NO_FACE,
                    details={"batch_id": batch_id},
                    batch_id=batch_id
                )
                return False
            
            # Validate faces using the same logic as in face_processing_service.py
            valid_faces = []
            invalid_faces = []
            number_of_detected_faces = len(face_objs)
            
            for i, item in enumerate(face_objs):
                face_image_array = item['face']
                w, h = face_image_array.shape[1], face_image_array.shape[0]
                
                current_app.logger.info(f"[Batch {batch_id}] Face {i+1} dimensions: {w}x{h}")
                
                # Check face size
                if w < 70 or h < 70:
                    current_app.logger.warning(f"[Batch {batch_id}] Face too small: {w}x{h}")
                    invalid_faces.append((face_image_array, i, ImageRejectionLogger.REASON_FACE_TOO_SMALL, {"width": w, "height": h}))
                    continue

                # Check for blurriness
                if FaceProcessingService.is_blurred(face_image_array, number_of_detected_faces):
                    current_app.logger.warning(f"[Batch {batch_id}] Face is blurry")
                    invalid_faces.append((face_image_array, i, ImageRejectionLogger.REASON_BLURRY, {}))
                    continue
                
                valid_faces.append((face_image_array, i))
            
            # Process results
            if len(valid_faces) > 1:
                current_app.logger.warning(f"[Batch {batch_id}] Multiple valid faces detected")
                ImageRejectionLogger.log_rejection(
                    image_path=source_path,
                    person_name=person_name,
                    reason=ImageRejectionLogger.REASON_MULTIPLE_FACES,
                    details={"valid_faces_count": len(valid_faces), "batch_id": batch_id},
                    batch_id=batch_id
                )
                return False
            elif len(valid_faces) == 0:
                # Log the specific rejection reason from invalid_faces
                if len(invalid_faces) >= 1:
                    # Get rejection reason from the first invalid face
                    _, _, reason, details = invalid_faces[0]
                    details["batch_id"] = batch_id
                    details["invalid_faces_count"] = len(invalid_faces)
                    ImageRejectionLogger.log_rejection(
                        image_path=source_path,
                        person_name=person_name,
                        reason=reason if len(invalid_faces) == 1 else ImageRejectionLogger.REASON_MULTIPLE_INVALID,
                        details=details,
                        batch_id=batch_id
                    )
                else:
                    ImageRejectionLogger.log_rejection(
                        image_path=source_path,
                        person_name=person_name,
                        reason=ImageRejectionLogger.REASON_NO_FACE,
                        details={"batch_id": batch_id},
                        batch_id=batch_id
                    )
                return False
            elif len(valid_faces) == 1:
                if len(invalid_faces) > 0:
                    current_app.logger.warning(f"[Batch {batch_id}] Multiple faces detected (1 valid, {len(invalid_faces)} invalid)")
                    ImageRejectionLogger.log_rejection(
                        image_path=source_path,
                        person_name=person_name,
                        reason=ImageRejectionLogger.REASON_MULTIPLE_FACES,
                        details={"valid_faces_count": 1, "invalid_faces_count": len(invalid_faces), "batch_id": batch_id},
                        batch_id=batch_id
                    )
                    return False
            
            # Get the valid face
            valid_face_array, face_index = valid_faces[0]
            face = face_objs[face_index]
            
            # Check if we have a valid face region
            if "facial_area" not in face:
                current_app.logger.warning(f"[Batch {batch_id}] No facial area found in detection result for {source_path}")
                return False
            
            # Load the image with OpenCV
            img = cv2.imread(source_path)
            if img is None:
                current_app.logger.error(f"[Batch {batch_id}] Failed to load image: {source_path}")
                return False
            
            # Extract face coordinates
            facial_area = face["facial_area"]
            x = facial_area["x"]
            y = facial_area["y"]
            w = facial_area["w"]
            h = facial_area["h"]
            
            # Add some margin (20%)
            margin = 0.2
            x_margin = int(w * margin)
            y_margin = int(h * margin)
            
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Calculate new coordinates with margin
            x1 = max(0, x - x_margin)
            y1 = max(0, y - y_margin)
            x2 = min(width, x + w + x_margin)
            y2 = min(height, y + h + y_margin)
            
            # Crop the face
            face_img = img[y1:y2, x1:x2]
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Save the cropped face
            cv2.imwrite(dest_path, face_img)
            
            current_app.logger.info(f"[Batch {batch_id}] Successfully extracted face from {source_path} to {dest_path}")
            return True
            
        except Exception as e:
            current_app.logger.error(f"[Batch {batch_id}] Error extracting face from {source_path}: {str(e)}")
            return False

    def _calculate_image_hash(self, image):
        """
        Calculate a perceptual hash of an image to identify duplicates
        
        Args:
            image: OpenCV image
            
        Returns:
            str: Hash of the image
        """
        # Resize the image to 8x8
        img_resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        if len(img_resized.shape) > 2:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_resized
        
        # Calculate average pixel value
        avg_pixel = img_gray.mean()
        
        # Create binary hash
        hash_str = ""
        for i in range(8):
            for j in range(8):
                hash_str += "1" if img_gray[i, j] > avg_pixel else "0"
        
        return hash_str 

    def _normalize_for_filename(self, text):
        """
        Normalize text to be used in filenames
        """
        if not text:
            return ""
        
        import re
        
        # Replace spaces with underscores
        text = text.replace(' ', '_')
        
        # Remove any characters that aren't alphanumeric or underscores
        text = re.sub(r'[^\w]', '', text)
        
        return text 

    def _ensure_ascii_path(self, text):
        """
        Ensure the path contains only ASCII characters to avoid encoding issues
        """
        import unicodedata
        import re
        
        # Normalize unicode characters
        normalized = unicodedata.normalize('NFKD', text)
        
        # Remove accents and convert to ASCII
        ascii_text = ''.join([c for c in normalized if not unicodedata.combining(c)])
        
        # Replace any remaining non-ASCII characters
        ascii_text = ascii_text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove any characters that aren't safe for filenames
        ascii_text = re.sub(r'[^\w_-]', '', ascii_text)

        return ascii_text

    def _archive_original(self, source_image_path, person_name, batch_id, extraction_result, validation_result=None):
        """
        Archive original SERP download with full metadata linkage.

        Args:
            source_image_path: Path to the original SERP download (BEFORE processing)
            person_name: Name of the person (for subdirectory)
            batch_id: Batch ID (for subdirectory)
            extraction_result: Dict with extraction info:
                {
                    'status': 'accepted'|'rejected'|'reference',
                    'face_file': 'extracted_face.jpg' or None,
                    'reason': 'rejection reason' or None,
                    'bbox': [x, y, w, h] or None,
                    'confidence': 0.98 or None
                }
            validation_result: Dict with validation info (optional):
                {
                    'reference_distance': 0.42,
                    'threshold': 0.75,
                    'passed': True,
                    'reason': None or 'Distance 0.82 > 0.75 threshold'
                }
        """
        try:
            # Create archive directory structure
            archive_dir = os.path.join(self.originals_archive_path, person_name, batch_id)
            os.makedirs(archive_dir, exist_ok=True)

            # Extract sequence number from source filename
            source_filename = os.path.basename(source_image_path)
            seq_num = self._extract_sequence_from_filename(source_filename)

            # Copy original SERP download to archive with standardized naming
            if seq_num:
                ext = os.path.splitext(source_filename)[1]
                archive_name = f"download_{seq_num:03d}{ext}"
            else:
                archive_name = source_filename

            archive_path = os.path.join(archive_dir, archive_name)

            # Only copy if source exists and archive doesn't already exist
            if os.path.exists(source_image_path) and not os.path.exists(archive_path):
                shutil.copy2(source_image_path, archive_path)
                current_app.logger.info(f"[Batch {batch_id}] Archived original: {archive_name}")

            # Create/update metadata JSON
            metadata_path = os.path.join(archive_dir, 'metadata.json')
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            # Build metadata entry
            entry = {
                'sequence': seq_num,
                'original_filename': source_filename,
                'archived_at': datetime.now().isoformat(),
                'extraction': extraction_result
            }

            # Add validation result if provided
            if validation_result:
                entry['validation'] = validation_result

            metadata[archive_name] = entry

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            status = extraction_result.get('status', 'unknown')
            current_app.logger.info(f"[Batch {batch_id}] Metadata updated: {archive_name} (status: {status})")

        except Exception as e:
            current_app.logger.error(f"[Batch {batch_id}] Error archiving original {source_image_path}: {str(e)}")

    def _extract_sequence_from_filename(self, filename):
        """Extract sequence number from filename"""
        import re
        match = re.search(r'_(\d+)\.(jpg|png|jpeg|gif|webp)$', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _generate_simple_gallery(self, person_name, batch_id, domain='serbia'):
        """
        Generate gallery showing source images with extracted faces and validation results.

        Args:
            person_name: Name of the person
            batch_id: Batch ID for organizing galleries
            domain: Domain (default: serbia)

        Returns:
            Path to generated gallery HTML or None if failed
        """
        try:
            import json
            from urllib.parse import quote

            # Get paths
            gallery_dir = os.path.join('storage/training-galleries', domain, person_name.lower(), batch_id)
            originals_dir = os.path.join('storage/serp_originals', domain, person_name)

            # Create gallery directory
            os.makedirs(gallery_dir, exist_ok=True)

            # Build extraction sources from SERP originals
            sources = {}
            total_extracted = 0
            total_accepted = 0

            if os.path.exists(originals_dir):
                for source_hash in sorted(os.listdir(originals_dir)):
                    source_path = os.path.join(originals_dir, source_hash)
                    if not os.path.isdir(source_path):
                        continue

                    metadata_path = os.path.join(source_path, 'metadata.json')
                    if not os.path.exists(metadata_path):
                        continue

                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)

                        source_accepted = []
                        source_rejected = []

                        # Build a map of sequence numbers to source images
                        seq_to_source = {}
                        for item in os.listdir(source_path):
                            if item.startswith('seq_') and item.endswith(('.jpg', '.jpeg', '.png')):
                                # Extract sequence number from "seq_NNN_filename.jpg"
                                parts = item.split('_', 2)
                                if len(parts) >= 2:
                                    try:
                                        seq_num = int(parts[1])
                                        seq_to_source[seq_num] = item
                                    except ValueError:
                                        pass

                        for filename, info in metadata.items():
                            if not filename.endswith(('.jpg', '.jpeg', '.png')):
                                continue

                            total_extracted += 1
                            seq_num = info.get('sequence')

                            # Get extraction and validation info from new structure
                            extraction = info.get('extraction', {})
                            validation = info.get('validation', {})
                            status = extraction.get('status', info.get('status', 'unknown'))
                            face_file = extraction.get('face_file')
                            reason = extraction.get('reason') or info.get('reason', 'Unknown')

                            # Get validation distance if available
                            distance = validation.get('reference_distance')
                            if distance is not None:
                                reason = f"Distance: {distance:.4f} (threshold: 0.75)"
                                if validation.get('reason'):
                                    reason = validation.get('reason')

                            if status in ('accepted', 'reference'):
                                total_accepted += 1
                                source_accepted.append({
                                    'filename': filename,
                                    'face_file': face_file,
                                    'distance': distance,
                                    'is_reference': extraction.get('is_reference', False)
                                })
                            else:
                                source_rejected.append({
                                    'filename': filename,
                                    'reason': reason,
                                    'distance': distance
                                })

                        if source_accepted or source_rejected:
                            sources[source_hash] = {
                                'accepted': source_accepted,
                                'rejected': source_rejected
                            }
                    except Exception as e:
                        current_app.logger.warning(f"[Gallery {batch_id}] Error reading metadata for {source_hash}: {str(e)}")

            # Generate HTML
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Results - {person_name.replace('_', ' ')}</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ text-align: center; color: #333; margin-bottom: 5px; }}
        .batch-info {{ text-align: center; margin: 10px 0; color: #666; font-size: 14px; }}
        .stats {{ display: flex; gap: 15px; justify-content: center; margin: 20px 0; flex-wrap: wrap; }}
        .stat-box {{ padding: 15px 25px; border-radius: 8px; text-align: center; min-width: 120px; }}
        .stat-box.total {{ background: #d1ecf1; border: 2px solid #17a2b8; }}
        .stat-box.accepted {{ background: #d4edda; border: 2px solid #28a745; }}
        .stat-box.rejected {{ background: #f8d7da; border: 2px solid #dc3545; }}
        .stat-value {{ font-size: 32px; font-weight: bold; margin-bottom: 3px; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .source-block {{ margin: 20px 0; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .source-header {{ font-size: 14px; font-weight: bold; color: #333; margin-bottom: 10px; padding: 8px; background: #f9f9f9; border-radius: 4px; }}
        .source-hash {{ font-family: monospace; font-size: 12px; color: #666; }}
        .source-image-display {{ margin: 15px 0; padding: 10px; background: #f5f5f5; border: 1px solid #ddd; border-radius: 4px; text-align: center; }}
        .source-image-display img {{ max-width: 100%; max-height: 400px; border-radius: 4px; }}
        .source-image-label {{ margin-top: 8px; font-size: 12px; color: #666; font-weight: bold; }}
        .extraction-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; }}
        .image-box {{ border-radius: 6px; padding: 6px; text-align: center; border: 2px solid #ddd; }}
        .image-box.accepted {{ border-color: #28a745; background: #f8fff8; }}
        .image-box.rejected {{ border-color: #dc3545; background: #fff8f8; opacity: 0.85; }}
        .image-box img {{ max-width: 100%; max-height: 120px; border-radius: 4px; }}
        .image-label {{ margin: 4px 0 0 0; font-size: 9px; color: #666; word-break: break-all; font-weight: bold; }}
        .rejection-reason {{ font-size: 8px; color: #dc3545; margin-top: 2px; font-style: italic; }}
        .empty-msg {{ text-align: center; color: #999; padding: 20px; font-style: italic; font-size: 14px; }}
        .subsection-title {{ font-size: 12px; font-weight: bold; color: #666; margin: 8px 0 6px 0; padding: 4px; background: #f0f0f0; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>Training Results: {person_name.replace('_', ' ')}</h1>
    <div class="batch-info">Batch ID: {batch_id}</div>

    <div class="stats">
        <div class="stat-box total">
            <div class="stat-value">{total_extracted}</div>
            <div class="stat-label">Total Extracted</div>
        </div>
        <div class="stat-box accepted">
            <div class="stat-value" style="color: #28a745;">{total_accepted}</div>
            <div class="stat-label">✓ Accepted ({(total_accepted * 100 // total_extracted) if total_extracted > 0 else 0}%)</div>
        </div>
        <div class="stat-box rejected">
            <div class="stat-value" style="color: #dc3545;">{total_extracted - total_accepted}</div>
            <div class="stat-label">✗ Rejected</div>
        </div>
    </div>
"""

            # Add source blocks
            if sources:
                for source_hash in sorted(sources.keys()):
                    source_info = sources[source_hash]
                    accepted_list = source_info['accepted']
                    rejected_list = source_info['rejected']
                    accepted_count = len(accepted_list)
                    rejected_count = len(rejected_list)

                    html += f"""
    <div class="source-block">
        <div class="source-header">
            Processing Batch: <span class="source-hash">{source_hash[:8]}...</span>
            ({accepted_count} ✓ {rejected_count} ✗)
        </div>
"""

                    # Show reference image first (highlighted)
                    reference_items = [i for i in accepted_list if i.get('is_reference')]
                    if reference_items:
                        ref_item = reference_items[0]
                        ref_filename = ref_item['filename']
                        ref_url = f"/training-galleries/serp_originals/{domain}/{person_name}/{source_hash}/{quote(ref_filename)}"
                        html += f"""
        <div style="margin: 15px 0; padding: 15px; background: #e3f2fd; border: 3px solid #1976d2; border-radius: 8px;">
            <h3 style="color: #1976d2; margin: 0 0 10px 0;">Reference Image (P18 from Wikidata)</h3>
            <div style="display: flex; gap: 20px; align-items: center;">
                <div style="text-align: center;">
                    <img src="{ref_url}" alt="Reference" style="max-width: 300px; max-height: 300px; border-radius: 4px; border: 2px solid #1976d2;">
                    <p style="margin: 5px 0; font-size: 12px; color: #1976d2;"><strong>Original Download</strong></p>
                </div>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 11px; color: #666;">All other images are compared against this reference (distance threshold: 0.75)</p>
        </div>
"""

                    # Show accepted faces (non-reference)
                    non_ref_accepted = [i for i in accepted_list if not i.get('is_reference')]
                    if non_ref_accepted:
                        html += '<div class="subsection-title">Accepted Faces ({} validated)</div>\n'.format(len(non_ref_accepted))
                        for item in non_ref_accepted:
                            filename = item['filename']
                            distance = item.get('distance')
                            img_url = f"/training-galleries/serp_originals/{domain}/{person_name}/{source_hash}/{quote(filename)}"

                            distance_str = f"Distance: {distance:.4f}" if distance is not None else ""

                            html += f"""
        <div style="margin-bottom: 15px; padding: 10px; background: #f8fff8; border: 2px solid #28a745; border-radius: 4px;">
            <div style="display: flex; gap: 15px; align-items: center;">
                <img src="{img_url}" alt="{filename}" style="max-width: 150px; max-height: 150px; border-radius: 4px;">
                <div>
                    <p style="margin: 0; color: #28a745; font-weight: bold;">ACCEPTED</p>
                    <p style="margin: 5px 0; font-size: 11px; color: #666;">{distance_str}</p>
                </div>
            </div>
        </div>
"""

                    # Show rejected faces
                    if rejected_list:
                        html += '<div class="subsection-title" style="margin-top: 30px;">Rejected Faces ({} not matching)</div>\n        <div class="extraction-grid">\n'.format(rejected_count)
                        for item in rejected_list:
                            filename = item['filename']
                            reason = item['reason']
                            distance = item.get('distance')
                            img_url = f"/training-galleries/serp_originals/{domain}/{person_name}/{source_hash}/{quote(filename)}"

                            distance_str = f"({distance:.4f})" if distance is not None else ""

                            html += f"""            <div class="image-box rejected">
                <img src="{img_url}" alt="{filename}" loading="lazy">
                <div class="image-label">{distance_str}</div>
                <div class="rejection-reason">{reason}</div>
            </div>
"""
                        html += '        </div>\n'

                    html += '    </div>\n'
            else:
                html += '<p class="empty-msg">No extraction sources found</p>'

            html += """
</body>
</html>
"""

            # Write HTML file
            gallery_path = os.path.join(gallery_dir, 'index.html')
            with open(gallery_path, 'w', encoding='utf-8') as f:
                f.write(html)

            current_app.logger.info(f"[Gallery {batch_id}] ✓ Gallery generated: {total_extracted} extracted, {total_accepted} accepted")
            return gallery_path

        except Exception as e:
            current_app.logger.error(f"[Gallery {batch_id}] Failed to generate gallery: {str(e)}", exc_info=True)
            return None

    def _generate_comparison_gallery(self, person_name, batch_id):
        """
        Generate HTML comparison gallery showing originals vs extracted faces

        Args:
            person_name: Name of the person
            batch_id: Batch ID
        """
        try:
            archive_dir = os.path.join(self.originals_archive_path, person_name, batch_id)
            metadata_path = os.path.join(archive_dir, 'metadata.json')

            if not os.path.exists(metadata_path):
                current_app.logger.warning(f"[Batch {batch_id}] No metadata found for gallery generation")
                return None

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Get extracted faces directory
            person_dir = os.path.join(self.training_pass_path, person_name)

            # Count stats
            accepted_count = sum(1 for m in metadata.values() if m['status'] == 'accepted')
            rejected_count = sum(1 for m in metadata.values() if m['status'] == 'rejected')
            total_count = len(metadata)

            # Generate HTML
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SERP Training Results - {person_name}</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .batch-info {{
            text-align: center;
            margin: 10px 0;
            color: #666;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin: 20px 0;
        }}
        .stat-box {{
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            min-width: 150px;
        }}
        .stat-box.accepted {{
            background: #d4edda;
            border: 2px solid #28a745;
        }}
        .stat-box.rejected {{
            background: #f8d7da;
            border: 2px solid #dc3545;
        }}
        .stat-box.total {{
            background: #d1ecf1;
            border: 2px solid #17a2b8;
        }}
        .stat-value {{
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .comparison-item {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .comparison-item.accepted {{
            border: 3px solid #28a745;
        }}
        .comparison-item.rejected {{
            border: 3px solid #dc3545;
            opacity: 0.7;
        }}
        .comparison-item h3 {{
            margin: 0 0 10px 0;
            font-size: 16px;
        }}
        .comparison-item h3.accepted {{
            color: #28a745;
        }}
        .comparison-item h3.rejected {{
            color: #dc3545;
        }}
        .image-pair {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        .image-box {{
            text-align: center;
        }}
        .image-box img {{
            max-width: 100%;
            max-height: 300px;
            border: 2px solid #ddd;
            border-radius: 4px;
        }}
        .image-box p {{
            margin: 5px 0 0 0;
            font-size: 12px;
            color: #666;
            font-weight: bold;
        }}
        .no-extraction {{
            grid-column: 1 / -1;
            text-align: center;
            padding: 40px;
            color: #999;
        }}
        .rejection-reason {{
            margin-top: 10px;
            padding: 10px;
            background: #fff3cd;
            border-radius: 4px;
            font-size: 12px;
            color: #856404;
        }}
    </style>
</head>
<body>
    <h1>SERP Training Results: {person_name}</h1>
    <div class="batch-info">Batch ID: {batch_id}</div>

    <div class="stats">
        <div class="stat-box total">
            <div class="stat-value">{total_count}</div>
            <div class="stat-label">Total Images</div>
        </div>
        <div class="stat-box accepted">
            <div class="stat-value">{accepted_count}</div>
            <div class="stat-label">Accepted ({accepted_count*100//total_count if total_count > 0 else 0}%)</div>
        </div>
        <div class="stat-box rejected">
            <div class="stat-value">{rejected_count}</div>
            <div class="stat-label">Rejected ({rejected_count*100//total_count if total_count > 0 else 0}%)</div>
        </div>
    </div>

    <div class="comparison-grid">
"""

            # Sort by sequence number
            sorted_items = sorted(metadata.items(), key=lambda x: x[1].get('sequence', 999))

            for filename, meta in sorted_items:
                seq_num = meta.get('sequence', '?')
                status = meta['status']
                reason = meta.get('reason', '')
                status_class = 'accepted' if status == 'accepted' else 'rejected'
                status_text = f"Sequence #{seq_num} - {'ACCEPTED ✓' if status == 'accepted' else 'REJECTED ✗'}"

                original_path = f"{person_name}/{batch_id}/{quote(filename)}"

                html += f"""
        <div class="comparison-item {status_class}">
            <h3 class="{status_class}">{status_text}</h3>
"""

                if status == 'accepted':
                    # Find corresponding extracted face
                    extracted_filename = None
                    if os.path.exists(person_dir):
                        for f in os.listdir(person_dir):
                            if f"_{seq_num}." in f:
                                extracted_filename = f
                                break

                    if extracted_filename:
                        extracted_path = f"../../trainingPass/{self.domain}/{person_name}/{quote(extracted_filename)}"
                        html += f"""
            <div class="image-pair">
                <div class="image-box">
                    <img src="{original_path}" alt="Original #{seq_num}" loading="lazy">
                    <p>ORIGINAL IMAGE</p>
                </div>
                <div class="image-box">
                    <img src="{extracted_path}" alt="Extracted #{seq_num}" loading="lazy">
                    <p>EXTRACTED FACE</p>
                </div>
            </div>
"""
                    else:
                        html += f"""
            <div class="image-pair">
                <div class="image-box">
                    <img src="{original_path}" alt="Original #{seq_num}" loading="lazy">
                    <p>ORIGINAL IMAGE</p>
                </div>
                <div class="no-extraction">
                    <p>⚠️ Extracted face not found</p>
                </div>
            </div>
"""
                else:
                    # Rejected - show only original with detailed reason
                    details_obj = meta.get('details', {})

                    # Format details for display
                    details_html = ""
                    if details_obj:
                        # Show verification attempts if available
                        if 'verification_attempts' in details_obj:
                            attempts = details_obj['verification_attempts']
                            details_html += "<br><br><strong>Verification Against References:</strong><br>"
                            details_html += "<table style='margin-left: 20px; border-collapse: collapse;'>"
                            details_html += "<tr style='background: #f0f0f0;'><th style='padding: 5px; border: 1px solid #ddd;'>Reference #</th><th style='padding: 5px; border: 1px solid #ddd;'>Distance</th><th style='padding: 5px; border: 1px solid #ddd;'>Matched?</th></tr>"
                            for attempt in attempts:
                                ref_seq = attempt.get('reference_seq', '?')
                                distance = attempt.get('distance', 999)
                                matched = attempt.get('matched', False)
                                match_icon = "✓" if matched else "✗"
                                match_color = "#28a745" if matched else "#dc3545"
                                details_html += f"<tr><td style='padding: 5px; border: 1px solid #ddd;'>#{ref_seq}</td><td style='padding: 5px; border: 1px solid #ddd;'>{distance:.4f}</td><td style='padding: 5px; border: 1px solid #ddd; color: {match_color};'>{match_icon}</td></tr>"
                            details_html += "</table>"
                            details_html += "<br><em>Threshold: 0.75 (lower distance = more similar)</em>"

                        if 'distance' in details_obj:
                            details_html += f"<br><strong>Similarity Distance:</strong> {details_obj['distance']:.4f} (threshold: 0.75)"
                        if 'width' in details_obj and 'height' in details_obj:
                            details_html += f"<br><strong>Face Dimensions:</strong> {details_obj['width']}x{details_obj['height']}px (minimum: 70x70)"
                        if 'valid_faces_count' in details_obj:
                            details_html += f"<br><strong>Valid Faces:</strong> {details_obj['valid_faces_count']}"
                        if 'invalid_faces_count' in details_obj:
                            details_html += f"<br><strong>Invalid Faces:</strong> {details_obj['invalid_faces_count']}"

                    html += f"""
            <div class="image-pair">
                <div class="image-box">
                    <img src="{original_path}" alt="Original #{seq_num}" loading="lazy">
                    <p>ORIGINAL IMAGE (REJECTED)</p>
                </div>
                <div class="no-extraction">
                    <p>❌ Not extracted</p>
                </div>
            </div>
"""
                    if reason:
                        html += f"""
            <div class="rejection-reason">
                <strong>Rejection Reason:</strong> {reason}{details_html}
            </div>
"""

                html += """
        </div>
"""

            html += """
    </div>
</body>
</html>
"""

            # Write HTML to archive directory
            gallery_path = os.path.join(archive_dir, 'index.html')
            with open(gallery_path, 'w', encoding='utf-8') as f:
                f.write(html)

            current_app.logger.info(f"[Batch {batch_id}] Generated comparison gallery at: {gallery_path}")

            # Return relative path for logging
            return f"storage/serp_originals/{person_name}/{batch_id}/index.html"

        except Exception as e:
            current_app.logger.error(f"[Batch {batch_id}] Error generating comparison gallery: {str(e)}")
            return None