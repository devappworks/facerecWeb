import os
from werkzeug.utils import secure_filename
from datetime import datetime
from threading import Thread
from io import BytesIO
import time
from app.services.face_processing_service import FaceProcessingService
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
                        return image_data  # Vrati original ako je već manja
                else:
                    # Vertikalna slika
                    if height > ImageService.MAX_IMAGE_SIZE[1]:
                        ratio = ImageService.MAX_IMAGE_SIZE[1] / height
                        new_height = ImageService.MAX_IMAGE_SIZE[1]
                        new_width = int(width * ratio)
                    else:
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

            querystring = {"query": f"{encoded_search_term}","limit":"100","size":"1024x768_and_more","type":"face","region":"us"}
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
            
            # Limit to maximum 50 images
            max_images = 70
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
        4. Stop processing when 40 images are reached for a person
        """
        # Push the application context
        with app_context:
            try:
                current_app.logger.info(f"[Batch {batch_id}] Starting DeepFace processing in background thread")
                
                if not saved_images:
                    current_app.logger.warning(f"[Batch {batch_id}] No images to process with DeepFace")
                    return
                
                # Create a list to track files that are being processed by this thread
                processing_files = []
                
                # Use normalized name for folder
                if hasattr(self, 'original_name') and hasattr(self, 'original_last_name'):
                    # Use normalized name for folder (not original)
                    normalized_name = self._ensure_ascii_path(self._normalize_for_filename(self.original_name))
                    normalized_last_name = self._ensure_ascii_path(self._normalize_for_filename(self.original_last_name))
                    person_name = f"{normalized_name}_{normalized_last_name}"
                    current_app.logger.info(f"[Batch {batch_id}] Using normalized name for folder: {person_name}")
                else:
                    # Extract person name from the filename of the first image
                    first_image = saved_images[0]
                    first_image_path = first_image["path"]
                    filename = os.path.basename(first_image_path)
                    name_parts = filename.split('_')
                    if len(name_parts) >= 2:
                        person_name = f"{name_parts[0]}_{name_parts[1]}"
                    else:
                        person_name = "unknown"
                    # Ensure ASCII-only path
                    person_name = self._ensure_ascii_path(person_name)
                    current_app.logger.info(f"[Batch {batch_id}] Extracted name from filename: {person_name}")
                
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
                
                for filename in os.listdir(training_dir):
                    if filename.startswith(f"{normalized_name}_{normalized_last_name}"):
                        file_path = os.path.join(training_dir, filename)
                        if os.path.isfile(file_path) and self._is_image_file(file_path):
                            image_files.append(file_path)
                            processing_files.append(file_path)
                
                if not image_files:
                    current_app.logger.warning(f"[Batch {batch_id}] No images found to process in {training_dir}")
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
                
                # Find the first three valid images
                valid_image_paths = []
                valid_image_dests = []
                max_reference_images = 3
                
                # Counter for images processed so far
                processed_count = len(existing_images)
                max_images_per_person = 40
                
                # Try each image until we find three valid ones or run out of images
                for image_path in image_files[:]:  # Create a copy of the list to safely modify it
                    if len(valid_image_paths) >= max_reference_images:
                        break
                        
                    try:
                        # Check if file still exists
                        if not os.path.exists(image_path):
                            current_app.logger.info(f"[Batch {batch_id}] Image no longer exists, skipping: {image_path}")
                            image_files.remove(image_path)
                            continue
                        
                        # Try to extract face from this image
                        image_filename = os.path.basename(image_path)
                        image_dest = os.path.join(person_dir, image_filename)
                        
                        face_extracted = self._extract_and_save_face(image_path, image_dest, batch_id)
                        
                        if face_extracted:
                            # We found a valid image!
                            valid_image_paths.append(image_path)
                            valid_image_dests.append(image_dest)
                            seq_num = extract_sequence_number(image_path)
                            current_app.logger.info(f"[Batch {batch_id}] Found valid reference image #{len(valid_image_paths)} (sequence #{seq_num}): {image_path}")
                            
                            # Increment processed count
                            processed_count += 1
                            current_app.logger.info(f"[Batch {batch_id}] Processed {processed_count}/{max_images_per_person} images for {person_name}")
                            
                            # Remove this image from the list of images to process
                            image_files.remove(image_path)
                            
                            # Remove the original image
                            try:
                                os.remove(image_path)
                                current_app.logger.info(f"[Batch {batch_id}] Removed valid reference image from original location: {image_path}")
                            except Exception as e:
                                current_app.logger.error(f"[Batch {batch_id}] Error removing valid reference image: {str(e)}")
                            
                            # Check if we've reached the maximum number of images
                            if processed_count >= max_images_per_person:
                                current_app.logger.info(f"[Batch {batch_id}] Reached maximum of {max_images_per_person} images for {person_name}, stopping processing")
                                
                                # Clean up any remaining images
                                for remaining_path in image_files:
                                    try:
                                        if os.path.exists(remaining_path):
                                            os.remove(remaining_path)
                                            current_app.logger.info(f"[Batch {batch_id}] Removed remaining image after reaching limit: {remaining_path}")
                                    except Exception as e:
                                        current_app.logger.error(f"[Batch {batch_id}] Error removing remaining image: {str(e)}")
                                
                                return
                        else:
                            # This image is not valid, delete it and try the next one
                            current_app.logger.warning(f"[Batch {batch_id}] Image not valid, trying next one: {image_path}")
                            try:
                                os.remove(image_path)
                                current_app.logger.info(f"[Batch {batch_id}] Removed invalid image: {image_path}")
                                image_files.remove(image_path)
                            except Exception as e:
                                current_app.logger.error(f"[Batch {batch_id}] Error removing invalid image: {str(e)}")
                    
                    except Exception as e:
                        current_app.logger.error(f"[Batch {batch_id}] Error processing potential reference image {image_path}: {str(e)}")
                        # Delete the image if there was an error
                        try:
                            if os.path.exists(image_path):
                                os.remove(image_path)
                                current_app.logger.info(f"[Batch {batch_id}] Removed image with error: {image_path}")
                                image_files.remove(image_path)
                        except Exception as remove_error:
                            current_app.logger.error(f"[Batch {batch_id}] Error removing image with error: {str(remove_error)}")
                
                # Check if we found at least one valid reference image
                if not valid_image_dests:
                    current_app.logger.error(f"[Batch {batch_id}] No valid reference images found, stopping processing")
                    
                    # Clean up any remaining images
                    for image_path in image_files:
                        try:
                            if os.path.exists(image_path):
                                os.remove(image_path)
                                current_app.logger.info(f"[Batch {batch_id}] Cleaned up remaining image: {image_path}")
                        except Exception as e:
                            current_app.logger.error(f"[Batch {batch_id}] Error cleaning up image: {str(e)}")
                    
                    return
                
                current_app.logger.info(f"[Batch {batch_id}] Found {len(valid_image_dests)} valid reference images")
                
                # Keep track of processed image hashes to avoid duplicates
                processed_hashes = set()
                
                # Calculate hashes of the reference images for comparison
                reference_images = []
                for i, image_dest in enumerate(valid_image_dests):
                    try:
                        img = cv2.imread(image_dest)
                        if img is None:
                            current_app.logger.error(f"[Batch {batch_id}] Failed to read reference image: {image_dest}")
                            continue
                        
                        img_hash = self._calculate_image_hash(img)
                        processed_hashes.add(img_hash)
                        reference_images.append({
                            'path': image_dest,
                            'hash': img_hash,
                            'image': img
                        })
                        current_app.logger.info(f"[Batch {batch_id}] Added reference image #{i+1}: {image_dest}")
                    except Exception as e:
                        current_app.logger.error(f"[Batch {batch_id}] Error calculating hash for reference image: {str(e)}")
                
                if not reference_images:
                    current_app.logger.error(f"[Batch {batch_id}] Failed to load any reference images, stopping processing")
                    return
                
                # Process each remaining image
                for image_path in image_files:
                    # Check if we've reached the maximum number of images
                    if processed_count >= max_images_per_person:
                        current_app.logger.info(f"[Batch {batch_id}] Reached maximum of {max_images_per_person} images for {person_name}, stopping processing")
                        
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
                        
                        # Try to verify with each reference image
                        is_match = False
                        matched_reference = None
                        
                        for i, ref_image in enumerate(reference_images):
                            try:
                                current_app.logger.info(f"[Batch {batch_id}] Comparing with reference image #{i+1}")
                                is_same_person = self._verify_faces(ref_image['path'], image_path, batch_id)
                                
                                if is_same_person:
                                    is_match = True
                                    matched_reference = ref_image
                                    current_app.logger.info(f"[Batch {batch_id}] Match found with reference image #{i+1}")
                                    break
                            except Exception as e:
                                current_app.logger.error(f"[Batch {batch_id}] Error in face verification with reference #{i+1}: {str(e)}")
                        
                        if is_match and matched_reference:
                            # Extract face and save to trainingPass/person_name
                            image_filename = os.path.basename(image_path)
                            image_dest = os.path.join(person_dir, image_filename)
                            
                            # Extract face from the image
                            face_extracted = self._extract_and_save_face(image_path, image_dest, batch_id)
                            
                            if not face_extracted:
                                # If face extraction failed, just delete the original image
                                current_app.logger.warning(f"[Batch {batch_id}] Face extraction failed, skipping: {image_path}")
                            else:
                                current_app.logger.info(f"[Batch {batch_id}] Verified match: Processed {image_path} to {image_dest}")
                                # Increment processed count
                                processed_count += 1
                                current_app.logger.info(f"[Batch {batch_id}] Processed {processed_count}/{max_images_per_person} images for {person_name}")
                        else:
                            current_app.logger.info(f"[Batch {batch_id}] Not a match with any reference image: {image_path}")
                        
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
                
            except Exception as e:
                current_app.logger.error(f"[Batch {batch_id}] Error in DeepFace processing thread: {str(e)}")
    
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
            threshold = 0.6  # Adjust as needed (lower = more strict)
            
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
                    invalid_faces.append((face_image_array, i))
                    continue
                
                # Check for blurriness
                if FaceProcessingService.is_blurred(face_image_array, number_of_detected_faces):
                    current_app.logger.warning(f"[Batch {batch_id}] Face is blurry")
                    invalid_faces.append((face_image_array, i))
                    continue
                
                valid_faces.append((face_image_array, i))
            
            # Process results
            if len(valid_faces) > 1:
                current_app.logger.warning(f"[Batch {batch_id}] Multiple valid faces detected")
                return False
            elif len(valid_faces) == 0:
                if len(invalid_faces) == 1:
                    current_app.logger.warning(f"[Batch {batch_id}] Face is blurry")
                elif len(invalid_faces) > 1:
                    current_app.logger.warning(f"[Batch {batch_id}] Multiple invalid faces detected")
                else:
                    current_app.logger.warning(f"[Batch {batch_id}] No face found")
                return False
            elif len(valid_faces) == 1:
                if len(invalid_faces) > 0:
                    current_app.logger.warning(f"[Batch {batch_id}] Multiple invalid faces detected")
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