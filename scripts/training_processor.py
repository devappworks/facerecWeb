
"""
Script za prebacivanje slika iz storage/trainingPass foldera u storage/recognized_faces_prod/serbia
i pokretanje face recognition nakon svakog foldera.
Ako folder ima 5 ili manje slika, briše se ceo folder.
"""

import os
import sys
import shutil
import time
import logging
from pathlib import Path

# Dodaj parent direktorijum u sys.path da bi mogli da importujemo app module
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Flask app imports
from app import create_app
from app.controllers.recognition_controller import RecognitionController

# Postavke logovnja
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TrainingProcessor:
    def __init__(self):
        self.source_base = "storage/trainingPassSerbia"
        self.target_dir = "storage/recognized_faces_prod/serbia"
        self.domain = "serbia"
        self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.min_images_threshold = 5  # Minimalni broj slika potreban za proces
        
        # Initialize Flask app context
        self.app = create_app()
        self.app_context = self.app.app_context()
        self.app_context.push()
        
    def __del__(self):
        """Cleanup Flask app context"""
        if hasattr(self, 'app_context'):
            self.app_context.pop()
        
    def is_image_file(self, filename):
        """Proverava da li je fajl slika na osnovu ekstenzije"""
        return Path(filename).suffix.lower() in self.allowed_extensions
    
    def ensure_target_directory_exists(self):
        """Kreira target direktorijum ako ne postoji"""
        os.makedirs(self.target_dir, exist_ok=True)
        logger.info(f"Target directory ensured: {self.target_dir}")
    
    def get_person_folders(self):
        """Vraća listu svih foldera u trainingPass direktorijumu"""
        if not os.path.exists(self.source_base):
            logger.error(f"Source directory does not exist: {self.source_base}")
            return []
        
        folders = []
        for item in os.listdir(self.source_base):
            item_path = os.path.join(self.source_base, item)
            if os.path.isdir(item_path):
                folders.append(item)
        
        logger.info(f"Found {len(folders)} person folders in {self.source_base}")
        return sorted(folders)
    
    def count_images_in_folder(self, person_folder):
        """Broji broj slika u folderu"""
        source_folder_path = os.path.join(self.source_base, person_folder)
        
        if not os.path.exists(source_folder_path):
            logger.error(f"Source folder does not exist: {source_folder_path}")
            return 0
        
        try:
            files = os.listdir(source_folder_path)
            image_files = [f for f in files if self.is_image_file(f)]
            return len(image_files)
        except Exception as e:
            logger.error(f"Error counting images in {source_folder_path}: {str(e)}")
            return 0
    
    def delete_folder(self, person_folder):
        """Briše ceo folder sa svim slikama"""
        source_folder_path = os.path.join(self.source_base, person_folder)
        
        try:
            if os.path.exists(source_folder_path):
                shutil.rmtree(source_folder_path)
                logger.info(f"Deleted folder: {person_folder}")
                return True
            else:
                logger.warning(f"Folder does not exist for deletion: {source_folder_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting folder {person_folder}: {str(e)}")
            return False
    
    def copy_images_from_folder(self, person_folder):
        """Prebacuje sve slike iz jednog person foldera u target direktorijum"""
        source_folder_path = os.path.join(self.source_base, person_folder)
        copied_count = 0
        skipped_count = 0
        already_exists_count = 0
        
        logger.info(f"Processing folder: {person_folder}")
        
        if not os.path.exists(source_folder_path):
            logger.error(f"Source folder does not exist: {source_folder_path}")
            return copied_count, skipped_count, already_exists_count
        
        # Dobij listu svih fajlova u folderu
        try:
            files = os.listdir(source_folder_path)
            image_files = [f for f in files if self.is_image_file(f)]
            total_images = len(image_files)
            
            logger.info(f"Found {total_images} images in {person_folder}")
            
            for image_file in image_files:
                source_path = os.path.join(source_folder_path, image_file)
                
                # Transformiši ime fajla u novi format
                transformed_filename = self.transform_filename(image_file)
                target_path = os.path.join(self.target_dir, transformed_filename)
                
                # Proveri da li slika već postoji na odredištu (sa transformisanim imenom)
                if os.path.exists(target_path):
                    logger.info(f"Image already exists, skipping: {image_file} -> {transformed_filename}")
                    already_exists_count += 1
                    continue
                
                try:
                    # Kopiraj sliku sa novim imenom (ne briši original za sada)
                    shutil.copy2(source_path, target_path)
                    copied_count += 1
                    logger.info(f"Copied: {image_file} -> {transformed_filename}")
                    
                    # Kratka pauza između kopiranja
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error copying {image_file}: {str(e)}")
                    skipped_count += 1
            
            # Proveri da li treba obrisati source folder
            processed_images = copied_count + already_exists_count
            if processed_images == total_images and skipped_count == 0:
                # Sve slike su uspešno obrađene (kopirane ili već postojale)
                logger.info(f"All {total_images} images processed successfully, deleting source folder: {person_folder}")
                if self.delete_folder(person_folder):
                    logger.info(f"Successfully deleted source folder: {person_folder}")
                else:
                    logger.error(f"Failed to delete source folder: {person_folder}")
            elif skipped_count > 0:
                logger.warning(f"Some images failed to copy ({skipped_count} errors), keeping source folder: {person_folder}")
            else:
                logger.info(f"Partial processing completed for {person_folder}")
            
        except Exception as e:
            logger.error(f"Error listing files in {source_folder_path}: {str(e)}")
        
        logger.info(f"Folder {person_folder} processed: {copied_count} copied, {skipped_count} skipped, {already_exists_count} already existed")
        return copied_count, skipped_count, already_exists_count
    
    def run_face_recognition(self):
        """Pokreće face recognition na serbia folderu"""
        logger.info("Starting face recognition process...")
        
        try:
            # Kreiraj test sliku za face recognition (koristimo prvu dostupnu sliku u target folderu)
            test_image_path = None
            if os.path.exists(self.target_dir):
                for file in os.listdir(self.target_dir):
                    if self.is_image_file(file):
                        test_image_path = os.path.join(self.target_dir, file)
                        break
            
            if not test_image_path:
                logger.warning("No test image found in target directory for face recognition")
                return {"status": "error", "message": "No test image available"}
            
            logger.info(f"Using test image: {test_image_path}")
            
            # Učitaj test sliku
            with open(test_image_path, 'rb') as f:
                test_image_bytes = f.read()
            
            # Pokreni face recognition
            result = RecognitionController.recognize_face(test_image_bytes, self.domain)
            logger.info("Face recognition completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error during face recognition: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def process_all_folders(self):
        """Glavna metoda koja obrađuje sve foldere"""
        logger.info("Starting training processor...")
        
        # Kreiraj target direktorijum
        self.ensure_target_directory_exists()
        
        # Dobij listu foldera
        person_folders = self.get_person_folders()
        
        if not person_folders:
            logger.error("No person folders found to process")
            return
        
        total_copied = 0
        total_skipped = 0
        total_already_exists = 0
        processed_folders = 0
        deleted_small_folders = 0  # Folderi obrisani zbog malog broja slika
        deleted_processed_folders = 0  # Folderi obrisani nakon uspešne obrade
        
        for folder in person_folders:
            logger.info(f"\n--- Processing folder {processed_folders + 1}/{len(person_folders)}: {folder} ---")
            
            # Prvo proveri broj slika u folderu
            image_count = self.count_images_in_folder(folder)
            logger.info(f"Folder {folder} contains {image_count} images")
            
            if image_count <= self.min_images_threshold:
                logger.info(f"Folder {folder} has {image_count} images (<= {self.min_images_threshold}), deleting entire folder")
                if self.delete_folder(folder):
                    deleted_small_folders += 1
                    logger.info(f"Successfully deleted small folder: {folder}")
                else:
                    logger.error(f"Failed to delete small folder: {folder}")
            else:
                logger.info(f"Folder {folder} has {image_count} images (> {self.min_images_threshold}), proceeding with processing")
                
                # Pre obrade - proveri da li folder još uvek postoji
                if not os.path.exists(os.path.join(self.source_base, folder)):
                    logger.info(f"Folder {folder} no longer exists (possibly deleted in previous run), skipping")
                    processed_folders += 1
                    continue
                
                # Kopiraj slike iz trenutnog foldera
                copied, skipped, already_exists = self.copy_images_from_folder(folder)
                total_copied += copied
                total_skipped += skipped
                total_already_exists += already_exists
                
                # Proveri da li je folder obrisan nakon obrade
                if not os.path.exists(os.path.join(self.source_base, folder)):
                    deleted_processed_folders += 1
                    logger.info(f"Source folder was deleted after processing: {folder}")
                
                if copied > 0:
                    # Pokreni face recognition nakon kopiranja slika
                    logger.info(f"Running face recognition after processing {folder}...")
                    recognition_result = self.run_face_recognition()
                    
                    if recognition_result.get("status") == "success":
                        logger.info(f"Face recognition successful for {folder}")
                        if "person" in recognition_result:
                            logger.info(f"Recognized person: {recognition_result['person']}")
                    else:
                        logger.warning(f"Face recognition had issues for {folder}: {recognition_result}")
                else:
                    logger.info(f"No new images copied from {folder}, skipping face recognition")
            
            processed_folders += 1
            
            # Kratka pauza između foldera
            time.sleep(2)
        
        logger.info(f"\n--- PROCESSING COMPLETE ---")
        logger.info(f"Processed folders: {processed_folders}/{len(person_folders)}")
        logger.info(f"Deleted small folders (with <= {self.min_images_threshold} images): {deleted_small_folders}")
        logger.info(f"Deleted processed folders (after successful processing): {deleted_processed_folders}")
        logger.info(f"Total deleted folders: {deleted_small_folders + deleted_processed_folders}")
        logger.info(f"Total images copied: {total_copied}")
        logger.info(f"Total images skipped (errors): {total_skipped}")
        logger.info(f"Total images already existed: {total_already_exists}")
        logger.info(f"Total images processed: {total_copied + total_skipped + total_already_exists}")

    def transform_filename(self, original_filename):
        """
        Transformiše ime fajla iz originalnog formata u novi format
        
        Original: Abraham_Nnamdi_Nwankwo_20250524_044003_66.jpg
        Novi: Abraham_Nnamdi_Nwankwo_2025-05-24_04400366.jpg
        """
        try:
            # Razdeli ime fajla na delove
            name_part, extension = os.path.splitext(original_filename)
            parts = name_part.split('_')
            
            if len(parts) < 4:
                # Ako format nije kako očekujemo, vrati originalno ime
                logger.warning(f"Unexpected filename format, keeping original: {original_filename}")
                return original_filename
            
            # Pronađi datum (YYYYMMDD format)
            date_index = -1
            for i, part in enumerate(parts):
                if len(part) == 8 and part.isdigit():
                    # Proveri da li je validan datum format YYYYMMDD
                    try:
                        year = int(part[:4])
                        month = int(part[4:6])
                        day = int(part[6:8])
                        if 2020 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                            date_index = i
                            break
                    except:
                        continue
            
            if date_index == -1:
                logger.warning(f"No valid date found in filename, keeping original: {original_filename}")
                return original_filename
            
            # Izvuci delove
            name_parts = parts[:date_index]  # Ime osobe
            date_part = parts[date_index]    # Datum YYYYMMDD
            remaining_parts = parts[date_index + 1:]  # Ostatak (vreme i broj)
            
            # Transformiši datum iz YYYYMMDD u YYYY-MM-DD
            transformed_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
            
            # Spoji preostale delove (vreme i brojevi)
            combined_suffix = ''.join(remaining_parts)
            
            # Kreiraj novo ime
            new_parts = name_parts + [transformed_date, combined_suffix]
            new_filename = '_'.join(new_parts) + extension
            
            logger.info(f"Transformed filename: {original_filename} -> {new_filename}")
            return new_filename
            
        except Exception as e:
            logger.error(f"Error transforming filename {original_filename}: {str(e)}")
            return original_filename

def main():
    """Glavna funkcija"""
    processor = None
    try:
        processor = TrainingProcessor()
        processor.process_all_folders()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise
    finally:
        # Ensure cleanup
        if processor:
            del processor

if __name__ == "__main__":
    main() 