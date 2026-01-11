import os
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class BatchManagementService:
    """
    Servis za upravljanje batch organizacijom slika za face recognition.
    Organizuje slike u batch-eve od po 5000 slika za optimizovanu pretragu.
    """
    
    IMAGES_PER_BATCH = 5000
    BATCH_BASE_PATH = 'storage/recognized_faces_batched'
    BATCH_METADATA_FILE = 'batch_metadata.json'
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    @staticmethod
    def clean_domain_for_path(domain: str) -> str:
        """Čisti domain string za korišćenje u putanjama"""
        # Koristi istu logiku kao RecognitionService
        domain = domain.split(':')[0]  # Ukloni port
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            domain = domain.replace(char, '_')
        return domain
    
    @staticmethod
    def is_image_file(filename: str) -> bool:
        """Proverava da li je fajl slika na osnovu ekstenzije"""
        return Path(filename).suffix.lower() in BatchManagementService.SUPPORTED_IMAGE_EXTENSIONS
    
    @staticmethod
    def get_batch_domain_path(domain: str) -> str:
        """Vraća putanju do batch foldera za dati domain"""
        clean_domain = BatchManagementService.clean_domain_for_path(domain)
        return os.path.join(BatchManagementService.BATCH_BASE_PATH, clean_domain)
    
    @staticmethod
    def get_batch_folder_name(batch_number: int) -> str:
        """Generiše ime batch foldera na osnovu broja"""
        return f"batch_{batch_number:04d}"
    
    @staticmethod
    def create_batch_metadata(domain: str, batches_info: List[Dict]) -> Dict:
        """Kreira metadata strukturu za batch-eve"""
        return {
            "domain": domain,
            "total_batches": len(batches_info),
            "images_per_batch": BatchManagementService.IMAGES_PER_BATCH,
            "last_updated": datetime.now().isoformat(),
            "created": datetime.now().isoformat(),
            "batches": batches_info
        }
    
    @staticmethod
    def save_batch_metadata(domain: str, metadata: Dict) -> None:
        """Čuva batch metadata u JSON fajl"""
        domain_path = BatchManagementService.get_batch_domain_path(domain)
        os.makedirs(domain_path, exist_ok=True)
        
        metadata_path = os.path.join(domain_path, BatchManagementService.BATCH_METADATA_FILE)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch metadata saved to: {metadata_path}")
    
    @staticmethod
    def load_batch_metadata(domain: str) -> Optional[Dict]:
        """Učitava batch metadata iz JSON fajla"""
        domain_path = BatchManagementService.get_batch_domain_path(domain)
        metadata_path = os.path.join(domain_path, BatchManagementService.BATCH_METADATA_FILE)
        
        if not os.path.exists(metadata_path):
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading batch metadata: {str(e)}")
            return None
    
    @staticmethod
    def get_all_images_from_source(source_path: str) -> List[str]:
        """Dobija listu svih slika iz source foldera"""
        if not os.path.exists(source_path):
            logger.warning(f"Source path does not exist: {source_path}")
            return []
        
        images = []
        try:
            for filename in os.listdir(source_path):
                if BatchManagementService.is_image_file(filename):
                    images.append(filename)
            
            # Sortiraj po datumu/vremenu za konzistentnost
            images.sort()
            logger.info(f"Found {len(images)} images in {source_path}")
            return images
            
        except Exception as e:
            logger.error(f"Error reading source directory {source_path}: {str(e)}")
            return []
    
    @staticmethod
    def split_images_into_batches(images: List[str]) -> List[List[str]]:
        """Deli listu slika u batch-eve od po IMAGES_PER_BATCH"""
        batches = []
        batch_size = BatchManagementService.IMAGES_PER_BATCH
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"Split {len(images)} images into {len(batches)} batches")
        return batches
    
    @staticmethod
    def create_pickle_file_for_batch(batch_path: str, batch_id: str) -> bool:
        """
        Kreira pickle fajl za batch folder pozivanjem DeepFace.find() sa test slikom
        
        Args:
            batch_path: Putanja do batch foldera
            batch_id: ID batch-a za logging
            
        Returns:
            bool: True ako je pickle fajl uspešno kreiran
        """
        try:
            logger.info(f"Creating pickle file for batch: {batch_id}")
            
            # Proveri da li batch folder postoji i ima slike
            if not os.path.exists(batch_path):
                logger.error(f"Batch path does not exist: {batch_path}")
                return False
            
            # Pronađi prvu sliku u batch-u za test
            test_image_path = None
            for filename in os.listdir(batch_path):
                if BatchManagementService.is_image_file(filename):
                    test_image_path = os.path.join(batch_path, filename)
                    break
            
            if not test_image_path:
                logger.error(f"No images found in batch {batch_id} for pickle creation")
                return False
            
            logger.info(f"Using test image for pickle creation: {os.path.basename(test_image_path)}")
            
            # Pozovi DeepFace.find() da kreira pickle fajl
            # Ovo će automatski kreirati representations_vgg_face.pkl u batch folderu
            from deepface import DeepFace
            
            _ = DeepFace.find(
                img_path=test_image_path,
                db_path=batch_path,
                model_name="VGG-Face",
                detector_backend="retinaface",
                distance_metric="cosine",
                enforce_detection=False,
                threshold=0.35,
                silent=False
            )
            
            # Proveri da li je pickle fajl kreiran (bilo koji .pkl fajl)
            pickle_files = [f for f in os.listdir(batch_path) if f.endswith('.pkl')]
            if pickle_files:
                logger.info(f"✅ Pickle file successfully created for batch {batch_id}: {pickle_files[0]}")
                return True
            else:
                logger.error(f"❌ No pickle file found in batch {batch_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating pickle file for batch {batch_id}: {str(e)}")
            return False

    @staticmethod
    def copy_batch_images(source_path: str, batch_images: List[str], batch_path: str, 
                         delete_originals: bool = False) -> Tuple[int, int, int]:
        """
        Kopira slike iz source-a u batch folder
        
        Args:
            source_path: Putanja do source foldera
            batch_images: Lista slika za kopiranje
            batch_path: Putanja do batch foldera
            delete_originals: Da li da briše originalne slike nakon kopiranja
        
        Returns:
            Tuple[copied_count, failed_count, deleted_count]
        """
        os.makedirs(batch_path, exist_ok=True)
        
        copied_count = 0
        failed_count = 0
        deleted_count = 0
        
        for image_filename in batch_images:
            try:
                source_file = os.path.join(source_path, image_filename)
                target_file = os.path.join(batch_path, image_filename)
                
                if os.path.exists(source_file):
                    # Kopiraj sliku
                    shutil.copy2(source_file, target_file)
                    copied_count += 1
                    
                    # Obriši original ako je traženo
                    if delete_originals:
                        try:
                            os.remove(source_file)
                            deleted_count += 1
                            logger.debug(f"Deleted original: {source_file}")
                        except Exception as delete_error:
                            logger.warning(f"Failed to delete original {source_file}: {str(delete_error)}")
                else:
                    logger.warning(f"Source file not found: {source_file}")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Error copying {image_filename}: {str(e)}")
                failed_count += 1
        
        logger.info(f"Batch copy complete: {copied_count} copied, {failed_count} failed, {deleted_count} deleted")
        return copied_count, failed_count, deleted_count

    @staticmethod
    def create_batches_from_source(source_path: str, domain: str, dry_run: bool = False, 
                                 delete_originals: bool = False, create_pickle_files: bool = True) -> Dict:
        """
        Glavna metoda za kreiranje batch strukture iz postojećeg foldera
        
        Args:
            source_path: Putanja do postojećeg foldera sa slikama
            domain: Domain identifikator
            dry_run: Ako je True, samo proverava šta bi se uradilo bez kopiranja
            delete_originals: Da li da briše originalne slike nakon kopiranja
            create_pickle_files: Da li da automatski kreira pickle fajlove
            
        Returns:
            Dict sa rezultatima operacije
        """
        logger.info(f"Creating batches from source: {source_path} for domain: {domain}")
        logger.info(f"Dry run mode: {dry_run}")
        logger.info(f"Delete originals: {delete_originals}")
        logger.info(f"Create pickle files: {create_pickle_files}")
        
        # Dobij sve slike iz source foldera
        all_images = BatchManagementService.get_all_images_from_source(source_path)
        
        if not all_images:
            return {
                "status": "error",
                "message": "No images found in source path",
                "details": {"source_path": source_path, "image_count": 0}
            }
        
        # Podeli slike u batch-eve
        image_batches = BatchManagementService.split_images_into_batches(all_images)
        
        if dry_run:
            # Samo vrati plan bez izvršavanja
            batch_plan = []
            for i, batch_images in enumerate(image_batches):
                batch_number = i + 1
                batch_folder = BatchManagementService.get_batch_folder_name(batch_number)
                batch_plan.append({
                    "batch_number": batch_number,
                    "batch_folder": batch_folder,
                    "image_count": len(batch_images),
                    "sample_images": batch_images[:5]  # Prikaži prvih 5 kao primer
                })
            
            return {
                "status": "success",
                "message": "Batch plan created (dry run)",
                "details": {
                    "total_images": len(all_images),
                    "total_batches": len(image_batches),
                    "images_per_batch": BatchManagementService.IMAGES_PER_BATCH,
                    "domain": domain,
                    "will_delete_originals": delete_originals,
                    "will_create_pickle_files": create_pickle_files,
                    "batch_plan": batch_plan
                }
            }
        
        # Kreiranje batch strukture
        domain_path = BatchManagementService.get_batch_domain_path(domain)
        os.makedirs(domain_path, exist_ok=True)
        
        batch_results = []
        total_copied = 0
        total_failed = 0
        total_deleted = 0
        pickle_success_count = 0
        
        for i, batch_images in enumerate(image_batches):
            batch_number = i + 1
            batch_folder = BatchManagementService.get_batch_folder_name(batch_number)
            batch_path = os.path.join(domain_path, batch_folder)
            
            logger.info(f"Processing batch {batch_number}/{len(image_batches)}: {len(batch_images)} images")
            
            # Kopiraj slike u batch folder
            copied_count, failed_count, deleted_count = BatchManagementService.copy_batch_images(
                source_path, batch_images, batch_path, delete_originals
            )
            
            total_copied += copied_count
            total_failed += failed_count
            total_deleted += deleted_count
            
            # Kreiraj pickle fajl za batch
            pickle_created = False
            if create_pickle_files and copied_count > 0:
                pickle_created = BatchManagementService.create_pickle_file_for_batch(batch_path, batch_folder)
                if pickle_created:
                    pickle_success_count += 1
            
            batch_info = {
                "batch_id": batch_folder,
                "batch_number": batch_number,
                "image_count": copied_count,
                "failed_count": failed_count,
                "deleted_count": deleted_count,
                "pickle_created": pickle_created,
                "created": datetime.now().isoformat(),
                "path": batch_path
            }
            batch_results.append(batch_info)
        
        # Kreiranje i čuvanje metadata
        metadata = BatchManagementService.create_batch_metadata(domain, batch_results)
        BatchManagementService.save_batch_metadata(domain, metadata)
        
        result = {
            "status": "success",
            "message": f"Successfully created {len(batch_results)} batches",
            "details": {
                "domain": domain,
                "total_images_processed": len(all_images),
                "total_images_copied": total_copied,
                "total_images_failed": total_failed,
                "total_images_deleted": total_deleted,
                "total_batches_created": len(batch_results),
                "pickle_files_created": pickle_success_count,
                "batch_base_path": domain_path,
                "options_used": {
                    "delete_originals": delete_originals,
                    "create_pickle_files": create_pickle_files
                },
                "batches": batch_results
            }
        }
        
        logger.info(f"Batch creation completed: {result}")
        return result
    
    @staticmethod
    def get_batch_info(domain: str) -> Dict:
        """Vraća informacije o postojećim batch-evima za domain"""
        metadata = BatchManagementService.load_batch_metadata(domain)
        
        if not metadata:
            return {
                "status": "not_found",
                "message": "No batch structure found for domain",
                "domain": domain
            }
        
        # Proveri da li batch folderi stvarno postoje
        domain_path = BatchManagementService.get_batch_domain_path(domain)
        existing_batches = []
        
        for batch_info in metadata.get("batches", []):
            batch_id = batch_info["batch_id"]
            batch_path = os.path.join(domain_path, batch_id)
            
            if os.path.exists(batch_path):
                # Prebroj stvaran broj slika u batch-u
                actual_image_count = len([
                    f for f in os.listdir(batch_path) 
                    if BatchManagementService.is_image_file(f)
                ])
                
                # Proveri da li stvarno postoji pickle fajl (bilo koji .pkl)
                pickle_files = [f for f in os.listdir(batch_path) if f.endswith('.pkl')]
                pickle_exists = len(pickle_files) > 0
                
                batch_info_copy = batch_info.copy()
                batch_info_copy["actual_image_count"] = actual_image_count
                batch_info_copy["exists"] = True
                batch_info_copy["pickle_created"] = pickle_exists  # Prepiši sa stvarnim stanjem
                if pickle_exists:
                    batch_info_copy["pickle_file"] = pickle_files[0]  # Dodaj ime pickle fajla
                existing_batches.append(batch_info_copy)
            else:
                batch_info_copy = batch_info.copy()
                batch_info_copy["exists"] = False
                existing_batches.append(batch_info_copy)
        
        return {
            "status": "success",
            "message": "Batch info retrieved",
            "metadata": metadata,
            "existing_batches": existing_batches,
            "batch_base_path": domain_path
        }
    
    @staticmethod
    def list_batch_domains() -> List[str]:
        """Lista svih domain-a koji imaju batch strukturu"""
        if not os.path.exists(BatchManagementService.BATCH_BASE_PATH):
            return []
        
        domains = []
        try:
            for item in os.listdir(BatchManagementService.BATCH_BASE_PATH):
                item_path = os.path.join(BatchManagementService.BATCH_BASE_PATH, item)
                if os.path.isdir(item_path):
                    # Proverava da li ima metadata fajl
                    metadata_path = os.path.join(item_path, BatchManagementService.BATCH_METADATA_FILE)
                    if os.path.exists(metadata_path):
                        domains.append(item)
            
            return sorted(domains)
            
        except Exception as e:
            logger.error(f"Error listing batch domains: {str(e)}")
            return [] 