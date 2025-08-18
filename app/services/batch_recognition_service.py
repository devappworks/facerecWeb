import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from deepface import DeepFace
from PIL import Image
from io import BytesIO

# Import postojećih servisa (ne menjam ih!)
from app.services.batch_management_service import BatchManagementService
from app.services.recognition_service import RecognitionService
from app.services.image_service import ImageService
from app.services.face_validation_service import FaceValidationService

logger = logging.getLogger(__name__)

class BatchRecognitionService:
    """
    Servis za batch face recognition koji paralelno pretražuje kroz batch-eve.
    Koristi postojeće komponente bez menjanja postojeće logike.
    """
    
    # Thread pool konfiguracija
    MAX_BATCH_THREADS = 3  # Maksimalno 3 batch-a simultano da ne preopteretimo sistem
    BATCH_TIMEOUT = 300    # 5 minuta timeout po batch-u
    
    @staticmethod
    def get_batch_folders(domain: str) -> List[str]:
        """
        Dobija listu svih batch foldera za dati domain koji imaju pickle fajlove
        
        Returns:
            Lista putanja do batch foldera sa pickle fajlovima
        """
        batch_info = BatchManagementService.get_batch_info(domain)
        
        if batch_info["status"] != "success":
            logger.warning(f"No batch structure found for domain: {domain}")
            return []
        
        batch_folders = []
        domain_path = BatchManagementService.get_batch_domain_path(domain)
        
        for batch_data in batch_info["existing_batches"]:
            if batch_data.get("exists", False):
                batch_path = os.path.join(domain_path, batch_data["batch_id"])
                
                # Proveri da li batch ima pickle fajl (bilo koji .pkl fajl)
                try:
                    pickle_files = [f for f in os.listdir(batch_path) if f.endswith('.pkl')]
                    if pickle_files:
                        batch_folders.append(batch_path)
                        logger.debug(f"Batch {batch_data['batch_id']} has pickle file: {pickle_files[0]}")
                    else:
                        logger.warning(f"Batch {batch_data['batch_id']} has no pickle file - skipping")
                except Exception as e:
                    logger.error(f"Error checking batch {batch_data['batch_id']}: {str(e)}")
        
        logger.info(f"Found {len(batch_folders)} batch folders with pickle files for domain: {domain}")
        return batch_folders
    
    @staticmethod
    def process_single_batch(image_path: str, batch_path: str, batch_id: str, 
                           model_name: str = "VGG-Face", detector_backend: str = "retinaface",
                           distance_metric: str = "cosine", threshold: float = 0.35) -> Dict:
        """
        Procesira jedan batch za face recognition
        
        Args:
            image_path: Putanja do test slike
            batch_path: Putanja do batch foldera
            batch_id: ID batch-a za logging
            model_name: DeepFace model
            detector_backend: Detector backend
            distance_metric: Distance metric
            threshold: Recognition threshold
            
        Returns:
            Dict sa rezultatima ili error-om
        """
        batch_start_time = time.time()
        thread_id = threading.current_thread().name
        
        logger.info(f"[{thread_id}] Starting batch {batch_id} recognition...")
        
        try:
            # Brzo proveri da li batch folder postoji i ima slike
            if not os.path.exists(batch_path):
                return {
                    "batch_id": batch_id,
                    "status": "error",
                    "message": f"Batch path does not exist: {batch_path}",
                    "results": [],
                    "processing_time": 0
                }
            
            # Prebroj slike u batch-u
            image_count = len([f for f in os.listdir(batch_path) 
                              if BatchManagementService.is_image_file(f)])
            
            if image_count == 0:
                return {
                    "batch_id": batch_id,
                    "status": "warning", 
                    "message": f"No images found in batch: {batch_id}",
                    "results": [],
                    "processing_time": 0
                }
            
            logger.info(f"[{thread_id}] Batch {batch_id}: Processing {image_count} images")
            
            # Koristi postojeći DeepFace.find (ISTA LOGIKA kao u RecognitionService!)
            dfs = DeepFace.find(
                img_path=image_path,
                db_path=batch_path,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                enforce_detection=False,
                threshold=threshold,
                silent=True  # Stišamo output za batch processing
            )
            
            processing_time = time.time() - batch_start_time
            
            logger.info(f"[{thread_id}] Batch {batch_id} completed in {processing_time:.2f}s")
            
            return {
                "batch_id": batch_id,
                "status": "success",
                "message": f"Batch processed successfully",
                "results": dfs,  # Vraća iste rezultate kao postojeći sistem
                "image_count": image_count,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - batch_start_time
            error_msg = f"Error processing batch {batch_id}: {str(e)}"
            logger.error(f"[{thread_id}] {error_msg}")
            
            return {
                "batch_id": batch_id,
                "status": "error",
                "message": error_msg,
                "results": [],
                "processing_time": processing_time
            }
    
    @staticmethod
    def combine_batch_results(batch_results: List[Dict]) -> List:
        """
        Kombinuje rezultate iz svih batch-eva u jednu listu
        Koristi istu strukturu kao postojeći sistem
        
        Args:
            batch_results: Lista rezultata iz batch-eva
            
        Returns:
            Kombinovana lista DataFrame-ova kao u postojećem sistemu
        """
        combined_results = []
        total_matches = 0
        
        for batch_result in batch_results:
            if batch_result["status"] == "success" and batch_result["results"]:
                # batch_result["results"] je lista DataFrame-ova kao u postojećem sistemu
                if isinstance(batch_result["results"], list):
                    combined_results.extend(batch_result["results"])
                    total_matches += sum(len(df) for df in batch_result["results"] if hasattr(df, '__len__'))
        
        logger.info(f"Combined results from {len(batch_results)} batches: {total_matches} total matches")
        return combined_results
    
    @staticmethod
    def recognize_face_batch(image_bytes, domain: str, max_threads: int = None) -> Dict:
        """
        Glavna metoda za batch face recognition
        Koristi postojeću logiku gde god je moguće!
        
        Args:
            image_bytes: Bytes test slike
            domain: Domain za pretragu
            max_threads: Maksimalni broj thread-ova (default je pametno izračunat)
            
        Returns:
            Dict sa rezultatima u istom formatu kao postojeći sistem
        """
        try:
            start_time = time.time()
            logger.info("Starting batch face recognition process")
            
            # KORISTI POSTOJEĆU LOGIKU za pripremu slike (isti kod kao u RecognitionService!)
            if hasattr(image_bytes, 'getvalue'):
                actual_bytes = image_bytes.getvalue()
                image_bytes.seek(0)
            else:
                actual_bytes = image_bytes
            
            # Dobij dimenzije originalne slike (isti kod!)
            original_image = Image.open(BytesIO(actual_bytes))
            original_width, original_height = original_image.size
            logger.info(f"Original image dimensions: {original_width}x{original_height}")
            
            # Smanji sliku (isti kod!)
            resized_image = ImageService.resize_image(actual_bytes)
            resized_pil = Image.open(resized_image)
            resized_width, resized_height = resized_pil.size
            logger.info(f"Resized image dimensions: {resized_width}x{resized_height}")
            
            # Očisti domain (isti kod!)
            clean_domain = RecognitionService.clean_domain_for_path(domain)
            
            # Kreiraj privremeni folder i sačuvaj sliku (isti kod!)
            temp_folder = os.path.join('storage/uploads', clean_domain)
            os.makedirs(temp_folder, exist_ok=True)
            
            image_path = os.path.join(temp_folder, f"temp_batch_recognition_{int(time.time() * 1000)}.jpg")
            with open(image_path, "wb") as f:
                f.write(resized_image.getvalue())
            logger.info(f"Resized image saved temporarily at: {image_path}")
            
            # VAŽNO: Pre-load model JEDNOM za sve threadove!
            logger.info("Pre-loading VGG-Face model to avoid thread competition...")
            try:
                _ = DeepFace.build_model("VGG-Face")
                logger.info("Model pre-loaded successfully")
            except Exception as e:
                logger.warning(f"Model pre-loading failed (continuing anyway): {str(e)}")
            
            # Izvadi lica (isti kod kao u RecognitionService!)
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend="retinaface",
                enforce_detection=False,
                normalize_face=True,
                align=True
            )
            
            if len(faces) == 0:
                logger.warning("No faces detected in image")
                return {"status": "error", "message": "No faces detected"}
            
            logger.info(f"Detected {len(faces)} faces")
            
            # PROCES VALIDACIJE LICA (koristi postojeću logiku!)
            valid_faces = []
            for i, face in enumerate(faces):
                face_info = RecognitionService.process_single_face(
                    face, i+1, image_path, original_width, original_height, resized_width, resized_height
                )
                if face_info:
                    valid_faces.append(face_info)
            
            # Finalna provera lica (postojeća logika!)
            final_valid_faces = FaceValidationService.process_face_filtering(valid_faces)
            
            # Early exit ako nema validnih lica nakon svih provera
            if len(final_valid_faces) == 0:
                print("🚫 Prekidam batch face recognition - nema validnih lica za obradu.")
                logger.info("Stopping batch face recognition - no valid faces to process after all checks")
                return {
                    "status": "no_faces",
                    "message": "No valid faces found after validation checks",
                    "total_faces_detected": len(faces),
                    "valid_faces_after_filtering": 0,
                    "batch_summary": {
                        "total_batches": 0,
                        "processed_batches": 0,
                        "failed_batches": 0
                    }
                }
            
            # Dobij batch foldere
            batch_folders = BatchRecognitionService.get_batch_folders(domain)
            
            if not batch_folders:
                return {
                    "status": "error",
                    "message": f"No batch structure found for domain: {domain}. Please run batch migration first."
                }
            
            # PAMETNO IZRAČUNAVANJE THREAD COUNT-a
            # UVEK ograniči na max 3 threada zbog resource contention
            if max_threads is None:
                # Auto-calculate: max 3 threada ili 1 thread po 2 batch-a
                optimal_threads = min(3, max(1, len(batch_folders) // 2))
            else:
                # User je zadao broj, ali ograniči na max 3 zbog performance
                optimal_threads = min(15, max_threads)
            
            optimal_threads = min(optimal_threads, len(batch_folders))  # Ne više od batch-eva
            
            logger.info(f"Processing {len(batch_folders)} batches with {optimal_threads} threads (performance-optimized)")
            logger.info(f"User requested: {max_threads}, system enforced max: 3, final: {optimal_threads}")
            if max_threads and max_threads > 3:
                logger.warning(f"Reduced threads from {max_threads} to {optimal_threads} to avoid resource contention")
            
            # PARALELNO PROCESIRANJE BATCH-EVA
            batch_results = []
            
            with ThreadPoolExecutor(max_workers=optimal_threads) as executor:
                # Pokreni sve batch-eve simultano
                future_to_batch = {}
                
                for batch_path in batch_folders:
                    batch_id = os.path.basename(batch_path)
                    future = executor.submit(
                        BatchRecognitionService.process_single_batch,
                        image_path, batch_path, batch_id
                    )
                    future_to_batch[future] = batch_id
                
                # Čekaj rezultate
                for future in as_completed(future_to_batch, timeout=BatchRecognitionService.BATCH_TIMEOUT):
                    batch_id = future_to_batch[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        
                        if result["status"] == "success":
                            matches = sum(len(df) for df in result["results"] if hasattr(df, '__len__'))
                            logger.info(f"✅ Batch {batch_id}: {matches} matches in {result['processing_time']:.2f}s")
                        else:
                            logger.warning(f"⚠️ Batch {batch_id}: {result['message']}")
                            
                    except Exception as e:
                        logger.error(f"❌ Batch {batch_id} failed: {str(e)}")
                        batch_results.append({
                            "batch_id": batch_id,
                            "status": "error", 
                            "message": str(e),
                            "results": []
                        })
            
            # Kombinuj rezultate iz svih batch-eva
            combined_results = BatchRecognitionService.combine_batch_results(batch_results)
            
            if not combined_results:
                return {
                    "status": "success",  # Nema match-ova NIJE greška
                    "message": "No matches found in any batch",
                    "batch_summary": {
                        "total_batches": len(batch_folders),
                        "processed_batches": len([r for r in batch_results if r["status"] == "success"]),
                        "failed_batches": len([r for r in batch_results if r["status"] == "error"])
                    }
                }
            
            # Log detalje pronađenih rezultata (koristi postojeću logiku!)
            RecognitionService.log_deepface_results(combined_results)
            
            # Filtriraj rezultate na osnovu validnih lica (postojeća logika!)
            filtered_results = RecognitionService.filter_recognition_results_by_valid_faces(
                combined_results, final_valid_faces, resized_width, resized_height
            )
            
            # Analiziraj rezultate (potpuno ista logika kao postojeći sistem!)
            final_result = RecognitionService.analyze_recognition_results(
                filtered_results,
                threshold=0.35,
                original_width=original_width,
                original_height=original_height,
                resized_width=resized_width,
                resized_height=resized_height
            )
            
            # Pripremi batch_details za JSON serialization (ukloni DataFrame objekte)
            json_safe_batch_details = []
            for batch_result in batch_results:
                safe_result = batch_result.copy()
                # Ukloni 'results' ključ koji sadrži DataFrame objekte
                if 'results' in safe_result:
                    results_summary = {
                        "dataframes_count": len(safe_result['results']) if isinstance(safe_result['results'], list) else 0,
                        "total_matches": 0
                    }
                    # Prebroj ukupan broj match-ova
                    if isinstance(safe_result['results'], list):
                        for df in safe_result['results']:
                            if hasattr(df, '__len__'):
                                results_summary["total_matches"] += len(df)
                    
                    safe_result['results_summary'] = results_summary
                    del safe_result['results']  # Ukloni DataFrame objekte
                
                json_safe_batch_details.append(safe_result)
            
            # Dodaj batch statistike (bez DataFrame objekata)
            final_result["batch_processing"] = {
                "total_processing_time": time.time() - start_time,
                "batch_summary": {
                    "total_batches": len(batch_folders),
                    "processed_batches": len([r for r in batch_results if r["status"] == "success"]),
                    "failed_batches": len([r for r in batch_results if r["status"] == "error"]),
                    "total_images_searched": sum(r.get("image_count", 0) for r in batch_results),
                    "batch_details": json_safe_batch_details,  # JSON-safe verzija
                    "thread_optimization": {
                        "requested_threads": max_threads,
                        "optimal_threads": optimal_threads,
                        "total_batches": len(batch_folders)
                    }
                }
            }
            
            logger.info(f"Batch recognition completed in {time.time() - start_time:.2f}s")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in batch face recognition: {str(e)}")
            return {
                "status": "error",
                "message": f"Batch recognition failed: {str(e)}"
            }
        finally:
            # Čišćenje privremene slike (isti kod!)
            try:
                if 'image_path' in locals() and os.path.exists(image_path):
                    os.remove(image_path)
                    logger.info(f"Cleaned up temporary file: {image_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")
    
    @staticmethod
    def get_batch_recognition_stats(domain: str) -> Dict:
        """
        Vraća statistike o batch strukturi za face recognition
        
        Args:
            domain: Domain za analizu
            
        Returns:
            Dict sa detaljnim statistikama
        """
        try:
            batch_info = BatchManagementService.get_batch_info(domain)
            
            if batch_info["status"] != "success":
                return {
                    "status": "not_available",
                    "message": f"No batch structure available for domain: {domain}",
                    "recommendation": "Run batch migration command first"
                }
            
            metadata = batch_info["metadata"]
            batches = batch_info["existing_batches"]
            
            # Prebrojava statistike
            total_images = sum(b.get("actual_image_count", 0) for b in batches if b.get("exists"))
            active_batches = len([b for b in batches if b.get("exists")])
            missing_batches = len([b for b in batches if not b.get("exists")])
            
            # Proceni performance
            estimated_search_time = active_batches * 2  # Pretpostavi 2s po batch-u
            parallel_search_time = max(2, estimated_search_time / BatchRecognitionService.MAX_BATCH_THREADS)
            
            return {
                "status": "available",
                "domain": domain,
                "total_images": total_images,
                "total_batches": metadata.get("total_batches", 0),
                "active_batches": active_batches,
                "missing_batches": missing_batches,
                "images_per_batch": metadata.get("images_per_batch", 0),
                "created": metadata.get("created"),
                "last_updated": metadata.get("last_updated"),
                "performance_estimate": {
                    "estimated_sequential_time": f"{estimated_search_time}s",
                    "estimated_parallel_time": f"{parallel_search_time:.1f}s",
                    "max_parallel_batches": BatchRecognitionService.MAX_BATCH_THREADS,
                    "speedup_factor": f"{estimated_search_time / parallel_search_time:.1f}x"
                },
                "batch_details": batches
            }
            
        except Exception as e:
            logger.error(f"Error getting batch recognition stats: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            } 