import os
import time
import logging
import cv2
import pandas as pd
from collections import defaultdict
from deepface import DeepFace
from PIL import Image
from io import BytesIO
import numpy as np

# Register AVIF and HEIF image format support
try:
    import pillow_avif
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # AVIF/HEIF support not installed

from app.services.image_service import ImageService
from app.services.face_processing_service import FaceProcessingService
from app.services.face_validation_service import FaceValidationService

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class RecognitionService:
    @staticmethod
    def clean_domain_for_path(domain):
        """Čisti domain string za korišćenje u putanjama"""
        # Ukloni port i nedozvoljene karaktere
        domain = domain.split(':')[0]  # Ukloni port
        # Zameni bilo koje nedozvoljene karaktere sa '_'
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            domain = domain.replace(char, '_')
        return domain

    @staticmethod
    def validate_face_confidence_and_eyes(face, index):
        """
        Validira confidence i koordinate očiju za lice
        
        Args:
            face (dict): Face objekat sa facial_area i confidence
            index (int): Indeks lica
            
        Returns:
            bool: True ako je lice validno
        """
        facial_area = face["facial_area"]
        confidence = face.get("confidence", 1)

        print(f"\n➡️ Lice {index}: {facial_area}, Confidence={confidence:.3f}")

        # Lowered confidence threshold from 0.995 to 0.97 to 0.95 to 0.85 to 0.65 for better recall
        if confidence >= 0.65:
            # Check if left_eye and right_eye coordinates are identical
            if FaceValidationService.has_identical_eye_coordinates(facial_area):
                left_eye = facial_area.get("left_eye")
                print(f"⚠️ Lice {index} ima identične koordinate za levo i desno oko ({left_eye}) - preskačem.")
                logger.info(f"Face {index} has identical left_eye and right_eye coordinates ({left_eye}) - skipping")
                return False

            print("✅ Validno lice - radim prepoznavanje.")
            return True
        else:
            print(f"⚠️ Niska sigurnost detekcije ({confidence:.3f} < 0.65) - preskačem ovo lice.")
            logger.info(f"Face {index} rejected - low confidence ({confidence:.3f} < 0.65)")
            return False



    @staticmethod
    def check_face_quality_and_create_info(cropped_face, facial_area, index, original_width, original_height, resized_width, resized_height, source_type="image"):
        """
        Proverava kvalitet lica (zamagljenost, kontrast, osvetljenost) i kreira info objekat ako je lice validno

        Args:
            cropped_face (np.array): Array slike
            facial_area (dict): Koordinate lica
            index (int): Indeks lica
            original_width (int): Širina originalne slike
            original_height (int): Visina originalne slike
            resized_width (int): Širina resized slike
            resized_height (int): Visina resized slike
            source_type (str): "image" for photos or "video" for video frames

        Returns:
            dict or None: Info objekat ako je lice validno, None ako nije
        """
        try:
            # Nova, poboljšana validacija kvaliteta lica
            is_quality_valid = RecognitionService.validate_face_quality(cropped_face, index, source_type)

            if not is_quality_valid:
                return None
            else:
                # Kreiraj info objekat sa originalnim koordinatama
                return FaceValidationService.create_face_info(
                    facial_area, index, original_width, original_height, resized_width, resized_height
                )

        except Exception as quality_error:
            logger.error(f"Error checking face quality for face {index}: {str(quality_error)}")
            print(f"❌ Greška pri proveri kvaliteta lica {index}: {str(quality_error)}")
            return None

    @staticmethod
    def process_single_face(face, index, image_path, original_width, original_height, resized_width, resized_height, source_type="image"):
        """
        Obrađuje jedno lice kroz sve validacije

        Args:
            face (dict): Face objekat
            index (int): Indeks lica
            image_path (str): Putanja do originalne slike
            original_width (int): Širina originalne slike
            original_height (int): Visina originalne slike
            resized_width (int): Širina resized slike
            resized_height (int): Visina resized slike
            source_type (str): "image" for photos or "video" for video frames

        Returns:
            dict or None: Info objekat validnog lica ili None
        """
        # Validacija confidence-a i koordinata očiju
        if not RecognitionService.validate_face_confidence_and_eyes(face, index):
            return None

        facial_area = face["facial_area"]

        # Crop lice samo za proveru blur-a (ne čuvamo sliku)
        img_cv = cv2.imread(image_path)
        x = facial_area["x"]
        y = facial_area["y"]
        w = facial_area["w"]
        h = facial_area["h"]
        cropped_face = img_cv[y:y+h, x:x+w]

        # Provera kvaliteta i kreiranje info objekta
        return RecognitionService.check_face_quality_and_create_info(
            cropped_face, facial_area, index, original_width, original_height, resized_width, resized_height, source_type
        )

    @staticmethod
    def filter_recognition_results_by_valid_faces(results, valid_faces, resized_width, resized_height):
        """
        Filtrira rezultate DeepFace.find na osnovu validnih lica
        
        Args:
            results: Rezultati DeepFace.find
            valid_faces (list): Lista validnih lica
            resized_width (int): Širina resized slike
            resized_height (int): Visina resized slike
            
        Returns:
            Filtrirani rezultati
        """
        if not valid_faces or not results:
            return results
        
        logger.info(f"Filtering recognition results based on {len(valid_faces)} valid faces")
        
        # Kreiraj koordinate validnih lica u resized formatu za poređenje
        valid_coordinates = []
        for face_info in valid_faces:
            resized_coords = face_info['resized_coordinates']
            valid_coordinates.append({
                'x': resized_coords['x'],
                'y': resized_coords['y'],
                'w': resized_coords['w'],
                'h': resized_coords['h'],
                'index': face_info['index']
            })
        
        filtered_results = []
        
        # DeepFace.find vraća listu DataFrame-ova
        if isinstance(results, list):
            for df in results:
                if hasattr(df, 'iterrows'):
                    filtered_rows = []
                    for _, row in df.iterrows():
                        try:
                            # Dobij koordinate iz rezultata
                            source_x = float(row['source_x'])
                            source_y = float(row['source_y'])
                            source_w = float(row['source_w'])
                            source_h = float(row['source_h'])
                            
                            # Proveri da li se poklapaju sa bilo kojim validnim licem
                            for valid_coord in valid_coordinates:
                                # Tolerancija za poređenje koordinata (u pikselima)
                                tolerance = 5
                                
                                if (abs(source_x - valid_coord['x']) <= tolerance and
                                    abs(source_y - valid_coord['y']) <= tolerance and
                                    abs(source_w - valid_coord['w']) <= tolerance and
                                    abs(source_h - valid_coord['h']) <= tolerance):
                                    
                                    filtered_rows.append(row)
                                    logger.info(f"Match found for valid face {valid_coord['index']} at coordinates ({source_x}, {source_y}, {source_w}, {source_h})")
                                    break
                        except Exception as e:
                            logger.warning(f"Error processing recognition result row: {str(e)}")
                            continue
                    
                    # Kreiraj novi DataFrame sa filtriranim redovima
                    if filtered_rows:
                        filtered_df = pd.DataFrame(filtered_rows)
                        filtered_results.append(filtered_df)
                    else:
                        # Dodaj prazan DataFrame da održimo strukturu
                        filtered_results.append(df.iloc[0:0])  # Prazan DataFrame sa istim kolonama
        
        logger.info(f"Filtered results: {len(filtered_results)} DataFrames with recognition matches")
        return filtered_results

    @staticmethod
    def recognize_face(image_bytes, domain, source_type="image"):
        """
        Prepoznaje lice iz prosleđene slike

        Args:
            image_bytes: Image data as bytes
            domain: Domain for recognition database
            source_type: "image" for photos (strict threshold) or "video" for video frames (lenient threshold)
        """
        try:
            logger.info("Starting face recognition process")
            start_time = time.time()

            # Prvo dobijamo dimenzije originalne slike
            from PIL import Image
            # Proverimo tip i izvučemo bytes ako je potrebno
            logger.info(f"image_bytes type: {type(image_bytes)}, has getvalue: {hasattr(image_bytes, 'getvalue')}")
            if hasattr(image_bytes, 'getvalue'):
                # Ako je BytesIO objekat
                actual_bytes = image_bytes.getvalue()
                logger.info(f"BytesIO case: actual_bytes length = {len(actual_bytes)}")
                image_bytes.seek(0)  # Reset pointer za slučaj da se koristi ponovo
            else:
                # Ako su već bytes
                actual_bytes = image_bytes
                logger.info(f"Bytes case: actual_bytes length = {len(actual_bytes) if actual_bytes else 0}")

            logger.info(f"About to open image with BytesIO, actual_bytes length: {len(actual_bytes) if actual_bytes else 0}")
            if actual_bytes:
                logger.info(f"First 20 bytes (hex): {actual_bytes[:20].hex()}")
                logger.info(f"First 20 bytes (repr): {repr(actual_bytes[:20])}")
            original_image = Image.open(BytesIO(actual_bytes))
            original_width, original_height = original_image.size
            logger.info(f"Original image dimensions: {original_width}x{original_height}")
            
            # Smanjimo veličinu slike - proslijedi bytes
            resized_image = ImageService.resize_image(actual_bytes)
            
            # Dobijamo dimenzije smanjene slike
            resized_pil = Image.open(resized_image)
            resized_width, resized_height = resized_pil.size
            logger.info(f"Resized image dimensions: {resized_width}x{resized_height}")
            
            # Očisti domain za putanju
            clean_domain = RecognitionService.clean_domain_for_path(domain)
            
            # Kreiraj privremeni folder za domain ako ne postoji
            temp_folder = os.path.join('storage/uploads', clean_domain)
            os.makedirs(temp_folder, exist_ok=True)
            
            # Sačuvaj smanjenu sliku privremeno
            image_path = os.path.join(temp_folder, f"temp_recognition_{int(time.time() * 1000)}.jpg")
            with open(image_path, "wb") as f:
                f.write(resized_image.getvalue())
            logger.info(f"Resized image saved temporarily at: {image_path}")

            # Model selection - using ArcFace for all domains (faster & more accurate)
            model_name = "ArcFace"
            detector_backend = "retinaface"
            distance_metric = "cosine"
            db_path = os.path.join('storage/recognized_faces_prod', clean_domain)

            # Extract faces
            logger.info(f"[STANDARD] Calling DeepFace.extract_faces with detector={detector_backend}")
            logger.info(f"[STANDARD] Image path: {image_path}, exists: {os.path.exists(image_path)}")
            logger.info(f"[STANDARD] Image dimensions: {original_width}x{original_height} -> {resized_width}x{resized_height}")

            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=detector_backend,
                enforce_detection=False,
                normalize_face=True,
                align=True
            )

            logger.info(f"[STANDARD] DeepFace.extract_faces returned {len(faces)} faces")
            for i, face in enumerate(faces):
                logger.info(f"[STANDARD] Face {i+1}: confidence={face.get('confidence', 1):.4f}, area={face.get('facial_area')}")

            if len(faces) == 0:
                print("❌ Nema nijednog lica.")
            else:
                print(f"✅ Pronađeno lica: {len(faces)}")

                # Lista za čuvanje informacija o validnim licima (ne čuvamo fizičke slike)
                valid_faces = []

                # Obradi svako lice kroz sve validacije
                for i, face in enumerate(faces):
                    face_info = RecognitionService.process_single_face(
                        face, i+1, image_path, original_width, original_height, resized_width, resized_height, source_type
                    )
                    if face_info:
                        valid_faces.append(face_info)

                # Finalna provera - size filtering for multi-person recognition
                # Changed size_threshold from 0.7 to 0.50 to 0.30 (30% of largest face)
                # This allows smaller but still valid faces to be recognized
                final_valid_faces = FaceValidationService.process_face_filtering(valid_faces, size_threshold=0.30)
                
                # Early exit ako nema validnih lica nakon svih provera
                if len(final_valid_faces) == 0:
                    print("🚫 Prekidam face recognition - nema validnih lica za obradu.")
                    logger.info("Stopping face recognition - no valid faces to process after all checks")
                    return {
                        "status": "no_faces",
                        "message": "No valid faces found after validation checks",
                        "recognized_faces": [],
                        "total_faces_detected": len(faces),
                        "valid_faces_after_filtering": 0
                    }
                
            try:
                # Use ArcFace for all domains (faster & more accurate)
                model_name = "ArcFace"
                use_batched = True
                logger.info(f"Using ArcFace model for {clean_domain} domain (faster & more accurate)")

                # Set threshold based on source type
                # Video frames have motion blur, compression artifacts, variable lighting
                # so they need a more lenient threshold than static photos
                if source_type == "video":
                    recognition_threshold = 0.55  # 45% confidence minimum for video frames
                    logger.info(f"Using VIDEO threshold: {recognition_threshold} (45% confidence minimum)")
                else:
                    recognition_threshold = 0.30  # 70% confidence minimum for static images
                    logger.info(f"Using IMAGE threshold: {recognition_threshold} (70% confidence minimum)")

                detector_backend = "retinaface"
                distance_metric = "cosine"
                db_path = os.path.join('storage/recognized_faces_prod', clean_domain)

                logger.info(f"Building {model_name} model...")
                _ = DeepFace.build_model(model_name)
                logger.info("Model built")
                logger.info("DB path: " + db_path)
                logger.info("Image Path: " + image_path)
                logger.info(f"Using batched mode: {use_batched}")
                
                # Izvršavamo prepoznavanje sa ili bez batched parametra
                dfs = DeepFace.find(
                    img_path=image_path,
                    db_path=db_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    distance_metric=distance_metric,
                    enforce_detection=False,
                    threshold=recognition_threshold,
                    silent=False,
                    batched=use_batched,  # KLJUČNI PARAMETAR!
                    refresh_database=False,  # Don't rebuild pkl on every request
                    align=True,  # Must match pkl settings
                    normalization='base',  # Must match pkl settings
                    expand_percentage=0  # Must match pkl settings
                )

                # DEBUG: Log raw DeepFace.find results
                logger.info(f"DEBUG: DeepFace.find returned type={type(dfs)}, len={len(dfs) if dfs else 0}")
                if dfs and len(dfs) > 0:
                    for face_idx, face_results in enumerate(dfs):
                        logger.info(f"DEBUG: Face {face_idx+1}: type={type(face_results)}, len={len(face_results) if hasattr(face_results, '__len__') else 'N/A'}")
                        if hasattr(face_results, '__len__') and len(face_results) > 0:
                            if isinstance(face_results, list):
                                for m_idx, m in enumerate(face_results[:3]):
                                    logger.info(f"DEBUG: Match {m_idx+1}: {m.get('identity', 'N/A')[:50]}... distance={m.get('distance', 'N/A')}")
                            elif hasattr(face_results, 'iterrows'):
                                for m_idx, (_, row) in enumerate(face_results.head(3).iterrows()):
                                    logger.info(f"DEBUG: Match {m_idx+1}: {row['identity'][:50]}... distance={row['distance']}")

                # Pozivamo odgovarajuće funkcije na osnovu use_batched promenljive
                if use_batched:
                    # BATCHED MODE: DeepFace vraća list of list of dicts
                    logger.info("Using BATCHED functions (List[List[Dict]])")
                    RecognitionService.log_deepface_results_batched(dfs)
                    filtered_dfs = RecognitionService.filter_recognition_results_by_valid_faces_batched(
                        dfs, final_valid_faces, resized_width, resized_height
                    )
                    result = RecognitionService.analyze_recognition_results_batched(
                        filtered_dfs,
                        threshold=recognition_threshold,  # Use the correct threshold for the model
                        original_width=original_width,
                        original_height=original_height,
                        resized_width=resized_width,
                        resized_height=resized_height
                    )
                else:
                    # STANDARD MODE: DeepFace vraća list of DataFrames (postojeće funkcije)
                    logger.info("Using STANDARD functions (list of DataFrames)")
                    RecognitionService.log_deepface_results(dfs)
                    filtered_dfs = RecognitionService.filter_recognition_results_by_valid_faces(
                        dfs, final_valid_faces, resized_width, resized_height
                    )
                    result = RecognitionService.analyze_recognition_results(
                        filtered_dfs,
                        threshold=recognition_threshold,  # Use the correct threshold for the model
                        original_width=original_width,
                        original_height=original_height,
                        resized_width=resized_width,
                        resized_height=resized_height
                    )
                logger.info(f"Recognition completed in {time.time() - start_time:.2f}s")
                return result
                
            except Exception as e:
                error_msg = f"Error during face recognition: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
        except Exception as e:
            logger.error(f"Error in recognize_face: {str(e)}")
            raise
        finally:
            # Čišćenje
            try:
                if 'image_path' in locals() and os.path.exists(image_path):
                    os.remove(image_path)
                    logger.info(f"Cleaned up temporary file: {image_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")

    @staticmethod
    def are_coordinates_similar(coord1, coord2, tolerance=10):
        """
        Proverava da li su koordinate dovoljno slične (u procentima).
        tolerance: razlika u procentima za x, y koordinate
        """
        if not coord1 or not coord2:
            return False
        
        x_diff = abs(coord1['x_percent'] - coord2['x_percent'])
        y_diff = abs(coord1['y_percent'] - coord2['y_percent'])
        
        return x_diff <= tolerance and y_diff <= tolerance
    
    @staticmethod
    def group_matches_by_coordinates(matches_with_coords, tolerance=10):
        """
        Grupira prepoznate osobe po sličnim koordinatama i zadržava samo onu sa najvećim confidence-om
        """
        if not matches_with_coords:
            return []
        
        grouped_matches = []
        used_indices = set()
        
        for i, match in enumerate(matches_with_coords):
            if i in used_indices:
                continue
                
            # Kreiraj grupu za trenutni match
            current_group = [match]
            used_indices.add(i)
            
            # Pronađi sve ostale matches sa sličnim koordinatama
            for j, other_match in enumerate(matches_with_coords):
                if j in used_indices:
                    continue
                    
                if RecognitionService.are_coordinates_similar(
                    match.get('face_coords'), 
                    other_match.get('face_coords'), 
                    tolerance
                ):
                    current_group.append(other_match)
                    used_indices.add(j)
            
            # Iz grupe izaberi match sa najmanjom distance (najvećim confidence-om)
            best_match_in_group = min(current_group, key=lambda x: x['distance'])
            grouped_matches.append(best_match_in_group)
            
            if len(current_group) > 1:
                logger.info(f"Grouped {len(current_group)} matches at similar coordinates, selected: {best_match_in_group['name']} (confidence: {round((1 - best_match_in_group['distance']) * 100, 2)}%)")
        
        return grouped_matches

    @staticmethod
    def analyze_recognition_results(results, threshold=0.4, original_width=None, original_height=None, resized_width=None, resized_height=None):
        """
        Analizira rezultate prepoznavanja i vraća najverovatnije ime.
        """
        name_scores = defaultdict(list)
        all_matches = defaultdict(list)
        face_coordinates_map = defaultdict(list)  # Nova mapa za koordinate
        matches_with_coords = []  # Lista svih match-ova sa koordinatama
        original_deepface_results = {}  # Čuva originalne DeepFace rezultate po imenu
        
        logger.info("Analyzing recognition results...")
        
        # Provera da li je results None ili prazan
        if results is None or len(results) == 0:
            logger.info("No results to analyze")
            return {"status": "error", "message": "No matches found"}
        
        try:
            logger.info(f"Results type: {type(results)}")
            
            # DeepFace.find vraća listu DataFrame-ova
            if isinstance(results, list):
                logger.info("Processing list of DataFrames")
                for df in results:
                    if hasattr(df, 'iterrows'):
                        for _, row in df.iterrows():
                            try:
                                distance = float(row['distance'])
                                full_path = row['identity']
                                
                                # Izvlačimo koordinate lica sa smanjene slike
                                face_coords = None
                                if all(dim is not None for dim in [original_width, original_height, resized_width, resized_height]):
                                    try:
                                        source_x = float(row['source_x'])
                                        source_y = float(row['source_y'])
                                        source_w = float(row['source_w'])
                                        source_h = float(row['source_h'])
                                        
                                        # Konvertujemo u procente originalne slike
                                        face_coords = {
                                            "x_percent": round((source_x / resized_width) * 100, 2),
                                            "y_percent": round((source_y / resized_height) * 100, 2),
                                            "width_percent": round((source_w / resized_width) * 100, 2),
                                            "height_percent": round((source_h / resized_height) * 100, 2)
                                        }
                                        logger.debug(f"Face coordinates: {face_coords}")
                                    except (KeyError, ValueError) as coord_error:
                                        logger.warning(f"Could not extract face coordinates: {coord_error}")
                                        face_coords = None
                                
                                # Izvlačimo ime osobe (sve do datuma)
                                if '\\' in full_path:  # Windows putanja
                                    filename = full_path.split('\\')[-1]
                                else:  # Unix putanja
                                    filename = full_path.split('/')[-1]
                                
                                # Uzimamo sve do prvog datuma (YYYYMMDD ili YYYY-MM-DD format)
                                name_parts = filename.split('_')
                                name = []
                                for part in name_parts:
                                    if len(part) >= 8 and (part[0:4].isdigit() or '-' in part):
                                        break
                                    name.append(part)
                                name = '_'.join(name)  # Koristimo donju crtu za spajanje
                                
                                normalized_name = name.strip()
                                
                                # Store match sa koordinatama za grupiranje
                                match_data = {
                                    'name': normalized_name,
                                    'distance': distance,
                                    'face_coords': face_coords,
                                    'full_path': full_path
                                }
                                matches_with_coords.append(match_data)
                                
                                # Čuvaj originalne DeepFace rezultate za svaku osobu
                                if normalized_name not in original_deepface_results:
                                    original_deepface_results[normalized_name] = []
                                original_deepface_results[normalized_name].append(dict(row))
                                
                                # Store all matches (za kompatibilnost)
                                all_matches[normalized_name].append(distance)
                                if face_coords:
                                    face_coordinates_map[normalized_name].append(face_coords)
                                logger.debug(f"Found match: {normalized_name} with distance {distance}")
                            except Exception as e:
                                logger.warning(f"Error processing row: {str(e)}")
                                continue
            else:
                logger.error(f"Unexpected results format: {type(results)}")
                return {"status": "error", "message": "Unexpected results format"}
                
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            return {"status": "error", "message": "Error processing recognition results"}

        # Grupiranje match-ova po koordinatama 
        logger.info(f"Total matches before grouping: {len(matches_with_coords)}")
        grouped_matches = RecognitionService.group_matches_by_coordinates(matches_with_coords, tolerance=10)
        logger.info(f"Total matches after grouping: {len(grouped_matches)}")
        
        # Kreiranje novih struktura podataka na osnovu grupiranih rezultata
        name_scores = defaultdict(list)
        all_matches = defaultdict(list)
        face_coordinates_map = defaultdict(list)
        
        for match in grouped_matches:
            name = match['name']
            distance = match['distance']
            face_coords = match['face_coords']
            
            # Store all matches
            all_matches[name].append(distance)
            if face_coords:
                face_coordinates_map[name].append(face_coords)
                
            # Store matches that pass threshold
            if distance < threshold:
                name_scores[name].append(distance)
                logger.debug(f"Grouped match passed threshold: {name} with distance {distance}")

        # Log summary of all matches found
        logger.info(f"\n{'='*50}")
        logger.info(f"RECOGNITION RESULTS:")
        logger.info(f"Total unique persons found: {len(all_matches)}")
        for name, distances in all_matches.items():
            avg_confidence = round((1 - sum(distances)/len(distances)) * 100, 2)
            logger.info(f"Person: {name}")
            logger.info(f"- Occurrences: {len(distances)}")
            logger.info(f"- Average confidence: {avg_confidence}%")
            logger.info(f"- Best confidence: {round((1 - min(distances)) * 100, 2)}%")
        logger.info(f"{'='*50}\n")

        # Process matches that passed threshold
        if not name_scores:
            logger.info(f"No matches found within threshold {threshold}")
            # Return all matches even if none passed threshold
            return {
                "status": "error",
                "message": "No matches within threshold",
                "all_detected_matches": [
                    {
                        "person_name": name,
                        "metrics": {
                            "occurrences": len(distances),
                            "average_distance": round(sum(distances) / len(distances), 4),
                            "min_distance": round(min(distances), 4),
                            "distances": distances
                        }
                    }
                    for name, distances in all_matches.items()
                ]
            }
        
        # Calculate statistics for matches that passed threshold
        # Using top 3 average for more robust matching
        name_statistics = {}
        for name, distances in name_scores.items():
            avg_distance = sum(distances) / len(distances)
            min_distance = min(distances)
            occurrences = len(distances)

            # Take average of top 3 matches (or all if fewer than 3)
            sorted_distances = sorted(distances)[:3]
            top3_avg_distance = sum(sorted_distances) / len(sorted_distances)

            # Calculate face size bonus: prioritize larger faces
            face_coords_list = face_coordinates_map.get(name, [])
            face_size_bonus = 0.0
            if face_coords_list:
                # Get first face coordinates (all should be for same face)
                coords = face_coords_list[0]
                if coords:
                    # Calculate face area as percentage of image
                    width_pct = coords.get('width_percent', 0)
                    height_pct = coords.get('height_percent', 0)
                    face_area = width_pct * height_pct  # Area as percentage
                    # Larger faces get a bonus (subtract from weighted score to lower it)
                    # Face area typically ranges from 1-100 (percent squared)
                    # Normalize to 0-0.15 range for bonus
                    face_size_bonus = (face_area / 100.0) * 0.15

            # Improved weighted score: top3 average + min distance - occurrence bonus - face size bonus
            # Lower score = better match
            weighted_score = (top3_avg_distance * 0.5) + (min_distance * 0.2) - (occurrences * 0.05) - face_size_bonus

            name_statistics[name] = {
                "occurrences": occurrences,
                "avg_distance": avg_distance,
                "min_distance": min_distance,
                "top3_avg_distance": top3_avg_distance,
                "face_size_bonus": face_size_bonus,
                "weighted_score": weighted_score,
                "distances": distances
            }

            logging.info(f"Threshold-passing matches for {name}:")
            logger.info(f"- Occurrences: {occurrences}")
            logger.info(f"- Average distance: {avg_distance:.4f}")
            logger.info(f"- Top 3 avg distance: {top3_avg_distance:.4f}")
            logger.info(f"- Min distance: {min_distance:.4f}")
            logger.info(f"- Face size bonus: {face_size_bonus:.4f}")
            logger.info(f"- Weighted score: {weighted_score:.4f}")
        
        # Find best match among threshold-passing matches
        best_match = min(name_statistics.items(), key=lambda x: x[1]['weighted_score'])
        best_name = best_match[0]
        stats = best_match[1]
        
        # Dodajemo ispis najboljeg podudaranja
        logger.info("\n" + "="*50)
        logger.info(f"BEST MATCH FOUND: {best_name}")
        logger.info(f"Confidence: {round((1 - stats['min_distance']) * 100, 2)}%")
        logger.info("="*50 + "\n")
        
        logger.info(f"Best match found: {best_name} with confidence {round((1 - stats['min_distance']) * 100, 2)}%")
        
        # Dobavi originalno ime osobe iz mapiranja
        from app.services.text_service import TextService
        original_person = TextService.get_original_text(best_name)
        
        # Ako je pronađeno originalno ime, koristi ga
        if original_person != best_name:
            logger.info(f"Found original name for {best_name}: {original_person}")
            display_name = original_person
        # Ako nije pronađeno originalno ime, a ime sadrži donju crtu, zameni je razmakom
        elif '_' in best_name:
            display_name = best_name.replace('_', ' ')
            logger.info(f"No mapping found, using formatted name: {display_name}")
        else:
            display_name = best_name

        # Sortiraj sve osobe po weighted_score i uzmi top 10
        # Sort by weighted score (lower is better)
        sorted_matches = sorted(name_statistics.items(), key=lambda x: x[1]['weighted_score'])
        top_matches = sorted_matches[:10]  # Limit to top 10 matches

        # Kreiraj niz svih prepoznatih osoba koje su prošle threshold (top 10)
        recognized_persons = []
        for person_name, person_stats in top_matches:
            original_person = TextService.get_original_text(person_name)
            if original_person != person_name:
                formatted_display_name = original_person
            elif '_' in person_name:
                formatted_display_name = person_name.replace('_', ' ')
            else:
                formatted_display_name = person_name
            
            # Uzmi samo prve koordinate za tu osobu (sve su iste jer se odnose na istu lokaciju na ulaznoj slici)
            coords_list = face_coordinates_map.get(person_name, [])
            face_coordinates = coords_list[0] if coords_list else None
            
            # Dodajemo objekat sa imenom i jednim setom koordinata
            person_obj = {
                "name": formatted_display_name,
                "face_coordinates": face_coordinates
            }
            recognized_persons.append(person_obj)
        
        logger.info(f"All recognized persons: {[p['name'] for p in recognized_persons]}")
        
        # Logiraj originalne DeepFace rezultate za finalne prepoznate osobe
        logger.info("\n" + "="*80)
        logger.info("ORIGINAL DEEPFACE RESULTS FOR FINAL RECOGNIZED PERSONS:")
        logger.info("="*80)
        for person_name in name_scores.keys():
            if person_name in original_deepface_results:
                logger.info(f"\nPerson: {person_name}")
                logger.info("-" * 50)
                for i, result in enumerate(original_deepface_results[person_name]):
                    logger.info(f"DeepFace Result #{i+1}:")
                    for key, value in result.items():
                        logger.info(f"  {key}: {value}")
                    logger.info("-" * 30)
        logger.info("="*80 + "\n")
        
        return {
            "status": "success",
            "message": f"Face recognized as: {display_name}",
            "person": display_name,  # Koristimo originalno ili formatirano ime
            "recognized_persons": recognized_persons,  # Novi niz objekata sa imenima i koordinatama
            "best_match": {
                "person_name": best_name,  # Originalno normalizovano ime
                "display_name": display_name,  # Ime za prikaz (originalno ili formatirano)
                "confidence_metrics": {
                    "occurrences": stats['occurrences'],
                    "average_distance": round(stats['avg_distance'], 4),
                    "min_distance": round(stats['min_distance'], 4),
                    "weighted_score": round(stats['weighted_score'], 4),
                    "confidence_percentage": round((1 - stats['min_distance']) * 100, 2),
                    "distances": stats['distances']
                }
            },
            "all_detected_matches": [
                {
                    "person_name": name,
                    "metrics": {
                        "occurrences": name_statistics[name]['occurrences'],
                        "average_distance": round(name_statistics[name]['avg_distance'], 4),
                        "min_distance": round(name_statistics[name]['min_distance'], 4),
                        "confidence_percentage": round((1 - name_statistics[name]['min_distance']) * 100, 2),
                        "distances": name_statistics[name]['distances'],
                        "weighted_score": round(name_statistics[name]['weighted_score'], 4)
                    }
                }
                for name, stats in top_matches  # Only include top 10 matches
            ]
        }

    @staticmethod
    def log_deepface_results(results):
        """
        Logiraj detaljno sve rezultate DeepFace.find pre filtriranja
        
        Args:
            results: Rezultati DeepFace.find (lista DataFrame-ova)
        """
        logger.info("\n" + "="*80)
        logger.info("DEEPFACE.FIND RESULTS - ALL FOUND MATCHES (PRE FILTRIRANJE)")
        logger.info("="*80)
        
        if not results or len(results) == 0:
            logger.info("❌ Nema rezultata od DeepFace.find")
            print("❌ Nema rezultata od DeepFace.find")
            return
        
        total_matches = 0
        all_persons = {}  # Dictionary za grupisanje po imenima
        
        # Analiziraj svaki DataFrame
        for df_index, df in enumerate(results):
            logger.info(f"\n📊 DataFrame {df_index + 1}:")
            print(f"\n📊 Analiziram DataFrame {df_index + 1}:")
            
            if hasattr(df, 'iterrows') and len(df) > 0:
                logger.info(f"   Broj pronađenih match-ova: {len(df)}")
                print(f"   Broj pronađenih match-ova: {len(df)}")
                
                for row_index, row in df.iterrows():
                    try:
                        # Izvuci osnovne informacije
                        identity_path = row['identity']
                        distance = float(row['distance'])
                        confidence = round((1 - distance) * 100, 2)
                        
                        # Koordinate lica
                        source_x = float(row['source_x'])
                        source_y = float(row['source_y'])
                        source_w = float(row['source_w'])
                        source_h = float(row['source_h'])
                        
                        # Ekstraktaj ime osobe iz putanje
                        if '\\' in identity_path:  # Windows putanja
                            filename = identity_path.split('\\')[-1]
                        else:  # Unix putanja
                            filename = identity_path.split('/')[-1]
                        
                        # Uzmi ime do prvog datuma
                        name_parts = filename.split('_')
                        person_name = []
                        for part in name_parts:
                            if len(part) >= 8 and (part[0:4].isdigit() or '-' in part):
                                break
                            person_name.append(part)
                        person_name = '_'.join(person_name)
                        
                        # Logiraj detalje match-a
                        logger.info(f"   ➡️ Match {row_index + 1}:")
                        logger.info(f"      👤 Osoba: {person_name}")
                        logger.info(f"      📁 Putanja: {identity_path}")
                        logger.info(f"      📏 Distance: {distance:.4f}")
                        logger.info(f"      🎯 Confidence: {confidence}%")
                        logger.info(f"      📍 Koordinate: x={source_x}, y={source_y}, w={source_w}, h={source_h}")
                        
                        print(f"   ➡️ Match {row_index + 1}: {person_name} - {confidence}% confidence")
                        
                        # Grupiši po imenima
                        if person_name not in all_persons:
                            all_persons[person_name] = []
                        all_persons[person_name].append({
                            'distance': distance,
                            'confidence': confidence,
                            'path': identity_path,
                            'coordinates': f"x={source_x}, y={source_y}, w={source_w}, h={source_h}"
                        })
                        
                        total_matches += 1
                        
                    except Exception as e:
                        logger.error(f"   ❌ Greška pri obradi row-a {row_index}: {str(e)}")
                        continue
                        
            else:
                logger.info("   📭 Prazan DataFrame")
                print("   📭 Prazan DataFrame")
        
        # Sumariziraj po osobama
        logger.info(f"\n📈 SUMARNI PREGLED:")
        logger.info(f"   🔢 Ukupno match-ova: {total_matches}")
        logger.info(f"   👥 Različitih osoba: {len(all_persons)}")
        
        print(f"\n📈 SUMARNI PREGLED:")
        print(f"   🔢 Ukupno match-ova: {total_matches}")
        print(f"   👥 Različitih osoba: {len(all_persons)}")
        
        if all_persons:
            logger.info(f"\n👤 OSOBE I NJIHOVI MATCH-OVI:")
            print(f"\n👤 OSOBE I NJIHOVI MATCH-OVI:")
            
            for person_name, matches in all_persons.items():
                avg_confidence = round(sum(match['confidence'] for match in matches) / len(matches), 2)
                best_confidence = round(max(match['confidence'] for match in matches), 2)
                
                logger.info(f"   🏷️  {person_name}:")
                logger.info(f"      📊 Broj match-ova: {len(matches)}")
                logger.info(f"      🎯 Prosečna sigurnost: {avg_confidence}%")
                logger.info(f"      ⭐ Najbolja sigurnost: {best_confidence}%")
                
                print(f"   🏷️  {person_name}: {len(matches)} match-ova (prosek: {avg_confidence}%, najbolja: {best_confidence}%)")
                
                # Logiraj sve match-ove za ovu osobu
                for i, match in enumerate(matches):
                    logger.info(f"      └─ Match {i+1}: {match['confidence']}% ({match['coordinates']})")
        
        logger.info("="*80 + "\n")
        print("="*50)

    @staticmethod
    def log_valid_faces(valid_faces):
        """
        Logiraj validna lica koja su prošla sve provere
        
        Args:
            valid_faces (list): Lista validnih lica
        """
        logger.info("\n" + "="*80)
        logger.info("VALIDNA LICA KOJA SU PROŠLA SVE PROVERE")
        logger.info("="*80)
        
        if not valid_faces or len(valid_faces) == 0:
            logger.info("❌ Nema validnih lica nakon svih provera")
            print("❌ Nema validnih lica nakon svih provera")
            return
        
        logger.info(f"✅ Broj validnih lica: {len(valid_faces)}")
        print(f"✅ Broj validnih lica: {len(valid_faces)}")
        
        for face_info in valid_faces:
            logger.info(f"\n   👤 Lice {face_info['index']}:")
            logger.info(f"      📏 Dimenzije: {face_info['width']}x{face_info['height']} (površina: {face_info['area']})")
            
            # Originalne koordinate
            orig_coords = face_info['original_coordinates']
            logger.info(f"      🎯 Originalne koordinate: x={orig_coords['x']}, y={orig_coords['y']}, w={orig_coords['w']}, h={orig_coords['h']}")
            
            # Resized koordinate (za poređenje sa DeepFace)
            resized_coords = face_info['resized_coordinates']
            logger.info(f"      🔍 Resized koordinate: x={resized_coords['x']}, y={resized_coords['y']}, w={resized_coords['w']}, h={resized_coords['h']}")
            
            print(f"   👤 Lice {face_info['index']}: {face_info['width']}x{face_info['height']} na poziciji ({resized_coords['x']}, {resized_coords['y']})")
        
        logger.info("="*80 + "\n")
        print("="*50)

    # =====================================
    # BATCHED=TRUE FUNCTIONS (list of dicts)
    # =====================================
    
    @staticmethod
    def log_deepface_results_batched(results):
        """
        Logiraj detaljno sve rezultate DeepFace.find pre filtriranja (BATCHED MODE - list of list of dicts)
        
        Args:
            results: Rezultati DeepFace.find sa batched=True (List[List[Dict[str, Any]]])
        """
        logger.info("\n" + "="*80)
        logger.info("DEEPFACE.FIND RESULTS (BATCHED MODE) - ALL FOUND MATCHES (PRE FILTRIRANJE)")
        logger.info("="*80)
        
        if not results or len(results) == 0:
            logger.info("❌ Nema rezultata od DeepFace.find (batched)")
            print("❌ Nema rezultata od DeepFace.find (batched)")
            return
        
        total_matches = 0
        all_persons = {}  # Dictionary za grupisanje po imenima
        
        logger.info(f"\n📊 Batched Results:")
        print(f"\n📊 Analiziram Batched Results:")
        logger.info(f"   Broj lica detektovanih: {len(results)}")
        print(f"   Broj lica detektovanih: {len(results)}")
        
        # Analiziraj svaku listu dictionary objekata (jedno po detektovanom licu)
        face_index = 0
        for face_results in results:
            logger.info(f"\n📊 Face {face_index + 1} matches:")
            print(f"\n📊 Face {face_index + 1} matches:")
            
            if not face_results or len(face_results) == 0:
                logger.info("   📭 Nema match-ova za ovo lice")
                print("   📭 Nema match-ova za ovo lice")
                face_index += 1
                continue
                
            logger.info(f"   Broj pronađenih match-ova: {len(face_results)}")
            print(f"   Broj pronađenih match-ova: {len(face_results)}")
            
            # Analiziraj svaki dict u listi za ovo lice
            for match_index, match_dict in enumerate(face_results):
                try:
                    # Izvuci osnovne informacije iz dict-a
                    identity_path = match_dict['identity']
                    distance = float(match_dict['distance'])
                    confidence = round((1 - distance) * 100, 2)
                    
                    # Koordinate lica
                    source_x = float(match_dict['source_x'])
                    source_y = float(match_dict['source_y'])
                    source_w = float(match_dict['source_w'])
                    source_h = float(match_dict['source_h'])
                    
                    # Ekstraktaj ime osobe iz putanje
                    if '\\' in identity_path:  # Windows putanja
                        filename = identity_path.split('\\')[-1]
                    else:  # Unix putanja
                        filename = identity_path.split('/')[-1]
                    
                    # Uzmi ime do prvog datuma
                    name_parts = filename.split('_')
                    person_name = []
                    for part in name_parts:
                        if len(part) >= 8 and (part[0:4].isdigit() or '-' in part):
                            break
                        person_name.append(part)
                    person_name = '_'.join(person_name)
                    
                    # Logiraj detalje match-a
                    logger.info(f"   ➡️ Match {match_index + 1} (batched):")
                    logger.info(f"      👤 Osoba: {person_name}")
                    logger.info(f"      📁 Putanja: {identity_path}")
                    logger.info(f"      📏 Distance: {distance:.4f}")
                    logger.info(f"      🎯 Confidence: {confidence}%")
                    logger.info(f"      📍 Koordinate: ({source_x}, {source_y}, {source_w}, {source_h})")
                    
                    print(f"   ➡️ Match {match_index + 1}: {person_name} - Confidence: {confidence}%")
                    
                    # Grupiši po imenima
                    if person_name not in all_persons:
                        all_persons[person_name] = []
                    all_persons[person_name].append({
                        'distance': distance,
                        'confidence': confidence,
                        'path': identity_path,
                        'coordinates': (source_x, source_y, source_w, source_h)
                    })
                    
                    total_matches += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing batched match {match_index + 1}: {str(e)}")
                    continue
            
            face_index += 1
        
        # Sumiranje rezultata
        logger.info(f"\n📈 SUMARNI PREGLED PRONAĐENIH OSOBA (BATCHED):")
        print(f"\n📈 SUMARNI PREGLED PRONAĐENIH OSOBA (BATCHED):")
        logger.info(f"   Ukupno match-ova: {total_matches}")
        logger.info(f"   Broj različitih osoba: {len(all_persons)}")
        print(f"   Ukupno match-ova: {total_matches}")
        print(f"   Broj različitih osoba: {len(all_persons)}")
        
        for person_name, matches in all_persons.items():
            avg_confidence = sum(match['confidence'] for match in matches) / len(matches)
            best_confidence = max(match['confidence'] for match in matches)
            logger.info(f"   👤 {person_name}: {len(matches)} match-ova, avg confidence: {avg_confidence:.1f}%, best: {best_confidence:.1f}%")
            print(f"   👤 {person_name}: {len(matches)} match-ova, avg confidence: {avg_confidence:.1f}%, best: {best_confidence:.1f}%")
        
        logger.info("="*80 + "\n")
        print("="*50)

    @staticmethod
    def filter_recognition_results_by_valid_faces_batched(results, valid_faces, resized_width, resized_height):
        """
        Filtrira rezultate DeepFace.find na osnovu validnih lica (BATCHED MODE - list of list of dicts)
        
        Args:
            results: Rezultati DeepFace.find sa batched=True (List[List[Dict[str, Any]]])
            valid_faces (list): Lista validnih lica
            resized_width (int): Širina resized slike
            resized_height (int): Visina resized slike
            
        Returns:
            Filtrirani rezultati (List[List[Dict[str, Any]]])
        """
        if not valid_faces or not results:
            return results
        
        logger.info(f"Filtering batched recognition results based on {len(valid_faces)} valid faces")
        
        # Kreiraj koordinate validnih lica u resized formatu za poređenje
        valid_coordinates = []
        for face_info in valid_faces:
            resized_coords = face_info['resized_coordinates']
            valid_coordinates.append({
                'x': resized_coords['x'],
                'y': resized_coords['y'],
                'w': resized_coords['w'],
                'h': resized_coords['h'],
                'index': face_info['index']
            })
        
        filtered_results = []
        total_original_matches = 0
        total_filtered_matches = 0
        
        # Batched results su lista lista dictionary objekata
        for face_results in results:
            face_filtered_matches = []
            
            if not face_results or len(face_results) == 0:
                filtered_results.append(face_filtered_matches)
                continue
            
            total_original_matches += len(face_results)
            
            for match_dict in face_results:
                try:
                    # Dobij koordinate iz rezultata
                    source_x = float(match_dict['source_x'])
                    source_y = float(match_dict['source_y'])
                    source_w = float(match_dict['source_w'])
                    source_h = float(match_dict['source_h'])
                    
                    # Proveri da li se poklapaju sa bilo kojim validnim licem
                    for valid_coord in valid_coordinates:
                        # Tolerancija za poređenje koordinata (u pikselima)
                        tolerance = 5
                        
                        if (abs(source_x - valid_coord['x']) <= tolerance and
                            abs(source_y - valid_coord['y']) <= tolerance and
                            abs(source_w - valid_coord['w']) <= tolerance and
                            abs(source_h - valid_coord['h']) <= tolerance):
                            
                            face_filtered_matches.append(match_dict)
                            total_filtered_matches += 1
                            logger.info(f"Batched match found for valid face {valid_coord['index']} at coordinates ({source_x}, {source_y}, {source_w}, {source_h})")
                            break
                except Exception as e:
                    logger.warning(f"Error filtering batched match: {str(e)}")
                    continue
            
            filtered_results.append(face_filtered_matches)
        
        logger.info(f"Filtered batched results: {total_original_matches} -> {total_filtered_matches} matches")
        return filtered_results

    @staticmethod  
    def analyze_recognition_results_batched(results, threshold=0.4, original_width=None, original_height=None, resized_width=None, resized_height=None):
        """
        Analizira rezultate prepoznavanja (BATCHED MODE - List[List[Dict]])
        Vraća isti format kao analyze_recognition_results ali radi sa List[List[Dict]] umesto DataFrames.
        """
        name_scores = defaultdict(list)
        all_matches = defaultdict(list)
        face_coordinates_map = defaultdict(list)
        matches_with_coords = []
        original_deepface_results = {}
        
        logger.info("Analyzing batched recognition results...")
        
        # Provera da li je results None ili prazan
        if results is None or len(results) == 0:
            logger.info("No batched results to analyze")
            return {"status": "error", "message": "No matches found"}
        
        try:
            logger.info(f"Batched results type: {type(results)}, faces count: {len(results)}")
            
            # Batched results su lista lista dictionary objekata
            total_matches_processed = 0
            for face_index, face_results in enumerate(results):
                if not face_results or len(face_results) == 0:
                    logger.info(f"Face {face_index + 1}: No matches")
                    continue
                
                logger.info(f"Face {face_index + 1}: Processing {len(face_results)} matches")
                
                for match_dict in face_results:
                    try:
                        distance = float(match_dict['distance'])
                        full_path = match_dict['identity']
                        
                        # Izvlačimo koordinate lica sa smanjene slike
                        face_coords = None
                        if all(dim is not None for dim in [original_width, original_height, resized_width, resized_height]):
                            try:
                                source_x = float(match_dict['source_x'])
                                source_y = float(match_dict['source_y'])
                                source_w = float(match_dict['source_w'])
                                source_h = float(match_dict['source_h'])
                                
                                # Konvertujemo u procente originalne slike
                                face_coords = {
                                    "x_percent": round((source_x / resized_width) * 100, 2),
                                    "y_percent": round((source_y / resized_height) * 100, 2),
                                    "width_percent": round((source_w / resized_width) * 100, 2),
                                    "height_percent": round((source_h / resized_height) * 100, 2)
                                }
                                logger.debug(f"Batched face coordinates: {face_coords}")
                            except (KeyError, ValueError) as coord_error:
                                logger.warning(f"Could not extract batched face coordinates: {coord_error}")
                                face_coords = None
                        
                        # Izvlačimo ime osobe (sve do datuma)
                        if '\\' in full_path:  # Windows putanja
                            filename = full_path.split('\\')[-1]
                        else:  # Unix putanja
                            filename = full_path.split('/')[-1]
                        
                        # Uzimamo sve do prvog datuma (YYYYMMDD ili YYYY-MM-DD format)
                        name_parts = filename.split('_')
                        name = []
                        for part in name_parts:
                            if len(part) >= 8 and (part[0:4].isdigit() or '-' in part):
                                break
                            name.append(part)
                        name = '_'.join(name)
                        
                        normalized_name = name.strip()
                        
                        # Store match sa koordinatama za grupiranje
                        match_data = {
                            'name': normalized_name,
                            'distance': distance,
                            'face_coords': face_coords,
                            'full_path': full_path
                        }
                        matches_with_coords.append(match_data)
                        
                        # Čuvaj originalne DeepFace rezultate za svaku osobu
                        if normalized_name not in original_deepface_results:
                            original_deepface_results[normalized_name] = []
                        original_deepface_results[normalized_name].append(match_dict)
                        
                        # Store all matches
                        all_matches[normalized_name].append(distance)
                        if face_coords:
                            face_coordinates_map[normalized_name].append(face_coords)
                        logger.debug(f"Found batched match: {normalized_name} with distance {distance}")
                        total_matches_processed += 1
                    except Exception as e:
                        logger.warning(f"Error processing batched match: {str(e)}")
                        continue
            
            logger.info(f"Total matches processed: {total_matches_processed}")
                    
        except Exception as e:
            logger.error(f"Error processing batched results: {str(e)}")
            return {"status": "error", "message": "Error processing batched recognition results"}

        # Skip grouping for video recognition - we want ALL matches for voting
        # The old grouping logic kept only ONE match per face location,
        # which defeated the purpose of multi-match voting
        logger.info(f"Total batched matches: {len(matches_with_coords)} (using all for voting)")

        # Kreiranje novih struktura podataka - use ALL matches, not grouped
        name_scores = defaultdict(list)
        all_matches = defaultdict(list)
        face_coordinates_map = defaultdict(list)

        for match in matches_with_coords:
            name = match['name']
            distance = match['distance']
            face_coords = match['face_coords']
            
            # Store all matches
            all_matches[name].append(distance)
            if face_coords:
                face_coordinates_map[name].append(face_coords)
                
            # Store matches that pass threshold
            if distance < threshold:
                name_scores[name].append(distance)
                logger.debug(f"Batched grouped match passed threshold: {name} with distance {distance}")

        # Log summary of all matches found (with training image paths for audit)
        logger.info(f"\n{'='*50}")
        logger.info(f"BATCHED RECOGNITION RESULTS:")
        logger.info(f"Total unique persons found: {len(all_matches)}")
        logger.info(f"Threshold: {threshold} (min confidence: {round((1-threshold)*100)}%)")
        for name, distances in all_matches.items():
            avg_confidence = round((1 - sum(distances)/len(distances)) * 100, 2)
            best_distance = min(distances)
            best_confidence = round((1 - best_distance) * 100, 2)
            passed_threshold = best_distance < threshold
            logger.info(f"Person: {name}")
            logger.info(f"  - Occurrences: {len(distances)}")
            logger.info(f"  - Average confidence: {avg_confidence}%")
            logger.info(f"  - Best confidence: {best_confidence}% {'[PASSED]' if passed_threshold else '[REJECTED - below threshold]'}")
            # Log the training images that matched this person
            if name in original_deepface_results:
                logger.info(f"  - Matched training images:")
                for match in sorted(original_deepface_results[name], key=lambda x: x['distance'])[:5]:  # Top 5 closest matches
                    match_conf = round((1 - match['distance']) * 100, 2)
                    img_path = match['identity'].split('/')[-1] if '/' in match['identity'] else match['identity'].split('\\')[-1]
                    logger.info(f"      {match_conf}% <- {img_path}")
        logger.info(f"{'='*50}\n")

        # Process matches that passed threshold
        if not name_scores:
            logger.info(f"No batched matches found within threshold {threshold}")
            # Return all matches even if none passed threshold
            return {
                "status": "error",
                "message": "No matches within threshold",
                "all_detected_matches": [
                    {
                        "person_name": name,
                        "metrics": {
                            "occurrences": len(distances),
                            "average_distance": round(sum(distances) / len(distances), 4),
                            "min_distance": round(min(distances), 4),
                            "distances": distances
                        }
                    }
                    for name, distances in all_matches.items()
                ]
            }
        
        # Calculate statistics for matches that passed threshold
        # For video recognition, we want to favor consistency (multiple matches) over single best match
        # For image recognition, single best match is more important
        name_statistics = {}
        for name, distances in name_scores.items():
            avg_distance = sum(distances) / len(distances)
            min_distance = min(distances)
            occurrences = len(distances)

            # Take average of top 3 matches (or all if fewer than 3)
            sorted_distances = sorted(distances)[:3]
            top3_avg_distance = sum(sorted_distances) / len(sorted_distances)

            # Improved weighted score formula:
            # - Uses top 3 average instead of single best (more robust)
            # - Gives stronger weight to occurrence count (consistency matters for video)
            # - Lower score = better match
            #
            # Old formula: weighted_score = (avg_distance * 0.4) + (min_distance * 0.3) - (occurrences * 0.1)
            # New formula: uses top3 average and stronger occurrence weight
            weighted_score = (top3_avg_distance * 0.5) + (min_distance * 0.2) - (occurrences * 0.05)

            name_statistics[name] = {
                "occurrences": occurrences,
                "avg_distance": avg_distance,
                "min_distance": min_distance,
                "top3_avg_distance": top3_avg_distance,
                "weighted_score": weighted_score,
                "distances": distances
            }

            logging.info(f"Batched threshold-passing matches for {name}:")
            logger.info(f"- Occurrences: {occurrences}")
            logger.info(f"- Average distance: {avg_distance:.4f}")
            logger.info(f"- Top 3 avg distance: {top3_avg_distance:.4f}")
            logger.info(f"- Min distance: {min_distance:.4f}")
            logger.info(f"- Weighted score: {weighted_score:.4f}")
        
        # Find best match among threshold-passing matches
        best_match = min(name_statistics.items(), key=lambda x: x[1]['weighted_score'])
        best_name = best_match[0]
        stats = best_match[1]
        
        # Dodajemo ispis najboljeg podudaranja
        logger.info("\n" + "="*50)
        logger.info(f"BEST BATCHED MATCH FOUND: {best_name}")
        logger.info(f"Confidence: {round((1 - stats['min_distance']) * 100, 2)}%")
        logger.info("="*50 + "\n")
        
        logger.info(f"Best batched match found: {best_name} with confidence {round((1 - stats['min_distance']) * 100, 2)}%")
        
        # Dobavi originalno ime osobe iz mapiranja
        from app.services.text_service import TextService
        original_person = TextService.get_original_text(best_name)
        
        # Ako je pronađeno originalno ime, koristi ga
        if original_person != best_name:
            logger.info(f"Found original name for {best_name}: {original_person}")
            display_name = original_person
        # Ako nije pronađeno originalno ime, a ime sadrži donju crtu, zameni je razmakom
        elif '_' in best_name:
            display_name = best_name.replace('_', ' ')
            logger.info(f"No mapping found, using formatted name: {display_name}")
        else:
            display_name = best_name

        # Sortiraj sve osobe po weighted_score i uzmi top 10
        # Sort by weighted score (lower is better)
        sorted_matches = sorted(name_statistics.items(), key=lambda x: x[1]['weighted_score'])
        top_matches = sorted_matches[:10]  # Limit to top 10 matches

        # Kreiraj niz svih prepoznatih osoba koje su prošle threshold (top 10)
        recognized_persons = []
        for person_name, person_stats in top_matches:
            original_person = TextService.get_original_text(person_name)
            if original_person != person_name:
                formatted_display_name = original_person
            elif '_' in person_name:
                formatted_display_name = person_name.replace('_', ' ')
            else:
                formatted_display_name = person_name
            
            # Uzmi samo prve koordinate za tu osobu
            coords_list = face_coordinates_map.get(person_name, [])
            face_coordinates = coords_list[0] if coords_list else None
            
            # Dodajemo objekat sa imenom i jednim setom koordinata
            person_obj = {
                "name": formatted_display_name,
                "face_coordinates": face_coordinates
            }
            recognized_persons.append(person_obj)
        
        logger.info(f"All recognized batched persons: {[p['name'] for p in recognized_persons]}")
        
        # Logiraj originalne DeepFace rezultate za finalne prepoznate osobe
        logger.info("\n" + "="*80)
        logger.info("ORIGINAL DEEPFACE BATCHED RESULTS FOR FINAL RECOGNIZED PERSONS:")
        logger.info("="*80)
        for person_name in name_scores.keys():
            if person_name in original_deepface_results:
                logger.info(f"\nPerson: {person_name}")
                logger.info("-" * 50)
                for i, result in enumerate(original_deepface_results[person_name]):
                    logger.info(f"DeepFace Batched Result #{i+1}:")
                    for key, value in result.items():
                        logger.info(f"  {key}: {value}")
                    logger.info("-" * 30)
        logger.info("="*80 + "\n")
        
        return {
            "status": "success",
            "message": f"Face recognized as: {display_name} (batched)",
            "person": display_name,
            "recognized_persons": recognized_persons,
            "best_match": {
                "person_name": best_name,
                "display_name": display_name,
                "confidence_metrics": {
                    "occurrences": stats['occurrences'],
                    "average_distance": round(stats['avg_distance'], 4),
                    "min_distance": round(stats['min_distance'], 4),
                    "weighted_score": round(stats['weighted_score'], 4),
                    "confidence_percentage": round((1 - stats['min_distance']) * 100, 2),
                    "distances": stats['distances']
                }
            },
            "all_detected_matches": [
                {
                    "person_name": name,
                    "metrics": {
                        "occurrences": name_statistics[name]['occurrences'],
                        "average_distance": round(name_statistics[name]['avg_distance'], 4),
                        "min_distance": round(name_statistics[name]['min_distance'], 4),
                        "confidence_percentage": round((1 - name_statistics[name]['min_distance']) * 100, 2),
                        "distances": name_statistics[name]['distances'],
                        "weighted_score": round(name_statistics[name]['weighted_score'], 4)
                    }
                }
                for name, stats in top_matches  # Only include top 10 matches
            ],
            "processing_mode": "batched"
        }

    @staticmethod
    def validate_face_quality(cropped_face, index, source_type="image"):
        """
        Dodatne validacije kvaliteta lica za eliminisanje lažno pozitivnih (lica u pozadini)

        Args:
            cropped_face (np.array): Array slike lica
            index (int): Indeks lica
            source_type (str): "image" for photos (strict) or "video" for video frames (lenient)

        Returns:
            bool: True ako je lice validnog kvaliteta
        """
        try:
            # Convert to uint8 if needed
            if cropped_face.dtype == 'float64' or cropped_face.dtype == 'float32':
                face_uint8 = (cropped_face * 255).astype(np.uint8)
            else:
                face_uint8 = cropped_face.astype(np.uint8)

            # Convert to grayscale
            gray = cv2.cvtColor(face_uint8, cv2.COLOR_BGR2GRAY)

            # 1. BLUR DETECTION - different thresholds for image vs video
            # Video frames have motion blur, compression artifacts, so use lower threshold
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = laplacian.var()

            # Use 75 for video (more lenient due to motion blur), 100 for images (strict)
            blur_threshold = 75 if source_type == "video" else 100
            if laplacian_var < blur_threshold:
                print(f"⚠️ Lice {index} je zamagljeno (Laplacian var: {laplacian_var:.2f} < {blur_threshold}) - odbacujem.")
                logger.info(f"Face {index} rejected - too blurry (Laplacian variance: {laplacian_var:.2f} < {blur_threshold}, source_type={source_type})")
                return False

            # 2. KONTRAST VALIDACIJA
            # Računam standardnu devijaciju piksela kao meru kontrasta
            contrast = gray.std()
            min_contrast = 25.0  # Minimalni kontrast

            if contrast < min_contrast:
                print(f"⚠️ Lice {index} ima slab kontrast ({contrast:.2f}) - odbacujem.")
                logger.info(f"Face {index} rejected - low contrast ({contrast:.2f})")
                return False

            # 3. JASNOĆA/OSVETLJENJE VALIDACIJA
            mean_brightness = gray.mean()

            # Odbaci lica koja su previše tamna ili previše svetla
            if mean_brightness < 30 or mean_brightness > 220:
                print(f"⚠️ Lice {index} ima lošu osvetljenost ({mean_brightness:.2f}) - odbacujem.")
                logger.info(f"Face {index} rejected - poor lighting (brightness: {mean_brightness:.2f})")
                return False

            # 4. DODATNA DETEKCIJA OŠTRINE PREKO GRADIJENTA
            # Sobel operator za detekciju ivica
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
            edge_density = np.mean(sobel_magnitude)

            min_edge_density = 15.0  # Minimalna gustina ivica
            if edge_density < min_edge_density:
                print(f"⚠️ Lice {index} ima malu gustinu ivica ({edge_density:.2f}) - odbacujem.")
                logger.info(f"Face {index} rejected - low edge density ({edge_density:.2f})")
                return False

            print(f"✅ Lice {index} prošlo sve kvalitativne provere ({source_type} mode):")
            print(f"   📏 Laplacian var: {laplacian_var:.2f} (>{blur_threshold})")
            print(f"   🎨 Kontrast: {contrast:.2f} (>25)")
            print(f"   💡 Osvetljenost: {mean_brightness:.2f} (30-220)")
            print(f"   🔍 Gustina ivica: {edge_density:.2f} (>15)")

            logger.info(f"Face {index} passed quality checks ({source_type}) - Laplacian: {laplacian_var:.2f}, Contrast: {contrast:.2f}, Brightness: {mean_brightness:.2f}, Edge density: {edge_density:.2f}")
            return True

        except Exception as e:
            logger.error(f"Error in face quality validation for face {index}: {str(e)}")
            print(f"❌ Greška u validaciji kvaliteta lica {index}: {str(e)}")
            return False

    @staticmethod
    def recognize_face_with_config(image_bytes, domain: str, config: dict):
        """
        Run face recognition with custom configuration for A/B testing

        Args:
            image_bytes: Image data (bytes or BytesIO)
            domain: Domain for database lookup
            config: Configuration dictionary with parameters:
                - model_name: Model to use (e.g., "VGG-Face", "Facenet512")
                - detector_backend: Detector (e.g., "retinaface")
                - distance_metric: Metric (e.g., "cosine")
                - recognition_threshold: Threshold for recognition
                - detection_confidence_threshold: Threshold for face detection
                - blur_threshold, contrast_threshold, etc.

        Returns:
            Recognition result dictionary
        """
        start_time = time.time()

        try:
            # Extract config parameters
            model_name = config.get("model_name", "VGG-Face")
            detector_backend = config.get("detector_backend", "retinaface")
            distance_metric = config.get("distance_metric", "cosine")
            recognition_threshold = config.get("recognition_threshold", 0.35)
            detection_confidence_threshold = config.get("detection_confidence_threshold", 0.995)

            logger.info(f"Running recognition with config: model={model_name}, threshold={recognition_threshold}, confidence={detection_confidence_threshold}")

            # Clean domain for path
            clean_domain = RecognitionService.clean_domain_for_path(domain)

            # Handle image bytes (same pattern as standard recognize_face method)
            if hasattr(image_bytes, 'getvalue'):
                # If BytesIO object, extract actual bytes
                actual_bytes = image_bytes.getvalue()
                image_bytes.seek(0)  # Reset pointer in case it's used again
            else:
                # If already bytes
                actual_bytes = image_bytes

            # Open image with fresh BytesIO
            image = Image.open(BytesIO(actual_bytes))
            original_width, original_height = image.size

            # Resize image - pass actual bytes
            resized_image = ImageService.resize_image(actual_bytes)
            resized_width, resized_height = Image.open(resized_image).size

            logger.info(f"Image resized from {original_width}x{original_height} to {resized_width}x{resized_height}")

            # Save temporary image
            temp_filename = f"temp_{int(time.time() * 1000)}.jpg"
            image_path = os.path.join("storage/temp", temp_filename)
            os.makedirs("storage/temp", exist_ok=True)

            with open(image_path, "wb") as f:
                f.write(resized_image.getvalue())
            logger.info(f"Resized image saved temporarily at: {image_path}")

            # Database path - use production DB for A/B testing (same as standard endpoint)
            db_path = os.path.join('storage/recognized_faces_prod', clean_domain)

            # Extract faces with configurable confidence threshold
            logger.info(f"[A/B TEST] Calling DeepFace.extract_faces with detector={detector_backend}")
            logger.info(f"[A/B TEST] Image path: {image_path}, exists: {os.path.exists(image_path)}")
            logger.info(f"[A/B TEST] Image dimensions: {original_width}x{original_height} -> {resized_width}x{resized_height}")

            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=detector_backend,
                enforce_detection=False,
                normalize_face=True,
                align=True
            )

            logger.info(f"[A/B TEST] DeepFace.extract_faces returned {len(faces)} faces")
            for i, face in enumerate(faces):
                logger.info(f"[A/B TEST] Face {i+1}: confidence={face.get('confidence', 1):.4f}, area={face.get('facial_area')}")

            if len(faces) == 0:
                print("❌ Nema nijednog lica.")
                logger.warning(f"[A/B TEST] No faces detected in image!")
                return {
                    "status": "no_faces",
                    "message": "No faces detected in the image",
                    "recognized_faces": [],
                    "total_faces_detected": 0,
                    "valid_faces_after_filtering": 0
                }
            else:
                print(f"✅ Pronađeno lica: {len(faces)}")
                logger.info(f"[A/B TEST] Processing {len(faces)} detected faces")

                # Process each face with configurable thresholds
                valid_faces = []

                for i, face in enumerate(faces):
                    facial_area = face["facial_area"]
                    confidence = face.get("confidence", 1)

                    print(f"\n➡️ Lice {i+1}: {facial_area}, Confidence={confidence:.3f}")
                    logger.info(f"[A/B TEST] Face {i+1}: confidence={confidence:.4f}, threshold={detection_confidence_threshold}")

                    # Use configurable confidence threshold
                    if confidence >= detection_confidence_threshold:
                        logger.info(f"[A/B TEST] Face {i+1}: PASSED confidence check")
                        # Check if left_eye and right_eye coordinates are identical
                        if FaceValidationService.has_identical_eye_coordinates(facial_area):
                            left_eye = facial_area.get("left_eye")
                            print(f"⚠️ Lice {i+1} ima identične koordinate za levo i desno oko ({left_eye}) - preskačem.")
                            logger.warning(f"[A/B TEST] Face {i+1}: FAILED - identical eye coordinates ({left_eye})")
                            continue

                        logger.info(f"[A/B TEST] Face {i+1}: PASSED eye coordinates check")

                        # Extract and validate face quality (same as standard method)
                        try:
                            img_cv = cv2.imread(image_path)
                            x = facial_area["x"]
                            y = facial_area["y"]
                            w = facial_area["w"]
                            h = facial_area["h"]
                            cropped_face = img_cv[y:y+h, x:x+w]

                            is_quality_valid = RecognitionService.validate_face_quality(cropped_face, i+1)
                            logger.info(f"[A/B TEST] Face {i+1}: quality validation result = {is_quality_valid}")

                            if is_quality_valid:
                                logger.info(f"[A/B TEST] Face {i+1}: PASSED quality check")
                                face_info = FaceValidationService.create_face_info(
                                    facial_area, i+1, original_width, original_height, resized_width, resized_height
                                )
                                if face_info:
                                    valid_faces.append(face_info)
                                    logger.info(f"[A/B TEST] Face {i+1}: added to valid_faces list")
                                else:
                                    logger.warning(f"[A/B TEST] Face {i+1}: FAILED - create_face_info returned None")
                            else:
                                logger.warning(f"[A/B TEST] Face {i+1}: FAILED quality validation")
                        except Exception as quality_error:
                            logger.error(f"[A/B TEST] Face {i+1}: ERROR during quality check: {str(quality_error)}")
                            continue
                    else:
                        print(f"⚠️ Niska sigurnost detekcije ({confidence:.3f} < {detection_confidence_threshold}) - preskačem ovo lice.")
                        logger.warning(f"[A/B TEST] Face {i+1}: FAILED confidence check ({confidence:.3f} < {detection_confidence_threshold})")

                logger.info(f"[A/B TEST] valid_faces count before final filtering: {len(valid_faces)}")
                # Final face filtering - 30% threshold to allow smaller but valid faces
                final_valid_faces = FaceValidationService.process_face_filtering(valid_faces, size_threshold=0.30)
                logger.info(f"[A/B TEST] final_valid_faces count after filtering: {len(final_valid_faces)}")

                # Early exit if no valid faces
                if len(final_valid_faces) == 0:
                    print("🚫 Prekidam face recognition - nema validnih lica za obradu.")
                    logger.info("Stopping face recognition - no valid faces to process after all checks")
                    return {
                        "status": "no_faces",
                        "message": "No valid faces found after validation checks",
                        "recognized_faces": [],
                        "total_faces_detected": len(faces),
                        "valid_faces_after_filtering": 0
                    }

            try:
                # Use batched mode from profile config (VGG-Face: True, ArcFace: False due to pickle issue)
                use_batched = config.get("batched", True)

                logger.info(f"Building {model_name} model...")
                _ = DeepFace.build_model(model_name)
                logger.info("Model built")
                logger.info("DB path: " + db_path)
                logger.info("Image Path: " + image_path)
                logger.info(f"Using batched mode: {use_batched}")

                # Run recognition with configurable parameters
                dfs = DeepFace.find(
                    img_path=image_path,
                    db_path=db_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    distance_metric=distance_metric,
                    enforce_detection=False,
                    threshold=recognition_threshold,
                    silent=False,
                    batched=use_batched,
                    refresh_database=False,  # Don't rebuild pkl on every request
                    align=True,  # Must match pkl settings
                    normalization='base',  # Must match pkl settings
                    expand_percentage=0  # Must match pkl settings
                )

                # Process results based on batched mode
                if use_batched:
                    logger.info("Using BATCHED functions (List[List[Dict]])")
                    RecognitionService.log_deepface_results_batched(dfs)
                    filtered_dfs = RecognitionService.filter_recognition_results_by_valid_faces_batched(
                        dfs, final_valid_faces, resized_width, resized_height
                    )
                    result = RecognitionService.analyze_recognition_results_batched(
                        filtered_dfs,
                        threshold=recognition_threshold,
                        original_width=original_width,
                        original_height=original_height,
                        resized_width=resized_width,
                        resized_height=resized_height
                    )
                else:
                    logger.info("Using STANDARD functions (list of DataFrames)")
                    RecognitionService.log_deepface_results(dfs)
                    filtered_dfs = RecognitionService.filter_recognition_results_by_valid_faces(
                        dfs, final_valid_faces, resized_width, resized_height
                    )
                    result = RecognitionService.analyze_recognition_results(
                        filtered_dfs,
                        threshold=recognition_threshold,
                        original_width=original_width,
                        original_height=original_height,
                        resized_width=resized_width,
                        resized_height=resized_height
                    )

                logger.info(f"Recognition completed in {time.time() - start_time:.2f}s")
                return result

            except Exception as e:
                error_msg = f"Error during face recognition: {str(e)}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "message": error_msg,
                    "recognized_faces": []
                }

        except Exception as e:
            logger.error(f"Error in recognize_face_with_config: {str(e)}")
            logger.exception("Full traceback:")
            return {
                "status": "error",
                "message": str(e),
                "recognized_faces": []
            }
        finally:
            # Cleanup
            try:
                if 'image_path' in locals() and os.path.exists(image_path):
                    os.remove(image_path)
                    logger.info(f"Cleaned up temporary file: {image_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")