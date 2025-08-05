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

        if confidence >= 0.99:
            # Check if left_eye and right_eye coordinates are identical
            if FaceValidationService.has_identical_eye_coordinates(facial_area):
                left_eye = facial_area.get("left_eye")
                print(f"⚠️ Lice {index} ima identične koordinate za levo i desno oko ({left_eye}) - preskačem.")
                logger.info(f"Face {index} has identical left_eye and right_eye coordinates ({left_eye}) - skipping")
                return False

            print("✅ Validno lice - radim prepoznavanje.")
            return True
        else:
            print("⚠️ Niska sigurnost detekcije - preskačem ovo lice.")
            return False



    @staticmethod
    def check_face_blur_and_create_info(cropped_face, facial_area, index, original_width, original_height, resized_width, resized_height):
        """
        Proverava zamagljenost lica i kreira info objekat ako je lice validno
        
        Args:
            cropped_face (np.array): Array slike
            facial_area (dict): Koordinate lica
            index (int): Indeks lica
            original_width (int): Širina originalne slike
            original_height (int): Visina originalne slike
            resized_width (int): Širina resized slike
            resized_height (int): Visina resized slike
            
        Returns:
            dict or None: Info objekat ako je lice validno, None ako nije
        """
        try:
            # Convert cropped face to format needed for blur detection
            # The is_blurred method expects normalized array (0-1 range)
            cropped_face_normalized = cropped_face.astype(np.float32) / 255.0
            
            # Check if face is blurry
            is_blurry = FaceProcessingService.is_blurred(cropped_face_normalized, 1)
            
            if is_blurry:
                print(f"⚠️ Lice {index} je zamagljeno - odbacujem.")
                logger.info(f"Face {index} is blurry - rejecting")
                return None
            else:
                print(f"✅ Lice {index} je oštro - dodajem u validne.")
                logger.info(f"Face {index} is sharp - adding to valid faces")
                # Kreiraj info objekat sa originalnim koordinatama
                return FaceValidationService.create_face_info(
                    facial_area, index, original_width, original_height, resized_width, resized_height
                )
                
        except Exception as blur_error:
            logger.error(f"Error checking blur for face {index}: {str(blur_error)}")
            print(f"❌ Greška pri proveri zamućenosti lica {index}: {str(blur_error)}")
            return None

    @staticmethod
    def process_single_face(face, index, image_path, original_width, original_height, resized_width, resized_height):
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
        
        # Provera zamagljenosti i kreiranje info objekta
        return RecognitionService.check_face_blur_and_create_info(
            cropped_face, facial_area, index, original_width, original_height, resized_width, resized_height
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
    def recognize_face(image_bytes, domain):
        """
        Prepoznaje lice iz prosleđene slike
        """
        try:
            logger.info("Starting face recognition process")
            start_time = time.time()
            
            # Prvo dobijamo dimenzije originalne slike
            from PIL import Image
            # Proverimo tip i izvučemo bytes ako je potrebno
            if hasattr(image_bytes, 'getvalue'):
                # Ako je BytesIO objekat
                actual_bytes = image_bytes.getvalue()
                image_bytes.seek(0)  # Reset pointer za slučaj da se koristi ponovo
            else:
                # Ako su već bytes
                actual_bytes = image_bytes
            
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
            
            #     # Definišemo parametre
            model_name = "VGG-Face"
            detector_backend = "retinaface"
            distance_metric = "cosine"
            db_path = os.path.join('storage/recognized_faces_prod', clean_domain)

            # Extract faces
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=detector_backend,
                enforce_detection=False,
                normalize_face=True,
                align=True
            )

            if len(faces) == 0:
                print("❌ Nema nijednog lica.")
            else:
                print(f"✅ Pronađeno lica: {len(faces)}")

                # Lista za čuvanje informacija o validnim licima (ne čuvamo fizičke slike)
                valid_faces = []

                # Obradi svako lice kroz sve validacije
                for i, face in enumerate(faces):
                    face_info = RecognitionService.process_single_face(
                        face, i+1, image_path, original_width, original_height, resized_width, resized_height
                    )
                    if face_info:
                        valid_faces.append(face_info)

                # Finalna provera - zadržati samo najveća lica
                final_valid_faces = FaceValidationService.process_face_filtering(valid_faces)
                
            try:
                # Definišemo parametre
                model_name = "VGG-Face"
                detector_backend = "retinaface"
                distance_metric = "cosine"
                db_path = os.path.join('storage/recognized_faces_prod', clean_domain)
                
                # KLJUČNA PROMENLJIVA - promeni ovu na True za batched mode
                use_batched = True
                
                logger.info("Building VGG-Face model...")
                _ = DeepFace.build_model("VGG-Face")
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
                    threshold=0.35,
                    silent=False,
                    batched=use_batched  # KLJUČNI PARAMETAR!
                )
                
                # Pozivamo odgovarajuće funkcije na osnovu use_batched promenljive
                if use_batched:
                    # BATCHED MODE: DeepFace vraća list of dicts
                    logger.info("Using BATCHED functions (list of dicts)")
                    RecognitionService.log_deepface_results_batched(dfs)
                    filtered_dfs = RecognitionService.filter_recognition_results_by_valid_faces_batched(
                        dfs, final_valid_faces, resized_width, resized_height
                    )
                    result = RecognitionService.analyze_recognition_results_batched(
                        filtered_dfs, 
                        threshold=0.35,
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
                        threshold=0.35,
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
        name_statistics = {}
        for name, distances in name_scores.items():
            avg_distance = sum(distances) / len(distances)
            min_distance = min(distances)
            occurrences = len(distances)
            
            weighted_score = (avg_distance * 0.4) + (min_distance * 0.3) - (occurrences * 0.1)
            
            name_statistics[name] = {
                "occurrences": occurrences,
                "avg_distance": avg_distance,
                "min_distance": min_distance,
                "weighted_score": weighted_score,
                "distances": distances
            }
            
            logging.info(f"Threshold-passing matches for {name}:")
            logger.info(f"- Occurrences: {occurrences}")
            logger.info(f"- Average distance: {avg_distance:.4f}")
            logger.info(f"- Min distance: {min_distance:.4f}")
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
        
        # Kreiraj niz svih prepoznatih osoba koje su prošle threshold
        recognized_persons = []
        for person_name in name_scores.keys():
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
                        "occurrences": len(distances),
                        "average_distance": round(sum(distances) / len(distances), 4),
                        "min_distance": round(min(distances), 4),
                        "confidence_percentage": round((1 - min(distances)) * 100, 2),
                        "distances": distances
                    }
                }
                for name, distances in all_matches.items()
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

        # Grupiranje match-ova po koordinatama
        logger.info(f"Total batched matches before grouping: {len(matches_with_coords)}")
        grouped_matches = RecognitionService.group_matches_by_coordinates(matches_with_coords, tolerance=10)
        logger.info(f"Total batched matches after grouping: {len(grouped_matches)}")
        
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
                logger.debug(f"Batched grouped match passed threshold: {name} with distance {distance}")

        # Log summary of all matches found
        logger.info(f"\n{'='*50}")
        logger.info(f"BATCHED RECOGNITION RESULTS:")
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
        name_statistics = {}
        for name, distances in name_scores.items():
            avg_distance = sum(distances) / len(distances)
            min_distance = min(distances)
            occurrences = len(distances)
            
            weighted_score = (avg_distance * 0.4) + (min_distance * 0.3) - (occurrences * 0.1)
            
            name_statistics[name] = {
                "occurrences": occurrences,
                "avg_distance": avg_distance,
                "min_distance": min_distance,
                "weighted_score": weighted_score,
                "distances": distances
            }
            
            logging.info(f"Batched threshold-passing matches for {name}:")
            logger.info(f"- Occurrences: {occurrences}")
            logger.info(f"- Average distance: {avg_distance:.4f}")
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
        
        # Kreiraj niz svih prepoznatih osoba koje su prošle threshold
        recognized_persons = []
        for person_name in name_scores.keys():
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
                        "occurrences": len(distances),
                        "average_distance": round(sum(distances) / len(distances), 4),
                        "min_distance": round(min(distances), 4),
                        "confidence_percentage": round((1 - min(distances)) * 100, 2),
                        "distances": distances
                    }
                }
                for name, distances in all_matches.items()
            ],
            "processing_mode": "batched"
        }