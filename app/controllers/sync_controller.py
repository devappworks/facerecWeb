import os
import shutil
import logging
from app.services.recognition_service import RecognitionService
from app.controllers.recognition_controller import RecognitionController
from app.services.background_service import BackgroundService
from app.services.kylo_service import KyloService
from app.services.text_service import TextService

logger = logging.getLogger(__name__)

class SyncController:
    # Dozvoljene ekstenzije slika
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}
    
    @staticmethod
    def is_image_file(filename):
        """Proverava da li je fajl slika na osnovu ekstenzije"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in SyncController.ALLOWED_EXTENSIONS
    
    @staticmethod
    def sync_faces(source_dir='storage/recognized_faces', target_dir='storage/recognized_faces_prod', test_image_path=None):
        """
        Sinhronizuje slike iz izvornog u ciljni direktorijum
        """
        try:
            logger.info(f"Započinjem sinhronizaciju iz {source_dir} u {target_dir}")
            
            # Osiguraj da ciljni direktorijum postoji
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                logger.info(f"Kreiran glavni ciljni direktorijum: {target_dir}")
            
            # Dobavi listu domen foldera
            if not os.path.exists(source_dir):
                logger.error(f"Izvorni direktorijum ne postoji: {source_dir}")
                return {"error": "Izvorni direktorijum ne postoji"}
                
            domain_folders = [f for f in os.listdir(source_dir) 
                             if os.path.isdir(os.path.join(source_dir, f))]
            logger.info(f"Pronađeni domeni: {domain_folders}")
            
            if not domain_folders:
                logger.warning("Nisu pronađeni domeni za sinhronizaciju!")
                return {"message": "Nisu pronađeni domeni za sinhronizaciju"}
            
            # Sinhronizuj svaki domen folder
            total_copied = 0
            results = []
            
            for domain_folder in domain_folders:
                domain_result = SyncController.sync_domain_folder(
                    domain_folder, 
                    source_dir, 
                    target_dir, 
                    test_image_path
                )
                total_copied += domain_result["copied_count"]
                results.append(domain_result)
            
            return {
                "message": f"Sinhronizacija završena. Ukupno kopirano {total_copied} slika.",
                "total_copied": total_copied,
                "domain_results": results
            }
            
        except Exception as e:
            logger.error(f"Greška pri sinhronizaciji: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def sync_domain_folder(domain_folder, source_dir, target_dir, test_image_path=None):
        """Sinhronizuje slike za jedan domen i briše originale nakon kopiranja"""
        source_domain_path = os.path.join(source_dir, domain_folder)
        target_domain_path = os.path.join(target_dir, domain_folder)
        
        # Konstante za limite slika
        MAX_TOTAL_IMAGES = 40
        MAX_DAILY_IMAGES = 3
        
        logger.info(f"Sinhronizujem domen {domain_folder}")
        logger.info(f"Izvorni folder: {source_domain_path}")
        logger.info(f"Ciljni folder: {target_domain_path}")
        
        # Osiguraj da ciljni folder postoji
        if not os.path.exists(target_domain_path):
            os.makedirs(target_domain_path)
            logger.info(f"Kreiran ciljni folder: {target_domain_path}")
        
        # Dobavi listu slika u izvornom i ciljnom folderu
        try:
            all_source_files = os.listdir(source_domain_path)
            # Filtriraj samo slike
            source_images = {f for f in all_source_files if SyncController.is_image_file(f)}
            logger.info(f"Ukupno fajlova: {len(all_source_files)}, od toga slika: {len(source_images)}")
        except Exception as e:
            logger.error(f"Greška pri čitanju izvornog foldera: {str(e)}")
            return {"domain": domain_folder, "error": str(e), "copied_count": 0}
            
        try:
            all_target_files = os.listdir(target_domain_path) if os.path.exists(target_domain_path) else []
            # Filtriraj samo slike
            target_images = {f for f in all_target_files if SyncController.is_image_file(f)}
            logger.info(f"Broj slika u ciljnom folderu: {len(target_images)}")
        except Exception as e:
            logger.error(f"Greška pri čitanju ciljnog foldera: {str(e)}")
            target_images = set()
        
        # Pronađi nove slike koje treba kopirati
        new_images = source_images - target_images
        logger.info(f"Broj novih slika za kopiranje: {len(new_images)}")
        
        if not new_images:
            logger.info(f"Nema novih slika za domen: {domain_folder}")
            return {"domain": domain_folder, "message": "Nema novih slika", "copied_count": 0}
        
        # Kopiraj nove slike i briši originale
        copied_count = 0
        copied_images = []
        skipped_images = []
        
        # Funkcija za izvlačenje imena osobe i datuma iz naziva fajla
        def extract_person_and_date(filename):
            try:
                # Pretpostavljamo format: ime_osobe_YYYYMMDD.jpg ili ime_osobe_YYYY-MM-DD.jpg
                parts = filename.split('_')
                
                # Tražimo deo koji izgleda kao datum
                date_part = None
                person_parts = []
                
                for part in parts:
                    # Proveri da li deo izgleda kao datum (YYYYMMDD ili YYYY-MM-DD)
                    if (len(part) >= 8 and part[0:4].isdigit()) or '-' in part:
                        date_part = part.split('.')[0]  # Ukloni ekstenziju ako je deo sa datumom
                        break
                    person_parts.append(part)
                
                # Ako nismo našli datum, vraćamo None
                if not date_part:
                    return None, None
                    
                person_name = '_'.join(person_parts)
                return person_name, date_part
            except Exception:
                return None, None
        
        # Funkcija za brojanje slika po osobi
        def count_images_for_person(person_name, target_images):
            count = 0
            for img in target_images:
                img_person, _ = extract_person_and_date(img)
                if img_person == person_name:
                    count += 1
            return count
        
        # Funkcija za brojanje slika po osobi i datumu
        def count_images_for_person_on_date(person_name, date_str, target_images):
            count = 0
            for img in target_images:
                img_person, img_date = extract_person_and_date(img)
                if img_person == person_name and img_date == date_str:
                    count += 1
            return count
        
        # Sortiraj slike po datumu (novije prvo) da bismo zadržali najnovije slike
        sorted_new_images = sorted(new_images, reverse=True)
        
        for image in sorted_new_images:
            source_path = os.path.join(source_domain_path, image)
            target_path = os.path.join(target_domain_path, image)
            
            # Proveri da li je fajl (ne folder) i da li je slika
            if os.path.isfile(source_path) and SyncController.is_image_file(image):
                try:
                    # Izvuci ime osobe i datum iz naziva fajla
                    person_name, date_str = extract_person_and_date(image)
                    
                    if not person_name or not date_str:
                        logger.warning(f"Nije moguće izvući ime osobe ili datum iz naziva fajla: {image}")
                        # Kopiraj sliku bez provere limita
                        shutil.copy2(source_path, target_path)
                        copied_count += 1
                        copied_images.append(image)
                        logger.info(f"Kopirana slika (bez provere limita): {image}")
                        
                        # Obriši original
                        if os.path.exists(target_path) and os.path.getsize(target_path) == os.path.getsize(source_path):
                            os.remove(source_path)
                            logger.info(f"Obrisan original: {image}")
                        continue
                    
                    # Proveri ukupan broj slika za osobu
                    total_images = count_images_for_person(person_name, target_images)
                    logger.info(f"Trenutni ukupan broj slika za {person_name}: {total_images}")
                    
                    if total_images >= MAX_TOTAL_IMAGES:
                        logger.info(f"Preskačem kopiranje: {person_name} već ima {MAX_TOTAL_IMAGES} sačuvanih slika.")
                        # Obriši original bez kopiranja
                        os.remove(source_path)
                        logger.info(f"Obrisan original (limit slika): {image}")
                        skipped_images.append({"image": image, "reason": "Person image limit reached"})
                        continue
                    
                    # Proveri broj slika za osobu na taj datum
                    daily_images = count_images_for_person_on_date(person_name, date_str, target_images)
                    logger.info(f"Trenutni broj dnevnih slika za {person_name} na datum {date_str}: {daily_images}")
                    
                    if daily_images >= MAX_DAILY_IMAGES:
                        logger.info(f"Preskačem kopiranje: Već postoji {MAX_DAILY_IMAGES} slika za {person_name} na datum {date_str}.")
                        # Obriši original bez kopiranja
                        os.remove(source_path)
                        logger.info(f"Obrisan original (dnevni limit): {image}")
                        skipped_images.append({"image": image, "reason": "Daily image limit reached"})
                        continue
                    
                    # Kopiraj sliku ako su svi limiti zadovoljeni
                    shutil.copy2(source_path, target_path)
                    copied_count += 1
                    copied_images.append(image)
                    logger.info(f"Kopirana slika: {image}")
                    
                    # Dodaj sliku u listu ciljnih slika za tačno brojanje
                    target_images.add(image)
                    
                    # Obriši original
                    if os.path.exists(target_path) and os.path.getsize(target_path) == os.path.getsize(source_path):
                        os.remove(source_path)
                        logger.info(f"Obrisan original: {image}")
                    else:
                        logger.warning(f"Kopiranje nije uspelo, original nije obrisan: {image}")
                    
                except Exception as e:
                    logger.error(f"Greška pri kopiranju/brisanju {image}: {str(e)}")
        
        logger.info(f"Kopirano i obrisano {copied_count} slika za domen: {domain_folder}")
        logger.info(f"Preskočeno {len(skipped_images)} slika zbog limita")
        
        # Inicijalizuj prepoznavanje lica nakon kopiranja
        recognition_result = None
        if copied_count > 0 and test_image_path and os.path.exists(test_image_path):
            try:
                # Učitaj test sliku
                with open(test_image_path, 'rb') as f:
                    test_image_bytes = f.read()
                
                # Pozovi prepoznavanje lica
                recognition_result = RecognitionController.recognize_face(test_image_bytes, domain_folder)
                logger.info(f"Inicijalizacija prepoznavanja lica za domen {domain_folder} uspešna")
            except Exception as e:
                logger.error(f"Greška pri inicijalizaciji prepoznavanja lica za domen {domain_folder}: {str(e)}")
                recognition_result = {"error": str(e)}
        
        return {
            "domain": domain_folder,
            "copied_count": copied_count,
            "copied_images": copied_images,
            "skipped_images": skipped_images,
            "recognition_result": recognition_result
        }
    
    @staticmethod
    def sync_faces_background(source_dir='storage/recognized_faces', target_dir='storage/recognized_faces_prod', test_image_path=None):
        """
        Pokreće sinhronizaciju u pozadini
        """
        # Pokreni sinhronizaciju u pozadini
        BackgroundService.run_in_background(
            SyncController.sync_faces,
            source_dir,
            target_dir,
            test_image_path
        )
        
        return {
            "message": "Sinhronizacija pokrenuta u pozadini",
            "source_dir": source_dir,
            "target_dir": target_dir
        }

    @staticmethod
    def sync_images_from_kylo(domain):
        """
        Kontroler za sinhronizaciju slika sa Kylo sistema.
        Koordinira proces preuzimanja, obrade i čuvanja slika.
        """
        try:
            # Preuzimanje slika sa Kylo API-ja
            images_data = KyloService.fetch_images_from_kylo()
            
            if not images_data:
                logger.info("Nema podataka sa Kylo API-ja.")
                return {"status": "success", "message": "Nema novih slika za obradu", "processed": 0}
            
            # Pokretanje asinhrone obrade slika
            result = KyloService.process_images_from_kylo(images_data, domain)
            
            return {
                "status": "success", 
                "message": f"Pokrenuta obrada {len(images_data)} slika", 
                "processed": len(images_data)
            }
            
        except Exception as e:
            logger.error(f"Greška u SyncController.sync_images_from_kylo: {str(e)}")
            return {"status": "error", "message": str(e)}

    @staticmethod
    def transfer_images_background(source_dir='storage/transfer_images', target_domain='media24', batch_size=30):
        """
        Pokreće transfer slika iz source_dir u storage/recognized_faces_prod/target_domain u pozadini
        """
        # Pokreni transfer u pozadini
        BackgroundService.run_in_background(
            SyncController.transfer_images,
            source_dir,
            target_domain,
            batch_size
        )
        
        return {
            "message": "Transfer slika pokrenut u pozadini",
            "source_dir": source_dir,
            "target_domain": target_domain,
            "batch_size": batch_size
        }

    @staticmethod
    def transfer_images(source_dir='storage/transfer_images', target_domain='media24', batch_size=30):
        """
        Prebacuje slike iz source_dir u storage/recognized_faces_prod/target_domain
        """
        try:
            logger.info(f"Započinjem transfer slika iz {source_dir} u storage/recognized_faces_prod/{target_domain}")
            
            # Osiguraj da izvorni direktorijum postoji
            if not os.path.exists(source_dir):
                logger.error(f"Izvorni direktorijum ne postoji: {source_dir}")
                return {"error": "Izvorni direktorijum ne postoji"}
            
            # Osiguraj da ciljni direktorijum postoji
            target_dir = f"storage/recognized_faces_prod/{target_domain}"
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                logger.info(f"Kreiran ciljni direktorijum: {target_dir}")
            
            # Dobavi listu slika u izvornom direktorijumu
            try:
                all_source_files = os.listdir(source_dir)
                # Filtriraj samo slike
                source_images = [f for f in all_source_files if SyncController.is_image_file(f)]
                logger.info(f"Ukupno fajlova: {len(all_source_files)}, od toga slika: {len(source_images)}")
            except Exception as e:
                logger.error(f"Greška pri čitanju izvornog direktorijuma: {str(e)}")
                return {"error": str(e), "transferred_count": 0}
            
            if not source_images:
                logger.info(f"Nema slika za transfer u direktorijumu: {source_dir}")
                return {"message": "Nema slika za transfer", "transferred_count": 0}
            
            # Putanja do test slike za prepoznavanje
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            test_image_path = os.path.join(script_dir, 'scripts', 'test_face.JPG')
            
            # Proveri da li test slika postoji
            if not os.path.exists(test_image_path):
                logger.warning(f"Test slika ne postoji na putanji: {test_image_path}")
                test_image_path = None
            
            # Prebaci slike u batch-evima
            total_transferred = 0
            total_batches = (len(source_images) + batch_size - 1) // batch_size  # Zaokruži na gore
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(source_images))
                batch_images = source_images[start_idx:end_idx]
                
                logger.info(f"Obrađujem batch {batch_num + 1}/{total_batches} ({len(batch_images)} slika)")
                
                # Prebaci slike iz batch-a
                batch_transferred = 0
                for image in batch_images:
                    source_path = os.path.join(source_dir, image)
                    
                    # Proveri da li je fajl (ne folder) i da li je slika
                    if os.path.isfile(source_path) and SyncController.is_image_file(image):
                        try:
                            # Izvuci ime osobe i datum iz naziva fajla
                            # Format: "Zsombor Kálnoki-Kis_2024-08-05_1739272302457.jpg"
                            filename_without_ext, ext = os.path.splitext(image)
                            parts = filename_without_ext.split('_')
                            
                            # Pretpostavljamo da je prvi deo ime osobe, drugi datum, a treći timestamp
                            if len(parts) >= 3:
                                original_person = parts[0]  # Originalno ime osobe
                                date_str = parts[1]  # Datum u formatu YYYY-MM-DD
                                timestamp = parts[2]  # Timestamp
                                
                                # Normalizuj ime osobe
                                normalized_person = TextService.normalize_text(original_person, save_mapping=True)
                                
                                logger.info(f"Izvučeno ime: '{original_person}', normalizovano: '{normalized_person}', datum: {date_str}")
                                
                                # Kreiraj novo ime fajla sa normalizovanim imenom
                                new_filename = f"{normalized_person}_{date_str}_{timestamp}{ext}"
                                target_path = os.path.join(target_dir, new_filename)
                                
                                logger.info(f"Novo ime fajla: {new_filename}")
                            else:
                                # Ako format nije očekivan, koristi originalno ime
                                logger.warning(f"Neočekivan format imena fajla: {image}")
                                target_path = os.path.join(target_dir, image)
                            
                            # Kopiraj sliku
                            shutil.copy2(source_path, target_path)
                            batch_transferred += 1
                            logger.info(f"Kopirana slika: {image} -> {os.path.basename(target_path)}")
                            
                            # Privremeno isključeno brisanje originalnih slika
                            # if os.path.exists(target_path) and os.path.getsize(target_path) == os.path.getsize(source_path):
                            #     os.remove(source_path)
                            #     logger.info(f"Obrisan original: {image}")
                            # else:
                            #     logger.warning(f"Kopiranje nije uspelo, original nije obrisan: {image}")
                            
                        except Exception as e:
                            logger.error(f"Greška pri kopiranju/obradi {image}: {str(e)}")
                
                total_transferred += batch_transferred
                logger.info(f"Prebačeno {batch_transferred} slika u batch-u {batch_num + 1}")
                
                # Inicijalizuj prepoznavanje lica nakon svakog batch-a
                if batch_transferred > 0 and test_image_path:
                    try:
                        # Učitaj test sliku
                        with open(test_image_path, 'rb') as f:
                            test_image_bytes = f.read()
                        
                        # Pozovi prepoznavanje lica
                        recognition_result = RecognitionController.recognize_face(test_image_bytes, target_domain)
                        logger.info(f"Inicijalizacija prepoznavanja lica za batch {batch_num + 1} uspešna")
                        logger.info(f"Rezultat prepoznavanja: {recognition_result}")
                    except Exception as e:
                        logger.error(f"Greška pri inicijalizaciji prepoznavanja lica za batch {batch_num + 1}: {str(e)}")
            
            logger.info(f"Transfer završen. Ukupno prebačeno {total_transferred} slika.")
            
            return {
                "message": f"Transfer završen. Ukupno prebačeno {total_transferred} slika.",
                "transferred_count": total_transferred,
                "total_batches": total_batches
            }
            
        except Exception as e:
            logger.error(f"Greška pri transferu slika: {str(e)}")
            return {"error": str(e)} 