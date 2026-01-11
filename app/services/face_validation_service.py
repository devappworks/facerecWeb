import os
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class FaceValidationService:
    
    @staticmethod
    def has_identical_eye_coordinates(facial_area):
        """
        Proverava da li su koordinate levog i desnog oka identične
        
        Args:
            facial_area (dict): Facial area sa koordinatama
            
        Returns:
            bool: True ako su koordinate identične
        """
        left_eye = facial_area.get("left_eye")
        right_eye = facial_area.get("right_eye")
        
        if left_eye and right_eye and left_eye == right_eye:
            return True
        return False
    
    @staticmethod
    def convert_coordinates_to_original(facial_area, resized_width, resized_height, original_width, original_height):
        """
        Konvertuje koordinate sa resized slike na originalnu sliku
        
        Args:
            facial_area (dict): Facial area sa koordinatama sa resized slike
            resized_width (int): Širina resized slike
            resized_height (int): Visina resized slike
            original_width (int): Širina originalne slike
            original_height (int): Visina originalne slike
            
        Returns:
            dict: Facial area sa koordinatama konvertovanim na originalnu sliku
        """
        # Računamo scale faktore
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height
        
        # Konvertujemo osnovne koordinate
        original_facial_area = {
            "x": int(facial_area["x"] * scale_x),
            "y": int(facial_area["y"] * scale_y),
            "w": int(facial_area["w"] * scale_x),
            "h": int(facial_area["h"] * scale_y)
        }
        
        # Konvertujemo koordinate očiju, nosa i usta ako postoje
        for feature in ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]:
            if feature in facial_area and facial_area[feature]:
                if isinstance(facial_area[feature], tuple) and len(facial_area[feature]) == 2:
                    original_facial_area[feature] = (
                        int(facial_area[feature][0] * scale_x),
                        int(facial_area[feature][1] * scale_y)
                    )
                else:
                    original_facial_area[feature] = facial_area[feature]
        
        return original_facial_area
    
    @staticmethod
    def create_face_info(facial_area, index, original_width, original_height, resized_width, resized_height):
        """
        Kreira informacije o licu sa koordinatama u originalnoj slici
        
        Args:
            facial_area (dict): Facial area sa koordinatama sa resized slike
            index (int): Indeks lica
            original_width (int): Širina originalne slike
            original_height (int): Visina originalne slike
            resized_width (int): Širina resized slike
            resized_height (int): Visina resized slike
            
        Returns:
            dict: Informacije o licu sa originalnim koordinatama
        """
        # Konvertuj koordinate na originalnu sliku
        original_facial_area = FaceValidationService.convert_coordinates_to_original(
            facial_area, resized_width, resized_height, original_width, original_height
        )
        
        return {
            'index': index,
            'original_coordinates': original_facial_area,
            'resized_coordinates': facial_area,
            'area': original_facial_area["w"] * original_facial_area["h"],
            'width': original_facial_area["w"],
            'height': original_facial_area["h"]
        }
    
    @staticmethod
    def create_image_info(cropped_face_path, facial_area, index):
        """
        Kreira informacije o slici za analizu (deprecated - koristiti create_face_info)
        
        Args:
            cropped_face_path (str): Putanja do slike
            facial_area (dict): Facial area sa dimenzijama
            index (int): Indeks lica
            
        Returns:
            dict: Informacije o slici
        """
        w = facial_area["w"]
        h = facial_area["h"]
        
        return {
            'path': cropped_face_path,
            'width': w,
            'height': h,
            'area': w * h,
            'index': index
        }
    
    @staticmethod
    def analyze_and_filter_by_size(items, size_threshold=0.7, absolute_min_area=30000):
        """
        Analizira lica ili slike i zadržava samo najveće
        
        Args:
            items (list): Lista sa informacijama o licima ili slikama
            size_threshold (float): Prag za zadržavanje (0.0-1.0)
            absolute_min_area (int): Apsolutna minimalna površina za zadržavanje
            
        Returns:
            tuple: (items_to_keep, items_to_delete)
        """
        if len(items) <= 1:
            return items, []
        
        # Proveravamo da li su items lica ili slike na osnovu strukture
        has_path = 'path' in items[0] if items else False
        item_type = "slika" if has_path else "lice"
        item_type_plural = "slika" if has_path else "lica"
        
        print(f"\n🔍 Analiziram {len(items)} {item_type_plural} za zadržavanje najvećih...")
        logger.info(f"Analyzing {len(items)} {item_type_plural} to keep only largest ones")
        
        # Pronađi najveći item po površini (width * height)
        largest_item = max(items, key=lambda item: item['area'])
        largest_area = largest_item['area']
        
        if has_path:
            print(f"📏 Najveća {item_type}: {largest_item['path']} ({largest_item['width']}x{largest_item['height']}, površina: {largest_area})")
            logger.info(f"Largest {item_type}: {largest_item['path']} ({largest_item['width']}x{largest_item['height']}, area: {largest_area})")
        else:
            print(f"📏 Najveće {item_type}: Lice {largest_item['index']} ({largest_item['width']}x{largest_item['height']}, površina: {largest_area})")
            logger.info(f"Largest {item_type}: Face {largest_item['index']} ({largest_item['width']}x{largest_item['height']}, area: {largest_area})")
        
        # Definišemo prag - zadržati items koji su najmanje X% veličine najvećeg
        min_required_area = largest_area * size_threshold
        
        print(f"📊 Kriterijum za zadržavanje:")
        print(f"   📈 Procenat od najvećeg: {size_threshold*100}% (minimalno {min_required_area:.0f} piksela)")

        logger.info(f"Filter criteria: {size_threshold*100}% of largest ({min_required_area:.0f} pixels)")
        
        items_to_keep = []
        items_to_delete = []
        
        for item_info in items:
            area = item_info['area']
            percentage_of_largest = (area / largest_area) * 100
            
            # Kriterijum: lice mora biti najmanje X% od najvećeg lica
            # Ovo osigurava da mala lica u pozadini budu filtirana
            meets_percentage_threshold = area >= min_required_area
            keep_item = meets_percentage_threshold
            
            if keep_item:
                items_to_keep.append(item_info)

                if has_path:
                    print(f"✅ Zadržavam: {item_info['path']} (površina: {area}, {percentage_of_largest:.1f}% od najvećeg)")
                else:
                    print(f"✅ Zadržavam: Lice {item_info['index']} (površina: {area}, {percentage_of_largest:.1f}% od najvećeg)")

                logger.info(f"Keeping item with area {area}: {percentage_of_largest:.1f}% od najvećeg")
            else:
                items_to_delete.append(item_info)
                if has_path:
                    print(f"❌ Brišem: {item_info['path']} (površina: {area}, {percentage_of_largest:.1f}% od najveće, ispod praga)")
                else:
                    print(f"❌ Odbacujem: Lice {item_info['index']} (površina: {area}, {percentage_of_largest:.1f}% od najvećeg, ispod praga)")

                logger.info(f"Rejecting item with area {area}: {percentage_of_largest:.1f}% of largest, below threshold")
        
        return items_to_keep, items_to_delete
    
    @staticmethod
    def delete_images(images_to_delete):
        """
        Briše listu slika
        
        Args:
            images_to_delete (list): Lista slika za brisanje
        """
        for img_info in images_to_delete:
            try:
                if os.path.exists(img_info['path']):
                    os.remove(img_info['path'])
                    logger.info(f"Deleted smaller face image: {img_info['path']}")
            except Exception as delete_error:
                logger.error(f"Error deleting smaller image {img_info['path']}: {str(delete_error)}")
    

    
    @staticmethod
    def process_face_filtering(face_infos, size_threshold=0.7, absolute_min_area=30000):
        """
        Kompletna obrada filtriranja lica po veličini
        
        Args:
            face_infos (list): Lista informacija o licima
            size_threshold (float): Prag za zadržavanje lica (0.0-1.0)
            absolute_min_area (int): Apsolutna minimalna površina za zadržavanje
            
        Returns:
            list: Lista zadržanih lica
        """
        if len(face_infos) > 1:
            faces_to_keep, faces_to_delete = FaceValidationService.analyze_and_filter_by_size(face_infos, size_threshold, absolute_min_area)
            
            print(f"🎯 Finalno zadržano {len(faces_to_keep)} od {len(face_infos)} lica")
            logger.info(f"Final result: kept {len(faces_to_keep)} out of {len(face_infos)} faces")
            
            return faces_to_keep
            
        elif len(face_infos) == 1:
            print(f"✅ Samo jedno lice pronađeno - preskačem analizu veličine: Lice {face_infos[0]['index']}")
            logger.info(f"Only one face found - skipping size analysis: Face {face_infos[0]['index']}")
            return face_infos
        else:
            print("❌ Nema validnih lica nakon svih provera.")
            logger.info("No valid faces after all checks")
            return [] 