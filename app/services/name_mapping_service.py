import os
import json
import logging
from threading import Lock

logger = logging.getLogger(__name__)

class NameMappingService:
    """
    Servis za mapiranje između originalnih i normalizovanih imena osoba.
    Koristi JSON fajl za čuvanje mapiranja.
    """
    
    # Putanja do JSON fajla sa mapiranjem
    MAPPING_FILE = 'storage/name_mapping.json'
    
    # Lock za thread-safe pristup fajlu
    _file_lock = Lock()
    
    @staticmethod
    def save_name_mapping(original_name, normalized_name):
        """
        Čuva mapiranje između originalnog i normalizovanog imena.
        
        Args:
            original_name (str): Originalno ime osobe
            normalized_name (str): Normalizovano ime osobe
        """
        try:
            with NameMappingService._file_lock:
                # Kreiraj direktorijum ako ne postoji
                os.makedirs(os.path.dirname(NameMappingService.MAPPING_FILE), exist_ok=True)
                
                # Učitaj postojeće mapiranje
                mapping = {}
                if os.path.exists(NameMappingService.MAPPING_FILE):
                    with open(NameMappingService.MAPPING_FILE, 'r', encoding='utf-8') as f:
                        try:
                            mapping = json.load(f)
                        except json.JSONDecodeError:
                            logger.error("Invalid JSON in name mapping file. Creating new mapping.")
                
                # Dodaj novo mapiranje
                if normalized_name not in mapping:
                    mapping[normalized_name] = original_name
                    logger.info(f"Added name mapping: {normalized_name} -> {original_name}")
                
                # Sačuvaj ažurirano mapiranje
                with open(NameMappingService.MAPPING_FILE, 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving name mapping: {str(e)}")
    
    @staticmethod
    def get_original_name(normalized_name):
        """
        Vraća originalno ime na osnovu normalizovanog imena.
        
        Args:
            normalized_name (str): Normalizovano ime osobe
            
        Returns:
            str: Originalno ime osobe ili normalizovano ime ako mapiranje nije pronađeno
        """
        try:
            with NameMappingService._file_lock:
                if os.path.exists(NameMappingService.MAPPING_FILE):
                    with open(NameMappingService.MAPPING_FILE, 'r', encoding='utf-8') as f:
                        try:
                            mapping = json.load(f)
                            return mapping.get(normalized_name, normalized_name)
                        except json.JSONDecodeError:
                            logger.error("Invalid JSON in name mapping file.")
                            return normalized_name
                return normalized_name
        
        except Exception as e:
            logger.error(f"Error getting original name: {str(e)}")
            return normalized_name
    
    @staticmethod
    def get_all_mappings():
        """
        Vraća sva mapiranja između normalizovanih i originalnih imena.
        
        Returns:
            dict: Rečnik sa mapiranjima {normalized_name: original_name}
        """
        try:
            with NameMappingService._file_lock:
                if os.path.exists(NameMappingService.MAPPING_FILE):
                    with open(NameMappingService.MAPPING_FILE, 'r', encoding='utf-8') as f:
                        try:
                            return json.load(f)
                        except json.JSONDecodeError:
                            logger.error("Invalid JSON in name mapping file.")
                            return {}
                return {}
        
        except Exception as e:
            logger.error(f"Error getting all name mappings: {str(e)}")
            return {} 