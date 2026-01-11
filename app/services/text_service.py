import unicodedata
import re
import logging
from app.services.name_mapping_service import NameMappingService

logger = logging.getLogger(__name__)

class TextService:
    @staticmethod
    def normalize_text(text, save_mapping=True):
        """
        Normalizuje tekst uklanjanjem specijalnih karaktera i zamenom razmaka sa donjom crtom.
        Takođe čuva mapiranje između originalnog i normalizovanog teksta.
        
        Args:
            text (str): Tekst za normalizaciju
            save_mapping (bool): Da li da sačuva mapiranje između originalnog i normalizovanog teksta
            
        Returns:
            str: Normalizovani tekst
        """
        if not text:
            return ""
        
        # Sačuvaj originalni tekst
        original_text = text
        
        # Normalizuj unicode karaktere
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        
        # Zameni razmake sa donjom crtom i ukloni specijalne karaktere
        text = re.sub(r'[^\w\s]', '', text)
        text = text.replace(' ', '_')
        
        # Sačuvaj mapiranje ako je potrebno
        if save_mapping and text != original_text:
            logger.info(f"Saving name mapping: {original_text} -> {text}")
            NameMappingService.save_name_mapping(original_text, text)
        
        return text
    
    @staticmethod
    def get_original_text(normalized_text):
        """
        Vraća originalni tekst na osnovu normalizovanog teksta.
        
        Args:
            normalized_text (str): Normalizovani tekst
            
        Returns:
            str: Originalni tekst ili normalizovani tekst ako mapiranje nije pronađeno
        """
        return NameMappingService.get_original_name(normalized_text) 