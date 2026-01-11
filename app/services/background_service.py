import threading
import logging

logger = logging.getLogger(__name__)

class BackgroundService:
    @staticmethod
    def run_in_background(func, *args, **kwargs):
        """
        Pokreće funkciju u pozadini koristeći threading
        """
        def wrapper():
            try:
                logger.info(f"Pokrenuta pozadinska funkcija: {func.__name__}")
                func(*args, **kwargs)
                logger.info(f"Završena pozadinska funkcija: {func.__name__}")
            except Exception as e:
                logger.error(f"Greška u pozadinskoj funkciji {func.__name__}: {str(e)}")
        
        thread = threading.Thread(target=wrapper)
        thread.daemon = True  # Osigurava da se thread završi kada se glavni program završi
        thread.start()
        
        return True 