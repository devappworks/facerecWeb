import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'vas-tajni-kljuc'
    UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max-size 
    
    # Excel processing configuration
    EXCEL_FILE_PATH = os.getenv('EXCEL_FILE_PATH', 'storage/excel/data.xlsx')
    IMAGE_STORAGE_PATH = os.getenv('IMAGE_STORAGE_PATH', 'storage/training/media24')
    
    # Google Custom Search API configuration
    SERPAPI_SEARCH_API_KEY = os.getenv('SERPAPI_SEARCH_API_KEY', 'af309518c81f312d3abcffb4fc2165e6ae6bd320b0d816911d0d1153ccea88c8')
    GOOGLE_SEARCH_CX = os.getenv('GOOGLE_SEARCH_CX', '444622b2b520b4d97')

    # Wikimedia image download configuration (Waterfall approach)
    TARGET_IMAGES_PER_PERSON = int(os.getenv('TARGET_IMAGES_PER_PERSON', '40'))
    WIKIMEDIA_MINIMUM_THRESHOLD = int(os.getenv('WIKIMEDIA_MINIMUM_THRESHOLD', '20'))
    # If Wikimedia provides >= TARGET_IMAGES_PER_PERSON: Skip SERP entirely (100% free!)
    # If Wikimedia provides >= WIKIMEDIA_MINIMUM_THRESHOLD: Top up with SERP to reach target
    # If Wikimedia provides < WIKIMEDIA_MINIMUM_THRESHOLD: Use SERP as primary source 