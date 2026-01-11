from flask import current_app
from app.services.excel_service import ExcelService
from app.services.image_service import ImageService

class ExcelController:
    def __init__(self):
        self.excel_service = ExcelService()
        self.image_service = ImageService()
    
    def process_excel_and_fetch_images(self):
        try:
            # Process Excel file and get the first row data
            excel_data = self.excel_service.process_excel_file()
            
            if not excel_data:
                return {"success": False, "message": "No data found in Excel file or file not found"}
            
            # Fetch and save images based on the extracted data
            image_results = self.image_service.fetch_and_save_images(
                excel_data['name'],
                excel_data['last_name'],
                excel_data['occupation'],
                original_name=excel_data.get('original_name', ''),
                original_last_name=excel_data.get('original_last_name', '')
            )
            
            return {
                "success": True,
                "data": excel_data,
                "images": image_results
            }
            
        except Exception as e:
            current_app.logger.error(f"Error processing Excel and fetching images: {str(e)}")
            return {"success": False, "message": str(e)} 