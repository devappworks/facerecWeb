import os
import logging
import datetime
import re
import glob
import subprocess
import json

logger = logging.getLogger(__name__)

class ImageManagementService:
    """
    Service for handling image management operations (edit, delete)
    """
    
    def __init__(self):
        self.base_storage_path = 'storage/recognized_faces_prod'
    
    def _find_file_by_pattern(self, filename, domain):
        """
        Find a file by exact match first, then by date pattern if not found
        
        Args:
            filename: The filename to search for
            domain: The domain identifier
            
        Returns:
            str: The actual filename found, or None if not found
        """
        try:
            domain_path = os.path.join(self.base_storage_path, domain)
            full_path = os.path.join(domain_path, filename)
            
            # First, try exact match
            if os.path.exists(full_path):
                return filename
            
            # If exact match not found, extract date pattern from filename
            # Pattern: extract from first date occurrence to end
            # Example: "Ljubica Spurej Jazbinsek_2022-12-16_1741786474299.jpg" -> "2022-12-16_1741786474299.jpg"
            date_pattern = r'(\d{4}-\d{2}-\d{2}_\d+\.\w+)$'
            match = re.search(date_pattern, filename)
            
            if match:
                date_part = match.group(1)
                logger.info(f"Extracted date pattern: {date_part} from filename: {filename}")
                
                # Search for files ending with this pattern
                search_pattern = os.path.join(domain_path, f"*{date_part}")
                matching_files = glob.glob(search_pattern)
                
                if len(matching_files) == 1:
                    # Found exactly one matching file
                    found_filename = os.path.basename(matching_files[0])
                    logger.info(f"Found unique match: {found_filename} for pattern: {date_part}")
                    return found_filename
                elif len(matching_files) > 1:
                    # Multiple matches found
                    filenames = [os.path.basename(f) for f in matching_files]
                    logger.warning(f"Multiple files found for pattern {date_part}: {filenames}")
                    return None
                else:
                    # No matches found
                    logger.warning(f"No files found for pattern: {date_part}")
                    return None
            else:
                logger.warning(f"Could not extract date pattern from filename: {filename}")
                return None
                
        except Exception as e:
            logger.error(f"Error finding file by pattern {filename}: {str(e)}")
            return None
    
    def delete_image(self, filename, domain):
        """
        Delete an image from the recognized_faces_prod directory
        and also remove the corresponding embedding from pgvector database.

        Args:
            filename: The filename of the image to delete
            domain: The domain identifier

        Returns:
            dict: Information about the deleted image
        """
        try:
            # Try to find the actual file
            actual_filename = self._find_file_by_pattern(filename, domain)

            if not actual_filename:
                logger.warning(f"Image not found for deletion: {filename}")
                return {
                    "deleted": False,
                    "filename": filename,
                    "reason": "File not found"
                }

            # Construct the full path to the actual image
            domain_path = os.path.join(self.base_storage_path, domain)
            full_path = os.path.join(domain_path, actual_filename)

            # Delete the embedding from pgvector database first
            embedding_result = self._delete_embedding_from_db(full_path)

            # Delete the file
            os.remove(full_path)
            logger.info(f"Successfully deleted image: {full_path}")

            return {
                "deleted": True,
                "filename": filename,
                "actual_filename": actual_filename,
                "path": full_path,
                "embedding_deleted": embedding_result.get('deleted_count', 0) > 0,
                "embedding_count": embedding_result.get('deleted_count', 0)
            }

        except Exception as e:
            logger.error(f"Error deleting image {filename}: {str(e)}")
            raise

    def _delete_embedding_from_db(self, image_path):
        """
        Delete embedding from pgvector database using subprocess to avoid
        TensorFlow/psycopg2 memory conflicts.

        Args:
            image_path: The path to the image (as stored in face_embeddings table)

        Returns:
            dict: Result of the deletion operation
        """
        try:
            script_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'scripts',
                'pgvector_delete_embedding.py'
            )

            input_data = json.dumps({'image_path': image_path})

            result = subprocess.run(
                ['python3', script_path],
                input=input_data,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
                timeout=30
            )

            if result.returncode == 0:
                output = json.loads(result.stdout)
                if output.get('status') == 'success':
                    logger.info(f"Deleted {output.get('deleted_count', 0)} embedding(s) for {image_path}")
                    return output
                else:
                    logger.warning(f"Failed to delete embedding: {output.get('error')}")
                    return {'deleted_count': 0, 'error': output.get('error')}
            else:
                logger.error(f"Subprocess error: {result.stderr}")
                return {'deleted_count': 0, 'error': result.stderr}

        except Exception as e:
            logger.error(f"Error deleting embedding from database: {str(e)}")
            return {'deleted_count': 0, 'error': str(e)}
    
    def edit_image(self, filename, person, domain):
        """
        Edit an image's person name and update metadata
        
        Args:
            filename: The filename of the image to edit
            person: The new person name for the image
            domain: The domain identifier
            
        Returns:
            dict: Information about the edited image
        """
        try:
            # Try to find the actual file
            actual_filename = self._find_file_by_pattern(filename, domain)
            
            if not actual_filename:
                logger.warning(f"Image not found for editing: {filename}")
                return {
                    "edited": False,
                    "filename": filename,
                    "reason": "File not found",
                    "message": f"Image not found: {filename}"
                }
            
            # Construct the full path to the actual image
            domain_path = os.path.join(self.base_storage_path, domain)
            full_path = os.path.join(domain_path, actual_filename)
            
            # Normalize the person name (remove special characters, preserve capitalization)
            normalized_person = self._normalize_person_name(person)
            
            # Extract parts from the original filename
            # Format: Radmila_Marinkovic_2025-03-25_1744708784728.jpg
            
            # Try to find the date pattern in the filename
            date_pattern = r'_(\d{4}-\d{2}-\d{2}_\d+)'
            match = re.search(date_pattern, actual_filename)
            
            if match:
                # If date pattern found, preserve the date and ID part
                date_id_part = match.group(1)
                file_extension = os.path.splitext(actual_filename)[1].lower()
                new_filename = f"{normalized_person}_{date_id_part}{file_extension}"
            else:
                # Fallback if the filename doesn't match expected pattern
                file_extension = os.path.splitext(actual_filename)[1].lower()
                timestamp = int(datetime.datetime.now().timestamp())
                today = datetime.datetime.now().strftime('%Y-%m-%d')
                new_filename = f"{normalized_person}_{today}_{timestamp}{file_extension}"
            
            # Check if the new filename would be the same as the old one (ignoring case)
            if actual_filename.lower() == new_filename.lower():
                # If they're the same, add a timestamp to make it unique
                timestamp = int(datetime.datetime.now().timestamp())
                name_part, ext_part = os.path.splitext(new_filename)
                new_filename = f"{name_part}_{timestamp}{ext_part}"
            
            new_full_path = os.path.join(domain_path, new_filename)
            
            # Rename the file
            import shutil
            shutil.copy2(full_path, new_full_path)  # Copy with metadata
            os.remove(full_path)  # Remove original
            
            logger.info(f"Successfully edited image: {actual_filename} -> {new_filename} (Person: {person})")
            
            return {
                "edited": True,
                "original_filename": filename,
                "actual_filename": actual_filename,
                "new_filename": new_filename,
                "original_path": full_path,
                "new_path": new_full_path,
                "person": person,
                "normalized_person": normalized_person,
                "message": f"Image successfully edited: {actual_filename} -> {new_filename}"
            }
            
        except Exception as e:
            logger.error(f"Error editing image {filename}: {str(e)}")
            raise

    def _normalize_person_name(self, person_name):
        """
        Normalize a person name by replacing spaces with underscores
        and transliterating non-English characters to their English equivalents
        
        Args:
            person_name: The person name to normalize
            
        Returns:
            str: The normalized person name
        """
        import re
        import unicodedata
        
        # Transliterate non-ASCII characters to their ASCII equivalents
        # For example: 'ć' -> 'c', 'č' -> 'c', 'đ' -> 'd', etc.
        normalized = ''
        for char in person_name:
            if ord(char) < 128:  # ASCII characters
                normalized += char
            else:
                # Special handling for Serbian characters
                if char == 'ć' or char == 'č':
                    normalized += 'c'
                elif char == 'đ':
                    normalized += 'd'
                elif char == 'š':
                    normalized += 's'
                elif char == 'ž':
                    normalized += 'z'
                elif char == 'Ć' or char == 'Č':
                    normalized += 'C'
                elif char == 'Đ':
                    normalized += 'D'
                elif char == 'Š':
                    normalized += 'S'
                elif char == 'Ž':
                    normalized += 'Z'
                else:
                    # For other non-ASCII characters, try to decompose them
                    # For example: 'é' -> 'e'
                    ascii_char = unicodedata.normalize('NFKD', char).encode('ASCII', 'ignore').decode('ASCII')
                    normalized += ascii_char if ascii_char else '_'
        
        # Replace spaces with underscores
        normalized = re.sub(r'\s+', '_', normalized)
        # Replace other problematic characters for filenames
        normalized = re.sub(r'[/\\:*?"<>|]', '_', normalized)
        # Replace multiple underscores with a single one
        normalized = re.sub(r'_+', '_', normalized)
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        return normalized