import os
import pandas as pd
from flask import current_app
import json
import datetime

class ExcelService:
    def __init__(self, excel_path=None, excel_path_occupation=None):
        self.excel_path = excel_path or os.getenv('EXCEL_FILE_PATH', 'storage/excel/data.xlsx')
        self.excel_path_occupation = excel_path_occupation or os.getenv('EXCEL_FILE_PATH_OCCUPATION', 'storage/excel/occupation.xlsx')

    def process_excel_file(self):
        """
        Process the Excel file:
        1. Read the first row
        2. Remove the first row
        3. Save the updated Excel file
        4. Return the data from the first row
        """
        try:
            # Check if file exists
            if not os.path.exists(self.excel_path):
                current_app.logger.error(f"Excel file not found at {self.excel_path}")
                return None
            
            # Read the Excel file
            df = pd.read_excel(self.excel_path)
            
            # Check if file has data
            if df.empty:
                current_app.logger.warning(f"Excel file is empty: {self.excel_path}")
                return None
            
            # Get the first row
            first_row = df.iloc[0]
            
            # Remove the first row
            df = df.iloc[1:].reset_index(drop=True)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.excel_path), exist_ok=True)
            
            # Save the updated Excel file
            df.to_excel(self.excel_path, index=False)
            
            # Get original values
            original_name = first_row['name'] if 'name' in first_row else ""
            original_last_name = first_row['last_name'] if 'last_name' in first_row else ""
            original_occupation = first_row['occupation'] if 'occupation' in first_row else ""
            
            # Normalize values for search
            normalized_name = self._normalize_text(original_name)
            normalized_last_name = self._normalize_text(original_last_name)
            normalized_occupation = self._normalize_text(original_occupation)
            
            # Create a key for the name mapping
            normalized_key = f"{normalized_name}_{normalized_last_name}"
            original_value = f"{original_name} {original_last_name}"
            
            # Save the original name to the mapping file
            self._save_name_mapping(normalized_key, original_value)
            
            # Return the extracted data
            return {
                'name': normalized_name,
                'last_name': normalized_last_name,
                'occupation': normalized_occupation,
                'original_name': original_name,
                'original_last_name': original_last_name,
                'original_occupation': original_occupation
            }
            
        except Exception as e:
            current_app.logger.error(f"Error processing Excel file: {str(e)}")
            raise Exception(f"Failed to process Excel file: {str(e)}")
    
    def _normalize_text(self, text):
        """
        Normalize text by removing special characters and converting to ASCII
        """
        if pd.isna(text) or text is None:
            return ""
        
        import unicodedata
        import re
        
        # Convert to string if not already
        text = str(text)
        
        # Normalize unicode characters
        normalized = unicodedata.normalize('NFKD', text)
        
        # Remove accents
        ascii_text = ''.join([c for c in normalized if not unicodedata.combining(c)])
        
        # Replace any remaining non-ASCII characters with closest ASCII equivalent
        ascii_text = unicodedata.normalize('NFKC', ascii_text.encode('ascii', 'ignore').decode('ascii'))
        
        # Remove any characters that aren't alphanumeric or spaces
        ascii_text = re.sub(r'[^\w\s]', '', ascii_text)
        
        return ascii_text.strip()

    def _save_name_mapping(self, normalized_key, original_value):
        """
        Save the mapping between normalized name and original name to a JSON file
        """
        mapping_file = os.path.join('storage', 'name_mapping.json')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
        
        # Load existing mapping if it exists
        mapping = {}
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
            except Exception as e:
                current_app.logger.error(f"Error loading name mapping file: {str(e)}")
        
        # Add new mapping
        mapping[normalized_key] = original_value
        
        # Save updated mapping
        try:
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            current_app.logger.info(f"Saved name mapping: {normalized_key} -> {original_value}")
        except Exception as e:
            current_app.logger.error(f"Error saving name mapping file: {str(e)}")

    def check_excel_file(self):
        """
        Proverava da li Excel fajl postoji i da li sadrži podatke
        
        Returns:
            dict: Rezultat provere Excel fajla
        """
        try:
            # Provera da li fajl postoji
            if not os.path.exists(self.excel_path_occupation):
                current_app.logger.error(f"Excel fajl nije pronađen na putanji: {self.excel_path_occupation}")
                return {
                    "success": False, 
                    "message": f"Excel fajl nije pronađen na putanji: {self.excel_path_occupation}"
                }
            
            # Čitanje Excel fajla
            try:
                df = pd.read_excel(self.excel_path_occupation)
            except Exception as e:
                current_app.logger.error(f"Greška prilikom čitanja Excel fajla: {str(e)}")
                return {
                    "success": False, 
                    "message": f"Greška prilikom čitanja Excel fajla: {str(e)}"
                }
            
            # Provera da li fajl ima podatke
            if df.empty:
                current_app.logger.warning(f"Excel fajl je prazan: {self.excel_path_occupation}")
                return {
                    "success": False, 
                    "message": "Excel fajl je prazan"
                }
            
            # Ako je sve u redu, vrati uspešan odgovor
            row_count = len(df)
            return {
                "success": True, 
                "message": f"Excel fajl postoji i sadrži {row_count} redova",
                "row_count": row_count,
                "file_path": self.excel_path_occupation
            }
            
        except Exception as e:
            current_app.logger.error(f"Greška prilikom provere Excel fajla: {str(e)}")
            return {
                "success": False, 
                "message": f"Greška: {str(e)}"
            }

    def start_processing_thread(self, check_result, country, occupation_filter=None):
        """
        Pokreće thread za obradu Excel fajla

        Args:
            check_result (dict): Rezultat provere Excel fajla
            country (str): Zemlja za koju se traže poznate ličnosti
            occupation_filter (str, optional): Comma-separated list of occupations to process

        Returns:
            dict: Status pokretanja thread-a
        """
        try:
            # Pokrenimo thread za obradu
            import threading
            from flask import current_app

            # Dobijamo app_context za korišćenje u thread-u
            app_context = current_app.app_context()

            thread = threading.Thread(
                target=self._process_excel_thread,
                args=(app_context, country, occupation_filter)
            )
            thread.daemon = True
            thread.start()

            # Dodajemo informaciju da je thread pokrenut
            result = check_result.copy()
            result["thread_started"] = True
            result["country"] = country
            result["occupation_filter"] = occupation_filter

            if occupation_filter:
                result["message"] = f"{result['message']}. Obrada je započeta u pozadini za zemlju: {country}, zanimanja: {occupation_filter}."
            else:
                result["message"] = f"{result['message']}. Obrada je započeta u pozadini za zemlju: {country}."

            return result

        except Exception as e:
            current_app.logger.error(f"Greška prilikom pokretanja thread-a: {str(e)}")
            return {
                "success": False,
                "message": f"Greška prilikom pokretanja thread-a: {str(e)}"
            }

    def _process_excel_thread(self, app_context, country, occupation_filter=None):
        """
        Metoda koja se izvršava u pozadinskom thread-u

        Args:
            app_context: Flask aplikacioni kontekst
            country (str): Zemlja za koju se traže poznate ličnosti
            occupation_filter (str, optional): Comma-separated list of occupations to process
        """
        with app_context:
            try:
                current_app.logger.info(f"Započeta obrada Excel fajla u pozadini: {self.excel_path_occupation} za zemlju: {country}")

                # Parse occupation filter
                allowed_occupations = None
                if occupation_filter:
                    allowed_occupations = [occ.strip() for occ in occupation_filter.split(',')]
                    current_app.logger.info(f"Filtering occupations: {allowed_occupations}")

                # Čitanje Excel fajla
                df = pd.read_excel(self.excel_path_occupation)

                # Obrada svakog reda
                for index, row in df.iterrows():
                    try:
                        occupation = row['Occupation']

                        # Skip if occupation filter is set and this occupation is not in the list
                        if allowed_occupations and occupation not in allowed_occupations:
                            current_app.logger.info(f"Skipping occupation {occupation} (not in filter)")
                            continue

                        current_app.logger.info(f"Obrada reda {index+1}/{len(df)}/{occupation}")
                        
                        from app.services.openai_service import OpenAIService
                        
                        schema = OpenAIService().get_celebrity_schema()
                        messages = [
                            {
                                "role": "system", 
                                "content": f"""
                                    You are a precise data assistant whose job is to generate a list of famous real people.

                                    TASK: Generate more then 20 most famous individuals for a given occupation and country.

                                    CONSTRAINTS:
                                    - Only real, widely recognized individuals.
                                    - No fictional, obscure, or duplicate entries.
                                    - Return more then 20 most famous individuals for a given occupation and country.
                                    - Your output will be passed directly into a structured function call.
                                """
                            },
                            {
                                "role": "user", 
                                "content": [
                                    { "type": "text", "text": f"Generate a list of famous {row['Occupation']} from {country}." }
                                ]
                            }
                        ]
                        response = OpenAIService().safe_openai_request(
                            model="gpt-4.1",
                            messages=messages,
                            temperature=0.2,
                            max_tokens=8000,
                            functions=[schema],
                            function_call={"name": "get_celebrity"}
                        )
                        
                        if response.choices and response.choices[0].message.function_call:
                            function_call = response.choices[0].message.function_call
                            arguments = json.loads(function_call.arguments)
                            current_app.logger.info(f"Response: {json.dumps(arguments, ensure_ascii=False, indent=4)}")
                            list_of_names = arguments.get('objects', [])

                        if hasattr(response, "usage"):
                            total_tokens = response.usage.total_tokens
                            prompt_tokens = response.usage.prompt_tokens
                            completion_tokens = response.usage.completion_tokens
                            current_app.logger.info(f"Token usage - total: {total_tokens}, prompt: {prompt_tokens}, completion: {completion_tokens}")
                        else:
                            current_app.logger.warning("Token usage data not found in response.")  


                        schema = OpenAIService().get_celebrity_schema()
                        messages = [
                            {
                                "role": "system", 
                                "content": f"""
                                    You are an expert validation assistant.

                                    Your job is to verify whether each person in a provided list is a real, widely recognized individual from a specific country and occupation.

                                    TASK:
                                    - For each person in the list, determine if they are verifiably known for the specified occupation and are originally from the specified country.

                                    REQUIREMENTS:
                                    1. Only mark individuals as valid if they are truly notable in that occupation (e.g., actor, politician, athlete) and from the specified country.
                                    2. If a person is not known for the given occupation or is not from the country, mark them as "invalid".
                                    3. Do not include fictional or obscure individuals.
                                    4. Return a structured list of names with a `valid` flag (true or false), and optionally a short reason if invalid.
                                """
                            },
                            {
                                "role": "user", 
                                "content": [
                                    { "type": "text", "text": f"""
                                        Validate the following list of people and check if each person is a famous {row['Occupation']} from {country}.

                                        List:
                                        {list_of_names}
                                        """ 
                                    }
                                ]
                            }
                        ]
                        response_for_validation = OpenAIService().safe_openai_request(
                            model="gpt-4.1",
                            messages=messages,
                            temperature=0.2,
                            max_tokens=8000,
                            functions=[schema],
                            function_call={"name": "get_celebrity"}
                        )

                        if response_for_validation.choices and response_for_validation.choices[0].message.function_call:
                            function_call = response_for_validation.choices[0].message.function_call
                            arguments = json.loads(function_call.arguments)
                            names_to_save = arguments['objects']
                            current_app.logger.info(f"response_for_validation: {json.dumps(names_to_save, ensure_ascii=False, indent=4)}")
                            
                            # Čuvanje imena u Excel
                            save_result = self.save_names_to_excel(names_to_save)
                            current_app.logger.info(f"Rezultat čuvanja imena: {save_result['message']}")
            
                        
                    except Exception as row_error:
                        current_app.logger.error(f"Greška prilikom obrade reda {index+1}: {str(row_error)}")
                
                current_app.logger.info(f"Završena obrada Excel fajla u pozadini: {self.excel_path_occupation} za zemlju: {country}")
                
            except Exception as e:
                current_app.logger.error(f"Greška u pozadinskoj obradi Excel fajla: {str(e)}") 

    def save_names_to_excel(self, names_to_save, file_path=None):
        """
        Jednostavna funkcija koja čuva imena u Excel fajl.
        
        Args:
            names_to_save (str ili list): JSON string ili lista sa imenima
            file_path (str, optional): Putanja do Excel fajla
        
        Returns:
            dict: Rezultat operacije
        """
        try:
            # Koristi podrazumevanu putanju ako nije navedena
            if file_path is None:
                file_path = os.path.join('storage', 'excel', 'data.xlsx')
            
            current_app.logger.info(f"Čuvanje imena u Excel: {file_path}")
            
            # Pripremi listu imena
            names_list = []
            
            # Ako je names_to_save JSON string, konvertuj ga u listu
            if isinstance(names_to_save, str):
                try:
                    names_list = json.loads(names_to_save)
                except:
                    # Ako nije validan JSON, tretiraj kao običan string
                    names_list = [names_to_save]
            elif isinstance(names_to_save, list):
                names_list = names_to_save
            
            # Kreiraj direktorijum ako ne postoji
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Učitaj postojeći Excel ili kreiraj novi DataFrame
            try:
                df_existing = pd.read_excel(file_path)
                current_app.logger.info(f"Učitan postojeći Excel fajl sa {len(df_existing)} redova")
            except:
                df_existing = pd.DataFrame(columns=["name", "last_name"])
                current_app.logger.info("Kreiran novi DataFrame")
            
            # Pripremi nove redove
            new_rows = []
            for full_name in names_list:
                if isinstance(full_name, str):
                    # Razdvoji ime i prezime
                    parts = full_name.strip().split(' ', 1)
                    if len(parts) >= 2:
                        ime, prezime = parts[0], parts[1]
                        new_rows.append({"name": ime, "last_name": prezime})
            
            # Dodaj nove redove u DataFrame
            if new_rows:
                df_new = pd.DataFrame(new_rows)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                
                # Sačuvaj Excel fajl
                df_combined.to_excel(file_path, index=False)
                current_app.logger.info(f"Sačuvan Excel fajl sa {len(df_combined)} redova")
                
                return {
                    "status": "success",
                    "message": f"Uspešno dodato {len(new_rows)} novih imena",
                    "added_count": len(new_rows)
                }
            else:
                current_app.logger.info("Nema novih imena za dodavanje")
                return {
                    "status": "success",
                    "message": "Nema novih imena za dodavanje",
                    "added_count": 0
                }
            
        except Exception as e:
            error_msg = f"Greška: {str(e)}"
            current_app.logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg
            } 