import openai
import time
import traceback
from dotenv import load_dotenv
import os
from openai import OpenAI

class OpenAIService:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running the application.")

        self.client = OpenAI(api_key=self.openai_api_key)    

    def safe_openai_request(self, *args, **kwargs):
        # Function to make OpenAI requests in a "safe" way, i.e. repeating requests in case of error, with exponential backoff logic
        max_retries = 7
        backoff_factor = 3
        request_timeout = 145
        for attempt in range(max_retries):
            try:
                print(f"[INFO] Sending OpenAI API request: args={args}, kwargs={kwargs}. Attempt {attempt + 1}")
                response = self.client.chat.completions.create(*args, **kwargs)
                return response
            except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    print(f"[ERROR] OpenAI API error encountered: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("[ERROR] OpenAI API error encountered. All retries exhausted.")
                    traceback.print_exc()
                    raise
            except Exception as e:
                wait_time = backoff_factor ** attempt
                print(f"[ERROR] Unexpected error: {str(e)}. Args: {args}, Kwargs: {kwargs}")
                traceback.print_exc()
                if "The server is overloaded or not ready yet" in str(e) and attempt < max_retries - 1:
                    print(f"[ERROR] Server overload. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    if attempt >= max_retries - 1:
                        print("[ERROR] Maximum retries reached. Raising exception.")
                    raise


    def get_moderation_schema(self):
        return {
            "name": "generate_metadata",
            "description": "Generate description, objects and metatags from image",
            "parameters": {
                "type": "object",
                "properties": {
                "description": {
                    "type": "string",
                    "description": "A short textual summary or description of the input image"
                },
                "alt": {
                    "type": "string",
                    "description": "A short, concise description of the image suitable for use as alt of the image"
                },
                "objects": {
                    "type": "array",
                    "items": {
                    "type": "string"
                    },
                    "description": "List of key objects, entities or items mentioned in the image"
                },
                "metatags": {
                    "type": "array",
                    "items": {
                    "type": "string"
                    },
                    "description": "List of relevant metatags for SEO or content categorization of the image"
                }
                },
                "required": ["description", "objects", "metatags"]
            }
        }

    def get_celebrity_schema(self):
        return {
            "name": "get_celebrity",
            "description": "Generate lists of names for giving occupation of celebrity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of celebrity names"
                    }
                },
                "required": ["objects"]
            }
        }

    def get_humanity_check_schema(self):
        return {
            "name": "get_humanity_check",
            "description": "Return true if the person is real human, otherwise return false.",
            "parameters": {
                "type": "object",
                "properties": {
                    "human": {
                        "type": "boolean",
                        "description": "Return true if the person is real human, otherwise return false."
                    }
                },
                "required": ["human"]
            }
        } 
    
    def get_human_names_schema(self):
        return {
            "name": "get_human_names",
            "description": "Return list of human names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of human names"
                    }
                },
                "required": ["names"]
            }
        } 
        