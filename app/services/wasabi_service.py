import os
from dotenv import load_dotenv
import json
import boto3
import logging
from botocore.exceptions import NoCredentialsError

# Koristite apsolutnu putanju do .env fajla
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
logger = logging.getLogger(__name__)
logger.info(f"Loading .env from: {dotenv_path}")
load_dotenv(dotenv_path)

# Pojednostavljeni izrazi za logovanje
access_key = os.getenv("S3_ACCESS_KEY_ID")
secret_key = os.getenv("S3_SECRET_ACCESS_KEY")
endpoint = os.getenv("S3_ENDPOINT")
region = os.getenv("S3_DEFAULT_REGION")

logger.info(f"S3_ACCESS_KEY_ID: {'*' * len(access_key) if access_key else 'Not set'}")
logger.info(f"S3_SECRET_ACCESS_KEY: {'*' * len(secret_key) if secret_key else 'Not set'}")
logger.info(f"S3_ENDPOINT: {endpoint or 'Not set'}")
logger.info(f"S3_DEFAULT_REGION: {region or 'Not set'}")

class WasabiService:
    @staticmethod
    def get_s3_client():
        # Hardkodirani kredencijali (samo za testiranje)
        access_key = "8LHO8NVLF7MKVUM85N9U"  # Zamenite sa stvarnim access key-em
        secret_key = "340KXYGepKfPJrnmsQ855yTHFZ5zAMI4zsgwNiHc"  # Zamenite sa stvarnim secret key-em
        endpoint = "https://s3.eu-central-2.wasabisys.com"
        region = "eu-central-2"
        
        logger.info(f"Using hardcoded S3 credentials")
        
        return boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint,
            region_name=region
        )
        
    @staticmethod
    def upload_to_s3(local_path, s3_bucket, s3_key):
        logger.info(f"Uploading {local_path} to S3 bucket: {s3_bucket}, key: {s3_key}")
        s3 = WasabiService.get_s3_client()
        try:
            s3.upload_file(local_path, s3_bucket, s3_key)
            logger.info(f"Successfully uploaded {local_path} to S3 as {s3_key}")
        except FileNotFoundError:
            logger.error(f"File not found: {local_path}")
            raise Exception(f"File not found: {local_path}")
        except NoCredentialsError:
            logger.error("S3 credentials not available.")
            raise Exception("S3 credentials not available.")
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise Exception(f"Error uploading to S3: {str(e)}")