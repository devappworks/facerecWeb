"""
Wikimedia Commons Image Service
Downloads images from Wikimedia Commons for celebrity training.
Provides FREE, properly-licensed images as primary source.
"""

import os
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime
from PIL import Image as PILImage
from io import BytesIO

logger = logging.getLogger(__name__)


class WikimediaImageService:
    """
    Service for downloading images from Wikimedia Commons.
    Uses Wikidata and Commons API to fetch properly-licensed celebrity photos.
    """

    COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
    WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

    # User agent for API requests (required by Wikimedia)
    USER_AGENT = "FaceRecognitionTrainingApp/1.0 (https://github.com/devappworks/facerecWeb)"

    @classmethod
    def get_images_for_person(cls, wikidata_id: str, primary_image_url: str,
                              person_name: str, storage_path: str,
                              target_count: int = 40) -> Dict:
        """
        Download images from Wikimedia Commons for a person.

        Args:
            wikidata_id: Wikidata entity ID (e.g., "Q5812")
            primary_image_url: Primary image URL from Wikidata P18 property
            person_name: Normalized person name for filenames
            storage_path: Directory to save images
            target_count: Maximum images to download (default 40)

        Returns:
            Dict with download results:
            {
                "success": True,
                "images_downloaded": 25,
                "images": [...],
                "sources": {
                    "primary": 1,
                    "commons_category": 24
                }
            }
        """
        try:
            logger.info(f"[Wikimedia] Fetching images for {person_name} (Wikidata: {wikidata_id})")

            os.makedirs(storage_path, exist_ok=True)

            images_downloaded = []
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 1. Download primary image from Wikidata P18
            primary_count = 0
            if primary_image_url:
                logger.info(f"[Wikimedia] Downloading primary image: {primary_image_url}")
                primary_result = cls._download_image_from_url(
                    primary_image_url,
                    person_name,
                    storage_path,
                    index=1,
                    timestamp=timestamp,
                    source="wikidata_p18"
                )

                if primary_result:
                    images_downloaded.append(primary_result)
                    primary_count = 1
                    logger.info(f"[Wikimedia] ✅ Primary image downloaded")
                else:
                    logger.warning(f"[Wikimedia] ❌ Failed to download primary image")

            # 2. Get Commons category for this person
            commons_category = None
            if wikidata_id:
                commons_category = cls._get_commons_category(wikidata_id)

                if commons_category:
                    logger.info(f"[Wikimedia] Found Commons category: {commons_category}")
                else:
                    logger.info(f"[Wikimedia] No Commons category found for {wikidata_id}")

            # 3. Download additional images from Commons category
            category_count = 0
            if commons_category and len(images_downloaded) < target_count:
                remaining = target_count - len(images_downloaded)
                logger.info(f"[Wikimedia] Fetching up to {remaining} images from category")

                category_images = cls._download_category_images(
                    commons_category,
                    person_name,
                    storage_path,
                    limit=remaining,
                    start_index=len(images_downloaded) + 1,
                    timestamp=timestamp
                )

                images_downloaded.extend(category_images)
                category_count = len(category_images)
                logger.info(f"[Wikimedia] Downloaded {category_count} images from category")

            total_count = len(images_downloaded)
            logger.info(f"[Wikimedia] ✅ Total downloaded: {total_count} images "
                       f"(primary: {primary_count}, category: {category_count})")

            return {
                "success": True,
                "images_downloaded": total_count,
                "images": images_downloaded,
                "sources": {
                    "primary": primary_count,
                    "commons_category": category_count
                },
                "commons_category": commons_category
            }

        except Exception as e:
            logger.error(f"[Wikimedia] Error fetching images for {person_name}: {str(e)}")
            return {
                "success": False,
                "images_downloaded": 0,
                "images": [],
                "error": str(e)
            }

    @classmethod
    def _get_commons_category(cls, wikidata_id: str) -> Optional[str]:
        """
        Get Wikimedia Commons category name from Wikidata entity.
        Uses the P373 property (Commons category).

        Args:
            wikidata_id: Wikidata entity ID (e.g., "Q5812")

        Returns:
            Category name (e.g., "Novak Đoković") or None if not found
        """
        try:
            query = f"""
            SELECT ?commonsCategory WHERE {{
              wd:{wikidata_id} wdt:P373 ?commonsCategory .
            }}
            """

            response = requests.get(
                cls.WIKIDATA_SPARQL_URL,
                params={
                    'query': query,
                    'format': 'json'
                },
                headers={'User-Agent': cls.USER_AGENT},
                timeout=15
            )

            if response.status_code != 200:
                logger.warning(f"[Wikimedia] Wikidata query failed with status {response.status_code}")
                return None

            data = response.json()
            results = data.get('results', {}).get('bindings', [])

            if results and len(results) > 0:
                category = results[0].get('commonsCategory', {}).get('value')
                return category

            return None

        except Exception as e:
            logger.error(f"[Wikimedia] Error getting Commons category for {wikidata_id}: {str(e)}")
            return None

    @classmethod
    def _download_category_images(cls, category_name: str, person_name: str,
                                  storage_path: str, limit: int = 40,
                                  start_index: int = 1, timestamp: str = None) -> List[Dict]:
        """
        Download images from a Wikimedia Commons category.

        Args:
            category_name: Commons category name
            person_name: Person name for filename
            storage_path: Directory to save images
            limit: Maximum images to download
            start_index: Starting index for filenames
            timestamp: Timestamp for filenames

        Returns:
            List of downloaded image info dicts
        """
        try:
            if not timestamp:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Get list of image files in the category
            image_files = cls._get_category_members(category_name, limit=limit)

            if not image_files:
                logger.info(f"[Wikimedia] No images found in category: {category_name}")
                return []

            logger.info(f"[Wikimedia] Found {len(image_files)} images in category, downloading...")

            downloaded_images = []

            for i, file_title in enumerate(image_files):
                if len(downloaded_images) >= limit:
                    break

                # Get direct image URL
                image_url = cls._get_image_url(file_title)

                if not image_url:
                    logger.warning(f"[Wikimedia] Could not get URL for {file_title}")
                    continue

                # Download image
                result = cls._download_image_from_url(
                    image_url,
                    person_name,
                    storage_path,
                    index=start_index + len(downloaded_images),
                    timestamp=timestamp,
                    source="commons_category"
                )

                if result:
                    downloaded_images.append(result)
                    logger.info(f"[Wikimedia] Downloaded {len(downloaded_images)}/{limit}: {file_title}")

            return downloaded_images

        except Exception as e:
            logger.error(f"[Wikimedia] Error downloading category images: {str(e)}")
            return []

    @classmethod
    def _get_category_members(cls, category_name: str, limit: int = 50) -> List[str]:
        """
        Get list of image file titles from a Commons category.

        Args:
            category_name: Category name (without "Category:" prefix)
            limit: Maximum files to retrieve

        Returns:
            List of file titles (e.g., ["File:Example.jpg", ...])
        """
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category_name}',
                'cmtype': 'file',
                'cmlimit': min(limit, 500)  # API max is 500
            }

            response = requests.get(
                cls.COMMONS_API_URL,
                params=params,
                headers={'User-Agent': cls.USER_AGENT},
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"[Wikimedia] Commons API failed with status {response.status_code}")
                return []

            data = response.json()
            members = data.get('query', {}).get('categorymembers', [])

            # Extract file titles
            file_titles = [member['title'] for member in members if 'title' in member]

            return file_titles[:limit]

        except Exception as e:
            logger.error(f"[Wikimedia] Error getting category members: {str(e)}")
            return []

    @classmethod
    def _get_image_url(cls, file_title: str) -> Optional[str]:
        """
        Get direct image URL from file title.

        Args:
            file_title: File title (e.g., "File:Example.jpg")

        Returns:
            Direct image URL or None
        """
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'prop': 'imageinfo',
                'iiprop': 'url',
                'titles': file_title
            }

            response = requests.get(
                cls.COMMONS_API_URL,
                params=params,
                headers={'User-Agent': cls.USER_AGENT},
                timeout=15
            )

            if response.status_code != 200:
                return None

            data = response.json()
            pages = data.get('query', {}).get('pages', {})

            for page_id, page_data in pages.items():
                imageinfo = page_data.get('imageinfo', [])
                if imageinfo and len(imageinfo) > 0:
                    return imageinfo[0].get('url')

            return None

        except Exception as e:
            logger.error(f"[Wikimedia] Error getting image URL for {file_title}: {str(e)}")
            return None

    @classmethod
    def _download_image_from_url(cls, url: str, person_name: str, storage_path: str,
                                 index: int, timestamp: str, source: str = "wikimedia") -> Optional[Dict]:
        """
        Download a single image from URL and save it.

        Args:
            url: Image URL
            person_name: Person name for filename
            storage_path: Directory to save image
            index: Image index for filename
            timestamp: Timestamp for filename
            source: Source identifier

        Returns:
            Dict with image info or None if failed
        """
        try:
            # Download image
            response = requests.get(
                url,
                headers={'User-Agent': cls.USER_AGENT},
                timeout=30,
                stream=True
            )

            if response.status_code != 200:
                logger.warning(f"[Wikimedia] Failed to download {url}: HTTP {response.status_code}")
                return None

            # Determine file extension
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                extension = '.jpg'
            elif 'png' in content_type:
                extension = '.png'
            elif 'webp' in content_type:
                extension = '.webp'
            else:
                # Try to get from URL
                if url.lower().endswith('.png'):
                    extension = '.png'
                elif url.lower().endswith('.webp'):
                    extension = '.webp'
                else:
                    extension = '.jpg'  # Default

            # Create filename
            filename = f"{person_name}_{timestamp}_{index}{extension}"
            file_path = os.path.join(storage_path, filename)

            # Save image
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify it's a valid image
            try:
                with PILImage.open(file_path) as img:
                    width, height = img.size

                    # Basic validation
                    if width < 50 or height < 50:
                        logger.warning(f"[Wikimedia] Image too small: {width}x{height}")
                        os.remove(file_path)
                        return None

                    return {
                        "filename": filename,
                        "path": file_path,
                        "source_url": url,
                        "source": source,
                        "size": os.path.getsize(file_path),
                        "dimensions": f"{width}x{height}"
                    }
            except Exception as img_error:
                logger.warning(f"[Wikimedia] Invalid image file: {str(img_error)}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return None

        except Exception as e:
            logger.error(f"[Wikimedia] Error downloading image from {url}: {str(e)}")
            return None
