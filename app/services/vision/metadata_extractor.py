"""
Image metadata extractor for EXIF, IPTC, and XMP data.
Extracts useful information from photo files to enhance analysis.
"""

import io
import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts EXIF, IPTC, and other metadata from images."""

    def extract_all(self, image_data: bytes) -> Dict[str, Any]:
        """
        Extract all available metadata from image bytes.

        Returns a dictionary with:
        - gps: GPS coordinates and location info
        - datetime: When the photo was taken
        - camera: Camera make/model/lens info
        - image: Dimensions, orientation, format
        - iptc: Copyright, author, caption, keywords
        """
        result = {
            "gps": None,
            "datetime": None,
            "camera": None,
            "image": None,
            "iptc": None
        }

        try:
            img = Image.open(io.BytesIO(image_data))

            # Basic image info
            result["image"] = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode
            }

            # Extract EXIF data
            exif_data = self._extract_exif(img)
            if exif_data:
                result["gps"] = exif_data.get("gps")
                result["datetime"] = exif_data.get("datetime")
                result["camera"] = exif_data.get("camera")
                if exif_data.get("orientation"):
                    result["image"]["orientation"] = exif_data["orientation"]

            # Extract IPTC data
            iptc_data = self._extract_iptc(img)
            if iptc_data:
                result["iptc"] = iptc_data

        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")

        # Remove None values for cleaner output
        return {k: v for k, v in result.items() if v is not None}

    def _extract_exif(self, img: Image.Image) -> Optional[Dict[str, Any]]:
        """Extract EXIF metadata from PIL Image."""
        try:
            exif = img._getexif()
            if not exif:
                return None

            result = {}

            # Parse all EXIF tags
            exif_dict = {}
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_dict[tag] = value

            # GPS Info
            gps_info = exif_dict.get("GPSInfo")
            if gps_info:
                gps = self._parse_gps(gps_info)
                if gps:
                    result["gps"] = gps

            # DateTime
            datetime_original = exif_dict.get("DateTimeOriginal") or exif_dict.get("DateTime")
            if datetime_original:
                result["datetime"] = {
                    "taken": str(datetime_original),
                    "digitized": str(exif_dict.get("DateTimeDigitized", ""))
                }

            # Camera info
            camera = {}
            if exif_dict.get("Make"):
                camera["make"] = str(exif_dict["Make"]).strip()
            if exif_dict.get("Model"):
                camera["model"] = str(exif_dict["Model"]).strip()
            if exif_dict.get("LensModel"):
                camera["lens"] = str(exif_dict["LensModel"]).strip()
            if exif_dict.get("FocalLength"):
                focal = exif_dict["FocalLength"]
                if hasattr(focal, 'numerator'):
                    camera["focal_length"] = f"{focal.numerator / focal.denominator}mm"
                else:
                    camera["focal_length"] = f"{focal}mm"
            if exif_dict.get("FNumber"):
                fnum = exif_dict["FNumber"]
                if hasattr(fnum, 'numerator'):
                    camera["aperture"] = f"f/{fnum.numerator / fnum.denominator}"
                else:
                    camera["aperture"] = f"f/{fnum}"
            if exif_dict.get("ISOSpeedRatings"):
                camera["iso"] = exif_dict["ISOSpeedRatings"]
            if exif_dict.get("ExposureTime"):
                exp = exif_dict["ExposureTime"]
                if hasattr(exp, 'numerator'):
                    if exp.numerator == 1:
                        camera["shutter_speed"] = f"1/{exp.denominator}s"
                    else:
                        camera["shutter_speed"] = f"{exp.numerator}/{exp.denominator}s"
                else:
                    camera["shutter_speed"] = f"{exp}s"

            if camera:
                result["camera"] = camera

            # Orientation
            if exif_dict.get("Orientation"):
                result["orientation"] = exif_dict["Orientation"]

            return result if result else None

        except Exception as e:
            logger.debug(f"Error extracting EXIF: {e}")
            return None

    def _parse_gps(self, gps_info: dict) -> Optional[Dict[str, Any]]:
        """Parse GPS EXIF data into coordinates."""
        try:
            gps_dict = {}
            for tag_id, value in gps_info.items():
                tag = GPSTAGS.get(tag_id, tag_id)
                gps_dict[tag] = value

            # Extract latitude
            lat = gps_dict.get("GPSLatitude")
            lat_ref = gps_dict.get("GPSLatitudeRef", "N")

            # Extract longitude
            lon = gps_dict.get("GPSLongitude")
            lon_ref = gps_dict.get("GPSLongitudeRef", "E")

            if lat and lon:
                lat_decimal = self._dms_to_decimal(lat, lat_ref)
                lon_decimal = self._dms_to_decimal(lon, lon_ref)

                if lat_decimal is not None and lon_decimal is not None:
                    result = {
                        "latitude": round(lat_decimal, 6),
                        "longitude": round(lon_decimal, 6)
                    }

                    # Altitude if available
                    alt = gps_dict.get("GPSAltitude")
                    if alt:
                        if hasattr(alt, 'numerator'):
                            result["altitude"] = round(alt.numerator / alt.denominator, 1)
                        else:
                            result["altitude"] = round(float(alt), 1)

                    return result

            return None

        except Exception as e:
            logger.debug(f"Error parsing GPS: {e}")
            return None

    def _dms_to_decimal(self, dms: tuple, ref: str) -> Optional[float]:
        """Convert degrees/minutes/seconds to decimal degrees."""
        try:
            def to_float(val):
                if hasattr(val, 'numerator'):
                    return val.numerator / val.denominator
                return float(val)

            degrees = to_float(dms[0])
            minutes = to_float(dms[1])
            seconds = to_float(dms[2])

            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

            if ref in ["S", "W"]:
                decimal = -decimal

            return decimal

        except Exception as e:
            logger.debug(f"Error converting DMS: {e}")
            return None

    def _extract_iptc(self, img: Image.Image) -> Optional[Dict[str, Any]]:
        """Extract IPTC metadata from PIL Image."""
        try:
            # Try to get IPTC from image info
            iptc_info = img.info.get("iptc") or img.info.get("photoshop")

            # Also try APP13 marker for IPTC
            if hasattr(img, 'applist'):
                for app in img.applist:
                    if app[0] == 'APP13':
                        # Contains IPTC data
                        pass

            result = {}

            # Check for XMP data which might contain useful info
            xmp = img.info.get("xmp") or img.info.get("XML:com.adobe.xmp")
            if xmp:
                # XMP is XML, could parse it for additional metadata
                # For now, just note it exists
                pass

            # Check image description from EXIF as fallback
            exif = img._getexif() if hasattr(img, '_getexif') else None
            if exif:
                exif_dict = {TAGS.get(k, k): v for k, v in exif.items()}

                if exif_dict.get("ImageDescription"):
                    result["caption"] = str(exif_dict["ImageDescription"])
                if exif_dict.get("Artist"):
                    result["author"] = str(exif_dict["Artist"])
                if exif_dict.get("Copyright"):
                    result["copyright"] = str(exif_dict["Copyright"])
                if exif_dict.get("UserComment"):
                    comment = exif_dict["UserComment"]
                    if isinstance(comment, bytes):
                        try:
                            # Try to decode UserComment
                            if comment.startswith(b'UNICODE\x00'):
                                comment = comment[8:].decode('utf-16')
                            elif comment.startswith(b'ASCII\x00\x00\x00'):
                                comment = comment[8:].decode('ascii')
                            else:
                                comment = comment.decode('utf-8', errors='ignore')
                        except:
                            comment = str(comment)
                    if comment and comment.strip():
                        result["user_comment"] = str(comment).strip()

            return result if result else None

        except Exception as e:
            logger.debug(f"Error extracting IPTC: {e}")
            return None


# Singleton instance
_extractor = None

def get_metadata_extractor() -> MetadataExtractor:
    """Get or create singleton MetadataExtractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = MetadataExtractor()
    return _extractor
