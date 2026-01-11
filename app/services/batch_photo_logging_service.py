"""
Batch Photo Recognition Logging Service

Logs all batch photo recognition results with detailed information including:
- Recognized person with confidence
- Top 3 matches for each photo
- Timestamp, filename, and metadata
- Exportable in CSV and JSON formats
"""

import os
import json
import csv
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class BatchPhotoLoggingService:
    """Service for logging batch photo recognition results"""

    LOG_BASE_PATH = "storage/logs/batch_recognition"

    def __init__(self):
        """Initialize logging service and ensure directories exist"""
        os.makedirs(self.LOG_BASE_PATH, exist_ok=True)

    def log_recognition_result(self,
                               filename: str,
                               domain: str,
                               recognition_result: Dict,
                               batch_id: Optional[str] = None) -> Dict:
        """
        Log a single photo recognition result with top 3 matches.

        Args:
            filename: Original filename of the photo
            domain: Domain used for recognition
            recognition_result: Recognition result from RecognitionService
            batch_id: Optional batch identifier for grouping multiple photos

        Returns:
            Dictionary with log entry details
        """
        try:
            # Generate batch_id if not provided
            if not batch_id:
                batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Extract recognition data
            log_entry = self._create_log_entry(
                filename=filename,
                domain=domain,
                recognition_result=recognition_result,
                batch_id=batch_id
            )

            # Save to JSON log
            self._append_to_json_log(batch_id, log_entry)

            logger.info(f"Logged recognition result for {filename} in batch {batch_id}")

            return {
                "success": True,
                "batch_id": batch_id,
                "log_entry": log_entry
            }

        except Exception as e:
            logger.error(f"Error logging recognition result: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _create_log_entry(self,
                          filename: str,
                          domain: str,
                          recognition_result: Dict,
                          batch_id: str) -> Dict:
        """
        Create a structured log entry from recognition result.

        Extracts top 3 matches with confidence scores.
        """
        timestamp = datetime.now().isoformat()

        # Extract primary result
        recognized = recognition_result.get('recognized', False) or recognition_result.get('success', False)
        primary_person = recognition_result.get('person', 'Unknown')
        primary_confidence = recognition_result.get('confidence')

        # Extract top 3 matches
        top_3_matches = self._extract_top_3_matches(recognition_result)

        # Extract metadata
        face_count = recognition_result.get('face_count', 0)
        processing_time = recognition_result.get('processing_time_ms')

        log_entry = {
            "timestamp": timestamp,
            "batch_id": batch_id,
            "filename": filename,
            "domain": domain,
            "recognized": recognized,
            "primary_result": {
                "person": primary_person,
                "confidence": round(primary_confidence, 2) if primary_confidence else None
            },
            "top_3_matches": top_3_matches,
            "metadata": {
                "face_count": face_count,
                "processing_time_ms": processing_time,
                "has_multiple_faces": face_count > 1 if face_count else False
            }
        }

        return log_entry

    def _extract_top_3_matches(self, recognition_result: Dict) -> List[Dict]:
        """
        Extract top 3 recognition matches from result.

        Returns list of up to 3 matches with person name and confidence.
        """
        top_matches = []

        # Format 1: 'all_detected_matches' from RecognitionService (preferred - has full data)
        if 'all_detected_matches' in recognition_result and recognition_result['all_detected_matches']:
            all_detected = recognition_result['all_detected_matches']

            for idx, match in enumerate(all_detected[:3], start=1):
                person_name = match.get('person_name', '')
                metrics = match.get('metrics', {})

                # Format person name
                if '_' in person_name:
                    display_name = person_name.replace('_', ' ')
                else:
                    display_name = person_name

                confidence = metrics.get('confidence_percentage')
                occurrences = metrics.get('occurrences')
                min_distance = metrics.get('min_distance')
                weighted_score = metrics.get('weighted_score')

                top_matches.append({
                    "rank": idx,
                    "person": display_name,
                    "confidence": round(confidence, 2) if confidence else None,
                    "occurrences": occurrences,
                    "min_distance": round(min_distance, 4) if min_distance else None,
                    "weighted_score": round(weighted_score, 4) if weighted_score else None
                })

            return top_matches

        # Format 2: 'best_match' with 'all_matches' or 'top_matches'
        if 'best_match' in recognition_result:
            best_match = recognition_result['best_match']

            # Add primary match
            if best_match:
                person = best_match.get('display_name') or best_match.get('person') or best_match.get('name')
                confidence = None
                occurrences = None
                min_distance = None
                weighted_score = None

                # Try to get confidence from various locations
                if 'confidence_metrics' in best_match:
                    cm = best_match['confidence_metrics']
                    confidence = cm.get('confidence_percentage')
                    occurrences = cm.get('occurrences')
                    min_distance = cm.get('min_distance')
                    weighted_score = cm.get('weighted_score')
                elif 'confidence' in best_match:
                    confidence = best_match['confidence']

                if person:
                    top_matches.append({
                        "rank": 1,
                        "person": person,
                        "confidence": round(confidence, 2) if confidence else None,
                        "occurrences": occurrences,
                        "min_distance": round(min_distance, 4) if min_distance else None,
                        "weighted_score": round(weighted_score, 4) if weighted_score else None
                    })

            # Get additional matches
            all_matches = recognition_result.get('all_matches', []) or recognition_result.get('top_matches', [])

            for idx, match in enumerate(all_matches[:2], start=2):  # Get next 2 matches
                person = match.get('person') or match.get('name')
                confidence = None

                if 'confidence_metrics' in match:
                    confidence = match['confidence_metrics'].get('confidence_percentage')
                elif 'confidence' in match:
                    confidence = match['confidence']

                if person:
                    top_matches.append({
                        "rank": idx,
                        "person": person,
                        "confidence": round(confidence, 2) if confidence else None
                    })

        # Format 3: Direct 'matches' array
        elif 'matches' in recognition_result:
            for idx, match in enumerate(recognition_result['matches'][:3], start=1):
                person = match.get('person') or match.get('name')
                confidence = match.get('confidence')

                if person:
                    top_matches.append({
                        "rank": idx,
                        "person": person,
                        "confidence": round(confidence, 2) if confidence else None
                    })

        # If no matches found but we have a primary result, use that
        if not top_matches and recognition_result.get('person'):
            top_matches.append({
                "rank": 1,
                "person": recognition_result['person'],
                "confidence": round(recognition_result.get('confidence', 0), 2) if recognition_result.get('confidence') else None
            })

        return top_matches

    def _append_to_json_log(self, batch_id: str, log_entry: Dict):
        """
        Append log entry to JSON log file.

        Each batch has its own JSON log file with an array of entries.
        """
        log_file = os.path.join(self.LOG_BASE_PATH, f"{batch_id}.json")

        # Load existing entries if file exists
        entries = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse existing log file {log_file}, creating new")
                entries = []

        # Append new entry
        entries.append(log_entry)

        # Save updated entries
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    def get_batch_log(self, batch_id: str) -> Optional[Dict]:
        """
        Retrieve all log entries for a specific batch.

        Args:
            batch_id: Batch identifier

        Returns:
            Dictionary with batch summary and entries, or None if not found
        """
        log_file = os.path.join(self.LOG_BASE_PATH, f"{batch_id}.json")

        if not os.path.exists(log_file):
            return None

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                entries = json.load(f)

            # Calculate summary statistics
            summary = self._calculate_batch_summary(entries)

            return {
                "batch_id": batch_id,
                "total_photos": len(entries),
                "summary": summary,
                "entries": entries
            }

        except Exception as e:
            logger.error(f"Error retrieving batch log {batch_id}: {str(e)}")
            return None

    def _calculate_batch_summary(self, entries: List[Dict]) -> Dict:
        """
        Calculate summary statistics for a batch of recognition results.
        """
        total = len(entries)
        recognized = sum(1 for e in entries if e.get('recognized'))

        # Count unique persons
        persons = set()
        for entry in entries:
            if entry.get('recognized') and entry.get('primary_result', {}).get('person'):
                persons.add(entry['primary_result']['person'])

        # Calculate average confidence
        confidences = [
            e.get('primary_result', {}).get('confidence')
            for e in entries
            if e.get('recognized') and e.get('primary_result', {}).get('confidence') is not None
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "total_photos": total,
            "recognized_count": recognized,
            "unrecognized_count": total - recognized,
            "recognition_rate": round((recognized / total * 100), 2) if total > 0 else 0,
            "unique_persons": len(persons),
            "persons_list": sorted(list(persons)),
            "avg_confidence": round(avg_confidence, 2)
        }

    def export_batch_to_csv(self, batch_id: str) -> Optional[str]:
        """
        Export batch log to CSV format.

        Args:
            batch_id: Batch identifier

        Returns:
            Path to CSV file, or None if export failed
        """
        batch_log = self.get_batch_log(batch_id)

        if not batch_log:
            return None

        csv_file = os.path.join(self.LOG_BASE_PATH, f"{batch_id}.csv")

        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow([
                    'Timestamp',
                    'Filename',
                    'Domain',
                    'Recognized',
                    'Primary Person',
                    'Primary Confidence (%)',
                    'Match #1 Person',
                    'Match #1 Confidence (%)',
                    'Match #2 Person',
                    'Match #2 Confidence (%)',
                    'Match #3 Person',
                    'Match #3 Confidence (%)',
                    'Face Count',
                    'Processing Time (ms)'
                ])

                # Write data rows
                for entry in batch_log['entries']:
                    top_matches = entry.get('top_3_matches', [])

                    # Pad top_matches to always have 3 entries
                    while len(top_matches) < 3:
                        top_matches.append({"person": None, "confidence": None})

                    row = [
                        entry.get('timestamp', ''),
                        entry.get('filename', ''),
                        entry.get('domain', ''),
                        'Yes' if entry.get('recognized') else 'No',
                        entry.get('primary_result', {}).get('person', ''),
                        entry.get('primary_result', {}).get('confidence', ''),
                        top_matches[0].get('person', ''),
                        top_matches[0].get('confidence', ''),
                        top_matches[1].get('person', ''),
                        top_matches[1].get('confidence', ''),
                        top_matches[2].get('person', ''),
                        top_matches[2].get('confidence', ''),
                        entry.get('metadata', {}).get('face_count', ''),
                        entry.get('metadata', {}).get('processing_time_ms', '')
                    ]

                    writer.writerow(row)

            logger.info(f"Exported batch {batch_id} to CSV: {csv_file}")
            return csv_file

        except Exception as e:
            logger.error(f"Error exporting batch to CSV: {str(e)}")
            return None

    def list_batches(self, limit: int = 50) -> List[Dict]:
        """
        List all available batch logs, sorted by date (newest first).

        Args:
            limit: Maximum number of batches to return

        Returns:
            List of batch summaries
        """
        try:
            # Get all JSON log files
            log_files = sorted(
                Path(self.LOG_BASE_PATH).glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )[:limit]

            batches = []
            for log_file in log_files:
                batch_id = log_file.stem
                batch_log = self.get_batch_log(batch_id)

                if batch_log:
                    batches.append({
                        "batch_id": batch_id,
                        "total_photos": batch_log['total_photos'],
                        "summary": batch_log['summary'],
                        "created_at": batch_log['entries'][0]['timestamp'] if batch_log['entries'] else None
                    })

            return batches

        except Exception as e:
            logger.error(f"Error listing batches: {str(e)}")
            return []

    def delete_batch_log(self, batch_id: str) -> bool:
        """
        Delete a batch log (both JSON and CSV if exists).

        Args:
            batch_id: Batch identifier

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            json_file = os.path.join(self.LOG_BASE_PATH, f"{batch_id}.json")
            csv_file = os.path.join(self.LOG_BASE_PATH, f"{batch_id}.csv")

            deleted = False

            if os.path.exists(json_file):
                os.remove(json_file)
                deleted = True
                logger.info(f"Deleted JSON log: {json_file}")

            if os.path.exists(csv_file):
                os.remove(csv_file)
                logger.info(f"Deleted CSV log: {csv_file}")

            return deleted

        except Exception as e:
            logger.error(f"Error deleting batch log {batch_id}: {str(e)}")
            return False
