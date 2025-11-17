"""
Video Face Recognition Service
Extracts frames from video and performs face recognition on each frame.
"""

import os
import cv2
import uuid
import json
import time
import psutil
import logging
from datetime import datetime
from typing import Dict, List, Optional
from flask import current_app
from threading import Thread

from app.services.recognition_service import RecognitionService

logger = logging.getLogger(__name__)


class VideoService:
    """Service for processing videos and recognizing faces in frames"""

    VIDEO_STORAGE = "storage/videos"
    FRAMES_STORAGE = "storage/video_frames"
    RESULTS_STORAGE = "storage/video_results"

    def __init__(self):
        """Initialize video service"""
        os.makedirs(self.VIDEO_STORAGE, exist_ok=True)
        os.makedirs(self.FRAMES_STORAGE, exist_ok=True)
        os.makedirs(self.RESULTS_STORAGE, exist_ok=True)

    def save_video(self, video_bytes: bytes, original_filename: str) -> Dict:
        """
        Save uploaded video file.

        Args:
            video_bytes: Video file bytes
            original_filename: Original filename

        Returns:
            Dictionary with video_id and file path
        """
        try:
            video_id = str(uuid.uuid4())[:12]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Get file extension
            ext = os.path.splitext(original_filename)[1] or '.mp4'
            filename = f"{timestamp}_{video_id}{ext}"

            video_path = os.path.join(self.VIDEO_STORAGE, filename)

            # Save video file
            with open(video_path, 'wb') as f:
                f.write(video_bytes)

            # Get video info
            video_info = self._get_video_info(video_path)

            logger.info(f"Saved video {video_id}: {video_info}")

            return {
                "success": True,
                "video_id": video_id,
                "video_path": video_path,
                "filename": filename,
                "size_mb": len(video_bytes) / (1024 * 1024),
                **video_info
            }

        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
            return {
                "success": False,
                "message": f"Error saving video: {str(e)}"
            }

    def _get_video_info(self, video_path: str) -> Dict:
        """
        Get video metadata using OpenCV.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video info
        """
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return {
                    "error": "Could not open video"
                }

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_seconds": round(duration, 2),
                "duration_formatted": self._format_duration(duration)
            }

        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return {"error": str(e)}

    def _format_duration(self, seconds: float) -> str:
        """Format duration as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def extract_frames(self, video_path: str, video_id: str,
                      interval_seconds: float = 3.0) -> Dict:
        """
        Extract frames from video at specified interval.

        Args:
            video_path: Path to video file
            video_id: Unique video identifier
            interval_seconds: Extract 1 frame every N seconds (default 3)

        Returns:
            Dictionary with extraction results
        """
        start_time = time.time()

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return {
                    "success": False,
                    "message": "Could not open video"
                }

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Calculate frame interval
            frame_interval = int(fps * interval_seconds)

            logger.info(f"Video: {video_id}, FPS: {fps}, Total frames: {total_frames}, "
                       f"Duration: {duration:.2f}s, Extracting every {frame_interval} frames "
                       f"({interval_seconds}s)")

            # Create directory for frames
            frames_dir = os.path.join(self.FRAMES_STORAGE, video_id)
            os.makedirs(frames_dir, exist_ok=True)

            extracted_frames = []
            frame_number = 0
            extracted_count = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Extract frame at interval
                if frame_number % frame_interval == 0:
                    timestamp = frame_number / fps if fps > 0 else 0

                    frame_filename = f"frame_{frame_number:06d}_t{timestamp:.2f}s.jpg"
                    frame_path = os.path.join(frames_dir, frame_filename)

                    # Save frame as JPEG
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

                    extracted_frames.append({
                        "frame_number": frame_number,
                        "timestamp": round(timestamp, 2),
                        "filename": frame_filename,
                        "path": frame_path
                    })

                    extracted_count += 1

                    if extracted_count % 10 == 0:
                        logger.info(f"Extracted {extracted_count} frames...")

                frame_number += 1

            cap.release()

            extraction_time = time.time() - start_time

            logger.info(f"Extracted {extracted_count} frames from {total_frames} total "
                       f"in {extraction_time:.2f}s")

            return {
                "success": True,
                "video_id": video_id,
                "total_frames": total_frames,
                "extracted_count": extracted_count,
                "extraction_time": round(extraction_time, 2),
                "frames": extracted_frames,
                "frames_dir": frames_dir,
                "video_info": {
                    "fps": fps,
                    "duration": round(duration, 2),
                    "interval_seconds": interval_seconds
                }
            }

        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return {
                "success": False,
                "message": f"Error extracting frames: {str(e)}"
            }

    def recognize_frames(self, video_id: str, domain: str,
                        extraction_result: Dict) -> Dict:
        """
        Perform face recognition on extracted frames.

        Args:
            video_id: Unique video identifier
            domain: Domain for face recognition
            extraction_result: Result from extract_frames()

        Returns:
            Dictionary with recognition results
        """
        start_time = time.time()

        try:
            frames = extraction_result.get('frames', [])

            if not frames:
                return {
                    "success": False,
                    "message": "No frames to process"
                }

            logger.info(f"Starting recognition on {len(frames)} frames for video {video_id}")

            # System monitoring
            process = psutil.Process()
            initial_cpu = psutil.cpu_percent(interval=None)
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

            results = []
            recognized_count = 0
            failed_count = 0
            persons_detected = set()

            for i, frame_info in enumerate(frames):
                frame_path = frame_info['path']
                frame_number = frame_info['frame_number']
                timestamp = frame_info['timestamp']

                try:
                    # Read frame
                    with open(frame_path, 'rb') as f:
                        frame_bytes = f.read()

                    # Recognize faces
                    recognition_result = RecognitionService.recognize_face(
                        frame_bytes,
                        domain
                    )

                    # Parse result
                    recognized = False
                    person_name = None
                    confidence = None

                    if recognition_result.get('success'):
                        person_name = recognition_result.get('person')
                        confidence = recognition_result.get('confidence')

                        if person_name and person_name != 'Unknown':
                            recognized = True
                            recognized_count += 1
                            persons_detected.add(person_name)

                    results.append({
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "filename": frame_info['filename'],
                        "recognized": recognized,
                        "person": person_name,
                        "confidence": confidence,
                        "raw_result": recognition_result
                    })

                    # Log progress every 10 frames
                    if (i + 1) % 10 == 0:
                        current_cpu = psutil.cpu_percent(interval=None)
                        current_memory = process.memory_info().rss / (1024 * 1024)

                        logger.info(
                            f"Processed {i + 1}/{len(frames)} frames | "
                            f"Recognized: {recognized_count} | "
                            f"CPU: {current_cpu:.1f}% | "
                            f"Memory: {current_memory:.1f} MB"
                        )

                except Exception as frame_error:
                    logger.error(f"Error processing frame {frame_number}: {str(frame_error)}")
                    failed_count += 1

                    results.append({
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "filename": frame_info['filename'],
                        "recognized": False,
                        "person": None,
                        "confidence": None,
                        "error": str(frame_error)
                    })

            # Final system stats
            final_cpu = psutil.cpu_percent(interval=None)
            final_memory = process.memory_info().rss / (1024 * 1024)
            avg_cpu = (initial_cpu + final_cpu) / 2
            memory_used = final_memory - initial_memory

            processing_time = time.time() - start_time
            fps_processed = len(frames) / processing_time if processing_time > 0 else 0

            # Save results to JSON
            result_file = os.path.join(
                self.RESULTS_STORAGE,
                f"{video_id}_results.json"
            )

            complete_result = {
                "video_id": video_id,
                "domain": domain,
                "processed_at": datetime.now().isoformat(),
                "extraction_info": extraction_result,
                "statistics": {
                    "total_frames": len(frames),
                    "recognized_frames": recognized_count,
                    "failed_frames": failed_count,
                    "recognition_rate": round((recognized_count / len(frames)) * 100, 2),
                    "unique_persons": len(persons_detected),
                    "persons_list": sorted(list(persons_detected))
                },
                "performance": {
                    "processing_time_seconds": round(processing_time, 2),
                    "frames_per_second": round(fps_processed, 2),
                    "avg_cpu_percent": round(avg_cpu, 2),
                    "memory_used_mb": round(memory_used, 2),
                    "final_memory_mb": round(final_memory, 2)
                },
                "results": results
            }

            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(complete_result, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Video {video_id} processing complete:\n"
                f"  Frames processed: {len(frames)}\n"
                f"  Recognized: {recognized_count} ({(recognized_count/len(frames)*100):.1f}%)\n"
                f"  Unique persons: {len(persons_detected)}\n"
                f"  Processing time: {processing_time:.2f}s ({fps_processed:.2f} FPS)\n"
                f"  CPU usage: {avg_cpu:.1f}%\n"
                f"  Memory used: {memory_used:.1f} MB"
            )

            return {
                "success": True,
                **complete_result
            }

        except Exception as e:
            logger.error(f"Error recognizing frames: {str(e)}")
            return {
                "success": False,
                "message": f"Error recognizing frames: {str(e)}"
            }

    def process_video(self, video_bytes: bytes, original_filename: str,
                     domain: str, interval_seconds: float = 3.0) -> Dict:
        """
        Complete video processing pipeline: save, extract, recognize.

        Args:
            video_bytes: Video file bytes
            original_filename: Original filename
            domain: Domain for face recognition
            interval_seconds: Extract 1 frame every N seconds

        Returns:
            Dictionary with complete processing results
        """
        try:
            # Step 1: Save video
            save_result = self.save_video(video_bytes, original_filename)

            if not save_result.get('success'):
                return save_result

            video_id = save_result['video_id']
            video_path = save_result['video_path']

            # Step 2: Extract frames
            extraction_result = self.extract_frames(
                video_path,
                video_id,
                interval_seconds
            )

            if not extraction_result.get('success'):
                return extraction_result

            # Step 3: Recognize faces
            recognition_result = self.recognize_frames(
                video_id,
                domain,
                extraction_result
            )

            return recognition_result

        except Exception as e:
            logger.error(f"Error in video processing pipeline: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing video: {str(e)}"
            }

    def process_video_async(self, video_bytes: bytes, original_filename: str,
                           domain: str, interval_seconds: float = 3.0) -> str:
        """
        Start async video processing in background thread.

        Args:
            video_bytes: Video file bytes
            original_filename: Original filename
            domain: Domain for face recognition
            interval_seconds: Extract 1 frame every N seconds

        Returns:
            video_id for tracking progress
        """
        # First save the video to get video_id
        save_result = self.save_video(video_bytes, original_filename)

        if not save_result.get('success'):
            raise Exception(save_result.get('message', 'Failed to save video'))

        video_id = save_result['video_id']
        video_path = save_result['video_path']

        # Start background processing
        app_context = current_app.app_context()
        thread = Thread(
            target=self._process_video_thread,
            args=(video_id, video_path, domain, interval_seconds, app_context)
        )
        thread.daemon = True
        thread.start()

        return video_id

    def _process_video_thread(self, video_id: str, video_path: str,
                             domain: str, interval_seconds: float, app_context):
        """Background thread for video processing"""
        with app_context:
            try:
                logger.info(f"Starting background processing for video {video_id}")

                # Extract frames
                extraction_result = self.extract_frames(
                    video_path,
                    video_id,
                    interval_seconds
                )

                if not extraction_result.get('success'):
                    logger.error(f"Frame extraction failed: {extraction_result}")
                    return

                # Recognize faces
                recognition_result = self.recognize_frames(
                    video_id,
                    domain,
                    extraction_result
                )

                if not recognition_result.get('success'):
                    logger.error(f"Face recognition failed: {recognition_result}")
                    return

                logger.info(f"Background processing complete for video {video_id}")

            except Exception as e:
                logger.error(f"Error in background video processing: {str(e)}")

    def get_video_result(self, video_id: str) -> Optional[Dict]:
        """
        Get processing results for a video.

        Args:
            video_id: Video identifier

        Returns:
            Dictionary with results or None if not found
        """
        try:
            result_file = os.path.join(
                self.RESULTS_STORAGE,
                f"{video_id}_results.json"
            )

            if not os.path.exists(result_file):
                return None

            with open(result_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error getting video result: {str(e)}")
            return None
