#!/usr/bin/env python3
"""
Test video recognition on vucic2.mp4
Process first 60 seconds, extract frame every 5 seconds
"""
import sys
import os
import cv2
import tempfile
from deepface import DeepFace
from pathlib import Path

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIDEO_PATH = '/home/photolytics/vucic2.mp4'
DOMAIN = 'serbia'
INTERVAL_SECONDS = 5.0
MAX_DURATION = 60.0  # Only process first 60 seconds

# Recognition settings for serbia domain (ArcFace)
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.50
DB_PATH = f'storage/recognized_faces_prod/{DOMAIN}'

def extract_frames(video_path, interval_seconds, max_duration):
    """Extract frames from video at specified intervals, up to max_duration"""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    frames = []
    timestamps = []

    current_time = 0.0
    while current_time <= max_duration:
        frame_number = int(current_time * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()

        if not ret:
            break

        frames.append(frame)
        timestamps.append(current_time)
        print(f"Extracted frame at {current_time:.1f}s (frame #{frame_number})")

        current_time += interval_seconds

    video.release()
    print(f"\nExtracted {len(frames)} frames from first {max_duration}s")
    return frames, timestamps

def recognize_frame(frame, frame_idx, timestamp):
    """Run face recognition on a single frame"""
    # Save frame to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, frame)

    try:
        print(f"\n--- Frame {frame_idx} at {timestamp:.1f}s ---")

        # Check if database exists
        if not os.path.exists(DB_PATH):
            print(f"ERROR: Database not found at {DB_PATH}")
            return None

        print(f"Using {MODEL_NAME} model with {DETECTOR_BACKEND} detector")
        print(f"Database: {DB_PATH}")
        print(f"Threshold: {THRESHOLD}")

        # Run DeepFace.find
        dfs = DeepFace.find(
            img_path=temp_path,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False,
            threshold=THRESHOLD,
            silent=False,
            batched=True
        )

        # With batched=True, dfs is a list of lists of dicts
        # dfs[i] contains matches for the i-th detected face
        # Each match is a dict with 'identity', 'distance', etc.

        if not dfs or len(dfs) == 0:
            print("No faces detected")
            return {
                'timestamp': timestamp,
                'recognized': False,
                'reason': 'No faces detected'
            }

        print(f"Detected {len(dfs)} face(s)")

        # Get the first face's matches
        first_face_matches = dfs[0]

        if not first_face_matches or len(first_face_matches) == 0:
            print("No matches found for first face")
            return {
                'timestamp': timestamp,
                'recognized': False,
                'reason': 'No matches found'
            }

        # Get best match (first one, lowest distance)
        best_match = first_face_matches[0]
        distance = best_match['distance']
        identity_path = best_match['identity']

        # Extract person name from filename
        # Format: PersonName_YYYY-MM-DD_ID.ext
        # We need to get everything before the date
        print(f"  Identity path: {identity_path}")
        filename = Path(identity_path).stem  # Get filename without extension
        print(f"  Filename (no ext): {filename}")
        parts = filename.split('_')
        print(f"  Parts: {parts}")

        # Find where the date starts (YYYY-MM-DD format)
        # The date part looks like "2025-05-29" so it starts with 4 digits and contains hyphens
        person_name_parts = []
        for i, part in enumerate(parts):
            # Check if this part looks like a date (YYYY-MM-DD)
            if '-' in part and len(part) >= 10:
                # This is the date part, everything before it is the person name
                person_name_parts = parts[:i]
                print(f"  Found date at index {i}: {part}")
                break

        person_name = '_'.join(person_name_parts) if person_name_parts else 'Unknown'
        print(f"  Extracted person name: {person_name}")

        # Calculate confidence (inverse of distance, scaled to percentage)
        confidence = (1 - distance) * 100

        print(f"✓ MATCH FOUND!")
        print(f"  Person: {person_name}")
        print(f"  Distance: {distance:.4f}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Total matches for first face: {len(first_face_matches)}")

        return {
            'timestamp': timestamp,
            'recognized': True,
            'person': person_name,
            'distance': float(distance),
            'confidence': float(confidence),
            'identity_path': identity_path,
            'total_faces': len(dfs),
            'total_matches': len(first_face_matches)
        }

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'timestamp': timestamp,
            'recognized': False,
            'reason': str(e)
        }
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

def main():
    print("="*60)
    print("Video Recognition Test: vucic2.mp4")
    print("="*60)
    print(f"Video: {VIDEO_PATH}")
    print(f"Domain: {DOMAIN}")
    print(f"Model: {MODEL_NAME}")
    print(f"Interval: {INTERVAL_SECONDS}s")
    print(f"Max duration: {MAX_DURATION}s")
    print("="*60)

    # Extract frames
    frames, timestamps = extract_frames(VIDEO_PATH, INTERVAL_SECONDS, MAX_DURATION)

    # Process each frame
    results = []
    for idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        result = recognize_frame(frame, idx, timestamp)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total frames processed: {len(results)}")
    recognized_count = sum(1 for r in results if r['recognized'])
    print(f"Recognized: {recognized_count}")
    print(f"Not recognized: {len(results) - recognized_count}")
    print(f"Recognition rate: {(recognized_count / len(results) * 100):.1f}%")

    if recognized_count > 0:
        print("\nRecognized persons:")
        persons = {}
        for r in results:
            if r['recognized']:
                person = r['person']
                if person not in persons:
                    persons[person] = []
                persons[person].append(f"{r['timestamp']:.1f}s")

        for person, timestamps in persons.items():
            print(f"  - {person}: {len(timestamps)} times ({', '.join(timestamps)})")

    print("="*60)

if __name__ == "__main__":
    main()
