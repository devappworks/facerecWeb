#!/usr/bin/env python3
"""
Comprehensive analysis of vucic2.mp4 video recognition
"""
import sys
import os
import cv2
import tempfile
from collections import defaultdict
from deepface import DeepFace
from pathlib import Path

VIDEO_PATH = '/home/photolytics/vucic2.mp4'
DOMAIN = 'serbia'
INTERVAL_SECONDS = 5.0
MAX_DURATION = 180.0  # First 3 minutes

# Recognition settings for serbia domain (ArcFace)
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.50
DB_PATH = f'storage/recognized_faces_prod/{DOMAIN}'

def extract_person_name(identity_path):
    """Extract person name from filename"""
    filename = Path(identity_path).stem
    parts = filename.split('_')
    person_name_parts = []
    for i, part in enumerate(parts):
        if '-' in part and len(part) >= 10:
            person_name_parts = parts[:i]
            break
    return '_'.join(person_name_parts) if person_name_parts else 'Unknown'

def analyze_frame(frame, frame_idx, timestamp):
    """Analyze a single frame and return detailed results"""
    # Save frame to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        temp_path = tmp.name
        cv2.imwrite(temp_path, frame)

    try:
        # Run DeepFace.find
        dfs = DeepFace.find(
            img_path=temp_path,
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False,
            threshold=THRESHOLD,
            silent=True,
            batched=True
        )

        if not dfs or len(dfs) == 0:
            return {
                'timestamp': timestamp,
                'num_faces': 0,
                'faces': []
            }

        faces_data = []
        for face_idx, face_matches in enumerate(dfs):
            face_info = {
                'face_index': face_idx,
                'num_matches': len(face_matches) if face_matches else 0,
                'matches': []
            }

            if face_matches and len(face_matches) > 0:
                # Get top 5 matches for analysis
                for match in face_matches[:5]:
                    person_name = extract_person_name(match['identity'])
                    face_info['matches'].append({
                        'person': person_name,
                        'distance': match['distance'],
                        'confidence': (1 - match['distance']) * 100
                    })

            faces_data.append(face_info)

        return {
            'timestamp': timestamp,
            'num_faces': len(dfs),
            'faces': faces_data
        }

    except Exception as e:
        return {
            'timestamp': timestamp,
            'error': str(e)
        }
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass

def main():
    print("="*80)
    print("Video Recognition Analysis: vucic2.mp4")
    print("="*80)

    # Get video info
    video = cv2.VideoCapture(VIDEO_PATH)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    print(f"Video duration: {duration:.1f}s")
    print(f"FPS: {fps:.1f}")
    print(f"Analyzing first {MAX_DURATION}s, every {INTERVAL_SECONDS}s")
    print(f"Model: {MODEL_NAME}, Threshold: {THRESHOLD}")
    print("="*80)

    # Extract and analyze frames
    all_results = []
    person_stats = defaultdict(lambda: {'count': 0, 'distances': [], 'timestamps': []})

    current_time = 0.0
    frame_num = 0

    while current_time <= min(MAX_DURATION, duration):
        frame_number = int(current_time * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()

        if not ret:
            break

        print(f"\n--- Analyzing frame at {current_time:.1f}s ---")
        result = analyze_frame(frame, frame_num, current_time)
        all_results.append(result)

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        elif result['num_faces'] == 0:
            print(f"  No faces detected")
        else:
            print(f"  Detected {result['num_faces']} face(s)")
            for face in result['faces']:
                if face['matches']:
                    best = face['matches'][0]
                    print(f"  Face {face['face_index']+1}: {best['person']} ({best['confidence']:.1f}%, dist={best['distance']:.3f})")

                    # Track stats
                    person_stats[best['person']]['count'] += 1
                    person_stats[best['person']]['distances'].append(best['distance'])
                    person_stats[best['person']]['timestamps'].append(current_time)

                    # Show alternative matches if any
                    if len(face['matches']) > 1:
                        alts = face['matches'][1:3]  # Show next 2 alternatives
                        alt_names = [f"{m['person']}({m['confidence']:.0f}%)" for m in alts]
                        print(f"       Alternatives: {', '.join(alt_names)}")
                else:
                    print(f"  Face {face['face_index']+1}: No matches")

        current_time += INTERVAL_SECONDS
        frame_num += 1

    video.release()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - Recognition Statistics")
    print("="*80)
    print(f"Total frames analyzed: {len(all_results)}")

    # Sort by count
    sorted_persons = sorted(person_stats.items(), key=lambda x: x[1]['count'], reverse=True)

    print(f"\nPersons detected (sorted by frequency):")
    print("-"*60)
    for person, stats in sorted_persons:
        avg_dist = sum(stats['distances']) / len(stats['distances'])
        min_dist = min(stats['distances'])
        max_dist = max(stats['distances'])
        avg_conf = (1 - avg_dist) * 100
        min_conf = (1 - max_dist) * 100
        max_conf = (1 - min_dist) * 100

        print(f"\n{person}:")
        print(f"  Count: {stats['count']} frames")
        print(f"  Confidence: {min_conf:.1f}% - {max_conf:.1f}% (avg: {avg_conf:.1f}%)")
        print(f"  Distance: {min_dist:.3f} - {max_dist:.3f} (avg: {avg_dist:.3f})")
        print(f"  Timestamps: {', '.join([f'{t:.0f}s' for t in stats['timestamps'][:10]])}" +
              ("..." if len(stats['timestamps']) > 10 else ""))

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    vucic_stats = person_stats.get('Aleksandar_Vucic', {'count': 0, 'distances': []})
    other_persons = [(p, s) for p, s in sorted_persons if p != 'Aleksandar_Vucic']

    print(f"\nAleksandar Vucic detected: {vucic_stats['count']} times")
    if vucic_stats['distances']:
        print(f"  Average distance: {sum(vucic_stats['distances'])/len(vucic_stats['distances']):.3f}")

    print(f"\nOther persons detected: {len(other_persons)}")
    for person, stats in other_persons[:5]:
        avg_dist = sum(stats['distances']) / len(stats['distances'])
        print(f"  - {person}: {stats['count']} times (avg distance: {avg_dist:.3f})")

    if other_persons:
        print("\n⚠️  ISSUE: Multiple different persons detected in what should be")
        print("    a video primarily featuring Aleksandar Vucic.")
        print("\n    Possible causes:")
        print("    1. Multiple faces in frames (other people in video)")
        print("    2. Different angles/lighting causing mismatches")
        print("    3. Similar-looking faces in database")
        print("    4. Threshold might be too loose for this video quality")

        # Check if we should recommend stricter threshold
        all_distances = []
        for p, s in sorted_persons:
            all_distances.extend(s['distances'])
        if all_distances:
            avg_all = sum(all_distances) / len(all_distances)
            print(f"\n    Average distance across all detections: {avg_all:.3f}")
            if avg_all > 0.4:
                print("    → Consider using a stricter threshold (e.g., 0.40 instead of 0.50)")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
