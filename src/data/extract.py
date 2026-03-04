"""
Extract keypoints from sign language videos
"""
import cv2
import numpy as np
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, SEQUENCE_PATH, SEQUENCE_LENGTH
from src.utils.extraction import mediapipe_detection, extract_keypoints, get_holistic_model

def extract_keypoints_from_videos():
    """
    Extract keypoints from sign language videos
    """
    print("="*60)
    print("EXTRACTING KEYPOINTS FROM DATASET")
    print("="*60)
    
    # Get all class folders
    classes = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != 'sequences']
    classes = sorted(classes)
    
    print(f"\nFound {len(classes)} classes")
    print(f"Sequence length: {SEQUENCE_LENGTH} frames")
    print(f"Output: {SEQUENCE_PATH}\n")
    
    # Create sequences directory
    SEQUENCE_PATH.mkdir(parents=True, exist_ok=True)
    
    # Initialize MediaPipe
    print("Initializing MediaPipe Holistic...")
    holistic = get_holistic_model()
    
    total_processed = 0
    
    for class_idx, class_name in enumerate(classes):
        class_path = DATA_DIR / class_name
        
        # Get all video files (.MOV and .MP4, case insensitive)
        videos = (list(class_path.glob('*.MOV')) + 
                 list(class_path.glob('*.mov')) +
                 list(class_path.glob('*.MP4')) + 
                 list(class_path.glob('*.mp4')))
        
        if not videos:
            print(f"[WARNING] No videos found in {class_name}")
            continue
            
        print(f"[{class_idx+1}/{len(classes)}] Processing '{class_name}' ({len(videos)} videos)")
        
        # Create class folder in sequences
        seq_class_path = SEQUENCE_PATH / class_name
        seq_class_path.mkdir(parents=True, exist_ok=True)
        
        # Process each video
        for video_idx, video_path in enumerate(videos):
            cap = cv2.VideoCapture(str(video_path))
            frames_data = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # MediaPipe detection
                image, results = mediapipe_detection(frame, holistic)
                
                # Extract keypoints
                keypoints = extract_keypoints(results)
                frames_data.append(keypoints)
            
            cap.release()
            
            # Skip if no frames extracted
            if len(frames_data) == 0:
                print(f"    [WARNING] No frames extracted from {video_path.name}")
                continue
            
            # Save sequence with all frames (no downsampling)
            sequence_data = np.array(frames_data)
            video_name = video_path.stem  # Filename without extension
            np.save(seq_class_path / f"{video_name}.npy", sequence_data)
            
            total_processed += 1
        
        print(f"  [OK] Saved {len(videos)} sequences")
    
    holistic.close()
    
    print(f"\n{'='*60}")
    print(f"[OK] EXTRACTION COMPLETE")
    print(f"  Total sequences: {total_processed}")
    print(f"  Classes: {len(classes)}")
    print(f"  Saved to: {SEQUENCE_PATH}")
    print(f"{'='*60}")

if __name__ == "__main__":
    extract_keypoints_from_videos()
