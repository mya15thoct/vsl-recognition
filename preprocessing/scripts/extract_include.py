"""
Extract keypoints from INCLUDE dataset (.MOV videos)
Adapted from extract_keypoints.py for INCLUDE format
"""
import cv2
import numpy as np
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, SEQUENCE_PATH, SEQUENCE_LENGTH
from utils.keypoint_extraction import mediapipe_detection, extract_keypoints, get_holistic_model

def extract_include_keypoints():
    """
    Extract keypoints from INCLUDE .MOV videos
    """
    print("="*60)
    print("EXTRACTING KEYPOINTS FROM INCLUDE DATASET")
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
        
        #Get all .MOV videos
        videos = list(class_path.glob('*.MOV'))
        
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
            
            # Process sequence
            if len(frames_data) > SEQUENCE_LENGTH:
                # Downsample to SEQUENCE_LENGTH
                indices = np.linspace(0, len(frames_data) - 1, SEQUENCE_LENGTH, dtype=int)
                frames_data = [frames_data[i] for i in indices]
            elif len(frames_data) < SEQUENCE_LENGTH:
                # Pad with last frame
                last_frame = frames_data[-1] if frames_data else np.zeros(1662)
                while len(frames_data) < SEQUENCE_LENGTH:
                    frames_data.append(last_frame.copy())
            
            # Save sequence
            sequence_data = np.array(frames_data[:SEQUENCE_LENGTH])
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
    extract_include_keypoints()
