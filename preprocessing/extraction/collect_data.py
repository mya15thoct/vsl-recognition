"""
Keypoint extraction from VSL_Isolated videos using MediaPipe
"""
import cv2
import numpy as np
import os
from pathlib import Path

# Import from parent module
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, SEQUENCE_PATH, SEQUENCE_LENGTH, NO_SEQUENCES
from utils.keypoint_extraction import mediapipe_detection, extract_keypoints, get_holistic_model


def collect_keypoints_from_videos(actions=None, num_sequences=NO_SEQUENCES, sequence_length=SEQUENCE_LENGTH):
    """
    Collect keypoint sequences from videos in VSL_Isolated folder
    
    Args:
        actions: List of action folder names to process (None = all folders)
        num_sequences: Number of sequences per action
        sequence_length: Number of frames per sequence
    """
    print("="*60)
    print("COLLECTING KEYPOINTS FROM VIDEOS")
    print("="*60)
    
    # Get all action folders
    if actions is None:
        actions = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != 'sequences']
        print(f"\nFound {len(actions)} action folders in {DATA_DIR}")
    
    # Create sequences directory
    SEQUENCE_PATH.mkdir(parents=True, exist_ok=True)
    
    # Initialize MediaPipe model
    print("\nInitializing MediaPipe Holistic...")
    holistic = get_holistic_model()
    
    print(f"\nProcessing {len(actions)} actions...")
    print(f"Sequences per action: {num_sequences}")
    print(f"Frames per sequence: {sequence_length}\n")
    
    total_processed = 0
    
    for action_idx, action in enumerate(actions):
        action_path = DATA_DIR / action
        
        # Check if action folder exists
        if not action_path.exists():
            print(f"[WARNING] Skipping {action}: folder not found")
            continue
        
        # Get video files in action folder
        video_files = list(action_path.glob('*.mp4')) + list(action_path.glob('*.avi'))
        
        if not video_files:
            print(f"[WARNING] Skipping {action}: no videos found")
            continue
        
        print(f"[{action_idx+1}/{len(actions)}] Processing '{action}' ({len(video_files)} videos)")
        
        for sequence in range(min(num_sequences, len(video_files))):
            video_path = video_files[sequence]
            
            # Read video
            cap = cv2.VideoCapture(str(video_path))
            frames_data = []
            
            # Extract frames
            frame_count = 0
            while cap.isOpened() and frame_count < sequence_length:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                
                # Extract keypoints
                keypoints = extract_keypoints(results)
                frames_data.append(keypoints)
                
                frame_count += 1
            
            cap.release()
            
            # Pad if needed
            while len(frames_data) < sequence_length:
                frames_data.append(np.zeros(1662))
            
            # Save sequence
            sequence_data = np.array(frames_data[:sequence_length])
            save_path = SEQUENCE_PATH / action / str(sequence)
            save_path.mkdir(parents=True, exist_ok=True)
            
            np.save(save_path / f"{sequence}.npy", sequence_data)
            total_processed += 1
        
        print(f"  [OK] Saved {min(num_sequences, len(video_files))} sequences")
    
    holistic.close()
    
    print(f"\n{'='*60}")
    print(f"[OK] COLLECTION COMPLETE")
    print(f"  Total sequences: {total_processed}")
    print(f"  Saved to: {SEQUENCE_PATH}")
    print(f"{'='*60}")
    
    return actions


if __name__ == "__main__":
    # Collect from all actions
    collect_keypoints_from_videos()
