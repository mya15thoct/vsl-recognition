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
    Collect keypoint sequences from frame images in VSL_Isolated folder
    
    Args:
        actions: List of action folder names to process (None = all folders)
        num_sequences: Number of sequences per action
        sequence_length: Number of frames per sequence
    """
    print("="*60)
    print("COLLECTING KEYPOINTS FROM FRAMES")
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
        
        # Get frame image files in action folder (including subdirectories)
        frame_files = sorted(list(action_path.glob('**/*.jpg')) + list(action_path.glob('**/*.png')))
        
        if not frame_files:
            print(f"[WARNING] Skipping {action}: no frames found")
            continue
        
        print(f"[{action_idx+1}/{len(actions)}] Processing '{action}' ({len(frame_files)} frames)")
        
        # Process multiple sequences from the same set of frames
        for sequence in range(num_sequences):
            frames_data = []
            
            # Sample frames evenly
            if len(frame_files) >= sequence_length:
                # Sample sequence_length frames evenly from all frames
                indices = np.linspace(0, len(frame_files)-1, sequence_length, dtype=int)
            else:
                # Use all available frames
                indices = list(range(len(frame_files)))
            
            # Read and process each frame
            for idx in indices:
                if idx < len(frame_files):
                    frame = cv2.imread(str(frame_files[idx]))
                    
                    if frame is not None:
                        # Make detections
                        image, results = mediapipe_detection(frame, holistic)
                        
                        # Extract keypoints
                        keypoints = extract_keypoints(results)
                        frames_data.append(keypoints)
            
            # Pad if needed
            while len(frames_data) < sequence_length:
                frames_data.append(np.zeros(1662))
            
            # Save sequence
            sequence_data = np.array(frames_data[:sequence_length])
            save_path = SEQUENCE_PATH / action / str(sequence)
            save_path.mkdir(parents=True, exist_ok=True)
            
            np.save(save_path / f"{sequence}.npy", sequence_data)
            total_processed += 1
        
        print(f"  [OK] Saved {num_sequences} sequences")
    
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
