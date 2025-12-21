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

from config import DATA_DIR, SEQUENCE_PATH, NO_SEQUENCES, TRIM_START_FRAMES, TRIM_END_FRAMES
from utils.keypoint_extraction import mediapipe_detection, extract_keypoints, get_holistic_model


def get_max_sequence_length(data_dir, trim_start=0, trim_end=0):
    """
    Scan dataset to find maximum sequence length after trimming
    
    Args:
        data_dir: Path to data directory
        trim_start: Frames to trim from start
        trim_end: Frames to trim from end
    
    Returns:
        Maximum sequence length found in dataset
    """
    max_length = 0
    
    # Get all action folders
    actions = [d for d in data_dir.iterdir() if d.is_dir() and d.name != 'sequences']
    
    for action_path in actions:
        # Get frame files
        frame_files = list(action_path.glob('**/*.jpg')) + list(action_path.glob('**/*.png'))
        
        if frame_files:
            # Calculate trimmed length
            total_frames = len(frame_files)
            if total_frames > (trim_start + trim_end):
                trimmed_length = total_frames - trim_start - trim_end
            else:
                trimmed_length = total_frames
            
            max_length = max(max_length, trimmed_length)
    
    return max_length


def collect_keypoints_from_videos(actions=None, num_sequences=NO_SEQUENCES):
    """
    Collect keypoint sequences from frame images in VSL_Isolated folder
    
    Args:
        actions: List of action folder names to process (None = all folders)
        num_sequences: Number of sequences per action
    """
    print("="*60)
    print("COLLECTING KEYPOINTS FROM FRAMES")
    print("="*60)
    
    # Get all action folders
    if actions is None:
        actions = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != 'sequences']
        print(f"\nFound {len(actions)} action folders in {DATA_DIR}")
    
    # Auto-detect maximum sequence length
    print("\n[INFO] Auto-detecting maximum sequence length from dataset...")
    sequence_length = get_max_sequence_length(DATA_DIR, TRIM_START_FRAMES, TRIM_END_FRAMES)
    print(f"[INFO] Maximum sequence length (after trimming): {sequence_length} frames")
    print(f"[INFO] Trimming: {TRIM_START_FRAMES} start frames + {TRIM_END_FRAMES} end frames")
    
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
            
            # Trim start and end frames (remove preparation/ending movements)
            total_frames = len(frame_files)
            if total_frames > (TRIM_START_FRAMES + TRIM_END_FRAMES):
                # Enough frames to trim
                trimmed_frames = frame_files[TRIM_START_FRAMES:-TRIM_END_FRAMES if TRIM_END_FRAMES > 0 else None]
            else:
                # Not enough frames, use all
                trimmed_frames = frame_files
            
            # Use ALL available frames (no sampling to preserve motion)
            indices = list(range(len(trimmed_frames)))
            
            # Read and process each frame
            for idx in indices:
                if idx < len(trimmed_frames):
                    frame = cv2.imread(str(trimmed_frames[idx]))
                    
                    if frame is not None:
                        # Make detections
                        image, results = mediapipe_detection(frame, holistic)
                        
                        # Extract keypoints
                        keypoints = extract_keypoints(results)
                        frames_data.append(keypoints)
            
            # Pad if needed (repeat last frame for natural padding)
            if len(frames_data) < sequence_length:
                last_frame = frames_data[-1] if frames_data else np.zeros(1662)
                while len(frames_data) < sequence_length:
                    frames_data.append(last_frame.copy())
            
            # Save sequence
            sequence_data = np.array(frames_data[:sequence_length])
            save_path = SEQUENCE_PATH / action  # Simplified: action/
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save directly as sequence_num.npy (not in subfolder)
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
