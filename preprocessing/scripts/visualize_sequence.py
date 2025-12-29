"""
Visualize keypoint sequences as video to verify extraction quality
"""
import numpy as np
import cv2
import sys
from pathlib import Path
import mediapipe as mp

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import SEQUENCE_PATH, COLORS, TOTAL_KEYPOINTS, POSE_LANDMARKS, FACE_LANDMARKS, HAND_LANDMARKS

# MediaPipe drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def draw_landmarks_from_keypoints(image, keypoints_array):
    """
    Draw landmarks on image from flattened keypoints array
    
    Args:
        image: Blank image to draw on
        keypoints_array: 1662-dimensional keypoints (pose+face+hands)
    
    Returns:
        Image with landmarks drawn
    """
    # Parse keypoints back to structured format
    pose_kp = keypoints_array[:POSE_LANDMARKS].reshape(33, 4)  # 33 × 4 (x,y,z,vis)
    face_kp = keypoints_array[POSE_LANDMARKS:POSE_LANDMARKS+FACE_LANDMARKS].reshape(468, 3)
    lh_kp = keypoints_array[POSE_LANDMARKS+FACE_LANDMARKS:POSE_LANDMARKS+FACE_LANDMARKS+HAND_LANDMARKS].reshape(21, 3)
    rh_kp = keypoints_array[POSE_LANDMARKS+FACE_LANDMARKS+HAND_LANDMARKS:].reshape(21, 3)
    
    h, w, _ = image.shape
    
    # Draw pose landmarks with config colors
    pose_color = COLORS['pose'][0]
    for landmark in pose_kp:
        if landmark[3] > 0.5:  # Check visibility
            x, y = int(landmark[0] * w), int(landmark[1] * h)
            cv2.circle(image, (x, y), 4, pose_color, -1)
    
    # Draw face landmarks with config colors
    face_color = COLORS['face'][0]
    for landmark in face_kp:
        if landmark[0] > 0:
            x, y = int(landmark[0] * w), int(landmark[1] * h)
            cv2.circle(image, (x, y), 1, face_color, -1)
    
    # Draw left hand with config colors
    lh_color = COLORS['left_hand'][0]
    for landmark in lh_kp:
        if landmark[0] > 0:
            x, y = int(landmark[0] * w), int(landmark[1] * h)
            cv2.circle(image, (x, y), 4, lh_color, -1)
    
    # Draw right hand with config colors
    rh_color = COLORS['right_hand'][0]
    for landmark in rh_kp:
        if landmark[0] > 0:
            x, y = int(landmark[0] * w), int(landmark[1] * h)
            cv2.circle(image, (x, y), 4, rh_color, -1)
    
    return image


def visualize_sequence(npy_file_path, output_video_path=None, fps=10, show_live=False):
    """
    Create video from .npy sequence file
    
    Args:
        npy_file_path: Path to .npy file
        output_video_path: Path to save video (optional)
        fps: Frames per second
        show_live: Display video in real-time (requires X11)
    """
    # Load sequence
    sequence = np.load(npy_file_path)
    print(f"Loaded sequence: {sequence.shape}")
    
    # Create visualizations directory
    npy_path = Path(npy_file_path)
    vis_dir = SEQUENCE_PATH.parent / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create video writer with organized naming
    if output_video_path is None:
        # Format: action_name_video_name.mp4
        action_name = npy_path.parent.name
        video_name = npy_path.stem
        output_video_path = str(vis_dir / f"{action_name}_{video_name}.mp4")
    
    # Video properties
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Creating video: {output_video_path}")
    if show_live:
        print("Press 'q' to stop live preview...")
    
    # Process each frame
    for frame_idx, keypoints in enumerate(sequence):
        # Create white background (255 = white)
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw landmarks
        image = draw_landmarks_from_keypoints(image, keypoints)
        
        # Add frame number (black text on white background)
        cv2.putText(image, f'Frame {frame_idx+1}/{len(sequence)}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Show live preview
        if show_live:
            cv2.imshow('Sequence Visualization - Press Q to stop', image)
            if cv2.waitKey(100) & 0xFF == ord('q'):  # 100ms delay between frames
                show_live = False
        
        # Write frame
        video_writer.write(image)
    
    video_writer.release()
    if show_live:
        cv2.destroyAllWindows()
    
    print(f"✓ Video saved: {output_video_path}")
    return output_video_path


def visualize_action(action_name, sequence_idx=0, show_live=False):
    """
    Visualize a specific sequence from an action
    
    Args:
        action_name: Name of the action folder
        sequence_idx: Index of sequence to visualize (or video filename without .npy)
        show_live: Display video in real-time (requires X11)
    """
    # Try as index first, then as filename
    if isinstance(sequence_idx, int) or sequence_idx.isdigit():
        # Index mode: find nth file
        npy_files = sorted((SEQUENCE_PATH / action_name).glob('*.npy'))
        if not npy_files:
            print(f"[ERROR] No .npy files found in {action_name}")
            return
        idx = int(sequence_idx)
        if idx >= len(npy_files):
            print(f"[ERROR] Index {idx} out of range (only {len(npy_files)} files)")
            return
        npy_path = npy_files[idx]
    else:
        # Filename mode
        npy_path = SEQUENCE_PATH / action_name / f"{sequence_idx}.npy"
    
    if not npy_path.exists():
        print(f"[ERROR] File not found: {npy_path}")
        return
    
    visualize_sequence(str(npy_path), show_live=show_live)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_sequence.py <action_name> [sequence_idx_or_filename] [--show]")
        print("  python visualize_sequence.py <path_to_npy_file> [--show]")
        print("\nOptions:")
        print("  --show    Display video in real-time (requires X11)")
        print("\nExamples:")
        print("  python visualize_sequence.py bad 0              # First video in 'bad' class")
        print("  python visualize_sequence.py bad MVI_5161       # Specific video by name")
        print("  python visualize_sequence.py bad 0 --show       # With live preview")
        print("\nOutput:")
        print("  Videos saved to: data/INCLUDE/visualizations/")
        print("  Naming format: <action>_<video>.mp4")
        sys.exit(1)
    
    # Check for --show flag
    show_live = '--show' in sys.argv
    
    if sys.argv[1].endswith('.npy'):
        # Direct .npy file path
        visualize_sequence(sys.argv[1], show_live=show_live)
    else:
        # Action name and sequence index/filename
        action_name = sys.argv[1]
        # Keep sequence_idx as string - visualize_action will handle int/string conversion
        sequence_idx = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '--show' else '0'
        visualize_action(action_name, sequence_idx, show_live=show_live)
