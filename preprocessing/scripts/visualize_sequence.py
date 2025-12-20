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


def visualize_sequence(npy_file_path, output_video_path=None, fps=10):
    """
    Create video from .npy sequence file
    
    Args:
        npy_file_path: Path to .npy file
        output_video_path: Path to save video (optional)
        fps: Frames per second
    """
    # Load sequence
    sequence = np.load(npy_file_path)
    print(f"Loaded sequence: {sequence.shape}")
    
    # Create video writer
    if output_video_path is None:
        output_video_path = str(Path(npy_file_path).parent / "visualization.mp4")
    
    # Video properties
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Creating video: {output_video_path}")
    
    # Process each frame
    for frame_idx, keypoints in enumerate(sequence):
        # Create blank image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw landmarks
        image = draw_landmarks_from_keypoints(image, keypoints)
        
        # Add frame number
        cv2.putText(image, f"Frame {frame_idx+1}/{len(sequence)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        video_writer.write(image)
    
    video_writer.release()
    print(f"✓ Video saved: {output_video_path}")
    return output_video_path


def visualize_action(action_name, sequence_idx=0):
    """
    Visualize a specific sequence from an action
    
    Args:
        action_name: Name of the action folder
        sequence_idx: Index of sequence to visualize
    """
    npy_path = SEQUENCE_PATH / action_name / str(sequence_idx) / f"{sequence_idx}.npy"
    
    if not npy_path.exists():
        print(f"[ERROR] File not found: {npy_path}")
        return
    
    output_path = SEQUENCE_PATH / action_name / f"sequence_{sequence_idx}_visualization.mp4"
    visualize_sequence(str(npy_path), str(output_path))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_sequence.py <action_name> [sequence_idx]")
        print("  python visualize_sequence.py <path_to_npy_file>")
        print("\nExample:")
        print("  python visualize_sequence.py videos 0")
        sys.exit(1)
    
    if sys.argv[1].endswith('.npy'):
        # Direct .npy file path
        visualize_sequence(sys.argv[1])
    else:
        # Action name and sequence index
        action_name = sys.argv[1]
        sequence_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        visualize_action(action_name, sequence_idx)
