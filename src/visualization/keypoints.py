"""
Visualize keypoints from .npy files with skeleton connections
"""
import numpy as np
import cv2
import argparse
from pathlib import Path

# MediaPipe connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (27, 29),
    (29, 31), (24, 26), (26, 28), (28, 30), (30, 32)
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]


def draw_keypoints_with_skeleton(keypoints, frame_idx, total_frames):
    """
    Draw keypoints with skeleton connections.
    
    Args:
        keypoints: Array of shape (1662,) - flattened keypoints
        frame_idx: Current frame index
        total_frames: Total number of frames
    
    Returns:
        Image with drawn keypoints and connections
    """
    # Create blank image
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    
    # Reshape keypoints
    # Pose: 0:132 (33 landmarks × 4)
    # Face: 132:1536 (468 landmarks × 3)
    # Hands: 1536:1662 (21×2 landmarks × 3)
    
    pose_kp = keypoints[:132].reshape(33, 4)[:, :2]  # Only x, y
    face_kp = keypoints[132:1536].reshape(468, 3)[:, :2]
    left_hand_kp = keypoints[1536:1599].reshape(21, 3)[:, :2]
    right_hand_kp = keypoints[1599:1662].reshape(21, 3)[:, :2]
    
    # Scale to image size
    h, w = img.shape[:2]
    pose_kp = (pose_kp * [w, h]).astype(int)
    face_kp = (face_kp * [w, h]).astype(int)
    left_hand_kp = (left_hand_kp * [w, h]).astype(int)
    right_hand_kp = (right_hand_kp * [w, h]).astype(int)
    
    # Draw pose connections
    for connection in POSE_CONNECTIONS:
        pt1 = tuple(pose_kp[connection[0]])
        pt2 = tuple(pose_kp[connection[1]])
        if all(pose_kp[connection[0]]) and all(pose_kp[connection[1]]):
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    
    # Draw pose points
    for point in pose_kp:
        if all(point):
            cv2.circle(img, tuple(point), 4, (0, 0, 255), -1)
    
    # Draw face points (no connections for simplicity)
    for point in face_kp:
        if all(point):
            cv2.circle(img, tuple(point), 1, (0, 255, 0), -1)
    
    # Draw left hand connections
    for connection in HAND_CONNECTIONS:
        pt1 = tuple(left_hand_kp[connection[0]])
        pt2 = tuple(left_hand_kp[connection[1]])
        if all(left_hand_kp[connection[0]]) and all(left_hand_kp[connection[1]]):
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)
    
    # Draw left hand points
    for point in left_hand_kp:
        if all(point):
            cv2.circle(img, tuple(point), 3, (255, 0, 255), -1)
    
    # Draw right hand connections
    for connection in HAND_CONNECTIONS:
        pt1 = tuple(right_hand_kp[connection[0]])
        pt2 = tuple(right_hand_kp[connection[1]])
        if all(right_hand_kp[connection[0]]) and all(right_hand_kp[connection[1]]):
            cv2.line(img, pt1, pt2, (255, 128, 0), 2)
    
    # Draw right hand points
    for point in right_hand_kp:
        if all(point):
            cv2.circle(img, tuple(point), 3, (255, 165, 0), -1)
    
    # Add frame info
    cv2.putText(img, f'Frame {frame_idx+1}/{total_frames}', 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    return img


def visualize_keypoints_sequence(npy_file, output_video='output.mp4', fps=30):
    """
    Visualize keypoints sequence from .npy file.
    
    Args:
        npy_file: Path to .npy file
        output_video: Output video path
        fps: Frames per second
    """
    # Load keypoints
    keypoints = np.load(npy_file)
    num_frames = len(keypoints)
    
    print(f"Loaded: {npy_file}")
    print(f"Frames: {num_frames}")
    print(f"Shape: {keypoints.shape}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (1280, 720))
    
    # Process each frame
    for i, frame_kp in enumerate(keypoints):
        img = draw_keypoints_with_skeleton(frame_kp, i, num_frames)
        out.write(img)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{num_frames} frames")
    
    out.release()
    print(f"\n✓ Video saved to: {output_video}")


def find_max_sequence_length(data_dir):
    """
    Find maximum sequence length in dataset.
    """
    data_path = Path(data_dir)
    max_frames = 0
    max_file = ''
    
    for action_folder in data_path.iterdir():
        if action_folder.is_dir():
            for npy_file in action_folder.glob('*.npy'):
                data = np.load(npy_file)
                num_frames = len(data)
                if num_frames > max_frames:
                    max_frames = num_frames
                    max_file = str(npy_file)
    
    print(f"Max sequence length: {max_frames}")
    print(f"File: {max_file}")
    return max_frames, max_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Path to .npy file to visualize')
    parser.add_argument('--output', default='keypoints_visualization.mp4', help='Output video path')
    parser.add_argument('--find-max', action='store_true', help='Find max sequence length')
    parser.add_argument('--data-dir', default='checkpoints', help='Data directory')
    
    args = parser.parse_args()
    
    if args.find_max:
        find_max_sequence_length(args.data_dir)
    elif args.file:
        visualize_keypoints_sequence(args.file, args.output)
    else:
        print("Usage:")
        print("  Find max: python visualize_keypoints.py --find-max")
        print("  Visualize: python visualize_keypoints.py --file path/to/file.npy")
