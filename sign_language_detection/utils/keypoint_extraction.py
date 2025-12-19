"""
Utility functions for keypoint extraction using MediaPipe
"""
import cv2
import numpy as np
import mediapipe as mp
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    """
    Perform MediaPipe detection on image
    
    Args:
        image: BGR image from OpenCV
        model: MediaPipe Holistic model instance
        
    Returns:
        image: RGB image
        results: MediaPipe detection results
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
    return image, results


def extract_keypoints(results):
    """
    Extract keypoint values from MediaPipe results
    
    Args:
        results: MediaPipe Holistic results
        
    Returns:
        Concatenated numpy array of all keypoints (1662 values)
    """
    # Pose: 33 landmarks × 4 values (x, y, z, visibility) = 132
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*4)
    
    # Face: 468 landmarks × 3 values (x, y, z) = 1404
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468*3)
    
    # Left hand: 21 landmarks × 3 values = 63
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    
    # Right hand: 21 landmarks × 3 values = 63
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])


def get_holistic_model(min_detection_confidence=None, min_tracking_confidence=None):
    """
    Create and return MediaPipe Holistic model
    
    Args:
        min_detection_confidence: Minimum detection confidence (default from config)
        min_tracking_confidence: Minimum tracking confidence (default from config)
        
    Returns:
        MediaPipe Holistic model instance
    """
    if min_detection_confidence is None:
        min_detection_confidence = MP_MIN_DETECTION_CONFIDENCE
    if min_tracking_confidence is None:
        min_tracking_confidence = MP_MIN_TRACKING_CONFIDENCE
    
    return mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
