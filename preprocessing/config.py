"""
Configuration file for Sign Language Action Detection
"""
from pathlib import Path
import numpy as np

# ==================== PATHS ====================
# Base directories
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "VSL_Isolated" / "frames"

# Data paths
SEQUENCE_PATH = DATA_DIR / "sequences"
MODEL_PATH = BASE_DIR / "models"
LOGS_PATH = BASE_DIR / "logs"

# Create directories if they don't exist
SEQUENCE_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# ==================== MEDIAPIPE SETTINGS ====================
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# ==================== DATA SETTINGS ====================
# SEQUENCE_LENGTH is auto-detected from dataset (max length after trimming)
# This ensures we capture all available motion frames without loss

# Number of sequences to collect per action
NO_SEQUENCES = 30

# Trim frames at start/end (preparation and ending movements)
TRIM_START_FRAMES = 5  # Skip first 5 frames
TRIM_END_FRAMES = 5    # Skip last 5 frames

# Actions to detect (will be populated from data folder)
ACTIONS = np.array([])  # Will be set dynamically from VSL_Isolated folders

# ==================== MODEL SETTINGS ====================
# Model architecture
LSTM_UNITS = [64, 128, 64]
DENSE_UNITS = [64, 32]

# Training parameters
EPOCHS = 2000
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ==================== KEYPOINT DIMENSIONS ====================
# MediaPipe outputs
POSE_LANDMARKS = 33 * 4  # 33 landmarks × (x, y, z, visibility) = 132
FACE_LANDMARKS = 468 * 3  # 468 landmarks × (x, y, z) = 1404  
HAND_LANDMARKS = 21 * 3   # 21 landmarks × (x, y, z) per hand = 63

TOTAL_KEYPOINTS = POSE_LANDMARKS + FACE_LANDMARKS + (HAND_LANDMARKS * 2)  # 1662

# ==================== COLORS FOR VISUALIZATION ====================
COLORS = {
    'face': ((80, 110, 10), (80, 256, 121)),
    'pose': ((80, 22, 10), (80, 44, 121)),
    'left_hand': ((121, 22, 76), (121, 44, 250)),
    'right_hand': ((245, 117, 66), (245, 66, 230))
}
