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
SEQUENCE_PATH = PROJECT_ROOT / "data" / "VSL_Isolated" / "sequences"  # Fixed: outside frames/
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
# Fixed sequence length (number of frames per sequence)
SEQUENCE_LENGTH = 130  # ~4.3 seconds @ 30fps

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
    'face': ((0, 255, 0), (0, 200, 0)),           # Bright green (face)
    'pose': ((0, 0, 255), (0, 0, 200)),           # Bright red (pose/body)
    'left_hand': ((255, 0, 0), (200, 0, 0)),      # Bright blue (left hand)
    'right_hand': ((255, 0, 255), (200, 0, 200))  # Bright magenta (right hand)
}

# ==================== MODEL & TRAINING CONFIG ====================
# Model architecture
MODEL_CONFIG = {
    'cnn_filters': [64, 128, 256],
    'cnn_dropout': 0.2,
    'lstm_units': [128, 64],
    'lstm_dropout': 0.3,
    'dense_units': 128,
    'dense_dropout': 0.5
}

# Training hyperparameters
TRAINING_CONFIG = {
    'batch_size': 8,  # Reduced from 32 to avoid OOM
    'epochs': 100,
    'learning_rate': 0.001,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 7,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15
}

# Paths
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

