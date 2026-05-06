"""
Configuration file for Sign Language Action Detection
"""
import os
from pathlib import Path

# ==================== PATHS ====================
DATA_DIR        = Path("/mnt/ngan/vsl_data")          # Raw videos (read-only)
IMAGE_DIR       = Path("/mnt/ngan/ISL-Frames-Data")   # Static image frames (read-only)
RECOGNITION_DIR = Path("/mnt/ngan/recognition")        # Word model base — READ ONLY

# ISL-Sequences: ALL isl-sentence outputs go here
ISL_SEQ_DIR = Path("/mnt/ngan/ISL-Sequences")

SEQUENCE_PATH     = RECOGNITION_DIR / "sequences"        # VSL word video keypoints (read-only)
ISL_SEQUENCE_PATH = ISL_SEQ_DIR / "word"                 # ISL image keypoints (per-word)

# Combined model outputs → ISL-Sequences (NOT recognition/)
ISL_CHECKPOINT_DIR = ISL_SEQ_DIR / "checkpoints"
ISL_LOGS_DIR       = ISL_SEQ_DIR / "logs"

# Create only ISL-Sequences subdirs
ISL_SEQUENCE_PATH.mkdir(parents=True, exist_ok=True)
ISL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ISL_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== MEDIAPIPE SETTINGS ====================
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# ==================== DATA SETTINGS ====================
SEQUENCE_LENGTH = None  # No fixed length - use all frames, pad at load time

# ==================== KEYPOINT DIMENSIONS ====================
POSE_LANDMARKS = 33 * 4   # 33 landmarks × (x, y, z, visibility) = 132
FACE_LANDMARKS = 468 * 3  # 468 landmarks × (x, y, z)             = 1404
HAND_LANDMARKS = 21 * 3   # 21 landmarks  × (x, y, z) per hand    = 63

TOTAL_KEYPOINTS = POSE_LANDMARKS + FACE_LANDMARKS + (HAND_LANDMARKS * 2)  # 1662

# ==================== COLORS FOR VISUALIZATION ====================
COLORS = {
    'face':       ((0, 255, 0),   (0, 200, 0)),
    'pose':       ((0, 0, 255),   (0, 0, 200)),
    'left_hand':  ((255, 0, 0),   (200, 0, 0)),
    'right_hand': ((255, 0, 255), (200, 0, 200))
}

# ==================== MODEL SELECTION ====================
# 'mlp'         → MLP branches + Cross-Part Gating + BiLSTM + Attention (hybrid.py)
# 'transformer' → Transformer Encoder branches + same downstream (transformer/model.py)
# Override via environment variable: MODEL_TYPE=transformer python main.py train
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'mlp')

# Base word model checkpoint/log dirs (READ ONLY — already trained)
CHECKPOINT_DIR = RECOGNITION_DIR / 'checkpoints' / MODEL_TYPE
LOGS_DIR       = RECOGNITION_DIR / 'logs'        / MODEL_TYPE

# ==================== TRAINING CONFIG ====================
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 500,
    'learning_rate': 0.001,
    'early_stopping_patience': 40,
    'reduce_lr_patience': 15,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15
}

