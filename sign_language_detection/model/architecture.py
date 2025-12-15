"""
Model architecture for Sign Language Action Detection
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import LSTM_UNITS, DENSE_UNITS, TOTAL_KEYPOINTS


def build_model(num_actions, lstm_units=None, dense_units=None):
    """
    Build LSTM model for action detection
    
    Args:
        num_actions: Number of action classes to predict
        lstm_units: List of LSTM layer units (default from config)
        dense_units: List of Dense layer units (default from config)
        
    Returns:
        Compiled Keras model
    """
    if lstm_units is None:
        lstm_units = LSTM_UNITS
    if dense_units is None:
        dense_units = DENSE_UNITS
    
    model = Sequential()
    
    # LSTM layers
    model.add(LSTM(lstm_units[0], return_sequences=True, activation='relu', 
                   input_shape=(30, TOTAL_KEYPOINTS)))
    model.add(LSTM(lstm_units[1], return_sequences=True, activation='relu'))
    model.add(LSTM(lstm_units[2], return_sequences=False, activation='relu'))
    
    # Dense layers
    model.add(Dense(dense_units[0], activation='relu'))
    model.add(Dense(dense_units[1], activation='relu'))
    
    # Output layer
    model.add(Dense(num_actions, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model


def get_tensorboard_callback(log_dir):
    """
    Create TensorBoard callback
    
    Args:
        log_dir: Directory for TensorBoard logs
        
    Returns:
        TensorBoard callback
    """
    return TensorBoard(log_dir=str(log_dir))
