"""
CNN Branches for Spatial Feature Extraction
Specialized CNN branches for different body parts (pose, face, hands)
Uses Conv1D layers to extract spatial patterns from keypoint sequences
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


def create_hand_branch(input_dim=126, name_prefix='hand'):
    """
    DEEP CNN branch for hands (most important for sign language)
    126 dims → 64 features (3 Conv1D layers)
    
    Args:
        input_dim: Input dimension (21 landmarks × 2 hands × 3 coords = 126)
        name_prefix: Prefix for layer names
    
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_dim,), name=f'{name_prefix}_input')
    
    # Reshape to (features, 1) for Conv1D
    x = layers.Reshape((input_dim, 1))(inputs)
    
    # Conv1D layers for spatial feature extraction
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', 
                      name=f'{name_prefix}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu',
                      name=f'{name_prefix}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu',
                      name=f'{name_prefix}_conv3')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn3')(x)
    
    # Global pooling to get fixed-size output
    x = layers.GlobalAveragePooling1D(name=f'{name_prefix}_gap')(x)
    
    # Final dense layer to get 64 features
    x = layers.Dense(64, activation='relu', name=f'{name_prefix}_dense')(x)
    
    return Model(inputs=inputs, outputs=x, name=f'{name_prefix}_branch')


def create_face_branch(input_dim=1404, name_prefix='face'):
    """
    DEEP CNN branch for face (important for expressions)
    1404 dims → 128 features (4 Conv1D layers - face has many points)
    
    Args:
        input_dim: Input dimension (468 landmarks × 3 coords = 1404)
        name_prefix: Prefix for layer names
    
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_dim,), name=f'{name_prefix}_input')
    
    # Reshape to (features, 1) for Conv1D
    x = layers.Reshape((input_dim, 1))(inputs)
    
    # Conv1D layers for spatial feature extraction
    x = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu',
                      name=f'{name_prefix}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu',
                      name=f'{name_prefix}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu',
                      name=f'{name_prefix}_conv3')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn3')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu',
                      name=f'{name_prefix}_conv4')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn4')(x)
    
    # Global pooling to get fixed-size output
    x = layers.GlobalAveragePooling1D(name=f'{name_prefix}_gap')(x)
    
    # Final dense layer to get 128 features
    x = layers.Dense(128, activation='relu', name=f'{name_prefix}_dense')(x)
    
    return Model(inputs=inputs, outputs=x, name=f'{name_prefix}_branch')


def create_pose_branch(input_dim=132, name_prefix='pose'):
    """
    SHALLOW CNN branch for pose (less important, fewer points)
    132 dims → 64 features (2 Conv1D layers)
    
    Args:
        input_dim: Input dimension (33 landmarks × 4 coords = 132)
        name_prefix: Prefix for layer names
    
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_dim,), name=f'{name_prefix}_input')
    
    # Reshape to (features, 1) for Conv1D
    x = layers.Reshape((input_dim, 1))(inputs)
    
    # Conv1D layers for spatial feature extraction
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                      name=f'{name_prefix}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu',
                      name=f'{name_prefix}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    
    # Global pooling to get fixed-size output
    x = layers.GlobalAveragePooling1D(name=f'{name_prefix}_gap')(x)
    
    # Final dense layer to get 64 features
    x = layers.Dense(64, activation='relu', name=f'{name_prefix}_dense')(x)
    
    return Model(inputs=inputs, outputs=x, name=f'{name_prefix}_branch')
