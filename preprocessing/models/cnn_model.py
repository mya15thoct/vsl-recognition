"""
CNN-Only Model (No temporal modeling)
Simple baseline: CNN spatial features + Global pooling + Classification
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

from .cnn_branches import create_hand_branch, create_face_branch, create_pose_branch


def create_cnn_model(num_classes, sequence_length, dropout=0.3):
    """
    CNN-Only Model for sign language recognition
    
    Architecture:
    1. Split keypoints → pose/face/hands  
    2. CNN branches extract spatial features per frame
    3. Concatenate features (256D)
    4. Global Average Pooling over time dimension
    5. Classification layers
    
    NO LSTM, NO Transformer - pure CNN baseline
    
    Args:
        num_classes: Number of action classes
        sequence_length: Number of frames per sequence
        dropout: Dropout rate
    
    Returns:
        Keras Model
    """
    # === INPUT ===
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    
    # === SPLIT KEYPOINTS ===
    # Pose: 0:132 (33 landmarks × 4 coords)
    # Face: 132:1536 (468 landmarks × 3 coords)  
    # Hands: 1536:1662 (21×2 landmarks × 3 coords)
    pose_keypoints = layers.Lambda(lambda x: x[:, :, :132], name='pose_split')(inputs)
    face_keypoints = layers.Lambda(lambda x: x[:, :, 132:1536], name='face_split')(inputs)
    hand_keypoints = layers.Lambda(lambda x: x[:, :, 1536:], name='hand_split')(inputs)
    
    # === CNN BRANCHES (per-frame spatial features) ===
    pose_branch = create_pose_branch(132, 'pose')     # → 64 features
    face_branch = create_face_branch(1404, 'face')    # → 128 features
    hand_branch = create_hand_branch(126, 'hand')     # → 64 features
    
    # TimeDistributed applies CNN to each frame independently
    pose_features = layers.TimeDistributed(pose_branch, name='pose_td')(pose_keypoints)
    face_features = layers.TimeDistributed(face_branch, name='face_td')(face_keypoints)
    hand_features = layers.TimeDistributed(hand_branch, name='hand_td')(hand_keypoints)
    
    # === FEATURE FUSION ===
    # Concatenate: (64 + 128 + 64) = 256 features per frame
    merged = layers.Concatenate(name='feature_fusion')([
        pose_features, 
        face_features, 
        hand_features
    ])
    # Shape: (batch, sequence_length, 256)
    
    # === SHARED DENSE LAYERS ===
    x = layers.TimeDistributed(
        layers.Dense(256, activation='relu'), name='shared_td1'
    )(merged)
    x = layers.Dropout(dropout)(x)
    
    x = layers.TimeDistributed(
        layers.Dense(128, activation='relu'), name='shared_td2'
    )(x)
    x = layers.Dropout(dropout)(x)
    # Shape: (batch, sequence_length, 128)
    
    # === GLOBAL POOLING (aggregate over time) ===
    # Average all frames into single feature vector
    x = layers.GlobalAveragePooling1D(name='global_pool')(x)
    # Shape: (batch, 128)
    
    # === CLASSIFICATION HEAD ===
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_Only_Model')
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("CNN-Only Model Test")
    print("=" * 60)
    
    import numpy as np
    
    # Create model
    print("\n1. Creating model...")
    model = create_cnn_model(num_classes=76, sequence_length=33)
    print("   ✓ Model created")
    
    # Summary
    print("\n2. Model summary:")
    model.summary()
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    test_input = np.random.rand(2, 33, 1662).astype(np.float32)
    test_output = model.predict(test_input, verbose=0)
    
    print(f"   Input shape:  {test_input.shape}")
    print(f"   Output shape: {test_output.shape}")
    print(f"   Output sum:   {test_output[0].sum():.4f} (should be ~1.0)")
    
    if test_output.shape == (2, 76) and abs(test_output[0].sum() - 1.0) < 0.01:
        print("\n   ✓ All tests passed!")
    else:
        print("\n   ✗ Test failed!")
