"""
Hybrid MLP+Bidirectional LSTM Model for Sign Language Recognition
- MLP (Dense) branches for feature extraction from each body part
- Bidirectional LSTM layers for temporal modeling (forward + backward)
- Multi-stream architecture with specialized branches
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

# Handle both relative and absolute imports
try:
    from .components import create_hand_branch, create_face_branch, create_pose_branch
except ImportError:
    from components import create_hand_branch, create_face_branch, create_pose_branch


def create_hybrid_multistream_model(num_classes, sequence_length):
    """
    Hybrid MLP+LSTM architecture:
    1. MLP (Dense) branches for feature extraction (varying depth based on importance)
    2. Feature fusion across body parts
    3. Shared Dense layers for cross-part interaction learning
    4. LSTM layers for temporal modeling
    
    Args:
        num_classes: Number of action classes
        sequence_length: Frames per sequence
    
    Returns:
        Keras Model
    """
    # === INPUT LAYER ===
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # === SPLIT KEYPOINTS ===
    # Pose: 0:132 (33 × 4)
    # Face: 132:1536 (468 × 3) 
    # Hands: 1536:1662 (21×2 × 3)
    pose_keypoints = layers.Lambda(lambda x: x[:, :, :132], name='pose_split')(x)
    face_keypoints = layers.Lambda(lambda x: x[:, :, 132:1536], name='face_split')(x)
    hand_keypoints = layers.Lambda(lambda x: x[:, :, 1536:], name='hand_split')(x)
    
    # === MLP BRANCHES (varying depth) ===
    pose_branch = create_pose_branch(132, 'pose')     # 2 Dense layers (shallow)
    face_branch = create_face_branch(1404, 'face')    # 4 Dense layers (deep)
    hand_branch = create_hand_branch(126, 'hand')     # 3 Dense layers (deep)
    
    # Apply MLP to each frame
    pose_features = layers.TimeDistributed(pose_branch, name='pose_features')(pose_keypoints)
    face_features = layers.TimeDistributed(face_branch, name='face_features')(face_keypoints)
    hand_features = layers.TimeDistributed(hand_branch, name='hand_features')(hand_keypoints)
    
    # === FEATURE FUSION ===
    # Concatenate: (64 + 128 + 64) = 256 dimensions
    merged = layers.Concatenate(name='feature_fusion')([
        pose_features, 
        face_features, 
        hand_features
    ])
    
    # === SHARED LAYERS (learn cross-part interactions) ===
    # These layers see all parts together and learn their relationships
    x = layers.TimeDistributed(
        layers.Dense(256, activation='relu', name='shared_interaction1'),
        name='shared_td1'
    )(merged)
    x = layers.Dropout(0.3)(x)
    
    x = layers.TimeDistributed(
        layers.Dense(128, activation='relu', name='shared_interaction2'),
        name='shared_td2'
    )(x)
    x = layers.Dropout(0.3)(x)
    
    # === TEMPORAL MODELING (Bidirectional LSTM) ===
    # Each LSTM uses half the units; forward+backward concatenated = same output dim
    # lstm1: 64×2 = 128 dims, lstm2: 32×2 = 64 dims
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name='bilstm1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False), name='bilstm2')(x)
    x = layers.Dropout(0.3)(x)
    
    # === CLASSIFICATION HEAD ===
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='MLP_BiLSTM_Model')
    return model


if __name__ == "__main__":
    import numpy as np
    
    print("Creating MLP+LSTM Model...")
    print("\nArchitecture:")
    print("  MLP Branches (Feature Extraction):")
    print("    - Hand:  3 Dense layers (deep) → 64 features")
    print("    - Face:  4 Dense layers (deep) → 128 features")
    print("    - Pose:  2 Dense layers (shallow) → 64 features")
    print("  Shared Dense Layers:")
    print("    - 2 layers for cross-part interaction learning")
    print("  Temporal Modeling:")
    print("    - 2 Bidirectional LSTM layers (64×2=128 → 32×2=64)")
    print()
    
    model = create_hybrid_multistream_model(num_classes=76, sequence_length=33)
    model.summary()
    
    # Test
    print("\nTesting prediction...")
    test_input = np.random.rand(2, 33, 1662)
    test_output = model.predict(test_input, verbose=0)
    
    print(f"\nTest passed ")
    print(f"Input:  {test_input.shape}")
    print(f"Output: {test_output.shape}")
    print(f"Probabilities sum: {test_output[0].sum():.4f}")
