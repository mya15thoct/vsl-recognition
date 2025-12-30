"""
Hybrid Multi-stream CNN model
- Specialized branches for each body part (varying depth)
- Shared fusion layers for cross-part interactions
- Best of both worlds: specialization + interaction learning
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


def create_hand_branch(input_dim=126, name_prefix='hand'):
    """
    DEEP branch for hands (most important for sign language)
    126 dims → 64 features (3 layers)
    """
    inputs = layers.Input(shape=(input_dim,), name=f'{name_prefix}_input')
    
    x = layers.Dense(256, activation='relu', name=f'{name_prefix}_dense1')(inputs)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='relu', name=f'{name_prefix}_dense2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='relu', name=f'{name_prefix}_dense3')(x)
    
    return Model(inputs=inputs, outputs=x, name=f'{name_prefix}_branch')


def create_face_branch(input_dim=1404, name_prefix='face'):
    """
    DEEP branch for face (important for expressions)
    1404 dims → 128 features (4 layers - face has many points)
    """
    inputs = layers.Input(shape=(input_dim,), name=f'{name_prefix}_input')
    
    x = layers.Dense(512, activation='relu', name=f'{name_prefix}_dense1')(inputs)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(256, activation='relu', name=f'{name_prefix}_dense2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='relu', name=f'{name_prefix}_dense3')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn3')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='relu', name=f'{name_prefix}_dense4')(x)
    
    return Model(inputs=inputs, outputs=x, name=f'{name_prefix}_branch')


def create_pose_branch(input_dim=132, name_prefix='pose'):
    """
    SHALLOW branch for pose (less important, fewer points)
    132 dims → 64 features (2 layers)
    """
    inputs = layers.Input(shape=(input_dim,), name=f'{name_prefix}_input')
    
    x = layers.Dense(128, activation='relu', name=f'{name_prefix}_dense1')(inputs)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='relu', name=f'{name_prefix}_dense2')(x)
    
    return Model(inputs=inputs, outputs=x, name=f'{name_prefix}_branch')


def create_hybrid_multistream_model(num_classes, sequence_length):
    """
    Hybrid architecture:
    1. Specialized branches (varying depth based on importance)
    2. Feature fusion
    3. Shared layers for cross-part interaction learning
    4. LSTM for temporal modeling
    
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
    
    # === SPECIALIZED BRANCHES (varying depth) ===
    pose_branch = create_pose_branch(132, 'pose')     # 2 layers (shallow)
    face_branch = create_face_branch(1404, 'face')    # 4 layers (deep)
    hand_branch = create_hand_branch(126, 'hand')     # 3 layers (deep)
    
    # Apply to each frame
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
    
    # === TEMPORAL MODELING (LSTM) ===
    x = layers.LSTM(128, return_sequences=True, name='lstm1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=False, name='lstm2')(x)
    x = layers.Dropout(0.3)(x)
    
    # === CLASSIFICATION HEAD ===
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='HybridMultiStreamModel')
    return model


if __name__ == "__main__":
    import numpy as np
    
    print("Creating Hybrid Multi-Stream Model...")
    print("\nArchitecture:")
    print("  Specialized Branches:")
    print("    - Hand:  3 layers (deep) → 64 features")
    print("    - Face:  4 layers (deep) → 128 features")
    print("    - Pose:  2 layers (shallow) → 64 features")
    print("  Shared Layers:")
    print("    - 2 layers for cross-part interaction learning")
    print("  Temporal:")
    print("    - 2 LSTM layers (128 → 64)")
    print()
    
    model = create_hybrid_multistream_model(num_classes=76, sequence_length=33)
    model.summary()
    
    # Test
    print("\nTesting prediction...")
    test_input = np.random.rand(2, 33, 1662)
    test_output = model.predict(test_input, verbose=0)
    
    print(f"\nTest passed ✓")
    print(f"Input:  {test_input.shape}")
    print(f"Output: {test_output.shape}")
    print(f"Probabilities sum: {test_output[0].sum():.4f}")
