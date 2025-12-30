"""
Hybrid Model: CNN + Transformer
- CNN branches for spatial feature extraction (pose, face, hands)
- Transformer encoder for temporal modeling
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

from .cnn_branches import create_hand_branch, create_face_branch, create_pose_branch
from .transformer_encoder import PositionalEncoding, transformer_encoder_block


def create_hybrid_model(num_classes, sequence_length, 
                       num_transformer_blocks=1,
                       num_heads=4, 
                       head_size=64,
                       ff_dim=512,
                       dropout=0.3):
    """
    Hybrid CNN + Transformer architecture:
    
    1. SPATIAL FEATURE EXTRACTION (CNN)
       - Specialized branches for pose/face/hands (varying depth)
       - Feature fusion
       - Shared layers for cross-part interaction
    
    2. TEMPORAL MODELING (Transformer)
       - Positional encoding
       - Transformer encoder blocks (multi-head attention)
       - Global pooling
    
    3. CLASSIFICATION
       - Dense layers + softmax
    
    Args:
        num_classes: Number of action classes
        sequence_length: Frames per sequence
        num_transformer_blocks: Number of stacked transformer blocks
        num_heads: Number of attention heads
        head_size: Dimension per attention head
        ff_dim: Feed-forward network hidden dimension
        dropout: Dropout rate
    
    Returns:
        Keras Model
    """
    # === INPUT LAYER ===
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    masked_input = layers.Masking(mask_value=0.0)(inputs)
    
    # Compute attention mask for Transformer
    # Mask shape: (batch, seq_len) -> padded positions = False, real positions = True
    mask = tf.keras.backend.not_equal(
        tf.reduce_sum(tf.abs(inputs), axis=-1), 0.0
    )
    # Expand for attention: (batch, 1, 1, seq_len)
    attention_mask = mask[:, tf.newaxis, tf.newaxis, :]
    
    x = masked_input
    
    # === SPLIT KEYPOINTS ===
    # Pose: 0:132 (33 × 4)
    # Face: 132:1536 (468 × 3) 
    # Hands: 1536:1662 (21×2 × 3)
    pose_keypoints = layers.Lambda(lambda x: x[:, :, :132], name='pose_split')(x)
    face_keypoints = layers.Lambda(lambda x: x[:, :, 132:1536], name='face_split')(x)
    hand_keypoints = layers.Lambda(lambda x: x[:, :, 1536:], name='hand_split')(x)
    
    # === CNN BRANCHES (Spatial Feature Extraction) ===
    pose_branch = create_pose_branch(132, 'pose')     # 2 layers → 64 features
    face_branch = create_face_branch(1404, 'face')    # 4 layers → 128 features
    hand_branch = create_hand_branch(126, 'hand')     # 3 layers → 64 features
    
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
    
    # === SHARED LAYERS (Cross-part Interaction) ===
    x = layers.TimeDistributed(
        layers.Dense(256, activation='relu', name='shared_interaction1'),
        name='shared_td1'
    )(merged)
    x = layers.Dropout(dropout)(x)
    
    x = layers.TimeDistributed(
        layers.Dense(256, activation='relu', name='shared_interaction2'),
        name='shared_td2'
    )(x)
    x = layers.Dropout(dropout)(x)
    
    # === TRANSFORMER ENCODER (Temporal Modeling) ===
    # Add positional information
    x = PositionalEncoding(sequence_length, 256, name='pos_encoding')(x)
    
    # Stack Transformer blocks with attention mask
    for i in range(num_transformer_blocks):
        x = transformer_encoder_block(
            x,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            mask=attention_mask,
            name_prefix=f'transformer_block{i+1}'
        )
    
    # === GLOBAL POOLING ===
    x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # === CLASSIFICATION HEAD ===
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Hybrid_CNN_Transformer')
    return model


if __name__ == "__main__":
    print("Creating Hybrid CNN + Transformer Model...")
    print("\nArchitecture:")
    print("  1. CNN Spatial Features:")
    print("     - Hand:  3 layers (deep) → 64 features")
    print("     - Face:  4 layers (deep) → 128 features")
    print("     - Pose:  2 layers (shallow) → 64 features")
    print("  2. Feature Fusion: 256 dimensions")
    print("  3. Transformer Temporal Modeling:")
    print("     - Positional Encoding")
    print("     - 2 Transformer blocks (Multi-head Attention)")
    print("  4. Classification")
    print()
    
    model = create_hybrid_model(num_classes=76, sequence_length=33)
    model.summary()
    
    # Test
    print("\nTesting prediction...")
    test_input = np.random.rand(2, 33, 1662)
    test_output = model.predict(test_input, verbose=0)
    
    print(f"\nTest passed ✓")
    print(f"Input:  {test_input.shape}")
    print(f"Output: {test_output.shape}")
    print(f"Probabilities sum: {test_output[0].sum():.4f}")
