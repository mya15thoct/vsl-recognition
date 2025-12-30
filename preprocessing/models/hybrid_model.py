"""
Hybrid Model: CNN + Transformer
Combines CNN spatial features with Transformer temporal modeling
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

from .cnn_branches import create_hand_branch, create_face_branch, create_pose_branch
from .transformer import PositionalEncoding, transformer_encoder_block


def create_hybrid_model(num_classes, sequence_length, 
                        num_transformer_blocks=2,
                        num_heads=4,
                        key_dim=64,
                        ff_dim=512,
                        dropout=0.3):
    """
    Hybrid CNN + Transformer Model
    
    Architecture:
    1. Split keypoints → pose/face/hands
    2. CNN branches (spatial features per frame)
    3. Feature fusion (256D)
    4. Positional encoding
    5. Transformer encoder blocks (temporal modeling)
    6. Global pooling
    7. Classification
    
    Args:
        num_classes: Number of action classes
        sequence_length: Frames per sequence
        num_transformer_blocks: Number of stacked Transformer blocks
        num_heads: Attention heads per block
        key_dim: Dimension per attention head
        ff_dim: Feed-forward hidden dimension
        dropout: Dropout rate
    
    Returns:
        Keras Model
    """
    # === INPUT ===
    inputs = layers.Input(shape=(sequence_length, 1662), name='input')
    
    # === SPLIT KEYPOINTS ===
    pose = layers.Lambda(lambda x: x[:, :, :132], name='pose_split')(inputs)
    face = layers.Lambda(lambda x: x[:, :, 132:1536], name='face_split')(inputs)
    hands = layers.Lambda(lambda x: x[:, :, 1536:], name='hand_split')(inputs)
    
    # === CNN BRANCHES (spatial features) ===
    pose_branch = create_pose_branch(132, 'pose')
    face_branch = create_face_branch(1404, 'face')
    hand_branch = create_hand_branch(126, 'hand')
    
    pose_feat = layers.TimeDistributed(pose_branch, name='pose_td')(pose)
    face_feat = layers.TimeDistributed(face_branch, name='face_td')(face)
    hand_feat = layers.TimeDistributed(hand_branch, name='hand_td')(hands)
    
    # === FEATURE FUSION ===
    merged = layers.Concatenate(name='fusion')([pose_feat, face_feat, hand_feat])
    # Shape: (batch, seq_len, 256)
    
    # === SHARED DENSE ===
    x = layers.TimeDistributed(layers.Dense(256, activation='relu'), name='shared1')(merged)
    x = layers.Dropout(dropout)(x)
    
    # === POSITIONAL ENCODING ===
    x = PositionalEncoding(sequence_length, 256, name='pos_enc')(x)
    
    # === TRANSFORMER ENCODER ===
    for i in range(num_transformer_blocks):
        x = transformer_encoder_block(
            x,
            num_heads=num_heads,
            key_dim=key_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f'transformer_{i+1}'
        )
    
    # === GLOBAL POOLING ===
    x = layers.GlobalAveragePooling1D(name='global_pool')(x)
    
    # === CLASSIFICATION ===
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='CNN_Transformer_Hybrid')


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    import numpy as np
    
    print("=" * 50)
    print("Hybrid Model Test")
    print("=" * 50)
    
    print("\n1. Creating model...")
    model = create_hybrid_model(num_classes=76, sequence_length=33)
    print("   ✓ Model created")
    
    print("\n2. Model summary:")
    model.summary()
    
    print("\n3. Testing forward pass...")
    x = np.random.rand(2, 33, 1662).astype(np.float32)
    y = model.predict(x, verbose=0)
    
    print(f"   Input:  {x.shape}")
    print(f"   Output: {y.shape}")
    print(f"   Sum:    {y[0].sum():.4f} (should be ~1.0)")
    
    if y.shape == (2, 76) and abs(y[0].sum() - 1.0) < 0.01:
        print("\n   ✓ All tests passed!")
    else:
        print("\n   ✗ Test failed!")
