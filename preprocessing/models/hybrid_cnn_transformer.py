"""
Hybrid CNN + Transformer Model
Uses CNN branches for spatial features + Transformer for temporal modeling
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

from .transformer import PositionalEncoding, transformer_encoder_block


def create_hand_branch(input_dim=126, name_prefix='hand'):
    """Hand branch: 3 layers → 64 features"""
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
    """Face branch: 4 layers → 128 features"""
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
    """Pose branch: 2 layers → 64 features"""
    inputs = layers.Input(shape=(input_dim,), name=f'{name_prefix}_input')
    x = layers.Dense(128, activation='relu', name=f'{name_prefix}_dense1')(inputs)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu', name=f'{name_prefix}_dense2')(x)
    return Model(inputs=inputs, outputs=x, name=f'{name_prefix}_branch')


def create_cnn_transformer_model(num_classes, sequence_length,
                                  num_transformer_blocks=2,
                                  num_heads=4,
                                  key_dim=64,
                                  ff_dim=512,
                                  dropout=0.3):
    """
    CNN + Transformer Hybrid Model
    
    Architecture:
    1. Split keypoints → pose/face/hands
    2. CNN branches (spatial features)
    3. Feature fusion (256D)
    4. Positional encoding
    5. Transformer encoder (temporal modeling)
    6. Global pooling
    7. Classification
    """
    # === INPUT ===
    inputs = layers.Input(shape=(sequence_length, 1662), name='input')
    
    # === SPLIT KEYPOINTS ===
    pose = layers.Lambda(lambda x: x[:, :, :132], name='pose_split')(inputs)
    face = layers.Lambda(lambda x: x[:, :, 132:1536], name='face_split')(inputs)
    hands = layers.Lambda(lambda x: x[:, :, 1536:], name='hand_split')(inputs)
    
    # === CNN BRANCHES ===
    pose_branch = create_pose_branch(132, 'pose')
    face_branch = create_face_branch(1404, 'face')
    hand_branch = create_hand_branch(126, 'hand')
    
    pose_feat = layers.TimeDistributed(pose_branch, name='pose_td')(pose)
    face_feat = layers.TimeDistributed(face_branch, name='face_td')(face)
    hand_feat = layers.TimeDistributed(hand_branch, name='hand_td')(hands)
    
    # === FEATURE FUSION ===
    merged = layers.Concatenate(name='fusion')([pose_feat, face_feat, hand_feat])
    
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
    
    return Model(inputs=inputs, outputs=outputs, name='CNN_Transformer_Model')


if __name__ == "__main__":
    import numpy as np
    print("Testing CNN + Transformer Model...")
    model = create_cnn_transformer_model(num_classes=76, sequence_length=33)
    model.summary()
    
    x = np.random.rand(2, 33, 1662).astype(np.float32)
    y = model.predict(x, verbose=0)
    print(f"\nInput: {x.shape}, Output: {y.shape}")
    print(f"✓ Test passed!")
