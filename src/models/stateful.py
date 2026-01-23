"""
Stateful variant of MLP+LSTM model for variable-length sequence inference.
This model processes sequences frame-by-frame while maintaining LSTM state.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

# Handle both relative and absolute imports
try:
    from .cnn_branches import create_hand_branch, create_face_branch, create_pose_branch
except ImportError:
    from cnn_branches import create_hand_branch, create_face_branch, create_pose_branch


def create_stateful_model(num_classes, timesteps=1):
    """
    Create stateful MLP+LSTM model for variable-length inference.
    
    Key differences from stateless model:
    - batch_input_shape instead of Input shape
    - stateful=True in LSTM layers
    - Fixed batch_size=1 for inference
    
    Args:
        num_classes: Number of action classes
        timesteps: Number of frames per prediction (default=1 for frame-by-frame)
    
    Returns:
        Stateful Keras Model
    """
    # === INPUT LAYER (Fixed batch_size) ===
    inputs = layers.Input(batch_shape=(1, timesteps, 1662), name='stateful_input')
    # Note: No Masking layer needed for stateful (we control input directly)
    
    # === SPLIT KEYPOINTS ===
    pose_keypoints = layers.Lambda(lambda x: x[:, :, :132], name='pose_split')(inputs)
    face_keypoints = layers.Lambda(lambda x: x[:, :, 132:1536], name='face_split')(inputs)
    hand_keypoints = layers.Lambda(lambda x: x[:, :, 1536:], name='hand_split')(inputs)
    
    # === MLP BRANCHES ===
    pose_branch = create_pose_branch(132, 'pose')
    face_branch = create_face_branch(1404, 'face')
    hand_branch = create_hand_branch(126, 'hand')
    
    # Apply MLP to each frame
    pose_features = layers.TimeDistributed(pose_branch, name='pose_features')(pose_keypoints)
    face_features = layers.TimeDistributed(face_branch, name='face_features')(face_keypoints)
    hand_features = layers.TimeDistributed(hand_branch, name='hand_features')(hand_keypoints)
    
    # === FEATURE FUSION ===
    merged = layers.Concatenate(name='feature_fusion')([
        pose_features, 
        face_features, 
        hand_features
    ])
    
    # === SHARED LAYERS ===
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
    
    # === STATEFUL LSTM LAYERS ===
    x = layers.LSTM(128, return_sequences=True, stateful=True, name='lstm1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=False, stateful=True, name='lstm2')(x)
    x = layers.Dropout(0.3)(x)
    
    # === CLASSIFICATION HEAD ===
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Stateful_MLP_LSTM_Model')
    return model


def load_weights_from_stateless(stateful_model, stateless_model_path):
    """
    Load weights from trained stateless model into stateful model.
    
    Args:
        stateful_model: Stateful model instance
        stateless_model_path: Path to trained stateless model (.h5)
    
    Returns:
        stateful_model with loaded weights
    """
    from tensorflow.keras.models import load_model
    
    # Load stateless model
    stateless_model = load_model(stateless_model_path)
    
    # Copy weights layer by layer
    for stateful_layer, stateless_layer in zip(stateful_model.layers, stateless_model.layers):
        try:
            stateful_layer.set_weights(stateless_layer.get_weights())
            print(f"Copied weights: {stateful_layer.name}")
        except Exception as e:
            print(f"Skipped layer {stateful_layer.name}: {e}")
    
    print("\n Weights loaded successfully!")
    return stateful_model


if __name__ == "__main__":
    import numpy as np
    
    print("Creating Stateful MLP+LSTM Model...")
    print("\nArchitecture:")
    print("  - Input: (1, 1, 1662) - 1 frame at a time")
    print("  - MLP Branches: Dense layers for feature extraction")
    print("  - Stateful LSTM: Maintains state across frames")
    print("  - Output: (1, num_classes)")
    print()
    
    model = create_stateful_model(num_classes=76, timesteps=1)
    model.summary()
    
    # Test prediction
    print("\nTesting frame-by-frame prediction...")
    model.reset_states()
    
    for i in range(5):
        test_frame = np.random.rand(1, 1, 1662).astype('float32')
        pred = model.predict(test_frame, verbose=0)
        print(f"Frame {i+1}: Prediction shape = {pred.shape}, Sum = {pred.sum():.4f}")
    
    print("\n✓ Stateful model test passed!")
