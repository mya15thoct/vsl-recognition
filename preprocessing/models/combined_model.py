"""
Combined CNN+LSTM model for sign language recognition
"""
import tensorflow as tf
from tensorflow.keras import layers
from .cnn_model import create_spatial_cnn


def create_sign_language_model(num_classes, sequence_length=130, keypoint_dim=1662):
    """
    End-to-end CNN+LSTM model
    
    Args:
        num_classes: Number of action classes (81)
        sequence_length: Frames per sequence (130)
        keypoint_dim: Keypoints per frame (1662)
    
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(sequence_length, keypoint_dim), name='sequence_input')
    
    # CNN: Extract spatial features
    cnn = create_spatial_cnn(input_dim=keypoint_dim, output_dim=256)
    
    # Apply CNN to each frame using TimeDistributed
    # (130, 1662) -> (130, 256)
    x = layers.TimeDistributed(cnn, name='spatial_features')(inputs)
    
    # LSTM: Learn temporal patterns
    # (130, 256) -> (64,)
    x = layers.LSTM(128, return_sequences=True, name='lstm1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=False, name='lstm2')(x)
    x = layers.Dropout(0.3)(x)
    
    # Classification head
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SignLanguageModel')
    return model


if __name__ == "__main__":
    import numpy as np
    
    # Test
    print("Creating Sign Language Model...")
    model = create_sign_language_model(num_classes=81)
    model.summary()
    
    # Test prediction
    print("\nTesting prediction...")
    test_input = np.random.rand(2, 130, 1662)
    test_output = model.predict(test_input, verbose=0)
    
    print(f"\nTest passed")
    print(f"Input shape:  {test_input.shape}")   # (2, 130, 1662)
    print(f"Output shape: {test_output.shape}")  # (2, 81)
    print(f"Sum of probabilities: {test_output[0].sum():.4f}")  # Should be ~1.0
