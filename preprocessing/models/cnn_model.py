"""
1D CNN extract spatial features from keypoints of a single frame
"""
import tensorflow as tf
from tensorflow.keras import layers


def create_spatial_cnn(input_dim=1662, output_dim=256):
    """
    CNN extract spatial features from single frame keypoints
    
    Args:
        input_dim: Number of keypoint features (1662)
        output_dim: Number of output features (256)
    
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_dim,), name='keypoint_input')
    
    # Reshape: (1662,) -> (1662, 1) for Conv1D
    x = layers.Reshape((input_dim, 1))(inputs)
    
    # Conv Block 1
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same', name='conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Conv Block 2
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Conv Block 3
    x = layers.Conv1D(output_dim, kernel_size=3, activation='relu', padding='same', name='conv3')(x)
    x = layers.BatchNormalization()(x)
    
    # Global pooling: (1662, 256) -> (256,)
    outputs = layers.GlobalAveragePooling1D(name='spatial_features')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SpatialCNN')
    return model


if __name__ == "__main__":
    import numpy as np
    
    # Test
    print("Creating Spatial CNN model...")
    cnn = create_spatial_cnn()
    cnn.summary()
    
    # Test prediction
    print("\nTesting prediction...")
    test_input = np.random.rand(2, 1662)
    test_output = cnn.predict(test_input, verbose=0)
    
    print(f"\nTest passed")
    print(f"Input shape:  {test_input.shape}")   # (2, 1662)
    print(f"Output shape: {test_output.shape}")  # (2, 256)
