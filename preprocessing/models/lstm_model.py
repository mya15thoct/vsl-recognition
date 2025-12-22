"""
LSTM học temporal patterns từ sequence of features
"""
import tensorflow as tf
from tensorflow.keras import layers


def create_temporal_lstm(sequence_length=130, feature_dim=256):
    """
    LSTM học temporal patterns từ feature sequence
    
    Args:
        sequence_length: Số frames (130)
        feature_dim: Số features per frame (256)
    
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(sequence_length, feature_dim), name='feature_sequence')
    
    # LSTM Block 1
    x = layers.LSTM(128, return_sequences=True, name='lstm1')(inputs)
    x = layers.Dropout(0.3)(x)
    
    # LSTM Block 2
    x = layers.LSTM(64, return_sequences=False, name='lstm2')(x)
    outputs = layers.Dropout(0.3)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TemporalLSTM')
    return model


if __name__ == "__main__":
    import numpy as np
    
    # Test
    print("Creating Temporal LSTM model...")
    lstm = create_temporal_lstm()
    lstm.summary()
    
    # Test prediction
    print("\nTesting prediction...")
    test_input = np.random.rand(2, 130, 256)
    test_output = lstm.predict(test_input, verbose=0)
    
    print(f"\n✅ Test passed!")
    print(f"Input shape:  {test_input.shape}")   # (2, 130, 256)
    print(f"Output shape: {test_output.shape}")  # (2, 64)
