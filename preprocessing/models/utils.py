"""
Model loading utilities with custom objects
Handles loading models with custom layers properly registered
"""
import tensorflow as tf
from .transformer_encoder import PositionalEncoding


def load_model_safe(model_path, compile=True):
    """
    Load Keras model with custom objects registered
    
    This is necessary for models that use custom layers like PositionalEncoding.
    Without registering custom objects, load_model() will fail with "unknown layer" error.
    
    Args:
        model_path: Path to .h5 or SavedModel file
        compile: Whether to compile the model (default True)
    
    Returns:
        Loaded Keras model
    
    Example:
        >>> model = load_model_safe('checkpoints/best_model.h5')
        >>> predictions = model.predict(X_test)
    """
    custom_objects = {
        'PositionalEncoding': PositionalEncoding
    }
    
    return tf.keras.models.load_model(
        model_path, 
        custom_objects=custom_objects,
        compile=compile
    )
