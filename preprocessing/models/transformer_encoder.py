"""
Transformer Encoder Components
Positional encoding and multi-head self-attention for temporal modeling
"""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class PositionalEncoding(layers.Layer):
    """
    Adds positional encoding to input sequences
    Uses sinusoidal encoding: PE(pos, 2i) = sin(pos/10000^(2i/d))
    
    This preserves the temporal order information that Transformer
    would otherwise lose due to parallel processing.
    """
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Precompute positional encodings
        pos_enc = self._get_positional_encoding(sequence_length, d_model)
        self.pos_enc = tf.constant(pos_enc, dtype=tf.float32)
    
    def _get_positional_encoding(self, sequence_length, d_model):
        """Generate sinusoidal positional encoding"""
        pos_enc = np.zeros((sequence_length, d_model))
        
        for pos in range(sequence_length):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
        
        return pos_enc
    
    def call(self, x):
        """Add positional encoding to input"""
        return x + self.pos_enc[:tf.shape(x)[1], :]


def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0.3, name_prefix='transformer'):
    """
    Single Transformer encoder block with:
    - Multi-head self-attention
    - Add & Norm (1st residual connection)
    - Feed-forward network
    - Add & Norm (2nd residual connection)
    
    Args:
        inputs: Input tensor (batch, seq_len, features)
        head_size: Dimension of each attention head
        num_heads: Number of attention heads
        ff_dim: Hidden dimension of feed-forward network
        dropout: Dropout rate
        name_prefix: Name prefix for layers
    
    Returns:
        Output tensor (same shape as input)
    """
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=dropout,
        name=f'{name_prefix}_mha'
    )(inputs, inputs)
    
    attention_output = layers.Dropout(dropout)(attention_output)
    
    # Add & Norm (1st residual)
    x1 = layers.Add()([inputs, attention_output])
    x1 = layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_ln1')(x1)
    
    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation='relu', name=f'{name_prefix}_ff1')(x1)
    ff_output = layers.Dropout(dropout)(ff_output)
    ff_output = layers.Dense(inputs.shape[-1], name=f'{name_prefix}_ff2')(ff_output)
    ff_output = layers.Dropout(dropout)(ff_output)
    
    # Add & Norm (2nd residual)
    x2 = layers.Add()([x1, ff_output])
    x2 = layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_ln2')(x2)
    
    return x2
