"""
Transformer Components for Temporal Modeling
Contains: PositionalEncoding, transformer_encoder_block
"""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding for sequence data.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Precompute positional encodings
        pe = self._create_positional_encoding(sequence_length, d_model)
        self.pe = tf.constant(pe, dtype=tf.float32)
    
    def _create_positional_encoding(self, seq_len, d_model):
        pe = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
        return pe
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model
        })
        return config


def transformer_encoder_block(x, num_heads, key_dim, ff_dim, dropout=0.1, name='transformer'):
    """
    Single Transformer encoder block.
    
    Architecture:
    1. Multi-Head Self-Attention
    2. Add & LayerNorm
    3. Feed-Forward Network
    4. Add & LayerNorm
    """
    # Multi-Head Self-Attention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
        name=f'{name}_mha'
    )(x, x)
    
    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.Add()([x, attn_output])
    out1 = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_ln1')(out1)
    
    # Feed-Forward Network
    ff_output = layers.Dense(ff_dim, activation='relu', name=f'{name}_ff1')(out1)
    ff_output = layers.Dropout(dropout)(ff_output)
    ff_output = layers.Dense(x.shape[-1], name=f'{name}_ff2')(ff_output)
    ff_output = layers.Dropout(dropout)(ff_output)
    
    out2 = layers.Add()([out1, ff_output])
    out2 = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_ln2')(out2)
    
    return out2
