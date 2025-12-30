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
    Adds position information to input embeddings.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, sequence_length, d_model, **kwargs):
        """
        Args:
            sequence_length: Maximum sequence length
            d_model: Embedding dimension
        """
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Precompute positional encodings
        pe = self._create_positional_encoding(sequence_length, d_model)
        self.pe = tf.constant(pe, dtype=tf.float32)
    
    def _create_positional_encoding(self, seq_len, d_model):
        """Generate sinusoidal positional encoding matrix."""
        pe = np.zeros((seq_len, d_model))
        
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
        
        return pe
    
    def call(self, x):
        """Add positional encoding to input."""
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]
    
    def get_config(self):
        """For model serialization."""
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
    2. Add & LayerNorm (residual)
    3. Feed-Forward Network
    4. Add & LayerNorm (residual)
    
    Args:
        x: Input tensor (batch, seq_len, d_model)
        num_heads: Number of attention heads
        key_dim: Dimension per attention head
        ff_dim: Hidden dimension in feed-forward network
        dropout: Dropout rate
        name: Layer name prefix
    
    Returns:
        Output tensor (same shape as input)
    """
    # Multi-Head Self-Attention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
        name=f'{name}_mha'
    )(x, x)  # Self-attention: query=key=value
    
    attn_output = layers.Dropout(dropout)(attn_output)
    
    # Add & Norm (1st residual)
    out1 = layers.Add()([x, attn_output])
    out1 = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_ln1')(out1)
    
    # Feed-Forward Network
    ff_output = layers.Dense(ff_dim, activation='relu', name=f'{name}_ff1')(out1)
    ff_output = layers.Dropout(dropout)(ff_output)
    ff_output = layers.Dense(x.shape[-1], name=f'{name}_ff2')(ff_output)
    ff_output = layers.Dropout(dropout)(ff_output)
    
    # Add & Norm (2nd residual)
    out2 = layers.Add()([out1, ff_output])
    out2 = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_ln2')(out2)
    
    return out2


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Transformer Components Test")
    print("=" * 50)
    
    # Test PositionalEncoding
    print("\n1. Testing PositionalEncoding...")
    pe = PositionalEncoding(sequence_length=33, d_model=256)
    x = tf.random.normal((2, 33, 256))
    y = pe(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {y.shape}")
    print(f"   ✓ PositionalEncoding OK")
    
    # Test transformer_encoder_block
    print("\n2. Testing transformer_encoder_block...")
    y2 = transformer_encoder_block(
        y, 
        num_heads=4, 
        key_dim=64, 
        ff_dim=512, 
        dropout=0.1, 
        name='test_transformer'
    )
    print(f"   Input:  {y.shape}")
    print(f"   Output: {y2.shape}")
    print(f"   ✓ transformer_encoder_block OK")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
