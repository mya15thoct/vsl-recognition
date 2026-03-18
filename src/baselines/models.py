"""
Baseline Models for Comparison
Paper: "Multi-Stream MLP–BiLSTM with Temporal Attention
        for Isolated Sign Language Recognition"

Baselines:
  lstm       – MediaPipe keypoints + stacked LSTM
  lstm_gru   – MediaPipe keypoints + LSTM then GRU
  empath     – EMPATH architecture (ensemble of 4 Transformers), adapted to our input
               Original: Hasan & Adnan, ICPR 2025 (https://github.com/kreyazulh/EMPATH)
               Adaptation: same architecture, input shape changed to (seq_len, 1662)
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

TOTAL_KEYPOINTS = 1662  # 132 pose + 1404 face + 126 hand


class LayerNorm(layers.Layer):
    """Manual LayerNorm to avoid _MklLayerNorm GPU dispatch bug in TF 2.x."""
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=(input_shape[-1],), initializer='ones',  trainable=True)
        self.beta  = self.add_weight(name='beta',  shape=(input_shape[-1],), initializer='zeros', trainable=True)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var  = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / tf.sqrt(var + self.epsilon) + self.beta


# ---------------------------------------------------------------------------
# B1 – MediaPipe + stacked LSTM   (simple keypoint-based baseline)
# ---------------------------------------------------------------------------

def create_lstm_baseline(num_classes, sequence_length):
    """
    MediaPipe keypoints → LayerNorm → stacked LSTM × 3 → Dense head.
    Replicates the standard LSTM approach used in keypoint-based SLR papers.
    """
    inp = layers.Input(shape=(sequence_length, TOTAL_KEYPOINTS), name='input')

    x = LayerNorm(name='input_norm')(inp)

    x = layers.LSTM(256, return_sequences=True, name='lstm1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128, return_sequences=True, name='lstm2')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=False, name='lstm3')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return Model(inp, out, name='lstm_baseline')


# ---------------------------------------------------------------------------
# B2 – MediaPipe + LSTM-GRU hybrid
# ---------------------------------------------------------------------------

def create_lstm_gru_baseline(num_classes, sequence_length):
    """
    MediaPipe keypoints → LayerNorm → LSTM → GRU → Dense head.
    """
    inp = layers.Input(shape=(sequence_length, TOTAL_KEYPOINTS), name='input')

    x = LayerNorm(name='input_norm')(inp)

    x = layers.LSTM(256, return_sequences=True, name='lstm1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128, return_sequences=True, name='lstm2')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(64, return_sequences=False, name='gru1')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return Model(inp, out, name='lstm_gru_baseline')


# ---------------------------------------------------------------------------
# B3 – EMPATH: Ensemble of 4 Transformers
# Original paper: Hasan & Adnan, ICPR 2025
# GitHub: https://github.com/kreyazulh/EMPATH
# Adaptation: input shape changed from (12, 92, 3) to (seq_len, 1662)
# ---------------------------------------------------------------------------

def _transformer_block(x, num_heads, ff_dim, dropout_rate, name_prefix):
    """Single Transformer encoder block (Multi-Head Attention + FFN)."""
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=x.shape[-1] // num_heads,
        name=f'{name_prefix}_mha'
    )(x, x)
    attn_out = layers.Dropout(dropout_rate)(attn_out)
    x = LayerNorm(name=f'{name_prefix}_ln1')(x + attn_out)

    ffn = layers.Dense(ff_dim, activation='relu', name=f'{name_prefix}_ffn1')(x)
    ffn = layers.Dropout(dropout_rate)(ffn)
    ffn = layers.Dense(x.shape[-1], name=f'{name_prefix}_ffn2')(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    x = LayerNorm(name=f'{name_prefix}_ln2')(x + ffn)
    return x


def _single_transformer(inp, num_classes, model_idx,
                         num_heads=4, ff_dim=256, num_layers=2, dropout=0.3):
    """One Transformer member of the ensemble."""
    x = layers.Dense(64, name=f'm{model_idx}_proj')(inp)
    for i in range(num_layers):
        x = _transformer_block(x, num_heads, ff_dim, dropout,
                                name_prefix=f'm{model_idx}_block{i}')
    x = layers.GlobalAveragePooling1D(name=f'm{model_idx}_gap')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='relu', name=f'm{model_idx}_dense')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation='softmax',
                       name=f'm{model_idx}_out')(x)
    return out


def create_empath_baseline(num_classes, sequence_length):
    """
    EMPATH: Ensemble of 4 Transformer models, outputs averaged.
    Architecture adapted from Hasan & Adnan (ICPR 2025).
    Input adapted from (12, 92, 3) → (sequence_length, 1662).
    """
    inp = layers.Input(shape=(sequence_length, TOTAL_KEYPOINTS), name='input')
    x = LayerNorm(name='input_norm')(inp)

    outputs = [
        _single_transformer(x, num_classes, model_idx=i)
        for i in range(4)
    ]

    if len(outputs) == 1:
        ensemble_out = outputs[0]
    else:
        ensemble_out = layers.Average(name='ensemble_avg')(outputs)

    return Model(inp, ensemble_out, name='empath_baseline')


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINES = {
    'lstm':     create_lstm_baseline,
    'lstm_gru': create_lstm_gru_baseline,
    'empath':   create_empath_baseline,
}

BASELINE_LABELS = {
    'lstm':     'MediaPipe + stacked LSTM',
    'lstm_gru': 'MediaPipe + LSTM-GRU',
    'empath':   'EMPATH (Ensemble Transformer, adapted)',
}
