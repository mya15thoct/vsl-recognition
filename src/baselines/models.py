"""
Baseline Models for Comparison
Paper: "Multi-Stream MLP–BiLSTM with Temporal Attention
        for Isolated Sign Language Recognition"

Baselines:
  lstm       – MediaPipe keypoints + stacked LSTM
  lstm_gru   – MediaPipe keypoints + LSTM then GRU
  hwgate     – HWGATE (skeleton graph attention) — retrained on our data
  empath     – EMPATH (ensemble transformer attention) — retrained on our data
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

TOTAL_KEYPOINTS = 1662  # 132 pose + 1404 face + 126 hand


# ---------------------------------------------------------------------------
# B1 – MediaPipe + stacked LSTM   (simple keypoint-based baseline)
# ---------------------------------------------------------------------------

def create_lstm_baseline(num_classes, sequence_length):
    """
    MediaPipe keypoints → LayerNorm → stacked LSTM × 3 → Dense head.
    Replicates the standard LSTM approach used in keypoint-based SLR papers.
    """
    inp = layers.Input(shape=(sequence_length, TOTAL_KEYPOINTS), name='input')

    x = layers.LayerNormalization(name='input_norm')(inp)

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

    x = layers.LayerNormalization(name='input_norm')(inp)

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
# Registry
# ---------------------------------------------------------------------------

BASELINES = {
    'lstm':     create_lstm_baseline,
    'lstm_gru': create_lstm_gru_baseline,
    # 'hwgate': create_hwgate,   ← to be added when repo is found
    # 'empath':  create_empath,  ← to be added when repo is found
}

BASELINE_LABELS = {
    'lstm':     'MediaPipe + stacked LSTM',
    'lstm_gru': 'MediaPipe + LSTM-GRU',
    # 'hwgate': 'HWGATE (retrained)',
    # 'empath':  'EMPATH (retrained)',
}
