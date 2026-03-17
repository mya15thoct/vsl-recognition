"""
Ablation Study Models
Paper: "Multi-Stream MLP–BiLSTM with Temporal Attention
        for Isolated Sign Language Recognition"

Variants:
  V0  Proposed       – Full model (Multi-Stream + Concat + BiLSTM + Temporal Attn)
  V1  Single-Stream  – No multi-stream: 1 shared MLP on all 1662 keypoints
  V2  + Gating       – Add softmax gating on top of V0 (shows gating hurts)
  V3  Uni-LSTM       – Replace Bidirectional LSTM with unidirectional LSTM
  V4  w/o Attn       – Replace temporal attention with last hidden state
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import sys
from pathlib import Path

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_components",
    Path(__file__).parent.parent / "models" / "components.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
create_pose_branch = _mod.create_pose_branch
create_face_branch = _mod.create_face_branch
create_hand_branch = _mod.create_hand_branch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _multistream_branches(x):
    """Split input and extract per-part features. Returns (pose, face, hand) feat tensors."""
    pose_kp = layers.Lambda(lambda t: t[:, :, :132],    name='pose_split')(x)
    face_kp = layers.Lambda(lambda t: t[:, :, 132:1536], name='face_split')(x)
    hand_kp = layers.Lambda(lambda t: t[:, :, 1536:],   name='hand_split')(x)

    pose_feat = layers.TimeDistributed(create_pose_branch(132,  'pose'), name='pose_features')(pose_kp)
    face_feat = layers.TimeDistributed(create_face_branch(1404, 'face'), name='face_features')(face_kp)
    hand_feat = layers.TimeDistributed(create_hand_branch(126,  'hand'), name='hand_features')(hand_kp)
    return pose_feat, face_feat, hand_feat


def _bilstm_attention_head(merged, num_classes, dropout=0.3):
    """Shared Dense → BiLSTM×2 → Temporal Attention → Head"""
    x = layers.TimeDistributed(
        layers.Dense(256, activation='relu', name='shared1'), name='shared_td1'
    )(merged)
    x = layers.Dropout(dropout)(x)
    x = layers.TimeDistributed(
        layers.Dense(128, activation='relu', name='shared2'), name='shared_td2'
    )(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name='bilstm1')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True), name='bilstm2')(x)
    x = layers.Dropout(dropout)(x)

    # Lambda trick: avoids Keras masked-Softmax shape bug with Softmax(axis=1)
    attn_scores = layers.TimeDistributed(
        layers.Dense(1, activation='tanh', name='attn_score'), name='attn_td'
    )(x)  # (B, T, 1)
    context = layers.Lambda(
        lambda inputs: tf.reduce_sum(
            inputs[0] * tf.nn.softmax(inputs[1], axis=1), axis=1
        ),
        name='context_vector'
    )([x, attn_scores])  # (B, features)

    x = layers.Dense(128, activation='relu', name='dense1')(context)
    x = layers.Dropout(0.5)(x)
    return layers.Dense(num_classes, activation='softmax', name='output')(x)


# ---------------------------------------------------------------------------
# V0 – Proposed model
# Multi-Stream + direct Concat + BiLSTM + Temporal Attention
# ---------------------------------------------------------------------------

def create_v0_baseline(num_classes, sequence_length):
    """Proposed model: Multi-Stream + Concat + BiLSTM + Temporal Attention"""
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    pose_feat, face_feat, hand_feat = _multistream_branches(x)

    merged = layers.Concatenate(name='feature_fusion')([pose_feat, face_feat, hand_feat])

    outputs = _bilstm_attention_head(merged, num_classes)
    return Model(inputs=inputs, outputs=outputs, name='V0_Proposed')


# ---------------------------------------------------------------------------
# V1 – Single-Stream
# Ablates: body-part specialization (multi-stream)
# Replace: 3 MLP branches → 1 shared MLP on all 1662 dims
# ---------------------------------------------------------------------------

def create_v1_single_stream(num_classes, sequence_length):
    """Single MLP on all 1662 keypoints (no body-part split)"""
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    def _single_mlp():
        inp = layers.Input(shape=(1662,), name='single_input')
        z = layers.Dense(512, activation='relu', name='s_dense1')(inp)
        z = layers.BatchNormalization(name='s_bn1')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.Dense(256, activation='relu', name='s_dense2')(z)
        z = layers.BatchNormalization(name='s_bn2')(z)
        z = layers.Dropout(0.2)(z)
        z = layers.Dense(256, activation='relu', name='s_dense3')(z)
        return Model(inputs=inp, outputs=z, name='single_stream_branch')

    merged = layers.TimeDistributed(_single_mlp(), name='single_features')(x)

    outputs = _bilstm_attention_head(merged, num_classes)
    return Model(inputs=inputs, outputs=outputs, name='V1_SingleStream')


# ---------------------------------------------------------------------------
# V2 – + Softmax Gating
# Ablates: effect of adding gating on top of V0
# Shows: softmax gating hurts because it forces zero-sum competition
# ---------------------------------------------------------------------------

def create_v2_with_gating(num_classes, sequence_length):
    """V0 + softmax gating (ablates: gating is counterproductive)"""
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    pose_feat, face_feat, hand_feat = _multistream_branches(x)

    # Cross-part context
    pose_ctx = layers.TimeDistributed(
        layers.Dense(64, activation='relu', name='pose_ctx_dense'), name='pose_ctx'
    )(layers.Concatenate(name='pose_ctx_input')([face_feat, hand_feat]))
    face_ctx = layers.TimeDistributed(
        layers.Dense(64, activation='relu', name='face_ctx_dense'), name='face_ctx'
    )(layers.Concatenate(name='face_ctx_input')([pose_feat, hand_feat]))
    hand_ctx = layers.TimeDistributed(
        layers.Dense(64, activation='relu', name='hand_ctx_dense'), name='hand_ctx'
    )(layers.Concatenate(name='hand_ctx_input')([pose_feat, face_feat]))

    pose_enriched = layers.Concatenate(name='pose_enriched')([pose_feat, pose_ctx])
    face_enriched = layers.Concatenate(name='face_enriched')([face_feat, face_ctx])
    hand_enriched = layers.Concatenate(name='hand_enriched')([hand_feat, hand_ctx])

    gate_input = layers.Concatenate(name='gate_input')([pose_enriched, face_enriched, hand_enriched])
    gate = layers.TimeDistributed(
        layers.Dense(3, activation='softmax', name='gate_dense'), name='body_part_gate'
    )(gate_input)

    pose_scale = layers.Lambda(lambda g: g[:, :, 0:1], name='pose_gate')(gate)
    face_scale = layers.Lambda(lambda g: g[:, :, 1:2], name='face_gate')(gate)
    hand_scale = layers.Lambda(lambda g: g[:, :, 2:3], name='hand_gate')(gate)

    pose_gated = layers.Multiply(name='pose_gated')([pose_feat, pose_scale])
    face_gated = layers.Multiply(name='face_gated')([face_feat, face_scale])
    hand_gated = layers.Multiply(name='hand_gated')([hand_feat, hand_scale])

    merged = layers.Concatenate(name='feature_fusion')([pose_gated, face_gated, hand_gated])

    outputs = _bilstm_attention_head(merged, num_classes)
    return Model(inputs=inputs, outputs=outputs, name='V2_WithGating')


# ---------------------------------------------------------------------------
# V3 – Unidirectional LSTM
# Ablates: bidirectionality
# Replace: Bidirectional(LSTM) → LSTM (same hidden units)
# ---------------------------------------------------------------------------

def create_v3_unidirectional_lstm(num_classes, sequence_length):
    """V0 with unidirectional LSTM instead of BiLSTM"""
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    pose_feat, face_feat, hand_feat = _multistream_branches(x)
    merged = layers.Concatenate(name='feature_fusion')([pose_feat, face_feat, hand_feat])

    x = layers.TimeDistributed(
        layers.Dense(256, activation='relu', name='shared1'), name='shared_td1'
    )(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.TimeDistributed(
        layers.Dense(128, activation='relu', name='shared2'), name='shared_td2'
    )(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(64, return_sequences=True, name='lstm1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32, return_sequences=True, name='lstm2')(x)
    x = layers.Dropout(0.3)(x)

    attn_scores = layers.TimeDistributed(
        layers.Dense(1, activation='tanh', name='attn_score'), name='attn_td'
    )(x)
    context = layers.Lambda(
        lambda inputs: tf.reduce_sum(
            inputs[0] * tf.nn.softmax(inputs[1], axis=1), axis=1
        ),
        name='context_vector'
    )([x, attn_scores])

    x = layers.Dense(128, activation='relu', name='dense1')(context)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return Model(inputs=inputs, outputs=outputs, name='V3_UniLSTM')


# ---------------------------------------------------------------------------
# V4 – w/o Temporal Attention
# Ablates: temporal attention
# Replace: weighted sum over frames → last hidden state of BiLSTM
# ---------------------------------------------------------------------------

def create_v4_no_temporal_attention(num_classes, sequence_length):
    """V0 with last hidden state instead of temporal attention"""
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    pose_feat, face_feat, hand_feat = _multistream_branches(x)
    merged = layers.Concatenate(name='feature_fusion')([pose_feat, face_feat, hand_feat])

    x = layers.TimeDistributed(
        layers.Dense(256, activation='relu', name='shared1'), name='shared_td1'
    )(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.TimeDistributed(
        layers.Dense(128, activation='relu', name='shared2'), name='shared_td2'
    )(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name='bilstm1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False), name='bilstm2')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return Model(inputs=inputs, outputs=outputs, name='V4_NoTemporalAttention')


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VARIANTS = {
    'v0_baseline':            create_v0_baseline,
    'v1_single_stream':       create_v1_single_stream,
    'v2_with_gating':         create_v2_with_gating,
    'v3_unidirectional_lstm': create_v3_unidirectional_lstm,
    'v4_no_temporal_attention': create_v4_no_temporal_attention,
}

VARIANT_LABELS = {
    'v0_baseline':              'Proposed (Multi-Stream + BiLSTM + Attn)',
    'v1_single_stream':         'w/o Multi-Stream (single MLP)',
    'v2_with_gating':           'w/ Softmax Gating',
    'v3_unidirectional_lstm':   'w/ Uni-LSTM (no Bi)',
    'v4_no_temporal_attention': 'w/o Temporal Attention',
}
