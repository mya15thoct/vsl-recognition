"""
Ablation Study Models
Paper: "Multi-Stream MLP–BiLSTM with Cross-Part Contextual Gating
        for Isolated Sign Language Recognition"

Variants:
  V0  Baseline       – Full model (Multi-Stream + Cross-Part Gating + BiLSTM + Temporal Attn)
  V1  Single-Stream  – No multi-stream: 1 shared MLP on all 1662 keypoints
  V2  w/o Cross-Part – Keep gating but remove cross-part ctx enrichment (gate from raw features)
  V3  w/o Gating     – Remove gating entirely, just concat branches
  V4  Uni-LSTM       – Replace Bidirectional LSTM with unidirectional LSTM
  V5  w/o Attn       – Replace temporal attention with last hidden state
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
# Shared helper: downstream layers shared by most variants
# (Shared Dense → BiLSTM → Temporal Attention → Head)
# ---------------------------------------------------------------------------

def _shared_bilstm_attention_head(merged, num_classes, dropout=0.3):
    """
    Shared downstream:
      merged (B,T,256) → Shared Layers → BiLSTM×2 → Temporal Attn → Head
    Returns: output tensor (B, num_classes)
    """
    x = layers.TimeDistributed(
        layers.Dense(256, activation='relu', name='shared1'), name='shared_td1'
    )(merged)
    x = layers.Dropout(dropout)(x)

    x = layers.TimeDistributed(
        layers.Dense(128, activation='relu', name='shared2'), name='shared_td2'
    )(x)
    x = layers.Dropout(dropout)(x)

    # BiLSTM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name='bilstm1')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True), name='bilstm2')(x)
    x = layers.Dropout(dropout)(x)

    # Temporal Attention — use single Lambda to avoid Keras masked-Softmax
    # shape bug: Softmax(axis=1) on a masked (B,T,1) tensor can produce
    # (B,T,T) in some Keras versions instead of (B,T,1).
    attn_scores = layers.TimeDistributed(
        layers.Dense(1, activation='tanh', name='attn_score'), name='attn_td'
    )(x)  # (B, T, 1)
    context = layers.Lambda(
        lambda inputs: tf.reduce_sum(
            inputs[0] * tf.nn.softmax(inputs[1], axis=1), axis=1
        ),
        name='context_vector'
    )([x, attn_scores])  # (B, features)

    # Head
    x = layers.Dense(128, activation='relu', name='dense1')(context)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return outputs


# ---------------------------------------------------------------------------
# V0 – Baseline (identical to hybrid.py for reproducibility)
# ---------------------------------------------------------------------------

def create_v0_baseline(num_classes, sequence_length):
    """Full model: Multi-Stream + Cross-Part Gating + BiLSTM + Temporal Attn"""
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    pose_kp = layers.Lambda(lambda t: t[:, :, :132],    name='pose_split')(x)
    face_kp = layers.Lambda(lambda t: t[:, :, 132:1536], name='face_split')(x)
    hand_kp = layers.Lambda(lambda t: t[:, :, 1536:],   name='hand_split')(x)

    pose_feat = layers.TimeDistributed(create_pose_branch(132,  'pose'), name='pose_features')(pose_kp)
    face_feat = layers.TimeDistributed(create_face_branch(1404, 'face'), name='face_features')(face_kp)
    hand_feat = layers.TimeDistributed(create_hand_branch(126,  'hand'), name='hand_features')(hand_kp)

    # Cross-part ctx
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

    outputs = _shared_bilstm_attention_head(merged, num_classes)
    return Model(inputs=inputs, outputs=outputs, name='V0_Baseline')


# ---------------------------------------------------------------------------
# V1 – Single-Stream (no multi-stream)
# Ablates: body-part specialization
# Replace: 3 separate MLP branches → 1 single MLP on all 1662 dims
# ---------------------------------------------------------------------------

def create_v1_single_stream(num_classes, sequence_length):
    """Single MLP on all 1662 keypoints (no pose/face/hand split)"""
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    # Single MLP branch (output 256 to match merged dim of baseline)
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

    outputs = _shared_bilstm_attention_head(merged, num_classes)
    return Model(inputs=inputs, outputs=outputs, name='V1_SingleStream')


# ---------------------------------------------------------------------------
# V2 – Multi-Stream w/o Cross-Part Gating
# Ablates: cross-part context enrichment before gating
# Replace: ctx vectors removed; gate computed from raw branch features directly
# ---------------------------------------------------------------------------

def create_v2_no_cross_part_gating(num_classes, sequence_length):
    """Multi-stream with gating, but gate computed from raw features (no ctx enrichment)"""
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    pose_kp = layers.Lambda(lambda t: t[:, :, :132],    name='pose_split')(x)
    face_kp = layers.Lambda(lambda t: t[:, :, 132:1536], name='face_split')(x)
    hand_kp = layers.Lambda(lambda t: t[:, :, 1536:],   name='hand_split')(x)

    pose_feat = layers.TimeDistributed(create_pose_branch(132,  'pose'), name='pose_features')(pose_kp)
    face_feat = layers.TimeDistributed(create_face_branch(1404, 'face'), name='face_features')(face_kp)
    hand_feat = layers.TimeDistributed(create_hand_branch(126,  'hand'), name='hand_features')(hand_kp)

    # Gate from raw features (no cross-part enrichment)
    gate_input = layers.Concatenate(name='gate_input')([pose_feat, face_feat, hand_feat])  # (B,T,256)
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

    outputs = _shared_bilstm_attention_head(merged, num_classes)
    return Model(inputs=inputs, outputs=outputs, name='V2_NoCrossPartGating')


# ---------------------------------------------------------------------------
# V3 – Multi-Stream w/o Gating entirely
# Ablates: gating mechanism (softmax weighting)
# Replace: skip gate; directly concatenate branch features
# ---------------------------------------------------------------------------

def create_v3_no_gating(num_classes, sequence_length):
    """Multi-stream branches concatenated directly (no gating at all)"""
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    pose_kp = layers.Lambda(lambda t: t[:, :, :132],    name='pose_split')(x)
    face_kp = layers.Lambda(lambda t: t[:, :, 132:1536], name='face_split')(x)
    hand_kp = layers.Lambda(lambda t: t[:, :, 1536:],   name='hand_split')(x)

    pose_feat = layers.TimeDistributed(create_pose_branch(132,  'pose'), name='pose_features')(pose_kp)
    face_feat = layers.TimeDistributed(create_face_branch(1404, 'face'), name='face_features')(face_kp)
    hand_feat = layers.TimeDistributed(create_hand_branch(126,  'hand'), name='hand_features')(hand_kp)

    # Direct concatenation, no gate
    merged = layers.Concatenate(name='feature_fusion')([pose_feat, face_feat, hand_feat])

    outputs = _shared_bilstm_attention_head(merged, num_classes)
    return Model(inputs=inputs, outputs=outputs, name='V3_NoGating')


# ---------------------------------------------------------------------------
# V4 – Unidirectional LSTM (no Bi)
# Ablates: bidirectionality
# Replace: Bidirectional(LSTM) → LSTM (same hidden units)
# ---------------------------------------------------------------------------

def _shared_lstm_attention_head(merged, num_classes, dropout=0.3):
    """Downstream with unidirectional LSTM (same hidden units as BiLSTM)"""
    x = layers.TimeDistributed(
        layers.Dense(256, activation='relu', name='shared1'), name='shared_td1'
    )(merged)
    x = layers.Dropout(dropout)(x)
    x = layers.TimeDistributed(
        layers.Dense(128, activation='relu', name='shared2'), name='shared_td2'
    )(x)
    x = layers.Dropout(dropout)(x)

    # Unidirectional LSTM (same hidden units as BiLSTM: 64 and 32)
    x = layers.LSTM(64, return_sequences=True, name='lstm1')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(32, return_sequences=True, name='lstm2')(x)
    x = layers.Dropout(dropout)(x)

    # Temporal Attention — same Lambda fix as _shared_bilstm_attention_head
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
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return outputs


def create_v4_unidirectional_lstm(num_classes, sequence_length):
    """Same as baseline but LSTM instead of BiLSTM"""
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    pose_kp = layers.Lambda(lambda t: t[:, :, :132],    name='pose_split')(x)
    face_kp = layers.Lambda(lambda t: t[:, :, 132:1536], name='face_split')(x)
    hand_kp = layers.Lambda(lambda t: t[:, :, 1536:],   name='hand_split')(x)

    pose_feat = layers.TimeDistributed(create_pose_branch(132,  'pose'), name='pose_features')(pose_kp)
    face_feat = layers.TimeDistributed(create_face_branch(1404, 'face'), name='face_features')(face_kp)
    hand_feat = layers.TimeDistributed(create_hand_branch(126,  'hand'), name='hand_features')(hand_kp)

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

    outputs = _shared_lstm_attention_head(merged, num_classes)
    return Model(inputs=inputs, outputs=outputs, name='V4_UniLSTM')


# ---------------------------------------------------------------------------
# V5 – w/o Temporal Attention
# Ablates: temporal attention weighting
# Replace: weighted sum over frames → last hidden state
# ---------------------------------------------------------------------------

def _shared_bilstm_last_hidden_head(merged, num_classes, dropout=0.3):
    """Downstream with BiLSTM but using last hidden state instead of attention"""
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
    # return_sequences=False → automatically takes last hidden state
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False), name='bilstm2')(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    return outputs


def create_v5_no_temporal_attention(num_classes, sequence_length):
    """Same as baseline but last hidden state instead of temporal attention"""
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    pose_kp = layers.Lambda(lambda t: t[:, :, :132],    name='pose_split')(x)
    face_kp = layers.Lambda(lambda t: t[:, :, 132:1536], name='face_split')(x)
    hand_kp = layers.Lambda(lambda t: t[:, :, 1536:],   name='hand_split')(x)

    pose_feat = layers.TimeDistributed(create_pose_branch(132,  'pose'), name='pose_features')(pose_kp)
    face_feat = layers.TimeDistributed(create_face_branch(1404, 'face'), name='face_features')(face_kp)
    hand_feat = layers.TimeDistributed(create_hand_branch(126,  'hand'), name='hand_features')(hand_kp)

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

    outputs = _shared_bilstm_last_hidden_head(merged, num_classes)
    return Model(inputs=inputs, outputs=outputs, name='V5_NoTemporalAttention')


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VARIANTS = {
    'v0_baseline':            create_v0_baseline,
    'v1_single_stream':       create_v1_single_stream,
    'v2_no_cross_part_gating': create_v2_no_cross_part_gating,
    'v3_no_gating':           create_v3_no_gating,
    'v4_unidirectional_lstm': create_v4_unidirectional_lstm,
    'v5_no_temporal_attention': create_v5_no_temporal_attention,
}

VARIANT_LABELS = {
    'v0_baseline':             'Baseline (full model)',
    'v1_single_stream':        'Single-Stream (no multi-stream)',
    'v2_no_cross_part_gating': 'Multi-Stream, w/o Cross-Part Gating',
    'v3_no_gating':            'Multi-Stream, w/o Gating entirely',
    'v4_unidirectional_lstm':  'LSTM (unidirectional, no Bi)',
    'v5_no_temporal_attention': 'w/o Temporal Attention',
}