"""
Fine-tune Word Model ENCODER on static images (1 frame per gloss).

Strategy:
  - Freeze BiLSTM + Temporal Attention + output head
  - Only train per-frame encoder: pose/face/hand MLP branches + shared dense layers
  - Static images padded with zeros to match sequence_length
  - Masking layer ignores zero-padded frames automatically

Why freeze BiLSTM?
  - BiLSTM learns temporal patterns from multi-frame videos
  - Static images have no temporal info → would corrupt BiLSTM if trained on them
  - Encoder (MLP branches) learns visual features per-frame → benefits from static images

Usage:
  python src/training/train_static_images.py
  python src/training/train_static_images.py --model_path /mnt/ngan/recognition/checkpoints/mlp/best_model
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import argparse
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import SEQUENCE_PATH, CHECKPOINT_DIR, SEQUENCE_LENGTH, IMAGE_DIR


# ─────────────────────────────────────────────────────────────────────────────
# FREEZE / UNFREEZE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Layers to freeze (temporal modeling — not useful for single-frame input)
FROZEN_LAYER_NAMES = [
    'shared_td1', 'shared_td2',
    'bilstm1', 'bilstm2',
    'attn_td', 'temporal_attention', 'attn_apply', 'context_vector',
    'dense1', 'output',
]

# Layers to train (per-frame visual encoder)
ENCODER_LAYER_NAMES = [
    'pose_features', 'face_features', 'hand_features',
]


def freeze_temporal_layers(model: tf.keras.Model) -> int:
    """Freeze BiLSTM, Attention, and head layers. Return count of frozen params."""
    frozen = 0
    for layer in model.layers:
        if any(name in layer.name for name in FROZEN_LAYER_NAMES):
            layer.trainable = False
            frozen += layer.count_params()
        else:
            layer.trainable = True
    return frozen


def print_trainable_summary(model: tf.keras.Model):
    total    = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    frozen   = total - trainable
    print(f"\n  Total params:    {total:,}")
    print(f"  Trainable:       {trainable:,}  ← encoder only")
    print(f"  Frozen:          {frozen:,}  ← BiLSTM + Attention + Head")

    print("\n  Layer trainability:")
    for layer in model.layers:
        if layer.count_params() > 0:
            status = "TRAIN" if layer.trainable else "FROZEN"
            print(f"    [{status}] {layer.name:35s} {layer.count_params():>10,} params")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_static_sequences(sequence_path=None, sequence_length=None):
    """
    Load only static image sequences (*_static.npy) from the sequences folder.

    Returns:
        X: (N, sequence_length, 1662) — padded to sequence_length
        y: (N,)  integer labels
        action_names: list of class names
    """
    sequence_path  = Path(sequence_path or SEQUENCE_PATH)
    sequence_length = sequence_length or SEQUENCE_LENGTH or 33

    print(f"\nLoading static sequences from {sequence_path}...")

    action_folders = sorted([d for d in sequence_path.iterdir() if d.is_dir()])
    action_names   = [d.name for d in action_folders]

    X, y = [], []

    for label_idx, folder in enumerate(action_folders):
        static_files = sorted(folder.glob('*_static.npy'))
        for npy_file in static_files:
            seq = np.load(npy_file).astype(np.float32)   # (1, 1662)
            T   = seq.shape[0]

            # Pad to sequence_length
            padded = np.zeros((sequence_length, 1662), dtype=np.float32)
            padded[:min(T, sequence_length)] = seq[:min(T, sequence_length)]

            X.append(padded)
            y.append(label_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"  Static samples: {len(X)}")
    print(f"  Classes:        {len(action_names)}")
    print(f"  Shape:          {X.shape}")

    return X, y, action_names


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_encoder_on_static(
    model_path:      str   = None,
    sequence_path:   str   = None,
    save_path:       str   = None,
    lr:              float = 1e-4,
    epochs:          int   = 100,
    batch_size:      int   = 32,
    val_split:       float = 0.1,
):
    """
    Load pretrained Word Model → freeze BiLSTM → fine-tune encoder on static images.

    Args:
        model_path:    Path to trained Word Model checkpoint.
        sequence_path: Root sequences folder (contains *_static.npy files).
        save_path:     Where to save the fine-tuned model.
        lr:            Learning rate (small — encoder already has good features).
        epochs:        Max training epochs.
        batch_size:    Batch size.
        val_split:     Validation fraction.
    """
    model_path = model_path or str(CHECKPOINT_DIR / 'best_model')
    save_path  = save_path  or str(CHECKPOINT_DIR / 'best_model_enriched')

    print("=" * 60)
    print("ENCODER FINE-TUNING ON STATIC IMAGES")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading Word Model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Detect sequence_length from model input shape
    seq_len = model.input_shape[1]
    print(f"  Sequence length from model: {seq_len}")

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y, action_names = load_static_sequences(sequence_path, sequence_length=seq_len)
    num_classes = len(action_names)
    y_cat = tf.keras.utils.to_categorical(y, num_classes)

    # ── Freeze BiLSTM / Attention / Head ──────────────────────────────────────
    print("\nFreezing temporal layers...")
    freeze_temporal_layers(model)
    print_trainable_summary(model)

    # ── Compile ───────────────────────────────────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-7, verbose=1),
        ModelCheckpoint(save_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("TRAINING (encoder only — BiLSTM frozen)")
    print(f"  Samples:    {len(X)}")
    print(f"  lr:         {lr}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"{'=' * 60}\n")

    history = model.fit(
        X, y_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n[DONE] Fine-tuned model saved to: {save_path}")
    print("[DONE] Next: python src/utils/build_dictionary.py to build Attention Dictionary")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune Word Model encoder on static images")
    parser.add_argument('--model_path',    type=str, default=None,
                        help='Path to trained Word Model (default: CHECKPOINT_DIR/best_model)')
    parser.add_argument('--sequence_path', type=str, default=None,
                        help='Root sequences folder (default: SEQUENCE_PATH in config)')
    parser.add_argument('--save_path',     type=str, default=None,
                        help='Output path for fine-tuned model (default: CHECKPOINT_DIR/best_model_enriched)')
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--batch_size',    type=int,   default=32)
    args = parser.parse_args()

    train_encoder_on_static(
        model_path=args.model_path,
        sequence_path=args.sequence_path,
        save_path=args.save_path,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
