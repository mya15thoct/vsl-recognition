"""
Mix static image keypoints with video keypoints for Word Model training.

Key insight:
  The Word Model already has a Masking(mask_value=0.0) layer.
  Static images (1 frame) can be padded with zeros to match sequence_length.
  Masking automatically ignores zero-padded frames during BiLSTM / Attention.
  → No separate model needed. Just mix with video data and train normally.

Flow:
  1. Load video keypoints  : shape (N_video,  T, 1662)
  2. Load static keypoints : shape (N_static, 1662)
  3. Pad static → zeros    : shape (N_static, T, 1662)  [only frame 0 is real]
  4. Concatenate datasets
  5. Train Word Model as usual
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

try:
    from models.hybrid import create_hybrid_multistream_model
except ImportError:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.hybrid import create_hybrid_multistream_model


# ─────────────────────────────────────────────────────────────────────────────
# PAD STATIC IMAGES TO MATCH VIDEO SEQUENCE LENGTH
# ─────────────────────────────────────────────────────────────────────────────

def pad_static_to_sequence(
    static_keypoints: np.ndarray,   # (N, 1662) — 1 frame per sample
    sequence_length:  int,
) -> np.ndarray:
    """
    Pad static image keypoints (1 frame) with zeros to match sequence_length.

    The Word Model's Masking(mask_value=0.0) layer automatically ignores
    zero frames during BiLSTM and Temporal Attention computation.

    Args:
        static_keypoints: shape (N, 1662)
        sequence_length:  target T (same as video training)

    Returns:
        padded: shape (N, T, 1662)  — frame 0 is real, rest are zeros
    """
    N = static_keypoints.shape[0]
    padded = np.zeros((N, sequence_length, 1662), dtype=np.float32)
    padded[:, 0, :] = static_keypoints   # only frame 0 is real
    return padded


# ─────────────────────────────────────────────────────────────────────────────
# MIX STATIC + VIDEO DATA
# ─────────────────────────────────────────────────────────────────────────────

def mix_static_and_video(
    X_video:   np.ndarray,   # (N_video,  T, 1662)
    y_video:   np.ndarray,   # (N_video,)  integer labels
    X_static:  np.ndarray,   # (N_static, 1662)
    y_static:  np.ndarray,   # (N_static,) integer labels
    sequence_length: int,
) -> tuple:
    """
    Combine video sequences and static images into one training dataset.

    Args:
        X_video, y_video:   Video keypoints and labels.
        X_static, y_static: Static image keypoints and labels.
        sequence_length:    Target sequence length T.

    Returns:
        X_mixed: shape (N_video + N_static, T, 1662)
        y_mixed: shape (N_video + N_static,)
    """
    X_static_padded = pad_static_to_sequence(X_static, sequence_length)

    X_mixed = np.concatenate([X_video, X_static_padded], axis=0)
    y_mixed = np.concatenate([y_video, y_static],        axis=0)

    # Shuffle
    idx = np.random.permutation(len(X_mixed))
    return X_mixed[idx], y_mixed[idx]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_word_model_with_static(
    X_video:        np.ndarray,
    y_video:        np.ndarray,
    X_static:       np.ndarray,
    y_static:       np.ndarray,
    num_classes:    int,
    sequence_length: int,
    pretrained_path: str = None,    # path to existing Word Model to fine-tune
    save_path:       str = 'checkpoints/word_model_enriched.h5',
    lr:              float = 1e-3,
    epochs:          int   = 200,
    batch_size:      int   = 32,
    val_split:       float = 0.1,
):
    """
    Train (or fine-tune) Word Model on mixed video + static image data.

    Args:
        X_video:         Video keypoints  (N_video,  T, 1662).
        y_video:         Video labels     (N_video,).
        X_static:        Static keypoints (N_static, 1662).
        y_static:        Static labels    (N_static,).
        num_classes:     Number of gloss classes.
        sequence_length: T — must match X_video shape[1].
        pretrained_path: If set, load & fine-tune existing model (lr should be small).
        save_path:       Where to save the best model.
        lr:              Learning rate.
        epochs:          Max epochs.
        batch_size:      Batch size.
        val_split:       Validation fraction.
    """
    # ── Mix data ──────────────────────────────────────────────────────────────
    X_mixed, y_mixed = mix_static_and_video(
        X_video, y_video, X_static, y_static, sequence_length
    )
    y_cat = tf.keras.utils.to_categorical(y_mixed, num_classes)

    print(f"[Dataset] Videos:  {len(X_video)}")
    print(f"[Dataset] Static:  {len(X_static)}")
    print(f"[Dataset] Mixed:   {len(X_mixed)} total samples")

    # ── Build or load model ───────────────────────────────────────────────────
    if pretrained_path:
        print(f"\n[Model] Loading pretrained model from {pretrained_path}")
        model = tf.keras.models.load_model(pretrained_path)
        lr = lr * 0.1   # smaller LR for fine-tuning
        print(f"[Model] Fine-tuning mode → lr reduced to {lr:.2e}")
    else:
        print("\n[Model] Building new Word Model...")
        model = create_hybrid_multistream_model(num_classes, sequence_length)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=10, min_lr=1e-6, verbose=1),
        ModelCheckpoint(save_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[Training] Starting...")
    history = model.fit(
        X_mixed, y_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n[Done] Best model saved to {save_path}")
    print("[Done] Next step: build Attention Dictionary from this model → Frozen → Sentence Model")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    NUM_CLASSES     = 100   # update to your actual number
    SEQUENCE_LENGTH = 33    # update to your actual sequence_length

    # ── Placeholder data — replace with your actual data loading ──────────────
    N_video  = 1000
    N_static = 500

    X_video  = np.random.rand(N_video,  SEQUENCE_LENGTH, 1662).astype(np.float32)
    y_video  = np.random.randint(0, NUM_CLASSES, size=(N_video,))

    X_static = np.random.rand(N_static, 1662).astype(np.float32)
    y_static = np.random.randint(0, NUM_CLASSES, size=(N_static,))

    # ── Train from scratch on mixed data ──────────────────────────────────────
    model, history = train_word_model_with_static(
        X_video=X_video,
        y_video=y_video,
        X_static=X_static,
        y_static=y_static,
        num_classes=NUM_CLASSES,
        sequence_length=SEQUENCE_LENGTH,
        pretrained_path=None,   # set path if fine-tuning existing model
        save_path='checkpoints/word_model_enriched.h5',
        lr=1e-3,
        epochs=200,
        batch_size=32,
    )
