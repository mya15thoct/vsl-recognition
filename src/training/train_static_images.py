"""
Train Word Model on combined vocabulary: video sequences + static images.

Handles the case where ISL image classes differ from video classes:
  - Merges both vocabularies into one combined class list
  - Transfers encoder + BiLSTM weights from old model (old vocabulary)
  - Attaches a new output head for the expanded vocabulary
  - Trains on mixed data: video sequences + padded image sequences
  - Uses class weights to handle imbalance (images have far fewer samples)

Usage:
  python src/training/train_static_images.py \
    --model_path  /mnt/ngan/recognition/checkpoints/best_model \
    --seq_path    /mnt/ngan/recognition/sequences \
    --image_seq_path /mnt/ngan/recognition/sequences \
    --save_path   /mnt/ngan/recognition/checkpoints/best_model_combined \
    --action_mapping_path /mnt/ngan/recognition/checkpoints/action_mapping.json
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import SEQUENCE_PATH, CHECKPOINT_DIR, SEQUENCE_LENGTH
from utils.augmentation import add_noise, spatial_jitter


# ─────────────────────────────────────────────────────────────────────────────
# STATIC IMAGE AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def augment_static_sequence(seq: np.ndarray, n: int = 10) -> list:
    """
    Generate n augmented copies of a static image sequence.
    Only augments the real frame (index 0); zero-padded frames are untouched.

    Augmentations applied:
      - Gaussian noise  (std=0.01)
      - Spatial jitter  (std=0.015)
      - Combined noise + jitter

    Args:
        seq: (T, 1662) padded static sequence (1 real frame + zeros)
        n:   number of augmented copies to generate

    Returns:
        List of n augmented sequences, each (T, 1662)
    """
    augmented = []
    methods = ['noise', 'jitter', 'both']

    for i in range(n):
        method = methods[i % len(methods)]
        aug = seq.copy()

        if method == 'noise' or method == 'both':
            aug = add_noise(aug, noise_std=0.008)
        if method == 'jitter' or method == 'both':
            aug = spatial_jitter(aug, jitter_std=0.015)

        augmented.append(aug)

    return augmented


# ─────────────────────────────────────────────────────────────────────────────
# LOAD VIDEO SEQUENCES
# ─────────────────────────────────────────────────────────────────────────────

def load_video_sequences(seq_path, action_mapping_path, sequence_length):
    """Load original video sequences using existing action_mapping.json."""
    print(f"\n[VIDEO] Loading from {seq_path}...")
    with open(action_mapping_path) as f:
        mapping = json.load(f)                              # {"0": "BRING", ...}
    action_names = [mapping[str(i)] for i in range(len(mapping))]

    X, y = [], []
    for label_idx, class_name in enumerate(action_names):
        folder = Path(seq_path) / class_name
        if not folder.exists():
            continue
        for npy in sorted(folder.glob('*.npy')):
            if '_static' in npy.stem:
                continue                                    # skip images here
            seq = np.load(npy).astype(np.float32)
            padded = np.zeros((sequence_length, 1662), dtype=np.float32)
            padded[:min(len(seq), sequence_length)] = seq[:min(len(seq), sequence_length)]
            X.append(padded)
            y.append(label_idx)

    print(f"  Video samples: {len(X)}  |  Classes: {len(action_names)}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), action_names


# ─────────────────────────────────────────────────────────────────────────────
# LOAD IMAGE SEQUENCES
# ─────────────────────────────────────────────────────────────────────────────

def load_image_sequences(seq_path, sequence_length, target_per_class=30):
    """
    Load static image sequences with DYNAMIC augmentation.

    Each ISL class is augmented until it reaches target_per_class samples.
    Classes with more real images need fewer augmentations.

    Args:
        target_per_class: Target number of samples per ISL class.
                          Set to 0 to disable augmentation.
    """
    print(f"\n[IMAGE] Loading (target {target_per_class} samples/class)...")
    folders = sorted([d for d in Path(seq_path).iterdir() if d.is_dir()])

    image_classes = []
    raw = []                    # [(class_name, array), ...]

    for folder in folders:
        static_files = sorted(folder.glob('*_static.npy'))
        if not static_files:
            continue
        image_classes.append(folder.name)

        # Load all real images for this class
        real_seqs = []
        for npy in static_files:
            seq = np.load(npy).astype(np.float32)
            padded = np.zeros((sequence_length, 1662), dtype=np.float32)
            padded[:min(len(seq), sequence_length)] = seq[:min(len(seq), sequence_length)]
            real_seqs.append(padded)
            raw.append((folder.name, padded))

        # Dynamic augmentation: fill up to target_per_class
        if target_per_class > 0:
            n_real    = len(real_seqs)
            n_needed  = max(0, target_per_class - n_real)
            if n_needed > 0:
                # Cycle through real images and augment
                for i in range(n_needed):
                    base = real_seqs[i % n_real]
                    aug  = augment_static_sequence(base, n=1)[0]
                    raw.append((folder.name, aug))

    # Count per class for reporting
    from collections import Counter
    counts = Counter(name for name, _ in raw)
    avg    = np.mean(list(counts.values())) if counts else 0

    print(f"  Image classes: {len(image_classes)}")
    print(f"  Image samples: {len(raw)}  (avg {avg:.1f}/class, target={target_per_class})")
    return raw, image_classes


# ─────────────────────────────────────────────────────────────────────────────
# MERGE VOCABULARIES
# ─────────────────────────────────────────────────────────────────────────────

def merge_vocabularies(video_names, image_names):
    """
    Merge video and image class lists into one combined vocabulary.
    Video classes keep their existing indices; new image classes are appended.

    Returns:
        all_names:    combined list (video_names + new_image_only_names)
        name_to_idx:  {class_name: new_index}
        new_classes:  list of image classes not in video vocabulary
    """
    video_set   = set(video_names)
    new_classes = [n for n in image_names if n not in video_set]

    all_names   = video_names + new_classes
    name_to_idx = {name: i for i, name in enumerate(all_names)}

    print(f"\n[VOCAB] Video classes:     {len(video_names)}")
    print(f"[VOCAB] Image-only classes: {len(new_classes)}")
    print(f"[VOCAB] Combined:           {len(all_names)}")
    if new_classes:
        print(f"        New classes (sample): {new_classes[:5]} ...")
    return all_names, name_to_idx, new_classes


# ─────────────────────────────────────────────────────────────────────────────
# BUILD EXPANDED MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_expanded_model(old_model: tf.keras.Model, new_num_classes: int) -> tf.keras.Model:
    """
    Clone old model architecture, transfer weights, attach new output head.

    Layers transferred (same weights):
      - All layers EXCEPT the final Dense output head

    New output head:
      - dense1 weights transferred (same size)
      - output: Dense(new_num_classes)  ← NEW, trained from scratch
    """
    # Re-use everything up to (but not including) the old output layer
    second_to_last = old_model.layers[-2].output      # output of 'dense1'

    new_output = layers.Dense(
        new_num_classes, activation='softmax', name='output'
    )(second_to_last)

    new_model = Model(inputs=old_model.input, outputs=new_output,
                      name='WordModel_Combined')

    # Copy weights from old model except output layer
    for new_layer in new_model.layers:
        try:
            old_layer = old_model.get_layer(new_layer.name)
            new_layer.set_weights(old_layer.get_weights())
        except (ValueError, Exception):
            pass                                       # output layer — skip

    print(f"\n[MODEL] Expanded output: {old_model.output_shape[-1]} → {new_num_classes} classes")
    return new_model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train_combined(
    model_path:          str,
    seq_path:            str,
    save_path:           str,
    action_mapping_path: str,
    sequence_length:     int   = None,
    lr:                  float = 5e-4,
    epochs:              int   = 200,
    batch_size:          int   = 32,
    val_split:           float = 0.1,
):
    seq_path    = seq_path    or str(SEQUENCE_PATH)
    save_path   = save_path   or str(CHECKPOINT_DIR / 'best_model_combined')
    action_mapping_path = action_mapping_path or str(CHECKPOINT_DIR / 'action_mapping.json')

    print("=" * 60)
    print("COMBINED VOCABULARY TRAINING (Video + Images)")
    print("=" * 60)

    # ── Load old model to get sequence_length ────────────────────────────────
    print(f"\nLoading base model from {model_path}...")
    old_model   = tf.keras.models.load_model(model_path)
    seq_len     = sequence_length or old_model.input_shape[1]
    print(f"  Sequence length: {seq_len}")

    # ── Load data ─────────────────────────────────────────────────────────────
    X_vid, y_vid, video_names = load_video_sequences(seq_path, action_mapping_path, seq_len)

    # Load RAW images (no augmentation yet) — needed for clean split
    raw_images_real, image_names = load_image_sequences(seq_path, seq_len,
                                                         target_per_class=0)  # no aug

    # ── Merge vocabularies ────────────────────────────────────────────────────
    all_names, name_to_idx, new_classes = merge_vocabularies(video_names, image_names)
    new_num_classes = len(all_names)

    # Index image labels
    X_img_real = np.array([x for _, x in raw_images_real], dtype=np.float32)
    y_img_real = np.array([name_to_idx[name] for name, _ in raw_images_real], dtype=np.int32)

    # ── Split image data BEFORE augmentation (avoid leakage) ─────────────────
    from sklearn.model_selection import train_test_split

    if len(X_img_real) > 1:
        X_img_tr, X_img_val, y_img_tr, y_img_val = train_test_split(
            X_img_real, y_img_real,
            test_size=val_split,
            random_state=42
            # no stratify — too many classes vs too few image samples
        )
    else:
        X_img_tr, y_img_tr = X_img_real, y_img_real
        X_img_val, y_img_val = X_img_real, y_img_real

    # ── Augment ONLY training image split ────────────────────────────────────
    samples_per_class = len(X_vid) / max(len(video_names), 1)
    target_per_class  = int(samples_per_class)
    print(f"\n  Auto target per class: {target_per_class} (avg video samples/class)")

    # Augment training images to reach target
    X_img_tr_aug, y_img_tr_aug = list(X_img_tr), list(y_img_tr)
    from collections import Counter
    counts = Counter(y_img_tr.tolist())
    for label_idx in np.unique(y_img_tr):
        n_real   = counts[label_idx]
        n_needed = max(0, target_per_class - n_real)
        idxs     = np.where(y_img_tr == label_idx)[0]
        for i in range(n_needed):
            base = X_img_tr[idxs[i % len(idxs)]]
            aug  = augment_static_sequence(base, n=1)[0]
            X_img_tr_aug.append(aug)
            y_img_tr_aug.append(label_idx)

    X_img_tr_aug = np.array(X_img_tr_aug, dtype=np.float32)
    y_img_tr_aug = np.array(y_img_tr_aug, dtype=np.int32)

    print(f"  Image train (after aug): {len(X_img_tr_aug)}")
    print(f"  Image val   (real only): {len(X_img_val)}")

    # ── Combine train + val separately ────────────────────────────────────────
    # Video split
    X_v_tr, X_v_val, y_v_tr, y_v_val = train_test_split(
        X_vid, y_vid, test_size=val_split, stratify=y_vid, random_state=42
    )

    X_train = np.concatenate([X_v_tr,  X_img_tr_aug], axis=0)
    y_train = np.concatenate([y_v_tr,  y_img_tr_aug], axis=0)
    X_val   = np.concatenate([X_v_val, X_img_val],    axis=0)
    y_val   = np.concatenate([y_v_val, y_img_val],    axis=0)

    # Shuffle train
    idx_tr  = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx_tr], y_train[idx_tr]

    y_train_cat = tf.keras.utils.to_categorical(y_train, new_num_classes)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,   new_num_classes)

    print(f"\n[DATA] Train: {len(X_train)}  Val: {len(X_val)}")
    print(f"       Video: {len(X_vid)}  |  Image real: {len(X_img_real)}")


    # ── Build expanded model ──────────────────────────────────────────────────
    model = build_expanded_model(old_model, new_num_classes)

    # Class weights computed from training set only
    cw_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict  = dict(enumerate(cw_array))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy'],
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=10, min_lr=1e-6, verbose=1),
        ModelCheckpoint(save_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("TRAINING on combined vocabulary")
    print(f"  Classes:    {new_num_classes}  ({len(video_names)} video + {len(new_classes)} new)")
    print(f"  Train:      {len(X_train)}  (video + augmented images)")
    print(f"  Val:        {len(X_val)}    (real only — no augmentation)")
    print(f"  lr:         {lr}")
    print(f"{'=' * 60}\n")

    model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=cw_dict,
        verbose=1,
    )

    # ── Save action mapping ───────────────────────────────────────────────────
    mapping_out = {str(i): name for i, name in enumerate(all_names)}
    mapping_path = Path(save_path).parent / 'action_mapping_combined.json'
    with open(mapping_path, 'w') as f:
        json.dump(mapping_out, f, indent=2)

    print(f"\n[DONE] Model saved to:          {save_path}")
    print(f"[DONE] Action mapping saved to: {mapping_path}")
    print(f"[DONE] Next: python src/utils/build_dictionary.py")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',          type=str,   required=True)
    parser.add_argument('--seq_path',            type=str,   default=None)
    parser.add_argument('--save_path',           type=str,   default=None)
    parser.add_argument('--action_mapping_path', type=str,   default=None)
    parser.add_argument('--lr',                  type=float, default=5e-4)
    parser.add_argument('--epochs',              type=int,   default=200)
    parser.add_argument('--batch_size',          type=int,   default=32)
    args = parser.parse_args()

    train_combined(
        model_path=args.model_path,
        seq_path=args.seq_path,
        save_path=args.save_path,
        action_mapping_path=args.action_mapping_path,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
