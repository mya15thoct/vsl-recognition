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

import gc
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
from config import SEQUENCE_PATH, CHECKPOINT_DIR, ISL_SEQUENCE_PATH, ISL_CHECKPOINT_DIR
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
# COLLECT VIDEO PATHS  (no data loading — avoids RAM spike)
# ─────────────────────────────────────────────────────────────────────────────

def collect_video_paths(seq_path, action_mapping_path):
    """Return file paths + labels without loading .npy data into RAM."""
    print(f"\n[VIDEO] Scanning {seq_path}...")
    with open(action_mapping_path) as f:
        mapping = json.load(f)
    action_names = [mapping[str(i)] for i in range(len(mapping))]

    paths, labels = [], []
    for label_idx, class_name in enumerate(action_names):
        folder = Path(seq_path) / class_name
        if not folder.exists():
            continue
        for npy in sorted(folder.glob('*.npy')):
            if '_static' in npy.stem:
                continue
            paths.append(str(npy))
            labels.append(label_idx)

    print(f"  Video samples: {len(paths)}  |  Classes: {len(action_names)}")
    return paths, np.array(labels, dtype=np.int32), action_names


# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR-BASED tf.data BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_video_ds(paths, labels, seq_len, num_classes):
    """Dataset that loads each .npy file on demand — O(1) RAM."""
    def gen():
        for path, label in zip(paths, labels):
            seq    = np.load(path).astype(np.float32)
            padded = np.zeros((seq_len, 1662), dtype=np.float32)
            padded[:min(len(seq), seq_len)] = seq[:min(len(seq), seq_len)]
            oh     = np.zeros(num_classes, dtype=np.float32)
            oh[label] = 1.0
            yield padded, oh

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(seq_len, 1662), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,),  dtype=tf.float32),
        )
    )


def _make_image_ds(X_real, y_real, num_classes, target_per_class, augment):
    """Dataset from real image arrays; augments on-the-fly if augment=True."""
    def gen():
        for label_idx in np.unique(y_real):
            real_seqs = X_real[y_real == label_idx]
            n_real    = len(real_seqs)
            n_needed  = max(0, target_per_class - n_real) if augment else 0
            oh        = np.zeros(num_classes, dtype=np.float32)
            oh[label_idx] = 1.0
            for seq in real_seqs:
                yield seq, oh
            for i in range(n_needed):
                yield augment_static_sequence(real_seqs[i % n_real], n=1)[0], oh

    seq_len = X_real.shape[1]
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(seq_len, 1662), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,),  dtype=tf.float32),
        )
    )


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
    Matching is case-insensitive: 'Actor', 'ACTOR', 'actor' all map to the
    same class (canonical name taken from the video side).

    Returns:
        all_names:    combined list (video_names + new_image_only_names)
        name_to_idx:  {class_name: new_index}  — includes case aliases
        new_classes:  list of image classes not in video vocabulary
    """
    video_lower = {n.lower(): n for n in video_names}   # lower → canonical

    new_classes = [n for n in image_names if n.lower() not in video_lower]

    all_names   = video_names + new_classes
    name_to_idx = {name: i for i, name in enumerate(all_names)}

    # Add aliases so image folder names (different case) resolve to correct index
    for img_name in image_names:
        if img_name not in name_to_idx:
            canonical = video_lower.get(img_name.lower())
            if canonical:
                name_to_idx[img_name] = name_to_idx[canonical]

    merged = len(image_names) - len(new_classes)
    print(f"\n[VOCAB] Video classes:      {len(video_names)}")
    print(f"[VOCAB] Image-only classes: {len(new_classes)}")
    print(f"[VOCAB] Merged (same word):  {merged}")
    print(f"[VOCAB] Combined:            {len(all_names)}")
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
    image_seq_path:      str   = None,
    sequence_length:     int   = None,
    lr:                  float = 5e-4,
    epochs:              int   = 200,
    batch_size:          int   = 32,
    val_split:           float = 0.1,
):
    seq_path       = seq_path       or str(SEQUENCE_PATH)
    image_seq_path = image_seq_path or str(ISL_SEQUENCE_PATH)
    save_path      = save_path      or str(ISL_CHECKPOINT_DIR / 'best_model_combined')
    action_mapping_path = action_mapping_path or str(CHECKPOINT_DIR / 'action_mapping.json')

    print("=" * 60)
    print("COMBINED VOCABULARY TRAINING (Video + Images)")
    print("=" * 60)

    # ── Load old model to get sequence_length ────────────────────────────────
    print(f"\nLoading base model from {model_path}...")
    old_model   = tf.keras.models.load_model(model_path)
    seq_len     = sequence_length or old_model.input_shape[1]
    print(f"  Sequence length: {seq_len}")

    # ── Collect video paths (no data loading) ────────────────────────────────
    vid_paths, vid_labels, video_names = collect_video_paths(seq_path, action_mapping_path)

    # ── Load RAW images into RAM (828 × seq_len × 1662 ≈ 1 GB — fine) ────────
    raw_images_real, image_names = load_image_sequences(image_seq_path, seq_len,
                                                         target_per_class=0)

    # ── Merge vocabularies ────────────────────────────────────────────────────
    all_names, name_to_idx, new_classes = merge_vocabularies(video_names, image_names)
    new_num_classes = len(all_names)

    # ── Collect extra video paths for new ISL classes ─────────────────────────
    new_vid_paths, new_vid_labels = [], []
    for class_name in new_classes:
        folder = Path(seq_path) / class_name
        if not folder.exists():
            continue
        label_idx = name_to_idx[class_name]
        for npy in sorted(folder.glob('*.npy')):
            if '_static' in npy.stem:
                continue
            new_vid_paths.append(str(npy))
            new_vid_labels.append(label_idx)

    if new_vid_paths:
        print(f"\n  [NEW CLASS VIDEOS] {len(new_vid_paths)} sequences for "
              f"{len(new_classes)} new ISL classes → added to training")
        vid_paths  = vid_paths + new_vid_paths
        vid_labels = np.concatenate([vid_labels,
                                     np.array(new_vid_labels, dtype=np.int32)])

    # ── Index image labels ────────────────────────────────────────────────────
    X_img_real = np.array([x for _, x in raw_images_real], dtype=np.float32)
    y_img_real = np.array([name_to_idx[name] for name, _ in raw_images_real],
                          dtype=np.int32)
    del raw_images_real; gc.collect()

    # ── Split image data BEFORE augmentation (avoid leakage) ─────────────────
    from sklearn.model_selection import train_test_split

    if len(X_img_real) > 1:
        X_img_tr, X_img_val, y_img_tr, y_img_val = train_test_split(
            X_img_real, y_img_real, test_size=val_split, random_state=42
        )
    else:
        X_img_tr,  y_img_tr  = X_img_real, y_img_real
        X_img_val, y_img_val = X_img_real, y_img_real
    del X_img_real; gc.collect()

    # ── Split video paths (stratified, no data loading) ───────────────────────
    paths_tr, paths_val, labels_tr, labels_val = train_test_split(
        vid_paths, vid_labels, test_size=val_split,
        stratify=vid_labels, random_state=42
    )

    # ── Compute class weights from all training labels ────────────────────────
    target_per_class = int(len(vid_paths) / max(len(video_names), 1))
    from collections import Counter
    img_aug_labels = []
    for lbl in np.unique(y_img_tr):
        n_real   = (y_img_tr == lbl).sum()
        n_needed = max(0, target_per_class - n_real)
        img_aug_labels.extend([lbl] * (n_real + n_needed))
    all_train_labels = np.concatenate([labels_tr,
                                       np.array(img_aug_labels, dtype=np.int32)])
    present_classes = np.unique(all_train_labels)
    cw_array = compute_class_weight('balanced', classes=present_classes,
                                    y=all_train_labels)
    # Fill ALL new_num_classes keys so Keras weight tensor covers every index
    cw_dict = {i: 1.0 for i in range(new_num_classes)}
    for cls, w in zip(present_classes, cw_array):
        cw_dict[int(cls)] = float(w)

    n_train = len(paths_tr) + len(img_aug_labels)
    n_val   = len(paths_val) + len(X_img_val)
    print(f"\n  Auto target per class: {target_per_class} (avg video samples/class)")
    print(f"  Image train (after aug): {len(img_aug_labels)}")
    print(f"  Image val   (real only): {len(X_img_val)}")
    print(f"\n[DATA] Train: {n_train}  Val: {n_val}")
    print(f"       Video paths: {len(vid_paths)}  |  Image real: {len(X_img_tr) + len(X_img_val)}")

    # ── Build tf.data datasets (generators — no full-data RAM spike) ──────────
    train_ds = (
        _make_video_ds(paths_tr, labels_tr, seq_len, new_num_classes)
        .concatenate(_make_image_ds(X_img_tr, y_img_tr, new_num_classes,
                                    target_per_class, augment=True))
        .shuffle(buffer_size=2000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        _make_video_ds(paths_val, labels_val, seq_len, new_num_classes)
        .concatenate(_make_image_ds(X_img_val, y_img_val, new_num_classes,
                                    target_per_class=0, augment=False))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    print("[RAM] Using generator pipeline — no full-data spike.")

    # ── Build expanded model ──────────────────────────────────────────────────
    model = build_expanded_model(old_model, new_num_classes)
    del old_model; gc.collect()

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
    print(f"  Train:      {n_train}  (video + augmented images)")
    print(f"  Val:        {n_val}    (real only — no augmentation)")
    print(f"  lr:         {lr}")
    print(f"{'=' * 60}\n")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
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
    parser.add_argument('--model_path',          type=str, required=True)
    parser.add_argument('--seq_path',            type=str, default=None)
    parser.add_argument('--image_seq_path',      type=str, default=None,
                        help='Path to ISL-Sequences (default: ISL_SEQUENCE_PATH from config)')
    parser.add_argument('--save_path',           type=str, default=None)
    parser.add_argument('--action_mapping_path', type=str, default=None)
    parser.add_argument('--lr',                  type=float, default=5e-4)
    parser.add_argument('--epochs',              type=int,   default=200)
    parser.add_argument('--batch_size',          type=int,   default=32)
    args = parser.parse_args()

    train_combined(
        model_path=args.model_path,
        seq_path=args.seq_path,
        image_seq_path=args.image_seq_path,
        save_path=args.save_path,
        action_mapping_path=args.action_mapping_path,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
