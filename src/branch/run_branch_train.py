"""
Train one branch variant (mlp or transformer) for one seed.
Called as subprocess by run_branch_comparison.py.

Usage:
  python src/branch/run_branch_train.py --branch mlp --seed 42
  python src/branch/run_branch_train.py --branch transformer --seed 0
"""
import sys
import os
import gc
import json
import time
import random
import argparse
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

import numpy as np


def configure_gpu():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"[GPU] {gpus[0].name}")
    else:
        print("[GPU] No GPU — using CPU")


def set_seed(seed):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f"[Seed] {seed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', choices=['mlp', 'transformer'], required=True)
    parser.add_argument('--seed',   type=int, required=True)
    parser.add_argument('--force',  action='store_true')
    args = parser.parse_args()

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['OMP_NUM_THREADS']        = '8'
    os.environ['MODEL_TYPE']             = args.branch

    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report

    from config import SEQUENCE_LENGTH, TRAINING_CONFIG, RECOGNITION_DIR
    from training.data_loader import load_sequences, split_data, create_tf_dataset

    configure_gpu()
    set_seed(args.seed)

    out_dir = RECOGNITION_DIR / 'branch_comparison' / args.branch / f'seed_{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    result_file = out_dir / 'results.json'
    if result_file.exists() and not args.force:
        print(f"[SKIP] Already done: {result_file}")
        return

    print(f"\n{'='*70}")
    print(f"  BRANCH : {args.branch}")
    print(f"  SEED   : {args.seed}")
    print(f"  OUT    : {out_dir}")
    print(f"{'='*70}\n")

    # ── Load & split data ─────────────────────────────────────────────────────
    print("[1/4] Loading data...")
    X, y, action_names, is_original = load_sequences(target_length=SEQUENCE_LENGTH)
    num_classes = len(action_names)

    print("[2/4] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_size=TRAINING_CONFIG['train_split'],
        val_size=TRAINING_CONFIG['val_split'],
        is_original=is_original,
    )
    del X, y, is_original
    gc.collect()

    sequence_length = X_train.shape[1]
    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)

    batch    = TRAINING_CONFIG['batch_size']
    train_ds = create_tf_dataset(X_train, y_train, batch_size=batch, shuffle=True)
    val_ds   = create_tf_dataset(X_val,   y_val,   batch_size=batch, shuffle=False)
    test_ds  = create_tf_dataset(X_test,  y_test,  batch_size=batch, shuffle=False)

    present_classes = np.unique(y_train)
    cw_array        = compute_class_weight('balanced', classes=present_classes, y=y_train)
    cw_dict         = {i: 1.0 for i in range(num_classes)}
    for cls, w in zip(present_classes, cw_array):
        cw_dict[int(cls)] = float(w)
    y_test_labels = y_test.copy()

    del X_train, X_val, X_test, y_train, y_val, y_test
    gc.collect()

    # ── Build model ───────────────────────────────────────────────────────────
    print("[3/4] Building model...")
    if args.branch == 'transformer':
        from models.transformer.model import create_hybrid_transformer_model
        model = create_hybrid_transformer_model(
            num_classes=num_classes, sequence_length=sequence_length
        )
    else:
        from models.hybrid import create_hybrid_multistream_model
        model = create_hybrid_multistream_model(
            num_classes=num_classes, sequence_length=sequence_length
        )

    model.compile(
        optimizer=Adam(learning_rate=TRAINING_CONFIG['learning_rate']),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy'],
    )
    model.summary()
    n_params = model.count_params()
    print(f"Parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[4/4] Training  (train={n_train} val={n_val} test={n_test})...")
    callbacks = [
        ModelCheckpoint(str(out_dir / 'best_model'), monitor='val_accuracy',
                        save_best_only=True, save_format='tf', verbose=1),
        EarlyStopping(monitor='val_accuracy',
                      patience=TRAINING_CONFIG['early_stopping_patience'],
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=TRAINING_CONFIG['reduce_lr_patience'], verbose=1),
    ]

    t0      = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=TRAINING_CONFIG['epochs'],
        callbacks=callbacks,
        class_weight=cw_dict,
        verbose=1,
    )
    train_min = round((time.time() - t0) / 60, 1)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_epoch   = int(np.argmax(history_dict['val_accuracy'])) + 1

    y_pred = np.concatenate([
        model(xb, training=False).numpy().argmax(axis=1)
        for xb, _ in test_ds
    ])
    report    = classification_report(
        y_test_labels, y_pred,
        labels=list(range(num_classes)),
        target_names=action_names,
        output_dict=True, zero_division=0,
    )
    macro_f1  = report['macro avg']['f1-score']
    macro_pre = report['macro avg']['precision']
    macro_rec = report['macro avg']['recall']

    print(f"\n{'─'*60}")
    print(f"  Branch    : {args.branch}  |  Seed: {args.seed}")
    print(f"  Accuracy  : {test_acc*100:.2f}%")
    print(f"  Macro F1  : {macro_f1*100:.2f}%")
    print(f"  Precision : {macro_pre*100:.2f}%")
    print(f"  Recall    : {macro_rec*100:.2f}%")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Time      : {train_min} min")
    print(f"  Params    : {n_params:,}")
    print(f"{'─'*60}\n")

    results = {
        'branch':          args.branch,
        'seed':            args.seed,
        'test_accuracy':   float(test_acc),
        'test_loss':       float(test_loss),
        'macro_f1':        float(macro_f1),
        'macro_precision': float(macro_pre),
        'macro_recall':    float(macro_rec),
        'best_epoch':      best_epoch,
        'train_time_min':  train_min,
        'n_params':        n_params,
    }
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)

    print(f"[SAVED] {result_file}")

    del model, y_pred
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == '__main__':
    main()
