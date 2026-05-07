"""
Ablation Study Runner
Paper: "Multi-Stream MLP–BiLSTM with Cross-Part Contextual Gating
        for Isolated Sign Language Recognition"

Usage:
  # Run all variants
  python src/ablation/run_ablation.py

  # Run specific variants only
  python src/ablation/run_ablation.py --variants v0_baseline v4_unidirectional_lstm

  # Dry-run: build models and print summaries without training
  python src/ablation/run_ablation.py --dry-run

Results saved to:
  /mnt/ngan/recognition/ablation/
    ├── v0_baseline/
    │   ├── best_model/           ← SavedModel (best val_accuracy)
    │   ├── final_model/          ← SavedModel (after all epochs)
    │   ├── action_mapping.json
    │   ├── history.json          ← per-epoch train/val loss & accuracy
    │   └── results.json          ← test accuracy, macro F1, params count
    ├── v1_single_stream/
    │   └── ...
    ├── ...
    └── ablation_summary.csv      ← table of all variants (updated after each run)
"""

import sys
import json
import gc
import random
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# ── path setup ─────────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

from config import (
    SEQUENCE_PATH, RECOGNITION_DIR, TRAINING_CONFIG, SEQUENCE_LENGTH
)
from training.data_loader import load_sequences, split_data, create_tf_dataset
from ablation.models import VARIANTS, VARIANT_LABELS

# ── output root ─────────────────────────────────────────────────────────────
ABLATION_DIR = RECOGNITION_DIR / 'ablation'
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = ABLATION_DIR / 'ablation_summary.csv'


# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------

def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"[GPU] Using {gpus[0].name}")
    else:
        print("[GPU] No GPU detected – using CPU")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Seed] Global seed set to {seed}")


# ---------------------------------------------------------------------------
# Data loading  (done once, shared across all variants)
# ---------------------------------------------------------------------------

def load_data():
    """Load, split, and create tf.data.Datasets. Returns datasets + metadata."""
    print("\n" + "=" * 70)
    print("LOADING DATA  (shared across all ablation variants)")
    print("=" * 70)

    X, y, action_names, is_original = load_sequences(target_length=SEQUENCE_LENGTH)
    num_classes = len(action_names)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_size=TRAINING_CONFIG['train_split'],
        val_size=TRAINING_CONFIG['val_split'],
        is_original=is_original
    )
    del X, y, is_original
    gc.collect()

    sequence_length = X_train.shape[1]
    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)

    print("\nCreating TF datasets...")
    train_ds = create_tf_dataset(X_train, y_train,
                                  batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
    val_ds   = create_tf_dataset(X_val,   y_val,
                                  batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)
    test_ds  = create_tf_dataset(X_test,  y_test,
                                  batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)

    # Compute class weights once (from training labels, before freeing numpy)
    present_classes = np.unique(y_train)
    cw_array = compute_class_weight(
        class_weight='balanced',
        classes=present_classes,
        y=y_train
    )
    num_classes = len(action_names)
    class_weight_dict = {i: 1.0 for i in range(num_classes)}
    for cls, w in zip(present_classes, cw_array):
        class_weight_dict[int(cls)] = float(w)

    # Keep y_test labels for per-class metrics (from numpy, before freeing)
    y_test_labels = y_test.copy()

    # Free numpy arrays
    del X_train, X_val, X_test, y_train, y_val, y_test
    gc.collect()

    print(f"\nData ready: train={n_train}  val={n_val}  test={n_test}")
    print(f"Sequence length: {sequence_length}  |  Classes: {num_classes}")

    return {
        'train_ds':          train_ds,
        'val_ds':            val_ds,
        'test_ds':           test_ds,
        'action_names':      action_names,
        'num_classes':       num_classes,
        'sequence_length':   sequence_length,
        'class_weight_dict': class_weight_dict,
        'y_test_labels':     y_test_labels,
        'n_train':           n_train,
        'n_val':             n_val,
        'n_test':            n_test,
    }


# ---------------------------------------------------------------------------
# Single-variant training
# ---------------------------------------------------------------------------

def train_variant(variant_key, data, dry_run=False, force=False):
    """
    Build, train, evaluate one ablation variant.
    Saves all outputs under ABLATION_DIR / variant_key /.
    Returns a result dict.
    """
    label  = VARIANT_LABELS[variant_key]
    outdir = ABLATION_DIR / variant_key
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"VARIANT: {variant_key}")
    print(f"  {label}")
    print("=" * 70)

    best_model_path = outdir / 'best_model'
    history_path    = outdir / 'history.json'
    history_dict    = None
    best_epoch      = None
    train_time_min  = None
    trained_fresh   = False

    # ── Case 1: training already completed (history.json exists) → re-evaluate only
    # Note: best_model/ can exist mid-training (ModelCheckpoint saves each epoch),
    # so we check history.json which is only written AFTER training finishes.
    if history_path.exists() and not dry_run and not force:
        print(f"[RESUME] Training already done – loading best_model and re-evaluating")
        model    = tf.keras.models.load_model(str(best_model_path))
        n_params = model.count_params()
        print(f"Total parameters: {n_params:,}")

        with open(history_path) as f:
            history_dict = json.load(f)
        best_epoch = int(np.argmax(history_dict['val_accuracy'])) + 1
        # Recover train_time_min from previous results.json if it exists
        prev_results = outdir / 'results.json'
        if prev_results.exists():
            with open(prev_results) as f:
                train_time_min = json.load(f).get('train_time_min')

    # ── Case 2: train from scratch ──────────────────────────────────────────
    else:
        build_fn = VARIANTS[variant_key]
        model = build_fn(
            num_classes=data['num_classes'],
            sequence_length=data['sequence_length']
        )
        model.compile(
            optimizer=Adam(learning_rate=TRAINING_CONFIG['learning_rate']),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        model.summary()
        n_params = model.count_params()
        print(f"\nTotal parameters: {n_params:,}")

        if dry_run:
            print("[DRY-RUN] Skipping training.")
            tf.keras.backend.clear_session()
            gc.collect()
            return {
                'variant':    variant_key,
                'label':      label,
                'n_params':   n_params,
                'test_acc':   None,
                'macro_f1':   None,
                'best_epoch': None,
                'train_time': None,
                'status':     'dry-run',
            }

        log_dir = ABLATION_DIR.parent / 'logs' / 'ablation' / variant_key
        log_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                filepath=str(outdir / 'best_model'),
                monitor='val_accuracy',
                save_best_only=True,
                save_format='tf',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=TRAINING_CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=TRAINING_CONFIG['reduce_lr_patience'],
                verbose=1
            ),
            TensorBoard(
                log_dir=str(log_dir / 'fit'),
                histogram_freq=0
            ),
        ]

        t0 = time.time()
        history = model.fit(
            data['train_ds'],
            validation_data=data['val_ds'],
            epochs=TRAINING_CONFIG['epochs'],
            callbacks=callbacks,
            class_weight=data['class_weight_dict'],
            verbose=1
        )
        train_time_min = round((time.time() - t0) / 60, 2)
        trained_fresh  = True

        history_dict = {k: [float(v) for v in vals]
                        for k, vals in history.history.items()}
        with open(outdir / 'history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)

        model.save(str(outdir / 'final_model'), save_format='tf')

        mapping = {i: name for i, name in enumerate(data['action_names'])}
        with open(outdir / 'action_mapping.json', 'w') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        best_epoch = int(np.argmax(history_dict['val_accuracy'])) + 1

    # ── Evaluate (shared by both cases) ─────────────────────────────────────
    test_loss, test_acc = model.evaluate(data['test_ds'], verbose=0)

    y_pred_chunks = []
    for x_batch, _ in data['test_ds']:
        probs = model(x_batch, training=False).numpy()
        y_pred_chunks.append(probs.argmax(axis=1))
        del probs
    y_pred = np.concatenate(y_pred_chunks)
    del y_pred_chunks
    y_true = data['y_test_labels']

    report    = classification_report(
        y_true, y_pred,
        labels=list(range(len(data['action_names']))),
        target_names=data['action_names'],
        output_dict=True,
        zero_division=0
    )
    macro_f1  = report['macro avg']['f1-score']
    macro_pre = report['macro avg']['precision']
    macro_rec = report['macro avg']['recall']

    print(f"\n{'─'*50}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Macro F1      : {macro_f1*100:.2f}%")
    print(f"  Macro Precision: {macro_pre*100:.2f}%")
    print(f"  Macro Recall  : {macro_rec*100:.2f}%")
    print(f"  Best Epoch    : {best_epoch}")
    print(f"  Train Time    : {train_time_min} min")
    print(f"  Parameters    : {n_params:,}")
    print(f"{'─'*50}")

    results = {
        'variant':         variant_key,
        'label':           label,
        'test_accuracy':   float(test_acc),
        'test_loss':       float(test_loss),
        'macro_f1':        float(macro_f1),
        'macro_precision': float(macro_pre),
        'macro_recall':    float(macro_rec),
        'best_epoch':      best_epoch,
        'train_time_min':  train_time_min,
        'n_params':        n_params,
        'status':          'done',
        'classification_report': report,
    }
    with open(outdir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if trained_fresh:
        del history
    del model, y_pred
    tf.keras.backend.clear_session()
    gc.collect()

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def update_summary(all_results):
    """Write/update ablation_summary.csv from list of result dicts."""
    rows = []
    for r in all_results:
        rows.append({
            'Variant':          r['variant'],
            'Label':            r['label'],
            'Test Acc (%)':     round(r['test_acc'] * 100, 2) if r.get('test_acc') is not None else '',
            'Macro F1 (%)':     round(r['macro_f1'] * 100, 2) if r.get('macro_f1') is not None else '',
            'Best Epoch':       r.get('best_epoch', ''),
            'Train Time (min)': r.get('train_time_min', ''),
            'Params':           r.get('n_params', ''),
            'Status':           r.get('status', ''),
        })
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[Summary] Updated: {SUMMARY_CSV}")
    print(df.to_string(index=False))


def _flatten_result(r):
    """Normalize keys from train_variant output for update_summary."""
    return {
        'variant':          r.get('variant'),
        'label':            r.get('label'),
        'test_acc':         r.get('test_accuracy') if 'test_accuracy' in r else r.get('test_acc'),
        'macro_f1':         r.get('macro_f1'),
        'macro_precision':  r.get('macro_precision'),
        'macro_recall':     r.get('macro_recall'),
        'best_epoch':       r.get('best_epoch'),
        'train_time_min':   r.get('train_time_min'),
        'n_params':         r.get('n_params'),
        'status':           r.get('status'),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run ablation study for VSL recognition paper'
    )
    parser.add_argument(
        '--variants', nargs='+',
        choices=list(VARIANTS.keys()),
        default=list(VARIANTS.keys()),
        help='Which variants to run (default: all)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Build models and print summaries without training'
    )
    parser.add_argument(
        '--skip-done', action='store_true',
        help='Skip variants that already have results.json (useful for resuming)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force retrain all variants from scratch, ignoring any saved models'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed (default: None = no seeding). Results saved to ablation/seed_{N}/ when specified'
    )
    args = parser.parse_args()

    configure_gpu()
    if args.seed is not None:
        set_seed(args.seed)
        global ABLATION_DIR, SUMMARY_CSV
        ABLATION_DIR = RECOGNITION_DIR / 'ablation' / f'seed_{args.seed}'
        ABLATION_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_CSV  = ABLATION_DIR / 'ablation_summary.csv'
        print(f"[Seed] Results → {ABLATION_DIR}")

    # Load data once
    data = load_data()

    all_results = []
    for vk in args.variants:
        result_file = ABLATION_DIR / vk / 'results.json'

        if args.skip_done and result_file.exists():
            print(f"\n[SKIP] {vk} – results.json already exists")
            with open(result_file) as f:
                r = json.load(f)
            all_results.append(_flatten_result(r))
            continue

        try:
            r = train_variant(vk, data, dry_run=args.dry_run, force=args.force)
            all_results.append(_flatten_result(r))
        except Exception as e:
            print(f"\n[ERROR] Variant {vk} failed: {e}")
            import traceback
            traceback.print_exc()
            tf.keras.backend.clear_session()
            gc.collect()
            all_results.append({
                'variant': vk, 'label': VARIANT_LABELS[vk],
                'test_acc': None, 'macro_f1': None,
                'best_epoch': None, 'train_time_min': None,
                'n_params': None, 'status': f'ERROR: {e}',
            })

        # Update summary after each variant (so partial results are always saved)
        update_summary(all_results)

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print(f"Results saved to: {ABLATION_DIR}")
    print("=" * 70)
    update_summary(all_results)

    # ── Clean final summary ───────────────────────────────────────────────────
    done = [r for r in all_results if r.get('test_acc') is not None]
    if done:
        print("\n" + "=" * 95)
        print("  FINAL SUMMARY")
        print("=" * 95)
        hdr = f"  {'Variant':<28} {'Test Acc':>9} {'Macro F1':>9} {'Precision':>10} {'Recall':>8} {'Best Ep':>8} {'Time(m)':>8} {'Params':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in done:
            acc  = f"{r['test_acc']*100:.2f}%"       if r.get('test_acc')      is not None else "—"
            f1   = f"{r['macro_f1']*100:.2f}%"       if r.get('macro_f1')      is not None else "—"
            prec = f"{r['macro_precision']*100:.2f}%" if r.get('macro_precision') is not None else "—"
            rec  = f"{r['macro_recall']*100:.2f}%"   if r.get('macro_recall')  is not None else "—"
            ep   = str(r.get('best_epoch', '—'))
            tm   = str(r.get('train_time_min', '—'))
            p    = f"{r['n_params']:,}" if r.get('n_params') else "—"
            print(f"  {r['variant']:<28} {acc:>9} {f1:>9} {prec:>10} {rec:>8} {ep:>8} {tm:>8} {p:>10}")
        print("=" * 95 + "\n")



if __name__ == '__main__':
    main()