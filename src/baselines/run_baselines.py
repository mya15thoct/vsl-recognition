"""
Baseline Comparison Runner
Paper: "Multi-Stream MLP–BiLSTM with Temporal Attention
        for Isolated Sign Language Recognition"

Usage:
  python src/baselines/run_baselines.py
  python src/baselines/run_baselines.py --baselines lstm lstm_gru
  python src/baselines/run_baselines.py --force --seed 42
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
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

from config import (
    SEQUENCE_PATH, RECOGNITION_DIR, TRAINING_CONFIG, SEQUENCE_LENGTH
)
from training.data_loader import load_sequences, split_data, create_tf_dataset
from baselines.models import BASELINES, BASELINE_LABELS

BASELINES_DIR = RECOGNITION_DIR / 'baselines'
BASELINES_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = BASELINES_DIR / 'baselines_summary.csv'


def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"[GPU] Using {gpus[0].name}")
    else:
        print("[GPU] No GPU detected – using CPU")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Seed] Global seed set to {seed}")


def load_data():
    print("\n" + "=" * 70)
    print("LOADING DATA")
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

    train_ds = create_tf_dataset(X_train, y_train, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
    val_ds   = create_tf_dataset(X_val,   y_val,   batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)
    test_ds  = create_tf_dataset(X_test,  y_test,  batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)

    present_classes = np.unique(y_train)
    cw_array = compute_class_weight('balanced', classes=present_classes, y=y_train)
    class_weight_dict = {i: 1.0 for i in range(num_classes)}
    for cls, w in zip(present_classes, cw_array):
        class_weight_dict[int(cls)] = float(w)
    y_test_labels = y_test.copy()

    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    del X_train, X_val, X_test, y_train, y_val, y_test
    gc.collect()

    print(f"Data ready: train={n_train}  val={n_val}  test={n_test}")
    print(f"Sequence length: {sequence_length}  |  Classes: {num_classes}")

    return {
        'train_ds': train_ds, 'val_ds': val_ds, 'test_ds': test_ds,
        'action_names': action_names, 'num_classes': num_classes,
        'sequence_length': sequence_length,
        'class_weight_dict': class_weight_dict,
        'y_test_labels': y_test_labels,
    }


def train_baseline(key, data, force=False):
    label  = BASELINE_LABELS[key]
    outdir = BASELINES_DIR / key
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"BASELINE: {key}  –  {label}")
    print("=" * 70)

    history_path    = outdir / 'history.json'
    best_model_path = outdir / 'best_model'
    train_time_min  = None

    if history_path.exists() and not force:
        print("[RESUME] Loading saved model for re-evaluation")
        model = tf.keras.models.load_model(str(best_model_path))
        with open(history_path) as f:
            history_dict = json.load(f)
        best_epoch = int(np.argmax(history_dict['val_accuracy'])) + 1
        prev = outdir / 'results.json'
        if prev.exists():
            with open(prev) as f:
                train_time_min = json.load(f).get('train_time_min')
    else:
        build_fn = BASELINES[key]
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
        print(f"Total parameters: {model.count_params():,}")

        callbacks = [
            ModelCheckpoint(str(best_model_path), monitor='val_accuracy',
                            save_best_only=True, save_format='tf', verbose=1),
            EarlyStopping(monitor='val_accuracy',
                          patience=TRAINING_CONFIG['early_stopping_patience'],
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=TRAINING_CONFIG['reduce_lr_patience'], verbose=1),
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

        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(outdir / 'history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)

        best_epoch = int(np.argmax(history_dict['val_accuracy'])) + 1
        del history

    n_params = model.count_params()
    test_loss, test_acc = model.evaluate(data['test_ds'], verbose=0)

    y_pred_chunks = []
    for x_batch, _ in data['test_ds']:
        probs = model(x_batch, training=False).numpy()
        y_pred_chunks.append(probs.argmax(axis=1))
    y_pred = np.concatenate(y_pred_chunks)
    y_true = data['y_test_labels']

    report    = classification_report(y_true, y_pred,
                                       labels=list(range(len(data['action_names']))),
                                       target_names=data['action_names'],
                                       output_dict=True, zero_division=0)
    macro_f1  = report['macro avg']['f1-score']
    macro_pre = report['macro avg']['precision']
    macro_rec = report['macro avg']['recall']

    print(f"\n{'─'*50}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Macro F1      : {macro_f1*100:.2f}%")
    print(f"  Precision     : {macro_pre*100:.2f}%")
    print(f"  Recall        : {macro_rec*100:.2f}%")
    print(f"  Best Epoch    : {best_epoch}")
    print(f"  Train Time    : {train_time_min} min")
    print(f"  Parameters    : {n_params:,}")
    print(f"{'─'*50}")

    results = {
        'baseline': key, 'label': label,
        'test_accuracy': float(test_acc), 'test_loss': float(test_loss),
        'macro_f1': float(macro_f1), 'macro_precision': float(macro_pre),
        'macro_recall': float(macro_rec),
        'best_epoch': best_epoch, 'train_time_min': train_time_min,
        'n_params': n_params, 'status': 'done',
        'classification_report': report,
    }
    with open(outdir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    del model, y_pred, y_pred_chunks
    tf.keras.backend.clear_session()
    gc.collect()
    return results


def update_summary(all_results):
    rows = [{
        'Baseline':         r['baseline'],
        'Label':            r['label'],
        'Test Acc (%)':     round(r['test_accuracy'] * 100, 2),
        'Macro F1 (%)':     round(r['macro_f1'] * 100, 2),
        'Precision (%)':    round(r['macro_precision'] * 100, 2),
        'Recall (%)':       round(r['macro_recall'] * 100, 2),
        'Best Epoch':       r['best_epoch'],
        'Train Time (min)': r['train_time_min'],
        'Params':           r['n_params'],
        'Status':           r['status'],
    } for r in all_results]
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[Summary] Updated: {SUMMARY_CSV}")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baselines', nargs='+', choices=list(BASELINES.keys()),
                        default=list(BASELINES.keys()))
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    configure_gpu()
    if args.seed is not None:
        set_seed(args.seed)
        global BASELINES_DIR, SUMMARY_CSV
        BASELINES_DIR = RECOGNITION_DIR / 'baselines' / f'seed_{args.seed}'
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_CSV = BASELINES_DIR / 'baselines_summary.csv'

    data = load_data()

    # Load existing results for baselines NOT in current run
    all_results = []
    for key in BASELINES:
        if key in args.baselines:
            continue
        result_file = BASELINES_DIR / key / 'results.json'
        if result_file.exists():
            with open(result_file) as f:
                all_results.append(json.load(f))

    for key in args.baselines:
        try:
            r = train_baseline(key, data, force=args.force)
            all_results.append(r)
        except Exception as e:
            print(f"\n[ERROR] {key} failed: {e}")
            import traceback; traceback.print_exc()
            tf.keras.backend.clear_session(); gc.collect()
        update_summary(all_results)

    print("\n" + "=" * 70)
    print("BASELINES COMPLETE")
    print(f"Results saved to: {BASELINES_DIR}")
    print("=" * 70)

    # ── Final summary table ───────────────────────────────────────────────────
    done = [r for r in all_results if r.get('test_accuracy') is not None]
    if done:
        print("\n" + "=" * 95)
        print("  FINAL SUMMARY")
        print("=" * 95)
        hdr = f"  {'Baseline':<28} {'Test Acc':>9} {'Macro F1':>9} {'Precision':>10} {'Recall':>8} {'Best Ep':>8} {'Time(m)':>8} {'Params':>10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in done:
            acc  = f"{r['test_accuracy']*100:.2f}%"
            f1   = f"{r['macro_f1']*100:.2f}%"
            prec = f"{r['macro_precision']*100:.2f}%"
            rec  = f"{r['macro_recall']*100:.2f}%"
            ep   = str(r.get('best_epoch', '—'))
            tm   = str(r.get('train_time_min', '—'))
            p    = f"{r['n_params']:,}" if r.get('n_params') else "—"
            print(f"  {r['baseline']:<28} {acc:>9} {f1:>9} {prec:>10} {rec:>8} {ep:>8} {tm:>8} {p:>10}")
        print("=" * 95 + "\n")


if __name__ == '__main__':
    main()
