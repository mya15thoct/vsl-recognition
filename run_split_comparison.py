"""
Run model with different data-split protocols to match comparison papers.

Split configs:
  paper6  : 80% train / 20% test  (val = 10% taken from train side → 72/8/20)
             Paper [6] uses 80-20, no explicit val → we use small val for early stopping
  paper10 : Random stratified split, same ratio as our default (70/15/15) but
             ensures stratification is always applied (same as our default actually)

Results saved to:
  results/split_comparison/paper6/seed_<N>.json
  results/split_comparison/paper10/seed_<N>.json

Usage:
  python run_split_comparison.py
  python run_split_comparison.py --seeds 42 0 1
  python run_split_comparison.py --configs paper6
"""
import sys
import json
import os
import gc
import random
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).parent
SRC_DIR      = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

RESULTS_DIR  = PROJECT_ROOT / 'results' / 'split_comparison'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Split configurations
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_CONFIGS = {
    'paper6': {
        'label':      'Paper [6] — 80% train / 20% test',
        'train_size': 0.72,   # 72% train  (80% minus 8% for val)
        'val_size':   0.08,   # 8%  val    (so total train+val = 80%)
        # → test = 1 - 0.72 - 0.08 = 0.20  (20%)
        'stratify':   True,
    },
    'paper10': {
        'label':      'Paper [10] — Random stratified split (70/15/15)',
        'train_size': 0.70,
        'val_size':   0.15,
        # → test = 0.15
        'stratify':   True,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f"[Seed] Set to {seed}")


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


# ─────────────────────────────────────────────────────────────────────────────
# Training with custom split
# ─────────────────────────────────────────────────────────────────────────────

def run_with_split(config_name: str, cfg: dict, seed: int):
    """Train + evaluate using a specific split configuration."""
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report

    from config import SEQUENCE_LENGTH, TRAINING_CONFIG, RECOGNITION_DIR
    from training.data_loader import load_sequences, split_data, create_tf_dataset
    from training.evaluator import evaluate_model

    label = cfg['label']
    print(f"\n{'='*70}")
    print(f"  CONFIG  : {config_name}")
    print(f"  SPLIT   : train={cfg['train_size']:.0%}  val={cfg['val_size']:.0%}  "
          f"test={1-cfg['train_size']-cfg['val_size']:.0%}")
    print(f"  SEED    : {seed}")
    print(f"{'='*70}\n")

    out_dir = RECOGNITION_DIR / 'split_comparison' / config_name / f'seed_{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("[1/5] Loading sequences...")
    X, y, action_names, is_original = load_sequences(target_length=SEQUENCE_LENGTH)
    num_classes = len(action_names)

    # ── Split ────────────────────────────────────────────────────────────────
    print("\n[2/5] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_size=cfg['train_size'],
        val_size=cfg['val_size'],
        random_state=seed,          # use seed as random_state for reproducibility
        is_original=is_original,
    )
    del X, y, is_original
    gc.collect()

    sequence_length = X_train.shape[1]
    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)

    # ── TF datasets ──────────────────────────────────────────────────────────
    print("\n[3/5] Creating TF datasets...")
    batch = TRAINING_CONFIG['batch_size']
    train_ds = create_tf_dataset(X_train, y_train, batch_size=batch, shuffle=True)
    val_ds   = create_tf_dataset(X_val,   y_val,   batch_size=batch, shuffle=False)
    test_ds  = create_tf_dataset(X_test,  y_test,  batch_size=batch, shuffle=False)

    cw_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict  = dict(enumerate(cw_array))
    y_test_labels = y_test.copy()

    del X_train, X_val, X_test, y_train, y_val, y_test
    gc.collect()

    # ── Build model ──────────────────────────────────────────────────────────
    print("\n[4/5] Building model...")
    from config import MODEL_TYPE
    if MODEL_TYPE == 'transformer':
        from models.transformer.model import create_hybrid_transformer_model
        model = create_hybrid_transformer_model(num_classes=num_classes,
                                                sequence_length=sequence_length)
    else:
        from models.hybrid import create_hybrid_multistream_model
        model = create_hybrid_multistream_model(num_classes=num_classes,
                                                sequence_length=sequence_length)

    model.compile(
        optimizer=Adam(learning_rate=TRAINING_CONFIG['learning_rate']),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy'],
    )
    model.summary()

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\n[5/5] Training  (train={n_train} val={n_val} test={n_test})...")
    callbacks = [
        ModelCheckpoint(str(out_dir / 'best_model'), monitor='val_accuracy',
                        save_best_only=True, save_format='tf', verbose=1),
        EarlyStopping(monitor='val_accuracy',
                      patience=TRAINING_CONFIG['early_stopping_patience'],
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=TRAINING_CONFIG['reduce_lr_patience'], verbose=1),
    ]

    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds,
              epochs=TRAINING_CONFIG['epochs'],
              callbacks=callbacks, class_weight=cw_dict, verbose=1)
    train_min = round((time.time() - t0) / 60, 1)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    y_pred = np.concatenate([
        model(xb, training=False).numpy().argmax(axis=1)
        for xb, _ in test_ds
    ])
    report   = classification_report(y_test_labels, y_pred,
                                      labels=list(range(num_classes)),
                                      target_names=action_names,
                                      output_dict=True, zero_division=0)
    macro_f1  = report['macro avg']['f1-score']
    macro_pre = report['macro avg']['precision']
    macro_rec = report['macro avg']['recall']

    print(f"\n{'─'*60}")
    print(f"  Config    : {config_name}  |  Seed: {seed}")
    print(f"  Split     : train={n_train} / val={n_val} / test={n_test}")
    print(f"  Accuracy  : {test_acc*100:.2f}%")
    print(f"  Macro F1  : {macro_f1*100:.2f}%")
    print(f"  Precision : {macro_pre*100:.2f}%")
    print(f"  Recall    : {macro_rec*100:.2f}%")
    print(f"  Time      : {train_min} min")
    print(f"{'─'*60}\n")

    # ── Save results ─────────────────────────────────────────────────────────
    results = {
        'config':      config_name,
        'label':       label,
        'seed':        seed,
        'split': {
            'train':   n_train,
            'val':     n_val,
            'test':    n_test,
            'train_pct': cfg['train_size'],
            'val_pct':   cfg['val_size'],
            'test_pct':  round(1 - cfg['train_size'] - cfg['val_size'], 2),
        },
        'accuracy':    float(test_acc),
        'macro_f1':    float(macro_f1),
        'precision':   float(macro_pre),
        'recall':      float(macro_rec),
        'train_min':   train_min,
    }

    # Save to /mnt (full detail)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save to results/split_comparison/ (easy access)
    dst_dir = RESULTS_DIR / config_name
    dst_dir.mkdir(exist_ok=True)
    with open(dst_dir / f'seed_{seed}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[SAVED] {dst_dir / f'seed_{seed}.json'}")

    del model, y_pred
    tf.keras.backend.clear_session()
    gc.collect()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run model with paper-matching split configurations'
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=[42, 0, 1],
        help='Seeds to run (default: 42 0 1)'
    )
    parser.add_argument(
        '--configs', nargs='+', choices=list(SPLIT_CONFIGS.keys()),
        default=list(SPLIT_CONFIGS.keys()),
        help='Split configs to run (default: all)'
    )
    args = parser.parse_args()

    # Setup env once
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['OMP_NUM_THREADS'] = '8'
    configure_gpu()

    print(f"\n{'='*70}")
    print(f"  SPLIT COMPARISON RUNNER")
    print(f"  Configs : {args.configs}")
    print(f"  Seeds   : {args.seeds}")
    print(f"  Results → {RESULTS_DIR}")
    print(f"{'='*70}\n")

    all_results = {}
    start = time.time()

    for config_name in args.configs:
        cfg = SPLIT_CONFIGS[config_name]
        all_results[config_name] = {}

        for seed in args.seeds:
            set_seed(seed)
            r = run_with_split(config_name, cfg, seed)
            all_results[config_name][seed] = r

    # ── Summary across seeds ──────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  FINAL SUMMARY (averaged across seeds)")
    print(f"{'='*70}")
    print(f"  {'Config':<12} {'Seeds':<16} {'Acc':<10} {'F1':<10} {'Pre':<10} {'Rec':<10}")
    print(f"  {'─'*65}")

    summary = {}
    for config_name, seed_results in all_results.items():
        accs  = [r['accuracy']  for r in seed_results.values()]
        f1s   = [r['macro_f1']  for r in seed_results.values()]
        pres  = [r['precision'] for r in seed_results.values()]
        recs  = [r['recall']    for r in seed_results.values()]

        summary[config_name] = {
            'seeds':     list(seed_results.keys()),
            'mean_acc':  float(np.mean(accs)),
            'std_acc':   float(np.std(accs)),
            'mean_f1':   float(np.mean(f1s)),
            'mean_pre':  float(np.mean(pres)),
            'mean_rec':  float(np.mean(recs)),
        }
        s = summary[config_name]
        seeds_str = str(list(seed_results.keys()))
        print(f"  {config_name:<12} {seeds_str:<16} "
              f"{s['mean_acc']*100:.2f}%    "
              f"{s['mean_f1']*100:.2f}%    "
              f"{s['mean_pre']*100:.2f}%    "
              f"{s['mean_rec']*100:.2f}%")

    print(f"{'='*70}")

    # Save summary
    summary_path = RESULTS_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")
    print(f"  Total time  : {round((time.time()-start)/60, 1)} min")


if __name__ == '__main__':
    main()
