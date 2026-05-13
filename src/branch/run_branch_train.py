"""
Train one branch variant (mlp or transformer) for one seed.
Called as subprocess by run_branch_comparison.py.

Reuses trainer.train_model() — same code path as the proposed model.

Usage:
  python src/branch/run_branch_train.py --branch transformer --seed 42
  python src/branch/run_branch_train.py --branch mlp --seed 0
"""
import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SRC_DIR))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS']        = '8'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', choices=['mlp', 'transformer'], required=True)
    parser.add_argument('--seed',   type=int, required=True)
    parser.add_argument('--force',  action='store_true')
    args = parser.parse_args()

    # Must be set before importing config
    os.environ['MODEL_TYPE'] = args.branch

    from config import RECOGNITION_DIR
    from training.pipeline  import set_seed, setup_gpu
    from training.trainer   import train_model
    from training.evaluator import evaluate_model

    out_dir = RECOGNITION_DIR / 'branch_comparison' / args.branch / f'seed_{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    result_file = out_dir / 'results.json'
    if result_file.exists() and not args.force:
        print(f"[SKIP] Already done: {result_file}")
        return

    setup_gpu()
    set_seed(args.seed)

    print(f"\n{'='*70}")
    print(f"  BRANCH : {args.branch}")
    print(f"  SEED   : {args.seed}")
    print(f"  OUT    : {out_dir}")
    print(f"{'='*70}\n")

    # ── Train (reuse existing trainer) ────────────────────────────────────────
    history, _ = train_model(
        checkpoint_dir=str(out_dir),
        logs_dir=str(out_dir / 'logs'),
        dataset_name=f'{args.branch}_seed{args.seed}',
    )

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    best_epoch   = int(np.argmax(history_dict['val_accuracy'])) + 1
    n_params     = history.model.count_params()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    eval_acc, macro_f1, macro_pre, macro_rec, _ = evaluate_model(
        model_path=str(out_dir / 'best_model'),
        action_mapping_path=str(out_dir / 'action_mapping.json'),
    )

    results = {
        'branch':          args.branch,
        'seed':            args.seed,
        'test_accuracy':   float(eval_acc),
        'macro_f1':        float(macro_f1),
        'macro_precision': float(macro_pre),
        'macro_recall':    float(macro_rec),
        'best_epoch':      best_epoch,
        'n_params':        int(n_params),
    }
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[SAVED] {result_file}")
    print(f"  Acc={eval_acc*100:.2f}%  F1={macro_f1*100:.2f}%  Epoch={best_epoch}")


if __name__ == '__main__':
    main()
