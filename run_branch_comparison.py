"""
MLP Branches vs Transformer Encoder Branches — multi-seed comparison.
Table III in the paper.

Each (branch, seed) runs as a SEPARATE SUBPROCESS for full GPU cleanup.

Results:
  /mnt/ngan/recognition/branch_comparison/mlp/seed_{N}/results.json
  /mnt/ngan/recognition/branch_comparison/transformer/seed_{N}/results.json
  results/branch_comparison/summary.json

Usage:
  python run_branch_comparison.py
  python run_branch_comparison.py --seeds 42 0 1
  python run_branch_comparison.py --branches mlp
  python run_branch_comparison.py --skip-done
"""
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).parent
SRC_DIR      = PROJECT_ROOT / 'src'
LOG_DIR      = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / 'results' / 'branch_comparison'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BRANCHES = ['mlp', 'transformer']
SEEDS    = [42, 0, 1]

BRANCH_LABELS = {
    'mlp':         'MLP (prop.)',
    'transformer': 'Trans. Enc.',
}


def run_subprocess(name: str, cmd: list, log_path: Path) -> int:
    print(f"\n{'─'*70}")
    print(f"  STARTING : {name}")
    print(f"  LOG      : {log_path}")
    print(f"{'─'*70}\n")

    t0 = time.time()
    with open(log_path, 'w') as lf:
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        proc.wait()

    elapsed = round((time.time() - t0) / 60, 1)
    status  = 'OK' if proc.returncode == 0 else f'FAILED (code {proc.returncode})'
    print(f"\n  [{status}] {name}  —  {elapsed} min")
    return proc.returncode


def collect_results():
    """Read all results.json files → build summary."""
    summary = {}
    for branch in BRANCHES:
        branch_results = []
        branch_dir = Path('/mnt/ngan/recognition/branch_comparison') / branch
        if not branch_dir.exists():
            continue
        for seed_dir in sorted(branch_dir.glob('seed_*')):
            rfile = seed_dir / 'results.json'
            if rfile.exists():
                with open(rfile) as f:
                    branch_results.append(json.load(f))

        if not branch_results:
            continue

        accs  = [r['test_accuracy']   for r in branch_results]
        f1s   = [r['macro_f1']        for r in branch_results]
        pres  = [r['macro_precision']  for r in branch_results]
        recs  = [r['macro_recall']     for r in branch_results]
        eps   = [r['best_epoch']       for r in branch_results]
        times = [r['train_time_min']   for r in branch_results]

        summary[branch] = {
            'label':      BRANCH_LABELS[branch],
            'seeds':      [r['seed'] for r in branch_results],
            'mean_acc':   float(np.mean(accs)),
            'std_acc':    float(np.std(accs)),
            'mean_f1':    float(np.mean(f1s)),
            'std_f1':     float(np.std(f1s)),
            'mean_pre':   float(np.mean(pres)),
            'mean_rec':   float(np.mean(recs)),
            'mean_epoch': float(np.mean(eps)),
            'mean_time':  float(np.mean(times)),
            'per_seed':   branch_results,
        }

        # Copy individual seed results
        dst = RESULTS_DIR / branch
        dst.mkdir(exist_ok=True)
        import shutil
        for r in branch_results:
            src = (Path('/mnt/ngan/recognition/branch_comparison')
                   / branch / f'seed_{r["seed"]}' / 'results.json')
            if src.exists():
                shutil.copy2(src, dst / f'seed_{r["seed"]}.json')

    return summary


def print_summary(summary: dict):
    print(f"\n\n{'='*80}")
    print("  BRANCH COMPARISON SUMMARY  (mean ± std over seeds)")
    print(f"{'='*80}")
    hdr = f"  {'Variant':<18} {'Seeds':<14} {'Acc':<14} {'F1':<12} {'Prec':<10} {'Recall':<10} {'Epoch':>7} {'Time':>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for branch, s in summary.items():
        seeds_str = str(s['seeds'])
        acc_str   = f"{s['mean_acc']*100:.2f}±{s['std_acc']*100:.2f}%"
        f1_str    = f"{s['mean_f1']*100:.2f}±{s['std_f1']*100:.2f}%"
        pre_str   = f"{s['mean_pre']*100:.2f}%"
        rec_str   = f"{s['mean_rec']*100:.2f}%"
        ep_str    = f"{s['mean_epoch']:.0f}"
        tm_str    = f"{s['mean_time']:.0f}"
        print(f"  {s['label']:<18} {seeds_str:<14} {acc_str:<14} {f1_str:<12} {pre_str:<10} {rec_str:<10} {ep_str:>7} {tm_str:>7}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare MLP vs Transformer branches across seeds.'
    )
    parser.add_argument('--branches', nargs='+', choices=BRANCHES, default=BRANCHES)
    parser.add_argument('--seeds',    nargs='+', type=int,          default=SEEDS)
    parser.add_argument('--skip-done', action='store_true',
                        help='Skip if results.json already exists')
    parser.add_argument('--force', action='store_true',
                        help='Pass --force to re-train even if done')
    args = parser.parse_args()

    start = time.time()
    print(f"\n{'='*70}")
    print(f"  BRANCH COMPARISON RUNNER")
    print(f"  Branches  : {args.branches}")
    print(f"  Seeds     : {args.seeds}")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    outcomes = {}

    for branch in args.branches:
        for seed in args.seeds:
            exp = f'{branch}_seed{seed}'

            result_file = (Path('/mnt/ngan/recognition/branch_comparison')
                           / branch / f'seed_{seed}' / 'results.json')

            if args.skip_done and result_file.exists():
                print(f"\n[SKIP] {exp} — results already exist")
                outcomes[exp] = 'SKIPPED'
                continue

            cmd = [
                sys.executable, '-u',
                str(SRC_DIR / 'branch' / 'run_branch_train.py'),
                '--branch', branch,
                '--seed', str(seed),
            ]
            if args.force:
                cmd.append('--force')

            log_path = LOG_DIR / f'branch_{branch}_seed{seed}.log'
            rc = run_subprocess(f'{branch} — seed {seed}', cmd, log_path)
            outcomes[exp] = 'OK' if rc == 0 else 'FAILED'

    # ── Final report ──────────────────────────────────────────────────────────
    elapsed = round((time.time() - start) / 60, 1)
    print(f"\n{'='*70}")
    print(f"  ALL DONE  —  total {elapsed} min")
    print(f"{'='*70}")
    print(f"  {'Experiment':<28} {'Status'}")
    print(f"  {'─'*40}")
    for exp, status in outcomes.items():
        icon = '[OK]' if status == 'OK' else ('[SKIP]' if status == 'SKIPPED' else '[FAIL]')
        print(f"  {exp:<28} {icon}")
    print(f"{'='*70}\n")

    # Collect + print summary
    summary = collect_results()
    if summary:
        print_summary(summary)
        summary_path = RESULTS_DIR / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved: {summary_path}")

    failed = [k for k, v in outcomes.items() if v == 'FAILED']
    if failed:
        print(f"\n  [WARNING] Failed: {failed}")
        sys.exit(1)


if __name__ == '__main__':
    main()
