"""
Master script: Run ALL experiments for seeds 0 and 1.

Order per seed:
  1. Proposed model      → run_full_pipeline()
  2. Baselines           → run_baselines.py  (lstm, lstm_gru, empath)
  3. Ablation study      → run_ablation.py   (all variants)

Each experiment runs as a SEPARATE SUBPROCESS so GPU memory is fully
released before the next one starts.

Results are saved automatically by each sub-script:
  /mnt/ngan/recognition/checkpoints/mlp/seed_<N>/
  /mnt/ngan/recognition/baselines/seed_<N>/
  /mnt/ngan/recognition/ablation/seed_<N>/

Usage:
  python run_all_seeds.py
  python run_all_seeds.py --seeds 0
  python run_all_seeds.py --seeds 0 1 --skip-done
"""
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
SRC_DIR      = PROJECT_ROOT / 'src'
LOG_DIR      = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# Centralized results directory (in project, tracked by git or easy to access)
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_subprocess(name: str, cmd: list[str], log_path: Path) -> int:
    """Run cmd as subprocess, tee output to log_path, return exit code."""
    print(f"\n{'─'*70}")
    print(f"  STARTING : {name}")
    print(f"  LOG      : {log_path}")
    print(f"  CMD      : {' '.join(cmd)}")
    print(f"{'─'*70}\n")

    t0 = time.time()
    with open(log_path, 'w') as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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


def python_cmd(*args) -> list[str]:
    return [sys.executable, '-u', *[str(a) for a in args]]


def collect_results(seed: int):
    """
    After all experiments for a seed complete, copy result files
    into RESULTS_DIR for centralized access.

    Structure:
      results/
        proposed/seed_0.json
        baselines/seed_0_summary.csv
        baselines/seed_0_lstm.json  ...
        ablation/seed_0_summary.csv
        ablation/seed_0_v0_baseline.json  ...
    """
    import shutil

    proposed_src  = Path(f'/mnt/ngan/recognition/checkpoints/mlp/seed_{seed}/results.json')
    baselines_src = Path(f'/mnt/ngan/recognition/baselines/seed_{seed}')
    ablation_src  = Path(f'/mnt/ngan/recognition/ablation/seed_{seed}')

    copied = []

    # ── Proposed ──────────────────────────────────────────────────────────────
    dst_dir = RESULTS_DIR / 'proposed'
    dst_dir.mkdir(exist_ok=True)
    if proposed_src.exists():
        dst = dst_dir / f'seed_{seed}.json'
        shutil.copy2(proposed_src, dst)
        copied.append(str(dst))

    # ── Baselines ─────────────────────────────────────────────────────────────
    dst_dir = RESULTS_DIR / 'baselines'
    dst_dir.mkdir(exist_ok=True)
    if baselines_src.exists():
        csv = baselines_src / 'baselines_summary.csv'
        if csv.exists():
            shutil.copy2(csv, dst_dir / f'seed_{seed}_summary.csv')
            copied.append(str(dst_dir / f'seed_{seed}_summary.csv'))
        for f in baselines_src.glob('*/results.json'):
            dst = dst_dir / f'seed_{seed}_{f.parent.name}.json'
            shutil.copy2(f, dst)
            copied.append(str(dst))

    # ── Ablation ──────────────────────────────────────────────────────────────
    dst_dir = RESULTS_DIR / 'ablation'
    dst_dir.mkdir(exist_ok=True)
    if ablation_src.exists():
        csv = ablation_src / 'ablation_summary.csv'
        if csv.exists():
            shutil.copy2(csv, dst_dir / f'seed_{seed}_summary.csv')
            copied.append(str(dst_dir / f'seed_{seed}_summary.csv'))
        for f in ablation_src.glob('*/results.json'):
            dst = dst_dir / f'seed_{seed}_{f.parent.name}.json'
            shutil.copy2(f, dst)
            copied.append(str(dst))

    print(f"\n[RESULTS] Collected {len(copied)} files → {RESULTS_DIR}")
    for p in copied:
        print(f"  {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runners
# ─────────────────────────────────────────────────────────────────────────────

def run_proposed(seed: int) -> int:
    """Train + evaluate the proposed model for one seed."""
    # We call pipeline.py directly so it handles its own GPU setup & cleanup
    script = SRC_DIR / 'training' / 'pipeline.py'
    # pipeline.py doesn't accept --seed yet via CLI, so we call it through
    # a tiny inline script that passes the seed argument.
    inline = (
        f"import sys; sys.path.insert(0, '{SRC_DIR}'); "
        f"from training.pipeline import run_full_pipeline; "
        f"run_full_pipeline(seed={seed})"
    )
    cmd      = python_cmd('-c', inline)
    log_path = LOG_DIR / f'proposed_seed{seed}.log'
    return run_subprocess(f'Proposed model — seed {seed}', cmd, log_path)


def run_baselines(seed: int) -> int:
    """Run all baselines for one seed."""
    cmd = python_cmd(
        SRC_DIR / 'baselines' / 'run_baselines.py',
        '--force',
        '--seed', str(seed),
    )
    log_path = LOG_DIR / f'baselines_seed{seed}.log'
    return run_subprocess(f'Baselines — seed {seed}', cmd, log_path)


def run_ablation(seed: int) -> int:
    """Run full ablation study for one seed."""
    cmd = python_cmd(
        SRC_DIR / 'ablation' / 'run_ablation.py',
        '--force',
        '--seed', str(seed),
    )
    log_path = LOG_DIR / f'ablation_seed{seed}.log'
    return run_subprocess(f'Ablation — seed {seed}', cmd, log_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run all experiments (proposed + baselines + ablation) for multiple seeds.'
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=SEEDS,
        help=f'Seeds to run (default: {SEEDS})'
    )
    parser.add_argument(
        '--skip-done', action='store_true',
        help='Skip an experiment if its results already exist'
    )
    args = parser.parse_args()

    start_time = time.time()
    print(f"\n{'='*70}")
    print(f"  MASTER EXPERIMENT RUNNER")
    print(f"  Seeds     : {args.seeds}")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # Track outcomes: { (seed, experiment) : 'OK' | 'FAILED' | 'SKIPPED' }
    outcomes = {}

    for seed in args.seeds:
        print(f"\n{'#'*70}")
        print(f"###  SEED {seed}  ###")
        print(f"{'#'*70}")

        # ── 1. Proposed model ───────────────────────────────────────────────
        exp = f'proposed_seed{seed}'
        proposed_result_dir = Path('/mnt/ngan/recognition/checkpoints/mlp')
        proposed_done = (proposed_result_dir / f'seed_{seed}' / 'results.json').exists()

        if args.skip_done and proposed_done:
            print(f"\n[SKIP] Proposed seed {seed} — results already exist")
            outcomes[exp] = 'SKIPPED'
        else:
            rc = run_proposed(seed)
            outcomes[exp] = 'OK' if rc == 0 else 'FAILED'

        # ── 2. Baselines ────────────────────────────────────────────────────
        exp = f'baselines_seed{seed}'
        baselines_done = (
            Path('/mnt/ngan/recognition/baselines') / f'seed_{seed}' / 'baselines_summary.csv'
        ).exists()

        if args.skip_done and baselines_done:
            print(f"\n[SKIP] Baselines seed {seed} — results already exist")
            outcomes[exp] = 'SKIPPED'
        else:
            rc = run_baselines(seed)
            outcomes[exp] = 'OK' if rc == 0 else 'FAILED'

        # ── 3. Ablation ─────────────────────────────────────────────────────
        exp = f'ablation_seed{seed}'
        ablation_done = (
            Path('/mnt/ngan/recognition/ablation') / f'seed_{seed}' / 'ablation_summary.csv'
        ).exists()

        if args.skip_done and ablation_done:
            print(f"\n[SKIP] Ablation seed {seed} — results already exist")
            outcomes[exp] = 'SKIPPED'
        else:
            rc = run_ablation(seed)
            outcomes[exp] = 'OK' if rc == 0 else 'FAILED'

        # ── Collect all results for this seed → results/ ────────────────────
        print(f"\n[Seed {seed}] Collecting results...")
        collect_results(seed)

    # ── Final report ─────────────────────────────────────────────────────────
    elapsed_total = round((time.time() - start_time) / 60, 1)

    print(f"\n\n{'='*70}")
    print(f"  ALL EXPERIMENTS DONE  —  total {elapsed_total} min")
    print(f"{'='*70}")
    print(f"  {'Experiment':<30} {'Status'}")
    print(f"  {'─'*50}")
    for exp, status in outcomes.items():
        icon = '[OK]' if status == 'OK' else ('[SKIP]' if status == 'SKIPPED' else '[FAIL]')
        print(f"  {exp:<30} {icon}")
    print(f"{'='*70}\n")

    # Save summary JSON
    summary_path = LOG_DIR / 'run_all_seeds_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'seeds': args.seeds,
            'started': datetime.now().isoformat(),
            'elapsed_min': elapsed_total,
            'outcomes': outcomes,
        }, f, indent=2)
    print(f"  Summary saved: {summary_path}")

    failed = [k for k, v in outcomes.items() if v == 'FAILED']
    if failed:
        print(f"\n  [WARNING] Failed experiments: {failed}")
        sys.exit(1)


if __name__ == '__main__':
    main()
