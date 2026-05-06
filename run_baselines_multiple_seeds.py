"""
Run all baselines (LSTM, LSTM-GRU, EMPATH) for seeds 0 and 1.
Seed 42 was already run separately.

Results saved to:
  /mnt/ngan/recognition/baselines/seed_0/
  /mnt/ngan/recognition/baselines/seed_1/

Usage:
  python run_baselines_multiple_seeds.py
  python run_baselines_multiple_seeds.py --baselines lstm lstm_gru
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / 'src'


def run_seed(seed: int, baselines: list[str] = None):
    cmd = [
        sys.executable, '-u',
        str(SRC_DIR / 'baselines' / 'run_baselines.py'),
        '--force',
        '--seed', str(seed),
    ]
    if baselines:
        cmd += ['--baselines'] + baselines

    print(f"\n{'#'*80}")
    print(f"### RUNNING BASELINES — SEED {seed} ###")
    print(f"{'#'*80}")
    print(f"CMD: {' '.join(cmd)}\n")

    # Run as subprocess so TF graph/GPU memory is fully released between seeds
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n[ERROR] Seed {seed} exited with code {result.returncode}")
    else:
        print(f"\n[DONE] Seed {seed} finished successfully")
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--baselines', nargs='+',
        choices=['lstm', 'lstm_gru', 'empath'],
        default=['lstm', 'lstm_gru', 'empath'],
        help='Which baselines to run (default: all 3)'
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int,
        default=[0, 1],
        help='Which seeds to run (default: 0 1)'
    )
    args = parser.parse_args()

    print(f"\nBaselines : {args.baselines}")
    print(f"Seeds     : {args.seeds}")
    print(f"Results   → /mnt/ngan/recognition/baselines/seed_<N>/\n")

    failed = []
    for seed in args.seeds:
        rc = run_seed(seed, args.baselines)
        if rc != 0:
            failed.append(seed)

    print(f"\n{'='*80}")
    print("ALL SEEDS DONE")
    if failed:
        print(f"[WARNING] Failed seeds: {failed}")
    else:
        print("[OK] All seeds completed successfully")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
