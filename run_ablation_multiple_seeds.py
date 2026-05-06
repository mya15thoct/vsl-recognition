"""
Run ablation study variants for seeds 0 and 1.
Seed 42 was already run separately.

Results saved to:
  /mnt/ngan/recognition/ablation/seed_0/
  /mnt/ngan/recognition/ablation/seed_1/

Usage:
  python run_ablation_multiple_seeds.py
  python run_ablation_multiple_seeds.py --variants v0_baseline v1_single_stream
  python run_ablation_multiple_seeds.py --seeds 0 1
"""
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / 'src'


def run_seed(seed: int, variants: list[str] = None):
    cmd = [
        sys.executable, '-u',
        str(SRC_DIR / 'ablation' / 'run_ablation.py'),
        '--force',
        '--seed', str(seed),
    ]
    if variants:
        cmd += ['--variants'] + variants

    print(f"\n{'#'*80}")
    print(f"### RUNNING ABLATION — SEED {seed} ###")
    print(f"{'#'*80}")
    print(f"CMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\n[ERROR] Seed {seed} exited with code {result.returncode}")
    else:
        print(f"\n[DONE] Seed {seed} finished successfully")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--variants', nargs='+',
        default=None,
        help='Which variants to run (default: all). E.g.: v0_baseline v1_single_stream'
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int,
        default=[0, 1],
        help='Which seeds to run (default: 0 1)'
    )
    args = parser.parse_args()

    print(f"\nVariants : {args.variants or 'all'}")
    print(f"Seeds    : {args.seeds}")
    print(f"Results  → /mnt/ngan/recognition/ablation/seed_<N>/\n")

    failed = []
    for seed in args.seeds:
        rc = run_seed(seed, args.variants)
        if rc != 0:
            failed.append(seed)

    print(f"\n{'='*80}")
    print("ALL ABLATION SEEDS DONE")
    if failed:
        print(f"[WARNING] Failed seeds: {failed}")
    else:
        print("[OK] All seeds completed successfully")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
