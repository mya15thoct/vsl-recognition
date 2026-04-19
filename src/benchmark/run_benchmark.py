"""
Benchmark pipeline: download → extract keypoints → train → report results.

Usage:
  # Run all 3 datasets
  python src/benchmark/run_benchmark.py --dataset all

  # Run one dataset
  python src/benchmark/run_benchmark.py --dataset include
  python src/benchmark/run_benchmark.py --dataset isl
  python src/benchmark/run_benchmark.py --dataset include24

  # Inspect folder structure after download (before extracting)
  python src/benchmark/run_benchmark.py --dataset include24 --explore

  # Skip download if data already exists locally
  python src/benchmark/run_benchmark.py --dataset include --skip-download

  # Skip keypoint extraction if sequences already exist
  python src/benchmark/run_benchmark.py --dataset include --skip-extract

KAGGLE SETUP (run once on the server):
  pip install kaggle
  # Download kaggle.json from https://www.kaggle.com/settings → API → Create New Token
  cp kaggle.json ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from benchmark.dataset_configs import DATASETS, BASE_DIR


# ---------------------------------------------------------------------------
# Kaggle helpers
# ---------------------------------------------------------------------------

def check_kaggle_setup():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("[ERROR] Kaggle API credentials not found.")
        print(f"  Expected: {kaggle_json}")
        print("  Steps:")
        print("    1. Go to https://www.kaggle.com/settings → API → Create New Token")
        print("    2. scp kaggle.json <server>:~/.kaggle/kaggle.json")
        print("    3. chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True


def download_dataset(cfg):
    raw_dir = cfg["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = list(raw_dir.iterdir())
    if existing:
        print(f"[SKIP DOWNLOAD] {raw_dir} already has {len(existing)} item(s).")
        return True

    slug = cfg["kaggle_slug"]
    if slug.startswith("UPDATE_ME"):
        print(f"[ERROR] kaggle_slug for '{cfg['name']}' is not set.")
        print("  Update dataset_configs.py with the correct Kaggle slug.")
        return False

    print(f"[DOWNLOAD] {cfg['name']}  ({slug})")
    print(f"  → {raw_dir}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug,
         "-p", str(raw_dir), "--unzip"],
    )
    if result.returncode != 0:
        print(f"[ERROR] kaggle download failed (exit {result.returncode})")
        return False
    print(f"[OK] Downloaded to {raw_dir}")
    return True


# ---------------------------------------------------------------------------
# Dataset structure detection
# ---------------------------------------------------------------------------

def _has_videos(folder, exts):
    for ext in exts:
        if any(folder.glob(f"*{ext}")) or any(folder.glob(f"*{ext.upper()}")):
            return True
    return False


def find_video_root(raw_dir, extensions, max_depth=4):
    """
    Walk the download directory to find the folder whose direct
    subdirectories contain video files (i.e., the class-level root).
    Returns the best candidate or None.
    """
    raw_dir = Path(raw_dir)
    best_dir, best_count = None, 0

    def _walk(d, depth):
        nonlocal best_dir, best_count
        if depth > max_depth or not d.is_dir():
            return
        count = sum(1 for sub in d.iterdir() if sub.is_dir() and _has_videos(sub, extensions))
        if count > best_count:
            best_count, best_dir = count, d
        for sub in d.iterdir():
            if sub.is_dir():
                _walk(sub, depth + 1)

    _walk(raw_dir, 0)

    if best_dir is None or best_count == 0:
        print(f"[WARNING] No class/video structure found under {raw_dir}")
        print("  Run with --explore to inspect the downloaded folder.")
        return None

    print(f"[AUTO-DETECT] Video root: {best_dir}  ({best_count} class folders)")
    return best_dir


def explore_structure(raw_dir, extensions, limit=30):
    video_root = find_video_root(raw_dir, extensions)
    if video_root is None:
        print(f"Contents of {raw_dir}:")
        for p in sorted(raw_dir.iterdir()):
            print(f"  {p.name}{'/' if p.is_dir() else ''}")
        return

    print(f"\nClass folders under {video_root}:")
    class_dirs = sorted(d for d in video_root.iterdir() if d.is_dir())
    for i, d in enumerate(class_dirs):
        n = sum(1 for f in d.iterdir()
                if f.suffix.lower() in {e.lower() for e in extensions})
        print(f"  {d.name}/  ({n} videos)")
        if i >= limit - 1 and len(class_dirs) > limit:
            print(f"  ... ({len(class_dirs) - limit} more classes)")
            break
    print(f"\nTotal: {len(class_dirs)} classes")


# ---------------------------------------------------------------------------
# Keypoint extraction
# ---------------------------------------------------------------------------

def extract_keypoints(video_root, seq_dir, extensions):
    import cv2
    import numpy as np
    from utils.extraction import get_holistic_model, mediapipe_detection
    from utils.extraction import extract_keypoints as _kp

    seq_dir = Path(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted(d for d in video_root.iterdir() if d.is_dir())
    print(f"\n[EXTRACT] {len(class_dirs)} classes  →  {seq_dir}")

    holistic = get_holistic_model()
    total_saved, total_skipped = 0, 0

    for cls_idx, class_dir in enumerate(class_dirs):
        videos = []
        for ext in extensions:
            videos += list(class_dir.glob(f"*{ext}"))
            videos += list(class_dir.glob(f"*{ext.upper()}"))

        if not videos:
            continue

        out_dir = seq_dir / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{cls_idx+1}/{len(class_dirs)}] {class_dir.name}: {len(videos)} videos")

        for video_path in videos:
            out_path = out_dir / f"{video_path.stem}.npy"
            if out_path.exists():
                total_skipped += 1
                continue

            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                _, results = mediapipe_detection(frame, holistic)
                frames.append(_kp(results))
            cap.release()

            if frames:
                np.save(out_path, np.array(frames, dtype=np.float32))
                total_saved += 1

    holistic.close()
    print(f"[OK] Extracted {total_saved} sequences  "
          f"({total_skipped} already existed, skipped)")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(cfg):
    from training.trainer import train_model

    seq_dir = cfg["seq_dir"]
    if not seq_dir.exists() or not any(seq_dir.iterdir()):
        print(f"[ERROR] No sequences found in {seq_dir}. Run extraction first.")
        return None

    history, test_acc = train_model(
        sequence_path=cfg["seq_dir"],
        checkpoint_dir=cfg["checkpoint_dir"],
        logs_dir=cfg["logs_dir"],
        dataset_name=cfg["name"],
    )
    return float(test_acc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark on public sign language datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", choices=["include", "isl", "include24", "all"], default="all",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract",  action="store_true")
    parser.add_argument(
        "--explore", action="store_true",
        help="Print dataset folder structure after download, then exit",
    )
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    if not args.skip_download and not check_kaggle_setup():
        sys.exit(1)

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = BASE_DIR / "results" / f"benchmark_{timestamp}.json"

    for key in targets:
        cfg = DATASETS[key]
        sep = "=" * 70
        print(f"\n{sep}\nDATASET: {cfg['name']}\n{sep}")

        # 1. Download
        if not args.skip_download:
            ok = download_dataset(cfg)
            if not ok:
                print(f"[SKIP] {key}: download failed.\n")
                continue

        # 2. Explore / detect structure
        if args.explore:
            explore_structure(cfg["raw_dir"], cfg["video_extensions"])
            continue

        video_root = find_video_root(cfg["raw_dir"], cfg["video_extensions"])
        if video_root is None:
            print(f"[SKIP] {key}: cannot detect video structure. "
                  "Run with --explore after downloading.")
            continue

        # 3. Extract keypoints
        if not args.skip_extract:
            extract_keypoints(video_root, cfg["seq_dir"], cfg["video_extensions"])

        # 4. Train + evaluate
        test_acc = train_and_evaluate(cfg)
        if test_acc is None:
            continue

        results[key] = {
            "dataset": cfg["name"],
            "paper": cfg["paper"],
            "test_accuracy": test_acc,
            "timestamp": timestamp,
        }
        print(f"\n[RESULT] {cfg['name']}: {test_acc * 100:.2f}%")

    # Save results
    if results:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        for r in results.values():
            print(f"  {r['dataset']:40s}  {r['test_accuracy']*100:.2f}%")
        print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
