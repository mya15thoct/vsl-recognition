"""
Download 3 Kaggle benchmark datasets to the local data directory.

Datasets:
  1. INCLUDE        — daskoushik/include          (263 ISL classes, ~4287 videos)
  2. ISL Video      — prasadshet/indian-sign-language-video-dataset (60 signs)
  3. INCLUDE-24 Med — linardur/include-24-medical-modified (24 medical classes)

Prerequisites:
  pip install kaggle
  Set Kaggle credentials via ONE of:
    (a) ~/.kaggle/kaggle.json  →  {"username":"...", "key":"..."}
    (b) Environment variables  →  KAGGLE_USERNAME, KAGGLE_KEY

Usage:
  python download_datasets.py                        # download all 3
  python download_datasets.py --dataset include      # download only INCLUDE
  python download_datasets.py --dataset isl          # ISL Video Dataset
  python download_datasets.py --dataset medical      # INCLUDE-24 Medical
  python download_datasets.py --skip_existing        # skip already-downloaded datasets
"""

import argparse
import os
import sys
import zipfile
import shutil
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DATA_DIR = Path("/home/islabworker2/mya/vsl-recognition/data")

DATASETS = {
    "include": {
        "kaggle_id":    "daskoushik/include",
        "target_dir":   BASE_DATA_DIR / "INCLUDE",
        "description":  "INCLUDE — 263 ISL classes, ~4287 videos",
        "zip_name":     "include.zip",
    },
    "isl": {
        "kaggle_id":    "prasadshet/indian-sign-language-video-dataset",
        "target_dir":   BASE_DATA_DIR / "ISL_Video",
        "description":  "Indian Sign Language Video Dataset — 60 signs",
        "zip_name":     "indian-sign-language-video-dataset.zip",
    },
    "medical": {
        "kaggle_id":    "linardur/include-24-medical-modified",
        "target_dir":   BASE_DATA_DIR / "INCLUDE_24_Medical",
        "description":  "INCLUDE-24 Medical Modified — 24 medical classes",
        "zip_name":     "include-24-medical-modified.zip",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def check_kaggle_credentials():
    """Verify Kaggle credentials are available."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key  = os.environ.get("KAGGLE_KEY")

    if kaggle_json.exists():
        print(f"[OK] Kaggle credentials found: {kaggle_json}")
        return True
    elif env_user and env_key:
        print("[OK] Kaggle credentials found via environment variables")
        return True
    else:
        print("\n[ERROR] Kaggle credentials not found!")
        print("  Option 1: Create ~/.kaggle/kaggle.json")
        print('    {"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}')
        print("    chmod 600 ~/.kaggle/kaggle.json")
        print()
        print("  Option 2: Set environment variables")
        print("    export KAGGLE_USERNAME=your_username")
        print("    export KAGGLE_KEY=your_api_key")
        print()
        print("  Get your API key at: https://www.kaggle.com/account")
        return False


def check_kaggle_installed():
    """Check if kaggle CLI is available."""
    import shutil
    if shutil.which("kaggle") is not None:
        return True
    print("[ERROR] kaggle CLI not found.")
    print("  Run: pip install kaggle")
    return False


def download_dataset(name: str, info: dict, skip_existing: bool = False):
    """Download a single Kaggle dataset and extract it."""
    print(f"\n{'=' * 65}")
    print(f"  Dataset : {info['description']}")
    print(f"  Kaggle  : https://www.kaggle.com/datasets/{info['kaggle_id']}")
    print(f"  Target  : {info['target_dir']}")
    print(f"{'=' * 65}")

    target_dir: Path = info["target_dir"]

    # Skip if already exists and not empty
    if skip_existing and target_dir.exists() and any(target_dir.iterdir()):
        n_items = sum(1 for _ in target_dir.rglob("*"))
        print(f"[SKIP] Already exists with {n_items} items — use --force to re-download")
        return True

    # Create staging area
    staging_dir = BASE_DATA_DIR / "_downloading" / name
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        import subprocess

        print(f"\n[DOWNLOADING] {info['kaggle_id']} ...")
        # Use kaggle CLI — works with ALL versions of the kaggle package
        result = subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", info["kaggle_id"],
             "-p", str(staging_dir),
             "--quiet"],
            check=True,
            capture_output=False,
        )

        # Find the downloaded zip
        zip_files = list(staging_dir.glob("*.zip"))
        if not zip_files:
            print("[ERROR] No zip file found after download")
            return False

        zip_path = zip_files[0]
        print(f"\n[EXTRACTING] {zip_path.name} → {target_dir} ...")
        target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            total = len(zf.namelist())
            print(f"  {total} files to extract...")
            zf.extractall(target_dir)

        # Clean up staging
        shutil.rmtree(staging_dir, ignore_errors=True)

        # Count extracted files
        video_exts = {".mp4", ".mov", ".avi", ".mkv"}
        n_videos = sum(
            1 for f in target_dir.rglob("*")
            if f.suffix.lower() in video_exts
        )
        n_dirs = sum(1 for d in target_dir.rglob("*") if d.is_dir())

        print(f"\n[OK] {info['description']}")
        print(f"     Extracted to : {target_dir}")
        print(f"     Class folders: {n_dirs}")
        print(f"     Video files  : {n_videos}")
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to download {name}: {e}")
        shutil.rmtree(staging_dir, ignore_errors=True)
        return False


def print_summary():
    """Print summary of all dataset directories."""
    print(f"\n{'=' * 65}")
    print("  DATASET SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Base directory: {BASE_DATA_DIR}\n")

    video_exts = {".mp4", ".mov", ".avi", ".mkv"}
    for name, info in DATASETS.items():
        target: Path = info["target_dir"]
        if target.exists():
            n_videos = sum(1 for f in target.rglob("*") if f.suffix.lower() in video_exts)
            n_classes = sum(1 for d in target.iterdir() if d.is_dir())
            status = f"{n_classes} classes, {n_videos} videos"
        else:
            status = "NOT DOWNLOADED"
        print(f"  [{name:8s}]  {status}")
        print(f"             {target}")

    print(f"\n{'=' * 65}")
    print("  NEXT STEP:")
    print("  python evaluate_per_dataset.py \\")
    print(f"    --model_path /home/islabworker2/mya/recognition/checkpoints/mlp/best_model \\")
    print(f"    --action_mapping /home/islabworker2/mya/recognition/checkpoints/mlp/action_mapping.json")
    print(f"{'=' * 65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global BASE_DATA_DIR, DATASETS
    parser = argparse.ArgumentParser(
        description="Download Kaggle datasets for ISL benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py                    # download all 3 datasets
  python download_datasets.py --dataset include  # download only INCLUDE
  python download_datasets.py --skip_existing    # skip already-downloaded
  python download_datasets.py --summary          # show download status only
        """
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip datasets that are already downloaded"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Only print download status summary, do not download"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=f"Override base data directory (default: {BASE_DATA_DIR})"
    )
    args = parser.parse_args()

    # Override base dir if specified
    if args.data_dir:
        BASE_DATA_DIR = Path(args.data_dir)
        for key in DATASETS:
            DATASETS[key]["target_dir"] = BASE_DATA_DIR / DATASETS[key]["target_dir"].name

    BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.summary:
        print_summary()
        return

    print("\n" + "=" * 65)
    print("  KAGGLE DATASET DOWNLOADER")
    print("  3 ISL Benchmark Datasets → Automatic Pull")
    print("=" * 65)

    # Validate prerequisites
    if not check_kaggle_installed():
        sys.exit(1)
    if not check_kaggle_credentials():
        sys.exit(1)

    # Select datasets to download
    if args.dataset == "all":
        to_download = list(DATASETS.keys())
    else:
        to_download = [args.dataset]

    print(f"\nWill download: {', '.join(to_download)}")
    print(f"Target base  : {BASE_DATA_DIR}")

    results = {}
    for name in to_download:
        info = DATASETS[name]
        skip = args.skip_existing and not args.force
        results[name] = download_dataset(name, info, skip_existing=skip)

    # Final summary
    print_summary()
    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"[WARNING] Failed datasets: {failed}")
        sys.exit(1)
    else:
        print("[OK] All downloads completed successfully!")


if __name__ == "__main__":
    main()
