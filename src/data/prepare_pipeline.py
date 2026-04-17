"""
Master script: Extract keypoints + Augment data in one go.

USAGE:
  # Only process static images (fast):
  python src/data/prepare_pipeline.py --mode images

  # Full pipeline: video + images + augment:
  python src/data/prepare_pipeline.py --mode all

  # Original video-only pipeline:
  python src/data/prepare_pipeline.py --mode videos

Options:
  --mode   : 'images' | 'videos' | 'all'  (default: all)
  --image_dir : path to static images root folder (default: DATA_DIR/images)
"""
import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.extract import extract_keypoints_from_videos, extract_keypoints_from_images
from src.data.check_distribution import check_distribution
from src.data.augment import augment_dataset
from src.config import IMAGE_DIR


def run_images(image_dir=None):
    image_dir = image_dir or IMAGE_DIR
    print("\n" + "=" * 70)
    print("STATIC IMAGE PIPELINE")
    print("=" * 70 + "\n")

    print("STEP 1/1: Extracting keypoints from static images...")
    print("-" * 70)
    extract_keypoints_from_images(image_dir=image_dir)

    print("\n\n" + "=" * 70)
    print("IMAGES EXTRACTION COMPLETE!")
    print("Static sequences saved — data_loader.py will pick them up automatically.")
    print("=" * 70)


def run_all(image_dir=None):
    image_dir = image_dir or IMAGE_DIR
    print("\n" + "=" * 70)
    print("FULL DATA PREPARATION PIPELINE")
    print("=" * 70 + "\n")

    # STEP 1: Extract keypoints from videos
    print("STEP 1/4: Extracting keypoints from videos...")
    print("-" * 70)
    extract_keypoints_from_videos()

    # STEP 2: Extract keypoints from static images
    print("\n\nSTEP 2/4: Extracting keypoints from static images...")
    print("-" * 70)
    extract_keypoints_from_images(image_dir=image_dir)

    # STEP 3: Check data distribution
    print("\n\nSTEP 3/4: Checking data distribution...")
    print("-" * 70)
    class_stats, all_sequences, all_lengths = check_distribution()

    # STEP 4: Augment data
    print("\n\nSTEP 4/4: Augmenting data to balance classes...")
    print("-" * 70)
    median_samples  = int(np.median(all_sequences))
    target_samples  = max(30, median_samples)

    print(f"\nAuto-selected target: {target_samples} samples per class")
    print("(Based on median samples per class, minimum 30)\n")

    augment_dataset(
        target_samples_per_class=target_samples,
        augmentation_methods=['noise', 'subsample', 'scale', 'jitter'],
        output_suffix='_aug',
        dry_run=False
    )

    print("\n\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 70)


def run_videos():
    print("\n" + "=" * 70)
    print("VIDEO-ONLY DATA PREPARATION PIPELINE")
    print("=" * 70 + "\n")

    print("STEP 1/3: Extracting keypoints from videos...")
    print("-" * 70)
    extract_keypoints_from_videos()

    print("\n\nSTEP 2/3: Checking data distribution...")
    print("-" * 70)
    class_stats, all_sequences, all_lengths = check_distribution()

    print("\n\nSTEP 3/3: Augmenting data to balance classes...")
    print("-" * 70)
    median_samples = int(np.median(all_sequences))
    target_samples = max(30, median_samples)

    print(f"\nAuto-selected target: {target_samples} samples per class")
    augment_dataset(
        target_samples_per_class=target_samples,
        augmentation_methods=['noise', 'subsample', 'scale', 'jitter'],
        output_suffix='_aug',
        dry_run=False
    )

    print("\n\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation pipeline")
    parser.add_argument(
        '--mode',
        choices=['images', 'videos', 'all'],
        default='all',
        help=(
            "images → extract static images only | "
            "videos → extract videos + augment | "
            "all    → extract videos + images + augment (default)"
        )
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help="Root folder of static images (default: DATA_DIR/images)"
    )
    args = parser.parse_args()

    if args.mode == 'images':
        run_images(image_dir=args.image_dir)
    elif args.mode == 'videos':
        run_videos()
    else:
        run_all(image_dir=args.image_dir)
