"""
Master script: Extract keypoints + Augment data in one go
Run this to prepare full dataset for training
"""
import sys
from pathlib import Path

# Add project root to path (go up 2 levels: data -> src -> root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("\n" + "="*70)
print("DATA PREPARATION PIPELINE")
print("="*70 + "\n")

# STEP 1: Extract keypoints from videos
print("STEP 1/3: Extracting keypoints from videos...")
print("-" * 70)
from src.data.extract import extract_keypoints_from_videos
extract_keypoints_from_videos()

# STEP 2: Check data distribution
print("\n\nSTEP 2/3: Checking data distribution...")
print("-" * 70)
from src.data.check_distribution import check_distribution
class_stats, all_sequences, all_lengths = check_distribution()

# STEP 3: Augment data (only if needed)
print("\n\nSTEP 3/3: Augmenting data to balance classes...")
print("-" * 70)
from src.data.augment import augment_dataset

# Calculate smart target based on data
import numpy as np
median_samples = int(np.median(all_sequences))
target_samples = max(30, median_samples)  # At least 30 samples per class

print(f"\nAuto-selected target: {target_samples} samples per class")
print("(Based on median samples per class, minimum 30)\n")

augment_dataset(
    target_samples_per_class=target_samples,
    augmentation_methods=['noise', 'subsample', 'scale'],
    output_suffix='_aug',
    dry_run=False
)

# Final summary
print("\n\n" + "="*70)
print("DATA PREPARATION COMPLETE!")
