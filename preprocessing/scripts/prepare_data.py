"""
Master script: Extract keypoints + Augment data in one go
Run this to prepare full dataset for training
"""
import sys
from pathlib import Path

# Add preprocessing to path
sys.path.append(str(Path(__file__).parent / "preprocessing"))

print("\n" + "="*70)
print("DATA PREPARATION PIPELINE")
print("="*70 + "\n")

# STEP 1: Extract keypoints from videos
print("STEP 1/3: Extracting keypoints from videos...")
print("-" * 70)
from preprocessing.scripts.extract_include import extract_include_keypoints
extract_include_keypoints()

# STEP 2: Check data distribution
print("\n\nSTEP 2/3: Checking data distribution...")
print("-" * 70)
from check_data_distribution import check_distribution
class_stats, all_sequences, all_lengths = check_distribution()

# STEP 3: Augment data (only if needed)
print("\n\nSTEP 3/3: Augmenting data to balance classes...")
print("-" * 70)
from preprocessing.scripts.augment_data import augment_dataset

# Calculate smart target based on data
import numpy as np
median_samples = int(np.median(all_sequences))
target_samples = max(15, median_samples)  # At least 15, or median

print(f"\nAuto-selected target: {target_samples} samples per class")
print("(Based on median samples per class)\n")

augment_dataset(
    target_samples_per_class=target_samples,
    augmentation_methods=['noise', 'subsample', 'scale'],
    output_suffix='_aug',
    dry_run=False
)

# Final summary
print("\n\n" + "="*70)
print("DATA PREPARATION COMPLETE!")
print("="*70)
print("Ready to train:")
print("  python preprocessing/training/run.py")
print("="*70 + "\n")
