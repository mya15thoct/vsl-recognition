"""
Data distribution checker for VSL Recognition dataset

Analyzes class distribution and provides statistics
"""
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import SEQUENCE_PATH


def check_distribution():
    """
    Check data distribution across all classes
    
    Returns:
        class_stats: Dictionary with class names and sample counts
        all_sequences: List of sample counts per class
        all_lengths: List of sequence lengths
    """
    print("=" * 70)
    print("DATA DISTRIBUTION CHECK")
    print("=" * 70)
    
    class_folders = sorted([d for d in SEQUENCE_PATH.iterdir() if d.is_dir()])
    
    class_stats = {}
    all_sequences = []
    all_lengths = []
    
    for idx, class_folder in enumerate(class_folders, 1):
        npy_files = list(class_folder.glob('*.npy'))
        num_samples = len(npy_files)
        
        class_stats[class_folder.name] = num_samples
        all_sequences.append(num_samples)
        
        # Get sequence lengths
        for npy_file in npy_files[:3]:  # Sample first 3 files
            try:
                seq = np.load(npy_file)
                all_lengths.append(len(seq))
            except:
                pass
        
        print(f"[{idx:3d}/{len(class_folders)}] {class_folder.name:30s} | {num_samples:3d} sequences")
    
    # Statistics
    all_sequences = np.array(all_sequences)
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Total classes: {len(class_folders)}")
    print(f"Total sequences: {all_sequences.sum()}")
    print(f"Min samples per class: {all_sequences.min()}")
    print(f"Max samples per class: {all_sequences.max()}")
    print(f"Mean samples per class: {all_sequences.mean():.2f}")
    print(f"Median samples per class: {np.median(all_sequences):.0f}")
    
    if all_lengths:
        print(f"\nSequence length range: {min(all_lengths)} - {max(all_lengths)} frames")
    
    print("=" * 70)
    
    return class_stats, all_sequences, all_lengths


if __name__ == "__main__":
    check_distribution()
