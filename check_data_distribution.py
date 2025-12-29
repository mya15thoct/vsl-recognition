"""
Check data distribution in INCLUDE sequences
"""
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / "preprocessing"))
from config import SEQUENCE_PATH

def check_distribution():
    """Check sequence distribution across classes"""
    print("=" * 70)
    print("DATA DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"Sequences path: {SEQUENCE_PATH}\n")
    
    # Get all class folders
    class_folders = sorted([d for d in SEQUENCE_PATH.iterdir() if d.is_dir()])
    
    if not class_folders:
        print("ERROR: No class folders found!")
        print(f"Path checked: {SEQUENCE_PATH}")
        return
    
    print(f"Total classes: {len(class_folders)}\n")
    
    # Collect statistics
    all_sequences = []
    all_lengths = []
    class_stats = []
    
    for idx, class_folder in enumerate(class_folders, 1):
        npy_files = list(class_folder.glob('*.npy'))
        num_sequences = len(npy_files)
        
        # Load first file to get sequence lengths
        lengths = []
        for npy_file in npy_files:
            seq = np.load(npy_file)
            lengths.append(len(seq))
        
        all_sequences.append(num_sequences)
        all_lengths.extend(lengths)
        
        avg_length = np.mean(lengths) if lengths else 0
        
        class_stats.append({
            'name': class_folder.name,
            'count': num_sequences,
            'min_len': min(lengths) if lengths else 0,
            'max_len': max(lengths) if lengths else 0,
            'avg_len': avg_length
        })
        
        # Print progress
        if num_sequences > 0:
            print(f"[{idx:3d}/{len(class_folders)}] {class_folder.name:30s} | "
                  f"Sequences: {num_sequences:3d} | "
                  f"Length: {min(lengths):3d}-{max(lengths):3d} (avg: {avg_length:.1f})")
        else:
            print(f"[{idx:3d}/{len(class_folders)}] {class_folder.name:30s} | "
                  f"⚠️  NO SEQUENCES")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    total_sequences = sum(all_sequences)
    print(f"\nTotal sequences: {total_sequences}")
    print(f"Total classes: {len(class_folders)}")
    print(f"\nSequences per class:")
    print(f"  Min: {min(all_sequences)}")
    print(f"  Max: {max(all_sequences)}")
    print(f"  Mean: {np.mean(all_sequences):.1f}")
    print(f"  Median: {np.median(all_sequences):.1f}")
    
    print(f"\nSequence lengths (frames):")
    print(f"  Min: {min(all_lengths)}")
    print(f"  Max: {max(all_lengths)}")
    print(f"  Mean: {np.mean(all_lengths):.1f}")
    print(f"  Median: {np.median(all_lengths):.1f}")
    
    # Check for imbalanced classes
    print(f"\n{'=' * 70}")
    print("POTENTIAL ISSUES")
    print("=" * 70)
    
    # Classes with too few samples
    min_threshold = 5
    few_samples = [s for s in class_stats if s['count'] < min_threshold]
    if few_samples:
        print(f"\n⚠️  {len(few_samples)} class(es) with < {min_threshold} sequences:")
        for s in few_samples[:10]:
            print(f"     {s['name']}: {s['count']} sequences")
        if len(few_samples) > 10:
            print(f"     ... and {len(few_samples) - 10} more")
    
    # Classes with no samples
    no_samples = [s for s in class_stats if s['count'] == 0]
    if no_samples:
        print(f"\n❌ {len(no_samples)} class(es) with NO sequences:")
        for s in no_samples[:10]:
            print(f"     {s['name']}")
        if len(no_samples) > 10:
            print(f"     ... and {len(no_samples) - 10} more")
    
    # Large variation in sequence lengths
    length_range = max(all_lengths) - min(all_lengths)
    if length_range > 100:
        print(f"\n⚠️  Large sequence length variation: {min(all_lengths)} - {max(all_lengths)} frames")
        print(f"     → Model will need padding, may affect performance")
    
    print("\n" + "=" * 70)
    
    return class_stats, all_sequences, all_lengths


if __name__ == "__main__":
    check_distribution()
