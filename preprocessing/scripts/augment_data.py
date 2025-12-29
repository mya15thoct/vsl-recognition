"""
Batch data augmentation for INCLUDE dataset sequences
Automatically balances classes by augmenting underrepresented classes
"""
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import SEQUENCE_PATH
from utils.augmentation import augment_sequence, get_augmentation_methods


def augment_dataset(
    target_samples_per_class: int = 15,
    augmentation_methods: list = None,
    output_suffix: str = '_aug',
    dry_run: bool = False
):
    """
    Augment dataset to balance classes
    
    Args:
        target_samples_per_class: Target number of samples per class
        augmentation_methods: List of augmentation methods to use
        output_suffix: Suffix for augmented files (e.g., '_aug1', '_aug2')
        dry_run: If True, only print what would be done without creating files
    """
    if augmentation_methods is None:
        # NOTE: DO NOT use 'reverse' for sign language - it breaks temporal semantics!
        augmentation_methods = ['noise', 'subsample', 'scale']
    
    print("=" * 70)
    print("DATA AUGMENTATION - BALANCING CLASSES")
    print("=" * 70)
    print(f"Target samples per class: {target_samples_per_class}")
    print(f"Augmentation methods: {', '.join(augmentation_methods)}")
    print(f"Output suffix: {output_suffix}")
    print(f"Dry run: {dry_run}")
    print("=" * 70 + "\n")
    
    # Get all class folders
    class_folders = sorted([d for d in SEQUENCE_PATH.iterdir() if d.is_dir()])
    
    total_original = 0
    total_augmented = 0
    
    for idx, class_folder in enumerate(class_folders, 1):
        npy_files = list(class_folder.glob('*.npy'))
        
        # Filter out previously augmented files
        original_files = [f for f in npy_files if output_suffix not in f.stem]
        num_original = len(original_files)
        total_original += num_original
        
        # Calculate how many augmentations needed
        num_needed = max(0, target_samples_per_class - num_original)
        
        if num_needed == 0:
            print(f"[{idx:3d}/{len(class_folders)}] {class_folder.name:30s} "
                  f"| {num_original:3d} samples → Skip (already enough)")
            continue
        
        # Distribute augmentations across available samples
        augs_per_sample = num_needed // num_original + (1 if num_needed % num_original else 0)
        
        print(f"[{idx:3d}/{len(class_folders)}] {class_folder.name:30s} "
              f"| {num_original:3d} → {target_samples_per_class:3d} samples "
              f"(+{num_needed} augmented)")
        
        if dry_run:
            total_augmented += num_needed
            continue
        
        # Augment files
        augmented_count = 0
        for file_idx, npy_file in enumerate(original_files):
            # Load original sequence
            sequence = np.load(npy_file)
            
            # Create augmentations
            for aug_idx in range(augs_per_sample):
                if augmented_count >= num_needed:
                    break
                
                # Select augmentation method cyclically
                method = augmentation_methods[aug_idx % len(augmentation_methods)]
                
                # Apply augmentation
                try:
                    augmented_seq = augment_sequence(sequence, method)
                    
                    # Save augmented sequence
                    aug_filename = f"{npy_file.stem}{output_suffix}{aug_idx + 1}.npy"
                    aug_path = class_folder / aug_filename
                    np.save(aug_path, augmented_seq)
                    
                    augmented_count += 1
                except Exception as e:
                    print(f"    [WARNING] Failed to augment {npy_file.name} with {method}: {e}")
            
            if augmented_count >= num_needed:
                break
        
        total_augmented += augmented_count
        print(f"    ✓ Created {augmented_count} augmented samples")
    
    # Summary
    print("\n" + "=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"Total classes: {len(class_folders)}")
    print(f"Original samples: {total_original}")
    print(f"Augmented samples created: {total_augmented}")
    print(f"Total samples after augmentation: {total_original + total_augmented}")
    print("=" * 70)
    
    if dry_run:
        print("\n⚠️  This was a DRY RUN - no files were created")
        print("Run without --dry-run to actually create augmented files")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment INCLUDE dataset')
    parser.add_argument('--target', type=int, default=15,
                       help='Target number of samples per class (default: 15)')
    parser.add_argument('--methods', nargs='+', 
                       choices=get_augmentation_methods(),
                       default=['noise', 'subsample', 'scale'],
                       help='Augmentation methods to use (NOTE: avoid reverse for sign language)')
    parser.add_argument('--suffix', type=str, default='_aug',
                       help='Suffix for augmented files (default: _aug)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without creating files')
    
    args = parser.parse_args()
    
    augment_dataset(
        target_samples_per_class=args.target,
        augmentation_methods=args.methods,
        output_suffix=args.suffix,
        dry_run=args.dry_run
    )
