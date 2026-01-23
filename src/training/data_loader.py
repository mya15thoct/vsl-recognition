"""
Load sequences and create train/val/test datasets
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import SEQUENCE_PATH


def load_sequences(sequence_path=SEQUENCE_PATH, max_sequences_per_action=None, target_length=None):
    """
    Load all sequences and labels, padding to uniform length
    
    Args:
        sequence_path: Path to sequences directory
        max_sequences_per_action: Limit sequences per action (None = all). Use for testing with limited RAM.
        target_length: Target sequence length for padding (None = auto-detect from data)
    
    Returns:
        X: np.array (num_samples, max_length, 1662) - padded sequences
        y: np.array (num_samples,) - label indices
        action_names: list - Action names
    """
    print("Loading sequences...")
    if max_sequences_per_action:
        print(f"  [LIMIT MODE] Max {max_sequences_per_action} sequences per action")
    
    # Get all action folders
    action_folders = sorted([d for d in sequence_path.iterdir() if d.is_dir()])
    action_names = [d.name for d in action_folders]
    
    X = []
    y = []
    sequence_lengths = []
    
    for label_idx, action_folder in enumerate(action_folders):
        npy_files = sorted(action_folder.glob('*.npy'))
        
        # Limit sequences if specified
        if max_sequences_per_action:
            npy_files = npy_files[:max_sequences_per_action]
        
        for npy_file in npy_files:
            seq = np.load(npy_file)
            X.append(seq)
            y.append(label_idx)
            sequence_lengths.append(len(seq))
        
        print(f"  [{label_idx+1}/{len(action_folders)}] {action_folder.name}: {len(npy_files)} sequences")
    
    # Determine padding length
    if target_length is not None:
        max_length = target_length
        print(f"\n  Using target sequence length: {max_length} frames")
    else:
        max_length = max(sequence_lengths)
        print(f"\n  Auto-detected max sequence length: {max_length} frames")
    
    print(f"  Sequence length stats:")
    print(f"    Min: {min(sequence_lengths)} frames")
    print(f"    Max: {max(sequence_lengths)} frames")
    print(f"    Mean: {np.mean(sequence_lengths):.1f} frames")
    
    # Pad or truncate all sequences to max_length
    print(f"\n  Padding/truncating all sequences to {max_length} frames...")
    X_padded = []
    for seq in X:
        if len(seq) < max_length:
            # Pad with zeros
            padding = np.zeros((max_length - len(seq), seq.shape[1]))
            seq_padded = np.vstack([seq, padding])
        elif len(seq) > max_length:
            # Truncate to max_length
            seq_padded = seq[:max_length]
        else:
            seq_padded = seq
        X_padded.append(seq_padded)
    
    X = np.array(X_padded)
    y = np.array(y)
    
    print(f"\nLoaded {len(X)} sequences")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Actions: {len(action_names)}")
    
    return X, y, action_names


def split_data(X, y, train_size=0.7, val_size=0.15, random_state=42):
    """
    Split into train/val/test with smart stratification
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Check class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_samples = class_counts.min()
    
    print(f"\nClass distribution:")
    print(f"  Total classes: {len(unique_classes)}")
    print(f"  Minimum samples per class: {min_samples}")
    print(f"  Maximum samples per class: {class_counts.max()}")
    print(f"  Average samples per class: {class_counts.mean():.1f}")
    
    # Check if stratification is possible
    # IMPORTANT: We need to check if classes will have enough samples AFTER first split
    # If a class has 3 samples and we do 70/30 split, it might become 2/1
    # Then the second split on the "1" will fail with stratify
    
    # Calculate minimum samples in temp set after first split
    temp_size = 1 - train_size  # 0.3 for default
    min_samples_in_temp = int(np.floor(min_samples * temp_size))
    
    # Need at least 2 samples in temp set to do stratified val/test split
    can_stratify = min_samples_in_temp >= 2
    
    if not can_stratify:
        # Find problematic classes (those that will have < 2 in temp)
        problematic = []
        for i in range(len(unique_classes)):
            samples_in_temp = int(np.floor(class_counts[i] * temp_size))
            if samples_in_temp < 2:
                problematic.append((unique_classes[i], class_counts[i], samples_in_temp))
        
        print(f"\n⚠️  WARNING: {len(problematic)} class(es) will have < 2 samples in val+test set:")
        print(f"     (After {train_size*100:.0f}/{temp_size*100:.0f} split)")
        for cls_idx, total_count, temp_count in problematic[:5]:  # Show first 5
            print(f"     Class {cls_idx}: {total_count} total → ~{temp_count} in val+test")
        if len(problematic) > 5:
            print(f"     ... and {len(problematic) - 5} more")
    
    # Decide split strategy
    if can_stratify:
        print(f"  ✓ Using STRATIFIED split (balanced distribution)")
        # Split train vs (val+test) - stratified
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=train_size, random_state=random_state, stratify=y
        )
        
        # Split val vs test - stratified
        val_ratio = val_size / (1 - train_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_ratio, random_state=random_state, stratify=y_temp
        )
    else:
        print(f"  ⚠️  Using RANDOM split (stratification not possible)")
        print(f"     → May cause class imbalance in train/val/test sets")
        
        # Split train vs (val+test) - non-stratified
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=train_size, random_state=random_state
        )
        
        # Split val vs test - non-stratified
        val_ratio = val_size / (1 - train_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_ratio, random_state=random_state
        )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_tf_dataset(X, y, batch_size=32, shuffle=True):
    """
    Create tf.data.Dataset
    """
    # One-hot encode labels
    num_classes = y.max() + 1
    y_onehot = tf.keras.utils.to_categorical(y, num_classes)
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y_onehot))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    # Test
    X, y, actions = load_sequences()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    train_ds = create_tf_dataset(X_train, y_train, batch_size=32)
    
    print(f"\nDataset created")
    for batch_X, batch_y in train_ds.take(1):
        print(f"   Batch X shape: {batch_X.shape}")  # (32, 130, 1662)
        print(f"   Batch y shape: {batch_y.shape}")  # (32, num_classes)
