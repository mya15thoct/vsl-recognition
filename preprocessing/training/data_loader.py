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


def load_sequences(sequence_path=SEQUENCE_PATH, max_sequences_per_action=None):
    """
    Load all sequences and labels
    
    Args:
        sequence_path: Path to sequences directory
        max_sequences_per_action: Limit sequences per action (None = all). Use for testing with limited RAM.
    
    Returns:
        X: np.array (num_samples, 130, 1662)
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
    
    for label_idx, action_folder in enumerate(action_folders):
        npy_files = sorted(action_folder.glob('*.npy'))
        
        # Limit sequences if specified
        if max_sequences_per_action:
            npy_files = npy_files[:max_sequences_per_action]
        
        for npy_file in npy_files:
            seq = np.load(npy_file)
            X.append(seq)
            y.append(label_idx)
        
        print(f"  [{label_idx+1}/{len(action_folders)}] {action_folder.name}: {len(npy_files)} sequences")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nLoaded {len(X)} sequences")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Actions: {len(action_names)}")
    
    return X, y, action_names


def split_data(X, y, train_size=0.7, val_size=0.15, random_state=42):
    """
    Split into train/val/test
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Split train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )
    
    # Split val vs test
    val_ratio = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_ratio, random_state=random_state, stratify=y_temp
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
