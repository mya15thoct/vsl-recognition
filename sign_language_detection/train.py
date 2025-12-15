"""
Training script for Sign Language Action Detection
"""
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Import from modules
import sys
sys.path.append(str(Path(__file__).parent))

from config import SEQUENCE_PATH, MODEL_PATH, LOGS_PATH, EPOCHS
from model.architecture import build_model, get_tensorboard_callback


def load_sequences(sequence_path=SEQUENCE_PATH):
    """
    Load all saved keypoint sequences
    
    Args:
        sequence_path: Path to sequences directory
    
    Returns:
        sequences: Numpy array of sequences
        labels: Numpy array of labels  
        actions: List of action names
    """
    print("="*60)
    print("LOADING SEQUENCES")
    print("="*60)
    
    # Get all action folders
    action_folders = [d for d in sequence_path.iterdir() if d.is_dir()]
    actions = [d.name for d in action_folders]
    
    print(f"\nFound {len(actions)} actions")
    
    sequences = []
    labels = []
    
    for action_idx, action in enumerate(actions):
        action_path = sequence_path / action
        
        # Get all sequence folders
        sequence_folders = [d for d in action_path.iterdir() if d.is_dir()]
        print(f"  {action}: {len(sequence_folders)} sequences")
        
        for sequence_folder in sequence_folders:
            # Load sequence file
            sequence_files = list(sequence_folder.glob('*.npy'))
            if sequence_files:
                sequence_data = np.load(sequence_files[0])
                sequences.append(sequence_data)
                labels.append(action_idx)
    
    print(f"\nTotal sequences loaded: {len(sequences)}")
    
    return np.array(sequences), np.array(labels), actions


def train_model(epochs=EPOCHS, test_size=0.2):
    """
    Train the action detection model
    
    Args:
        epochs: Number of training epochs
        test_size: Proportion of data for testing
    """
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Load data
    X, y, actions = load_sequences()
    
    # Convert labels to categorical
    y_cat = to_categorical(y).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=test_size, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} sequences")
    print(f"  Testing: {len(X_test)} sequences")
    print(f"  Actions: {len(actions)}")
    
    # Build model
    print("\nBuilding model...")
    model = build_model(num_actions=len(actions))
    model.summary()
    
    # Setup callbacks
    log_dir = LOGS_PATH / 'training'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    tb_callback = get_tensorboard_callback(log_dir)
    
    checkpoint_path = MODEL_PATH / 'action_model_best.h5'
    checkpoint_callback = ModelCheckpoint(
        str(checkpoint_path),
        monitor='val_categorical_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    print(f"\nStarting training for {epochs} epochs...")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        callbacks=[tb_callback, checkpoint_callback],
        verbose=1
    )
    
    # Save final model
    final_model_path = MODEL_PATH / 'action_model_final.h5'
    model.save(str(final_model_path))
    
    # Save action labels
    labels_path = MODEL_PATH / 'actions.npy'
    np.save(labels_path, np.array(actions))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Best model: {checkpoint_path}")
    print(f"  Final model: {final_model_path}")
    print(f"  Action labels: {labels_path}")
    print(f"  TensorBoard logs: {log_dir}")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir={log_dir}")
    print("="*60)
    
    return model, history, actions


if __name__ == "__main__":
    # Train model
    model, history, actions = train_model()
