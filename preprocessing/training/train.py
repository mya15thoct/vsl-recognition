"""
Main training script
"""
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from models.combined_model import create_sign_language_model
from training.data_loader import load_sequences, split_data, create_tf_dataset
from config import TRAINING_CONFIG, CHECKPOINT_DIR, LOGS_DIR


def train_model():
    """Main training function"""
    
    print("="*70)
    print("SIGN LANGUAGE RECOGNITION - TRAINING")
    print("="*70)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    X, y, action_names = load_sequences()  # Load ALL sequences
    num_classes = len(action_names)
    
    # 2. Split data
    print("\n[2/5] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_size=TRAINING_CONFIG['train_split'],
        val_size=TRAINING_CONFIG['val_split']
    )
    
    # 3. Create datasets
    print("\n[3/5] Creating TensorFlow datasets...")
    train_ds = create_tf_dataset(X_train, y_train, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
    val_ds = create_tf_dataset(X_val, y_val, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)
    test_ds = create_tf_dataset(X_test, y_test, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)
    
    # 4. Build model
    print("\n[4/5] Building model...")
    model = create_sign_language_model(num_classes=num_classes)
    
    model.compile(
        optimizer=Adam(learning_rate=TRAINING_CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 5. Setup callbacks
    print("\n[5/5] Setting up training...")
    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(CHECKPOINT_DIR / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=TRAINING_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=TRAINING_CONFIG['reduce_lr_patience'],
            verbose=1
        ),
        TensorBoard(
            log_dir=str(LOGS_DIR / 'fit'),
            histogram_freq=1
        )
    ]
    
    # 6. Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TRAINING_CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70 + "\n")
    
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    
    # 8. Save final model
    model.save(str(CHECKPOINT_DIR / 'final_model.h5'))
    
    # 9. Save action mapping
    mapping = {i: name for i, name in enumerate(action_names)}
    with open(CHECKPOINT_DIR / 'action_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\nTraining complete")
    print(f"   Best model: {CHECKPOINT_DIR / 'best_model.h5'}")
    print(f"   Final model: {CHECKPOINT_DIR / 'final_model.h5'}")
    print(f"   TensorBoard: tensorboard --logdir={LOGS_DIR}")
    
    return history, test_acc


if __name__ == "__main__":
    train_model()
