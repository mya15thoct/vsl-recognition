"""
Evaluate model accuracy on test set
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from config import SEQUENCE_PATH, MODEL_PATH

def load_sequences(sequence_path=SEQUENCE_PATH):
    """Load all saved keypoint sequences"""
    action_folders = [d for d in sequence_path.iterdir() if d.is_dir()]
    actions = [d.name for d in action_folders]
    
    print(f"Found {len(actions)} actions")
    
    sequences = []
    labels = []
    
    for action_idx, action in enumerate(actions):
        action_path = sequence_path / action
        sequence_folders = [d for d in action_path.iterdir() if d.is_dir()]
        
        for sequence_folder in sequence_folders:
            sequence_files = list(sequence_folder.glob('*.npy'))
            if sequence_files:
                sequence_data = np.load(sequence_files[0])
                sequences.append(sequence_data)
                labels.append(action_idx)
    
    print(f"Total sequences loaded: {len(sequences)}")
    return np.array(sequences), np.array(labels), actions

def evaluate_model(model_path=None):
    """Evaluate trained model on test set"""
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load data
    X, y, actions = load_sequences()
    y_cat = to_categorical(y).astype(int)
    
    # Same split as training (important: use same random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )
    
    print(f"\nTest set size: {len(X_test)} sequences")
    print(f"Number of classes: {len(actions)}")
    
    # Load model
    if model_path is None:
        model_path = MODEL_PATH / 'action_model_best.h5'
    
    print(f"\nLoading model: {model_path}")
    model = load_model(str(model_path))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    
    # Get predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Accuracy (sklearn): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test_classes, y_pred_classes, 
                                target_names=actions, zero_division=0))
    
    # Per-class accuracy
    print("\n" + "="*60)
    print("PER-CLASS ACCURACY")
    print("="*60)
    for i, action in enumerate(actions):
        mask = y_test_classes == i
        if mask.sum() > 0:
            class_acc = accuracy_score(
                y_test_classes[mask], 
                y_pred_classes[mask]
            )
            support = mask.sum()
            print(f"{action}: {class_acc:.2%} ({support} samples)")
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(max(12, len(actions)), max(10, len(actions))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=actions, yticklabels=actions)
    plt.title(f'Confusion Matrix - Test Accuracy: {test_accuracy:.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    output_path = MODEL_PATH / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Confusion matrix saved to: {output_path}")
    
    # Show plot
    try:
        plt.show()
    except:
        print("[WARNING] Could not display plot (no display available)")
    
    print("\n" + "="*60)
    print(f"[OK] Evaluation complete!")
    print("="*60)
    
    return test_accuracy

if __name__ == "__main__":
    evaluate_model()
