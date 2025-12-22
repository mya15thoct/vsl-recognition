"""
Evaluate model and generate metrics
"""
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from training.data_loader import load_sequences, split_data
from config import CHECKPOINT_DIR


def evaluate_model(model_path=None):
    """
    Evaluate model on test set
    """
    # Load best model
    if model_path is None:
        model_path = CHECKPOINT_DIR / 'best_model.h5'
    
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Load data
    print("\nLoading test data...")
    X, y, action_names = load_sequences()
    _, _, X_test, _, _, y_test = split_data(X, y)
    
    # Predict
    print("\nPredicting...")
    y_pred = model.predict(X_test, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Metrics
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, y_pred_classes, target_names=action_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=action_names, yticklabels=action_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    output_path = CHECKPOINT_DIR / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nConfusion matrix saved: {output_path}")
    
    # Accuracy
    accuracy = (y_test == y_pred_classes).mean()
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    return accuracy, cm


if __name__ == "__main__":
    evaluate_model()
