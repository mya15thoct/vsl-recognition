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
        model_path = CHECKPOINT_DIR / 'best_model'
    
    print(f"Loading model: {model_path}")
    print(f"Model path exists: {Path(model_path).exists()}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check if model file exists at: {model_path}")
        print(f"  2. Verify the model was saved correctly")
        print(f"  3. Try retraining the model")
        raise
    
    # Get model's expected sequence length from input shape
    model_input_shape = model.input_shape
    expected_seq_length = model_input_shape[1]  # (None, seq_length, features)
    print(f"\nModel input shape: {model_input_shape}")
    print(f"Expected sequence length: {expected_seq_length} frames")
    
    # Load data with target sequence length matching the model
    print("\nLoading test data...")
    X, y, action_names, is_original = load_sequences(target_length=expected_seq_length)
    _, _, X_test, _, _, y_test = split_data(X, y, is_original=is_original)
    
    # Verify shapes match
    print(f"\nShape verification:")
    print(f"  Model expects: {model_input_shape}")
    print(f"  Data shape: {X_test.shape}")
    if X_test.shape[1:] != model_input_shape[1:]:
        raise ValueError(f"Shape mismatch! Model expects {model_input_shape[1:]}, but data has {X_test.shape[1:]}")
    
    # Predict
    print("\nPredicting...")
    y_pred = model.predict(X_test, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get unique classes in test set (some may be missing due to random split)
    unique_classes_in_test = np.unique(y_test)
    present_action_names = [action_names[i] for i in unique_classes_in_test]
    
    # Check if all classes are present
    if len(unique_classes_in_test) < len(action_names):
        missing_classes = set(range(len(action_names))) - set(unique_classes_in_test)
        print(f"\nWARNING: {len(missing_classes)} class(es) not in test set (due to random split):")
        for cls_idx in sorted(list(missing_classes))[:5]:
            print(f"     Class {cls_idx}: {action_names[cls_idx]}")
        if len(missing_classes) > 5:
            print(f"     ... and {len(missing_classes) - 5} more")
    
    # Metrics
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    report = classification_report(y_test, y_pred_classes,
                                   labels=unique_classes_in_test,
                                   target_names=present_action_names,
                                   output_dict=True,
                                   zero_division=0)
    print(classification_report(y_test, y_pred_classes,
                                labels=unique_classes_in_test,
                                target_names=present_action_names,
                                zero_division=0))

    accuracy    = report.get('accuracy', report.get('micro avg', {}).get('f1-score', 0.0))
    macro_f1    = report['macro avg']['f1-score']
    macro_pre   = report['macro avg']['precision']
    macro_rec   = report['macro avg']['recall']

    print("=" * 50)
    print(f"  Test Accuracy  : {accuracy*100:.2f}%")
    print(f"  Macro F1       : {macro_f1*100:.2f}%")
    print(f"  Macro Precision: {macro_pre*100:.2f}%")
    print(f"  Macro Recall   : {macro_rec*100:.2f}%")
    print("=" * 50)

    # Confusion matrix (only for classes in test set)
    cm = confusion_matrix(y_test, y_pred_classes, labels=unique_classes_in_test)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    # ── 1. Full normalized confusion matrix ───────────────────────────────
    n = len(present_action_names)
    fig_size = max(24, n * 0.18)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(cm_norm, ax=ax, cmap='Blues', vmin=0, vmax=1,
                xticklabels=present_action_names,
                yticklabels=present_action_names,
                linewidths=0, annot=False)
    ax.set_title(f'Normalized Confusion Matrix ({n} classes)', fontsize=14, pad=12)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.tick_params(axis='x', labelsize=5, rotation=90)
    ax.tick_params(axis='y', labelsize=5, rotation=0)
    plt.tight_layout()
    out_full = Path(__file__).parent.parent / 'visualization' / 'confusion_matrix_full.png'
    plt.savefig(out_full, dpi=200)
    plt.close()
    print(f"Confusion matrix (full) saved: {out_full}")

    # ── 2. Top confused pairs ─────────────────────────────────────────────
    TOP_N = 20
    off_diag = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                off_diag.append((cm[i, j], present_action_names[i], present_action_names[j]))
    off_diag.sort(reverse=True)
    top = off_diag[:TOP_N]

    if top:
        labels_top  = [f"{t} → {p}" for _, t, p in top]
        counts_top  = [c for c, _, _ in top]
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars = ax2.barh(labels_top[::-1], counts_top[::-1], color='steelblue')
        ax2.bar_label(bars, padding=3, fontsize=9)
        ax2.set_xlabel('Misclassification Count')
        ax2.set_title(f'Top {TOP_N} Confused Class Pairs')
        ax2.tick_params(axis='y', labelsize=9)
        plt.tight_layout()
        out_top = Path(__file__).parent.parent / 'visualization' / 'confusion_matrix_top_confused.png'
        plt.savefig(out_top, dpi=150)
        plt.close()
        print(f"Confusion matrix (top confused) saved: {out_top}")

    return accuracy, macro_f1, macro_pre, macro_rec, cm


if __name__ == "__main__":
    evaluate_model()
