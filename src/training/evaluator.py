import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import argparse
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import CHECKPOINT_DIR, SEQUENCE_PATH, TRAINING_CONFIG


def load_sequences_with_mapping(sequence_path, action_mapping_path, seq_length):
    """Load sequences using a fixed action_mapping (not auto-discover)."""
    with open(action_mapping_path) as f:
        mapping = json.load(f)                      # {"0": "BRING", ...}
    action_names = [mapping[str(i)] for i in range(len(mapping))]
    name_to_idx  = {name: i for i, name in enumerate(action_names)}

    seq_path = Path(sequence_path)
    X, y, is_original = [], [], []

    for class_name, label_idx in name_to_idx.items():
        folder = seq_path / class_name
        if not folder.exists():
            continue
        for npy in sorted(folder.glob('*.npy')):
            seq = np.load(npy).astype(np.float32)
            padded = np.zeros((seq_length, 1662), dtype=np.float32)
            padded[:min(len(seq), seq_length)] = seq[:min(len(seq), seq_length)]
            X.append(padded)
            y.append(label_idx)
            is_original.append('_static' not in npy.stem and '_aug' not in npy.stem)

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32),
            action_names,
            np.array(is_original, dtype=bool))


def evaluate_model(model_path=None, action_mapping_path=None, sequence_path=None):
    model_path          = model_path or str(CHECKPOINT_DIR / 'best_model')
    sequence_path       = sequence_path or str(SEQUENCE_PATH)
    action_mapping_path = action_mapping_path or str(CHECKPOINT_DIR / 'action_mapping.json')

    print(f"Loading model from: {model_path}")
    model   = tf.keras.models.load_model(model_path)
    seq_len = model.input_shape[1]
    n_out   = model.output_shape[-1]
    print(f"  Input length: {seq_len} frames  |  Output classes: {n_out}")

    # Load data with correct class mapping
    print(f"\nLoading sequences from: {sequence_path}")
    print(f"Action mapping:         {action_mapping_path}")
    X, y, action_names, is_original = load_sequences_with_mapping(
        sequence_path, action_mapping_path, seq_len
    )

    # Test split: only original (non-augmented) samples, last 15%
    orig_idx = np.where(is_original)[0]
    n_test   = max(1, int(len(orig_idx) * TRAINING_CONFIG['test_split']))
    test_idx = orig_idx[-n_test:]
    X_test   = X[test_idx]
    y_test   = y[test_idx]
    print(f"  Test set: {len(X_test)} samples  (original only)")

    # Predict
    print("\nPredicting...")
    y_pred         = model.predict(X_test, batch_size=64, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Only report classes that appear in test set
    unique_classes = np.unique(y_test)
    class_names    = [action_names[i] for i in unique_classes]

    print("\n========== EVALUATION RESULTS ==========")
    report = classification_report(
        y_test, y_pred_classes,
        labels=unique_classes,
        target_names=class_names,
        output_dict=True, zero_division=0
    )
    print(classification_report(
        y_test, y_pred_classes,
        labels=unique_classes,
        target_names=class_names,
        zero_division=0
    ))

    accuracy  = report.get('accuracy', 0.0)
    macro_f1  = report['macro avg']['f1-score']
    macro_pre = report['macro avg']['precision']
    macro_rec = report['macro avg']['recall']

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"F1 Score : {macro_f1 * 100:.2f}%")
    print(f"Precision: {macro_pre * 100:.2f}%")
    print(f"Recall   : {macro_rec * 100:.2f}%")
    print("=========================================")

    # Confusion matrix
    cm      = confusion_matrix(y_test, y_pred_classes, labels=unique_classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    n        = len(class_names)
    fig_size = max(24, n * 0.18)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(cm_norm, ax=ax, cmap='Blues', vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0, annot=False)
    ax.set_title(f'Confusion Matrix ({n} classes)')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.tick_params(axis='x', labelsize=5, rotation=90)
    ax.tick_params(axis='y', labelsize=5, rotation=0)
    plt.tight_layout()

    save_dir = Path(model_path).parent
    out_full = save_dir / 'confusion_matrix_combined.png'
    plt.savefig(out_full, dpi=200)
    plt.close()
    print(f"Confusion matrix saved: {out_full}")

    return accuracy, macro_f1, macro_pre, macro_rec, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',     type=str, default=None)
    parser.add_argument('--action_mapping', type=str, default=None)
    parser.add_argument('--sequence_path',  type=str, default=None)
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        action_mapping_path=args.action_mapping,
        sequence_path=args.sequence_path,
    )
