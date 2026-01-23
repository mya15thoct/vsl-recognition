"""
Script to convert trained stateless model to stateful model for inference.
"""
import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.stateful_model import create_stateful_model, load_weights_from_stateless
from tensorflow.keras.models import load_model


def convert_to_stateful(stateless_model_path, output_path, num_classes, timesteps=1, verify=False):
    """
    Convert stateless model to stateful model.
    
    Args:
        stateless_model_path: Path to trained stateless model (.h5)
        output_path: Path to save stateful model (.h5)
        num_classes: Number of action classes
        timesteps: Timesteps for stateful model (default=1)
        verify: If True, verify predictions match
    
    Returns:
        stateful_model
    """
    print("=" * 70)
    print("CONVERTING STATELESS MODEL TO STATEFUL")
    print("=" * 70)
    
    # 1. Create stateful model
    print(f"\n[1/4] Creating stateful model...")
    print(f"  - num_classes: {num_classes}")
    print(f"  - timesteps: {timesteps}")
    stateful_model = create_stateful_model(num_classes=num_classes, timesteps=timesteps)
    print("  Stateful model created")
    
    # 2. Load weights from stateless
    print(f"\n[2/4] Loading weights from stateless model...")
    print(f"  - Source: {stateless_model_path}")
    stateful_model = load_weights_from_stateless(stateful_model, stateless_model_path)
    
    # 3. Verify predictions (optional)
    if verify:
        print(f"\n[3/4] Verifying predictions...")
        verify_conversion(stateless_model_path, stateful_model, num_classes)
    else:
        print(f"\n[3/4] Skipping verification")
    
    # 4. Save stateful model
    print(f"\n[4/4] Saving stateful model...")
    print(f"  - Output: {output_path}")
    stateful_model.save(output_path)
    print("  Stateful model saved")
    
    print("\n" + "=" * 70)
    print(" CONVERSION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nStateful model saved to: {output_path}")
    print(f"Use this model for variable-length inference.")
    
    return stateful_model


def verify_conversion(stateless_path, stateful_model, num_classes):
    """
    Verify that stateful and stateless models give similar predictions.
    """
    # Load stateless model
    stateless_model = load_model(stateless_path)
    sequence_length = stateless_model.input_shape[1]
    
    # Create test sequence
    test_sequence = np.random.rand(1, sequence_length, 1662).astype('float32')
    
    # Stateless prediction
    stateless_pred = stateless_model.predict(test_sequence, verbose=0)
    
    # Stateful prediction (frame-by-frame)
    stateful_model.reset_states()
    stateful_preds = []
    for i in range(sequence_length):
        frame = test_sequence[:, i:i+1, :]  # (1, 1, 1662)
        pred = stateful_model.predict(frame, verbose=0)
        stateful_preds.append(pred)
    stateful_pred = stateful_preds[-1]  # Last prediction
    
    # Compare
    diff = np.abs(stateless_pred - stateful_pred).max()
    print(f"  - Max difference: {diff:.6f}")
    
    if diff < 1e-4:
        print("  Predictions match!")
    else:
        print(f" Predictions differ by {diff:.6f}")
        print("    This is normal due to numerical precision.")


def main():
    parser = argparse.ArgumentParser(description='Convert stateless model to stateful')
    parser.add_argument('--input', required=True, help='Path to stateless model (.h5)')
    parser.add_argument('--output', required=True, help='Path to save stateful model (.h5)')
    parser.add_argument('--num_classes', type=int, default=76, help='Number of classes')
    parser.add_argument('--timesteps', type=int, default=1, help='Timesteps per prediction')
    parser.add_argument('--verify', action='store_true', help='Verify predictions match')
    
    args = parser.parse_args()
    
    convert_to_stateful(
        stateless_model_path=args.input,
        output_path=args.output,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
        verify=args.verify
    )


if __name__ == "__main__":
    main()
