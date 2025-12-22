"""
Run full training and evaluation pipeline
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from training.train import train_model
from training.evaluate import evaluate_model


def run_full_pipeline():
    """
    Run complete pipeline: train and evaluate
    """
    print("="*70)
    print("FULL TRAINING & EVALUATION PIPELINE")
    print("="*70)
    
    # Step 1: Train model
    print("\n" + "="*70)
    print("STEP 1: TRAINING")
    print("="*70)
    history, train_test_acc = train_model()
    
    # Step 2: Evaluate on test set
    print("\n" + "="*70)
    print("STEP 2: DETAILED EVALUATION")
    print("="*70)
    eval_acc, confusion_matrix = evaluate_model()
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED")
    print("="*70)
    print(f"Training completed successfully")
    print(f"Final test accuracy: {eval_acc*100:.2f}%")
    print(f"Confusion matrix saved to checkpoints/confusion_matrix.png")
    print("="*70)
    
    return history, eval_acc, confusion_matrix


if __name__ == "__main__":
    run_full_pipeline()
