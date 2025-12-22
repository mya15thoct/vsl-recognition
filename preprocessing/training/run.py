"""
Run full training and evaluation pipeline
"""
import sys
from pathlib import Path
import tensorflow as tf

sys.path.append(str(Path(__file__).parent.parent))

from training.train import train_model
from training.evaluate import evaluate_model


def setup_gpu():
    """
    Configure GPU settings
    """
    print("="*70)
    print("GPU CONFIGURATION")
    print("="*70)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        try:
            # Enable memory growth (prevent TensorFlow from allocating all GPU memory)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
            
            # Set visible devices (optional, use first GPU)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"Using GPU: {gpus[0].name}")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU detected - training will use CPU (slower)")
    
    print("="*70 + "\n")
    
    return gpus


def run_full_pipeline():
    """
    Run complete pipeline: train and evaluate
    """
    # Step 0: Setup GPU
    setup_gpu()
    
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
