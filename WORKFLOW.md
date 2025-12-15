# Project Workflow and Usage Guide

## Complete Workflow

The sign language detection project follows this pipeline:

```
1. Data Preparation → 2. Keypoint Extraction → 3. Model Training → 4. Inference
```

---

## Step-by-Step Execution

### Step 1: Data Preparation (Optional)

**If you don't have dataset yet:**

```bash
cd data
python prepare_dataset.py
```

**What it does:**
- Crawls Vietnamese Sign Language videos from QIPEDC website
- Downloads videos and extracts frames
- Creates `VSL_Isolated/` dataset structure
- Generates dictionary.txt mapping class IDs to words

**Output:**
- `data/VSL_Isolated/videos/` - Downloaded videos
- `data/VSL_Isolated/frames/` - Extracted frames
- `data/VSL_Isolated/dictionary.txt` - Class mappings

---

### Step 2: Extract Keypoints from Videos

```bash
cd sign_language_detection
python -m data.collect_data
```

**What it does:**
- Uses MediaPipe Holistic to detect pose, face, and hand landmarks
- Processes each video frame-by-frame
- Extracts 1662 keypoints per frame (pose + face + hands)
- Saves sequences of 30 frames per video
- Creates .npy files with keypoint arrays

**Output:**
- `data/VSL_Isolated/sequences/[action]/[sequence_id]/[sequence_id].npy`

**Example:**
```
data/VSL_Isolated/sequences/
├── 000001/
│   ├── 0/0.npy
│   ├── 1/1.npy
│   └── ...
├── 000002/
│   └── ...
```

---

### Step 3: Train Model

```bash
cd sign_language_detection
python train.py
```

**What it does:**
- Loads all keypoint sequences from Step 2
- Splits data into train/test (80/20)
- Builds LSTM model architecture
- Trains for 2000 epochs (configurable in config.py)
- Saves best model and final model

**Model Architecture:**
```
Input (30 frames × 1662 keypoints)
    ↓
LSTM Layer 1 (64 units)
    ↓
LSTM Layer 2 (128 units)
    ↓
LSTM Layer 3 (64 units)
    ↓
Dense Layer 1 (64 units)
    ↓
Dense Layer 2 (32 units)
    ↓
Output (num_actions with softmax)
```

**Output:**
- `sign_language_detection/models/action_model_best.h5` - Best validation accuracy
- `sign_language_detection/models/action_model_final.h5` - Final epoch model
- `sign_language_detection/models/actions.npy` - Action label mappings
- `sign_language_detection/logs/training/` - TensorBoard logs

**During training, you'll see:**
```
Epoch 1/2000
loss: 2.5431 - accuracy: 0.2341 - val_accuracy: 0.3211
Epoch 2/2000
loss: 1.8234 - accuracy: 0.4512 - val_accuracy: 0.5123
...
```

---

### Step 4: Run Inference

**Webcam (Real-time):**
```bash
cd sign_language_detection
python inference.py
```

**Video File:**
```bash
python inference.py --video path/to/video.mp4
```

**What it does:**
- Loads trained model
- Captures webcam/video frames
- Extracts keypoints using MediaPipe
- Collects 30-frame sequences
- Predicts action for each sequence
- Displays prediction with confidence score

**Press 'q' to quit**

---

## Accuracy Evaluation

### Method 1: Training Metrics (Automatic)

The `train.py` script already provides accuracy metrics:

```python
# Training shows:
- Training accuracy per epoch
- Validation accuracy per epoch
- Best model is saved based on highest val_accuracy
```

**View training progress:**
```bash
cd sign_language_detection
tensorboard --logdir=logs/training
```

Then open: http://localhost:6006

### Method 2: Test Set Evaluation

Create a new evaluation script `evaluate.py`:

```python
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
    
    return np.array(sequences), np.array(labels), actions

def evaluate_model(model_path=None):
    """Evaluate trained model"""
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Load data
    X, y, actions = load_sequences()
    y_cat = to_categorical(y).astype(int)
    
    # Same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )
    
    # Load model
    if model_path is None:
        model_path = MODEL_PATH / 'action_model_best.h5'
    
    print(f"\nLoading model: {model_path}")
    model = load_model(str(model_path))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Accuracy (sklearn): {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, 
                                target_names=actions, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=actions, yticklabels=actions)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    output_path = MODEL_PATH / 'confusion_matrix.png'
    plt.savefig(output_path)
    print(f"\nConfusion matrix saved to: {output_path}")
    plt.show()
    
    return test_accuracy

if __name__ == "__main__":
    evaluate_model()
```

**Run evaluation:**
```bash
cd sign_language_detection
python evaluate.py
```

**Output:**
```
Test Accuracy: 0.8523
Classification Report:
              precision    recall  f1-score   support
    000001       0.85      0.90      0.87        10
    000002       0.88      0.82      0.85         8
    ...
    
Confusion matrix saved to: models/confusion_matrix.png
```

### Method 3: Per-Class Accuracy

Add to `evaluate.py` for detailed per-class metrics:

```python
# Per-class accuracy
for i, action in enumerate(actions):
    mask = y_test_classes == i
    if mask.sum() > 0:
        class_acc = accuracy_score(
            y_test_classes[mask], 
            y_pred_classes[mask]
        )
        print(f"{action}: {class_acc:.2%}")
```

---

## Configuration

Edit `sign_language_detection/config.py` to adjust:

```python
# Training parameters
EPOCHS = 2000              # Number of training epochs
BATCH_SIZE = 32           # Batch size
LEARNING_RATE = 0.001     # Learning rate

# Data parameters
SEQUENCE_LENGTH = 30      # Frames per sequence
NO_SEQUENCES = 30         # Sequences per action

# Model architecture
LSTM_UNITS = [64, 128, 64]
DENSE_UNITS = [64, 32]
```

---

## Quick Start Commands

```bash
# Complete workflow from scratch
cd data
python prepare_dataset.py          # 1. Prepare dataset

cd ../sign_language_detection
python -m data.collect_data         # 2. Extract keypoints
python train.py                     # 3. Train model
python inference.py                 # 4. Test with webcam

# View training progress
tensorboard --logdir=logs/training

# Evaluate accuracy
python evaluate.py                  # Create this file first
```

---

## Troubleshooting

**Issue: Low accuracy**
- Increase `EPOCHS` in config.py
- Collect more sequences per action
- Increase `SEQUENCE_LENGTH` for longer videos
- Check data quality

**Issue: Overfitting**
- Add dropout layers to model
- Reduce model complexity
- Add data augmentation
- Increase train/test split

**Issue: Slow training**
- Reduce `EPOCHS`
- Decrease `SEQUENCE_LENGTH`
- Use GPU if available
- Reduce number of actions
