# VSL Recognition - Sign Language Recognition

CNN + LSTM model for Vietnamese Sign Language recognition using MediaPipe keypoints.

## Quick Start

```bash
# 1. Prepare dataset (extract + augment)
python main.py data prepare

# 2. Train model
python main.py train

# 3. Run inference
python -m src.inference.realtime --mode webcam
```

## Project Structure

```
vsl-recognition/
├── src/                          # Main source code
│   ├── data/                     # Data preparation
│   │   ├── extract.py            # Keypoint extraction from videos
│   │   ├── augment.py            # Data augmentation
│   │   ├── check_distribution.py # Dataset statistics
│   │   └── prepare_pipeline.py   # Full data prep pipeline
│   │
│   ├── models/                   # Model architectures
│   │   ├── components.py         # CNN/MLP branches
│   │   ├── hybrid.py             # MLP+LSTM hybrid model
│   │   ├── stateful.py           # Stateful variant for inference
│   │   └── converter.py          # Model conversion utilities
│   │
│   ├── training/                 # Training pipeline
│   │   ├── data_loader.py        # Data loading utilities
│   │   ├── trainer.py            # Training logic
│   │   ├── evaluator.py          # Evaluation logic
│   │   └── pipeline.py           # Full training pipeline
│   │
│   ├── inference/                # Inference scripts
│   │   └── realtime.py           # Real-time inference (webcam/video)
│   │
│   ├── visualization/            # Visualization tools
│   │   ├── keypoints.py          # Visualize extracted keypoints
│   │   └── sequences.py          # Visualize sequences
│   │
│   ├── utils/                    # Shared utilities
│   │   ├── extraction.py         # MediaPipe keypoint extraction
│   │   ├── augmentation.py       # Augmentation methods
│   │   ├── inference_utils.py    # Inference utilities
│   │   └── viz_utils.py          # Visualization utilities
│   │
│   └── config.py                 # Configuration
│
├── data/                         # Dataset (gitignored)
│   ├── raw/                      # Raw videos
│   └── sequences/                # Extracted keypoints
│
├── main.py                       # Main CLI entry point
├── requirements.txt
└── README.md
```

## Model Architecture

```
Input (33 frames, 1662 keypoints)
    ↓
Split → Pose/Face/Hands
    ↓
MLP Branches (TimeDistributed Dense)
    ↓
Concatenate (256D)
    ↓
Shared Dense Layers
    ↓
LSTM (128 → 64)
    ↓
Dense → Softmax (76 classes)
```

## Dataset

- **INCLUDE**: Vietnamese Sign Language
- 76 classes, 1,166 videos
- Automatically balanced through augmentation

## Usage

### Data Preparation

```bash
# Full pipeline (extract + check + augment)
python main.py data prepare

# Just check distribution
python main.py data check
```

### Training

```bash
# Full training pipeline
python main.py train

# Or run directly
python -m src.training.pipeline
```

### Inference

```bash
# Webcam
python -m src.inference.realtime --mode webcam

# Video file
python -m src.inference.realtime --mode video --video input.mp4
```

### Visualization

```bash
# Visualize keypoints
python -m src.visualization.keypoints --class "hello"

# Visualize sequences
python -m src.visualization.sequences
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- MediaPipe
- OpenCV
- NumPy, Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```
