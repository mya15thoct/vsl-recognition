# Sign Language Recognition

CNN + LSTM model for sign language recognition using MediaPipe keypoints.

## Quick Start

```bash
# 1. Extract keypoints from videos
cd preprocessing
python scripts/extract_include.py

# 2. Train model
cd training
python run.py

# 3. Evaluate
python evaluate.py
```

## Project Structure

```
preprocessing/
├── models/
│   └── hybrid_model.py   # CNN + LSTM
├── training/
│   ├── train.py          # Training
│   ├── evaluate.py       # Evaluation
│   └── run.py            # Entry point
├── scripts/
│   ├── extract_include.py  # Keypoint extraction
│   └── inference.py        # Real-time inference
└── config.py
```

## Model Architecture

```
Input (33 frames, 1662 keypoints)
    ↓
Split → Pose/Face/Hands
    ↓
CNN Branches (TimeDistributed)
    ↓
Concatenate (256D)
    ↓
LSTM (128 → 64)
    ↓
Dense → Softmax (76 classes)
```

## Dataset

- **INCLUDE**: Indian Sign Language
- 76 classes, 1,166 videos

## Inference

```bash

# Video file
python -m scripts.inference --mode video --video input.mp4
```
