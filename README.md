# VSL Recognition - Sign Language Recognition

CNN + LSTM model for Vietnamese Sign Language recognition using MediaPipe keypoints.

## Quick Start

```bash
# 1. Prepare dataset (extract + augment)
python -m src.data.prepare_pipeline

# 2. Train model
python -m src.training.pipeline

# 3. Run inference
python -m src.inference.realtime --mode webcam
```

## Project Structure

```
vsl-recognition/
├── src/                          # Main source code (renamed from preprocessing/)
│   ├── data/                     # Data preparation
│   │   ├── extract.py            # Keypoint extraction
│   │   ├── augment.py            # Data augmentation
│   │   ├── check_distribution.py # Dataset statistics
│   │   └── prepare_pipeline.py   # Full pipeline
│   │
│   ├── models/                   # Model architectures
│   │   ├── components.py         # CNN/MLP branches
│   │   ├── hybrid.py             # MLP+LSTM model
│   │   ├── stateful.py           # Stateful variant
│   │   └── converter.py          # Model converter
│   │
│   ├── training/                 # Training pipeline
│   │   ├── data_loader.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── pipeline.py
│   │
│   ├── inference/                # Inference
│   │   └── realtime.py
│   │
│   ├── visualization/            # Visualization
│   │   ├── keypoints.py
│   │   └── sequences.py
│   │
│   ├── utils/                    # Utilities
│   │   ├── extraction.py
│   │   ├── augmentation.py
│   │   ├── inference_utils.py
│   │   └── viz_utils.py
│   │
│   └── config.py
│
├── data/                         # Dataset
└── requirements.txt
```

## Model Architecture

```
Input (33 frames, 1662 keypoints)
    ↓
MLP Branches → LSTM → Softmax (76 classes)
```

## Usage

### Data Preparation
```bash
python -m src.data.prepare_pipeline
```

### Training
```bash
python -m src.training.pipeline
```

### Inference
```bash
python -m src.inference.realtime --mode webcam
```

## Requirements

```bash
pip install -r requirements.txt
```
