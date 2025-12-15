# Sign Language Action Detection

Python modules converted from Jupyter notebooks for sign language action detection using MediaPipe and LSTM.

## Project Structure

```
sign_language_detection/
├── config.py              # Configuration and paths
├── utils/                 # Utility modules
│   ├── keypoint_extraction.py
│   └── visualization.py
├── data/                  # Data processing
│   └── collect_data.py
├── model/                 # Model architecture
│   └── architecture.py
├── train.py              # Training script
├── inference.py          # Inference script
└── requirements.txt      # Dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Data is automatically configured to use `../data/VSL_Isolated/`

## Usage

### 1. Collect Keypoints from Videos

```bash
cd sign_language_detection
python -m data.collect_data
```

This will process all videos in `../data/VSL_Isolated/` folder and save keypoint sequences.

### 2. Train Model

```bash
python train.py
```

Trains the LSTM model on collected sequences. Model saved to `models/`.

### 3. Run Inference

**Webcam (real-time):**
```bash
python inference.py
```

**Video file:**
```bash
python inference.py --video path/to/video.mp4
```

Press 'q' to quit.

### View Training Progress

```bash
tensorboard --logdir=logs/training
```

## Configuration

Edit `config.py` to modify:
- Data paths
- Model parameters (LSTM units, epochs, etc.)
- MediaPipe settings
- Sequence length

## Files Created

- `models/action_model_best.h5` - Best model (highest validation accuracy)
- `models/action_model_final.h5` - Final model after all epochs
- `models/actions.npy` - Action label mappings
- `data/VSL_Isolated/sequences/` - Extracted keypoint sequences
- `logs/training/` - TensorBoard training logs
