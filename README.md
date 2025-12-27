# Vietnamese Sign Language Detection

Sign language action detection for Vietnamese Sign Language (VSL) using MediaPipe and LSTM.

## Project Structure

```
vsl-recognition/
├── preprocessing/               # MediaPipe keypoint extraction
│   ├── scripts/                 # Validation scripts
│   │   └── validate_extraction.py
│   ├── extraction/              # Data extraction module
│   │   └── collect_data.py      # Extract keypoints from videos
│   ├── utils/                   # Utility functions (MediaPipe)
│   │   ├── keypoint_extraction.py
│   │   └── visualization.py
│   └── config.py                # Configuration
│
└── data/                        # Dataset (gitignored)
    ├── prepare_dataset.py       # Dataset preparation from QIPEDC
    └── VSL_Isolated/            # Videos, frames, sequences
```

## Installation

### Option 1: Conda (Recommended for Linux)

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate vsl_preprocessing
```

Or use the setup script:
```bash
chmod +x setup_linux.sh
./setup_linux.sh
```

### Option 2: Pip

```bash
cd preprocessing
pip install -r requirements.txt
```

## Usage

### 1. Validate Extraction Quality

Check MediaPipe extraction quality:

```bash
cd preprocessing
python -m scripts.validate_extraction
```

### 2. Prepare Dataset (Optional)

If you don't have a dataset yet:

```bash
cd data
python prepare_dataset.py
```

### 3. Extract Keypoints

```bash
cd preprocessing
python -m scripts.extract_keypoints
```

This processes all videos in `../data/VSL_Isolated/` and saves keypoint sequences.

### 4. Train Model

```bash
cd preprocessing/training
python run.py
```

This will:
- Train the CNN+LSTM model
- Save best model to `checkpoints/best_model.h5`
- Generate training logs for TensorBoard

### 5. Evaluate Model

```bash
cd preprocessing/training
python evaluate.py
```

This will:
- Test the model on test set
- Generate classification report
- Save confusion matrix to `checkpoints/confusion_matrix.png`

### 6. Run Inference (Real-time Predictions)

**Using webcam:**
```bash
cd preprocessing
python -m scripts.inference --mode webcam
```

**Using video file:**
```bash
python -m scripts.inference --mode video --video path/to/video.mp4
```

**Save output video:**
```bash
python -m scripts.inference --mode video --video input.mp4 --output result.mp4
```

Press `q` to quit.

## Technology Stack

- **MediaPipe Holistic**: Extract keypoints (pose, face, hands)
- **LSTM**: Sequential action classification model
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Video and image processing

## Dataset

Dataset is crawled from [QIPEDC](https://qipedc.moet.gov.vn) - Vietnamese Sign Language Dictionary.

## Additional Information

- For detailed module documentation: [sign_language_detection/README.md](sign_language_detection/README.md)
- Dataset preparation script: [data/prepare_dataset.py](data/prepare_dataset.py)

## License

Research and educational purposes.
