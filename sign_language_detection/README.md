# MediaPipe Keypoint Extraction Project

Focused on **MediaPipe extraction quality validation** only.

## Project Structure (Extraction Only)

```
sign_language_detection/
├── config.py                      # Configuration
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
├── scripts/                       # Validation & utility scripts ⭐
│   ├── __init__.py
│   └── validate_extraction.py    # MediaPipe quality validation
├── data/                          # Data processing
│   ├── __init__.py
│   └── collect_data.py            # Extract keypoints with MediaPipe
└── utils/                         # Utility functions
    ├── __init__.py
    ├── keypoint_extraction.py     # MediaPipe utilities
    └── visualization.py           # Visualization functions
```

## What Was Removed

The following LSTM-related files were removed to focus on extraction:
- ❌ `model/` - LSTM architecture folder
- ❌ `models/` - Trained models folder
- ❌ `logs/` - TensorBoard logs
- ❌ `train.py` - LSTM training script
- ❌ `evaluate.py` - LSTM evaluation script
- ❌ `inference.py` - LSTM inference script

## Usage

### 1. Validate MediaPipe Extraction Quality

```bash
cd sign_language_detection
python -m scripts.validate_extraction
```

**This will:**
- Check 10 random videos from dataset
- Show keypoints overlay for first video
- Calculate detection rate, confidence, consistency
- Provide quality assessment

**Metrics checked:**
- ✅ Detection Rate (>90% is good)
- ✅ Confidence Score (>0.7 is good)
- ✅ Consistency (>0.6 is good)

### 2. Collect Keypoints (After Validation)

```bash
python -m data.collect_data
```

**This will:**
- Process all videos in `../data/VSL_Isolated/`
- Extract 1662 keypoints per frame using MediaPipe
- Save sequences to `../data/VSL_Isolated/sequences/`

## MediaPipe Configuration

Edit `config.py`:

```python
# MediaPipe settings
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# Data settings
SEQUENCE_LENGTH = 30      # Frames per sequence
NO_SEQUENCES = 30         # Sequences per action
```

## Workflow

```
1. Validate extraction quality
   python -m scripts.validate_extraction
   
2. If quality ≥80% good:
   python -m data.collect_data
   
3. Check output:
   data/VSL_Isolated/sequences/
```

## Next Steps (Future)

After ensuring good extraction quality:
- Add LSTM model for classification
- Train on extracted keypoints
- Evaluate accuracy
- Deploy for inference

## Notes

- Focus: **MediaPipe extraction quality only**
- No LSTM training in this phase
- Ensure extraction is >80% good before proceeding
