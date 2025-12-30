# Model Architecture

## Hybrid Multi-Stream CNN+LSTM

### Overview
The model combines **specialized feature extraction** for each body part with **shared layers** for cross-part interaction learning.

### Architecture Details

```
Input: (sequence_length, 1662)
    ↓
┌─────────────────────────────────────────┐
│          Split Keypoints                │
├──────────┬──────────┬───────────────────┤
│ Pose     │ Face     │ Hands             │
│ (132)    │ (1404)   │ (126)             │
└────┬─────┴────┬─────┴────┬──────────────┘
     │          │          │
┌────▼────┐ ┌──▼────┐ ┌───▼────┐
│Pose CNN │ │Face   │ │Hand    │
│2 layers │ │CNN    │ │CNN     │
│         │ │4 layers│ │3 layers│
│→ 64 dim │ │→128 dim│ │→ 64 dim│
└────┬────┘ └───┬────┘ └───┬────┘
     │          │           │
     └──────────┼───────────┘
                │
         ┌──────▼──────┐
         │  Concat     │
         │  (256 dim)  │
         └──────┬──────┘
                │
         ┌──────▼──────────┐
         │ Shared Layers   │
         │ (2 layers)      │
         │ Learn cross-part│
         │ interactions    │
         └──────┬──────────┘
                │
         ┌──────▼──────┐
         │ LSTM (128)  │
         │ LSTM (64)   │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │ Dense (128) │
         │ Softmax(76) │
         └─────────────┘
```

### Key Features

**1. Specialized Branches (varying depth)**
- **Pose:** 2 layers (shallow) - 33 keypoints, less complex
- **Face:** 4 layers (deepest) - 468 keypoints, captures subtle expressions
- **Hands:** 3 layers (deep) - most important for sign language

**2. Shared Interaction Layers**
- 2 Dense layers after concatenation
- Learn relationships between hand shapes + facial expressions + body pose
- Critical for understanding complete sign meaning

**3. Temporal Modeling**
- 2 LSTM layers (128 → 64 units)
- Capture movement patterns over time
- Essential for signs with similar shapes but different motions

### Parameters
- Input: (33 frames, 1662 keypoints)
- Output: 76 classes (sign language words)
- Total parameters: ~500K (optimized for 1450 training samples)
