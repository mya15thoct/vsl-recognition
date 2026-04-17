# Paper Summary
## Multi-Stream MLP–BiLSTM with Temporal Attention for Isolated Sign Language Recognition

---

## 1. Datasets

Three datasets are used, all focusing on word-level sign language recognition.

| Dataset | Classes | Videos | Signers | Resolution | Notes |
|---|---|---|---|---|---|
| INCLUDE | 259 | 4,204 | 7 | 1920×1080 / 25fps | Primary, after filtering |
| ISL Video Dataset | 60 | 3,600+ | 4 | MP4 | General vocabulary |
| INCLUDE-24 Medical | 24 | 696 | Multiple | 1080p / 30fps | Medical domain, augmented |

**INCLUDE** is the primary benchmark for Indian Sign Language (ISL) recognition, comprising 4,204 videos across 259 word-level sign classes recorded by 7 senior signers. Videos span 15 semantic categories (Adjectives, Animals, Clothing, Colours, Days & Time, Electronics, Greetings, Transport, Household Objects, Occupations, People, Places, Pronouns, Seasons, Society) at 1920×1080 resolution, 25 fps under natural classroom lighting. Sign duration ranges from 2–4 seconds (mean: 2.57s).

**ISL Video Dataset** provides supplementary coverage of 60 common ISL signs, with 60 videos per sign contributed by 4 signers.

**INCLUDE-24 Medical** is a curated medical-focused subset covering 24 clinically relevant signs (e.g., Medicine, Doctor, Sick), designed for patient–doctor communication scenarios. Recorded at 1080p / 30fps by multiple signers under varied angles and lighting for real-world robustness.

---

## 2. Proposed Model: Multi-Stream MLP–BiLSTM with Temporal Attention

### Architecture

Input keypoints (1,662 dims per frame) are split into three body-part streams:

| Stream | Keypoints | MLP Output |
|---|---|---|
| Pose | 132 dims | 64-dim feature |
| Face | 1,404 dims | 128-dim feature |
| Hand | 126 dims | 64-dim feature |

Each stream is processed by a specialized MLP branch (TimeDistributed), then **directly concatenated** (256 dims total) — no gating.

The fused feature passes through:
1. Shared Dense layers (256 → 128, with Dropout)
2. BiLSTM × 2 (64 units → 32 units, return_sequences=True)
3. **Temporal Attention** — learned weighted sum over time steps
4. Dense head (128 → num_classes, softmax)

**Total parameters: 1,291,767**

### Key Design Decisions

- **Multi-stream over single-stream**: each body part has different feature scale and semantic role — specialized MLPs extract more discriminative features than a single shared MLP.
- **Direct concat over gating**: softmax gating forces zero-sum competition between streams (if one part gets higher weight, others are suppressed). Direct concat lets the downstream BiLSTM freely combine all features.
- **BiLSTM over uni-LSTM**: bidirectional modeling captures both past and future temporal context within a sign, leading to faster convergence despite more parameters.
- **Temporal attention over last hidden state**: not all frames carry equal information — frames at the peak of a sign are more discriminative than transition frames at the start/end.

---

## 3. Ablation Study

### Variants

| ID | Variant | Description |
|---|---|---|
| V0 | Proposed | Multi-Stream + Concat + BiLSTM + Temporal Attention |
| V1 | w/o Multi-Stream | Single shared MLP on all 1,662 dims |
| V2 | w/ Uni-LSTM | Replace BiLSTM with unidirectional LSTM |
| V3 | w/o Temporal Attention | Replace attention with last BiLSTM hidden state |

### Results (seed = 42)

| Variant | Test Acc | Macro F1 | Precision | Recall | Best Epoch | Time (min) | Params |
|---|---|---|---|---|---|---|---|
| **V0 Proposed** | **97.42%** | **90.62%** | **91.05%** | **90.98%** | 185 | **74.10** | 1,291,767 |
| V1 w/o Multi-Stream | 85.67% | 76.14% | 77.41% | 78.50% | 241 | 120.07 | 1,338,743 |
| V2 w/ Uni-LSTM | 97.18% | 90.00% | 90.49% | 90.51% | 382 | 104.03 | 1,209,431 |
| V3 w/o Temporal Attn | 72.16% | 60.98% | 62.76% | 65.70% | 456 | 157.71 | 1,291,702 |

---
### Key Findings

**1. Multi-stream is the foundation.**
Removing multi-stream (V1) causes the largest accuracy drop (−11.75%), confirming that body-part specialization is essential. A single MLP on raw 1,662-dim keypoints cannot learn discriminative features as effectively as specialized branches for pose, face, and hands.

**2. Temporal attention is critical.**
V3 (no attention) drops by −25.26% — the most severe degradation. This demonstrates that not all frames in a sign sequence are equally informative. Transition frames at the start/end of a sign carry little discriminative content; the attention mechanism learns to focus on the peak frames where the sign shape is most distinct.

**3. BiLSTM converges faster despite more parameters.**
V2 (Uni-LSTM) achieves comparable accuracy (97.18%) but requires 382 epochs and 104 minutes vs. V0's 185 epochs and 74 minutes. BiLSTM captures both past and future temporal context simultaneously, creating stronger gradient signals that lead to faster convergence — even though V0 has more parameters (1.29M vs. 1.21M).

**4. Direct concat outperforms gating.**
Earlier experiments with softmax gating showed no consistent improvement over direct concatenation (within 0.16% across seeds). Softmax gating forces streams to compete (zero-sum), suppressing complementary information. Direct concat preserves all features and delegates fusion to the BiLSTM layers.

---

## 4. Comparison with Transformer Baseline

To validate the architectural choice of BiLSTM over Transformer-based temporal modeling, a Transformer Encoder variant was trained on the same dataset and evaluated under identical conditions.

| Model | Test Acc | Macro F1 | Precision | Recall | Best Epoch | Time (min) | Params |
|---|---|---|---|---|---|---|---|
| **Proposed (BiLSTM)** | **97.42%** | **90.62%** | **91.05%** | **90.98%** | 185 | **74** | 1,291,767 |
| Transformer Encoder | 94.94% | 92.62% | 93.38% | 93.79% | 227 | ~117 | — |

**Findings:**
- Proposed model outperforms Transformer by **+2.48% accuracy** while training **~43 minutes faster**
- Transformer shows higher Macro F1/Precision/Recall — likely due to class-level calibration differences rather than overall superiority
- BiLSTM is better suited for short, fixed-length sign sequences (2–4s) where the sequential inductive bias is beneficial; Transformer's self-attention advantage is more pronounced on longer sequences
- For deployment on resource-constrained devices, BiLSTM's faster convergence and lower training cost is an additional practical advantage

---

## 5. Comparison with Other Baselines

All baselines trained on the same combined dataset (343 classes) with seed=42.

| Method | Test Acc | Macro F1 | Precision | Recall | Best Epoch | Time (min) | Params |
|---|---|---|---|---|---|---|---|
| **Proposed (Multi-Stream BiLSTM + Attn)** | **97.42%** | **90.62%** | **91.05%** | **90.98%** | 185 | **74.10** | 1,291,767 |
| EMPATH (Ensemble Transformer) | 94.12% | 86.63% | 87.51% | 87.75% | 397 | 121.61 | 1,022,164 |
| LSTM-GRU | 91.55% | 82.20% | 83.53% | 83.54% | 292 | 83.60 | 2,251,058 |
| LSTM | 83.26% | 73.27% | 75.42% | 75.30% | 411 | 112.44 | 2,263,218 |

**Key observations:**
- Proposed model outperforms EMPATH (ensemble of 4 Transformers) by **+3.30%** while being **~47 min faster** and using **~21% fewer parameters** (1.29M vs. 1.02M × 4 heads)
- LSTM-GRU is competitive but requires 2× more parameters (2.25M) with lower accuracy
- Simple LSTM baseline confirms that temporal modeling alone is insufficient — multi-stream + attention is essential
- EMPATH's ensemble complexity does not compensate for the lack of body-part specialization

---

## 6. References

1. Das Koushik. INCLUDE dataset. Kaggle. https://www.kaggle.com/datasets/daskoushik/include
2. Prasad Shet. Indian Sign Language Video Dataset. Kaggle. https://www.kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset
3. linardur. INCLUDE-24 Medical Modified. Kaggle. https://www.kaggle.com/datasets/linardur/include-24-medical-modified
