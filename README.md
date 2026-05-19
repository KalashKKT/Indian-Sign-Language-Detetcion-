#  ISL Recognition using Video Vision Transformers + LSTM

A comprehensive deep learning system for **Indian Sign Language (ISL) recognition** using **Video Vision Transformers (VideoMAEv2-Huge)** and **Bidirectional LSTM networks**.

The system extracts robust spatio-temporal video features using a pretrained transformer backbone and models temporal sign dynamics with LSTM layers for accurate ISL classification.

Supports:

- Training on ISL datasets
- Real-time webcam inference
- Video file processing
- Feature visualization
- Temporal smoothing
- Performance analysis

---

##  System Overview

Architecture:

```text
Video Input
      ↓
Video Preprocessing
(Resize + Frame Extraction)
      ↓
VideoMAEv2-Huge
(Feature Extraction)
      ↓
Bidirectional LSTM
(Temporal Sequence Learning)
      ↓
Classification Layer
      ↓
Predicted ISL Sign
```

---

#  Features

###  Video Vision Transformer

- Uses **VideoMAEv2-Huge**
- Robust extraction of spatial and temporal representations
- Pretrained on large-scale video datasets

###  LSTM Sequence Modeling

- Bidirectional LSTM captures temporal dependencies
- Learns sign motion progression effectively

###  Feature Visualization

Includes:

- PCA projection
- t-SNE visualization
- Feature magnitude analysis
- Threshold-based feature inspection

###  Real-time Detection

- Webcam-based live inference
- Temporal prediction smoothing
- Confidence filtering
- FPS monitoring

###  Batch Processing

- Efficient loading and training on large datasets
- GPU accelerated pipeline

---

#  Model Architecture

### Video Preprocessing

- Resize frames to **224×224**
- Extract **16 frames per video**
- Normalize and prepare temporal sequence

### Feature Extraction

Backbone:

**VideoMAEv2-Huge**

Extracted feature dimension:

```text
1024
```

### Sequence Analysis

Bidirectional LSTM:

- Hidden size: 512
- Layers: 2
- Dropout: 0.3

### Classification

Final dense layer outputs:

```text
ISL Sign Class Prediction
```

---

#  Dataset Structure

Dataset organization:

```text
isl/
│
├── dataset.csv
│
└── isl sign/
      └── basic_words/

            ├── HELLO/
            ├── GOOD/
            ├── BYE/
            ├── HOUSE/
            ├── MORNING/
            ├── NICE/
            ├── THANK_YOU/
            ├── WELCOME/
            ├── WORK/
            ├── YES/

            └── SMALL_DATASET/

                  ├── ATTENTION/
                  ├── DO_NOT/
                  ├── LATER/
                  ├── MORE/
                  ├── NO/
                  ├── NOW/
                  ├── PLEASE/
                  ├── REGRET/
                  ├── TOMORROW/
                  ├── WHAT/
                  ├── WHERE/
                  ├── WHO/
                  └── YESTERDAY/
```

### Dataset Filtering

The system automatically filters and uses:

 Basic ISL words

Excludes:

 Alphabet classes (A–Z)

 Other unsupported categories

---

#  Installation

Clone repository:

```bash
git clone https://github.com/KalashKKT/ISL_Recognition.git

cd ISL_Recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure dataset follows the required structure.

---

#  Usage

## 1. Train Model

```bash
python train_isl_model.py
```

Training automatically:

- Loads dataset
- Extracts video features
- Trains VideoMAE + LSTM
- Generates feature visualizations
- Saves trained weights

Output:

```text
isl_video_transformer_model.pth
```

---

## 2. Real-time Webcam Detection

Default webcam:

```bash
python realtime_isl_detector.py
```

Specific camera:

```bash
python realtime_isl_detector.py --camera 1
```

Save inference output:

```bash
python realtime_isl_detector.py --save --output output.mp4
```

---

## 3. Process Existing Video

```bash
python realtime_isl_detector.py --video path/to/video.mp4
```

Save output:

```bash
python realtime_isl_detector.py --video video.mp4 --save
```

---

#  Configuration

Key hyperparameters:

| Parameter | Value |
|-----------|--------|
| Max Frames | 16 |
| Frame Size | 224×224 |
| Batch Size | 4 |
| Learning Rate | 1e−4 |
| Epochs | 10 |
| Feature Size | 1024 |
| LSTM Hidden Size | 512 |
| LSTM Layers | 2 |
| Dropout | 0.3 |

---

#  Feature Analysis

The system generates:

### PCA Visualization

Projects high-dimensional feature vectors into 2D space.

### t-SNE Visualization

Captures nonlinear feature structures.

### Feature Magnitude Distribution

Analyzes representation strength.

### Threshold Analysis

Helps identify informative feature regions.

Generated output:

```text
feature_distribution.png
```

---

#  Evaluation Metrics

Training and validation include:

- Loss curves
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class metrics

---

#  Real-Time Enhancements

Implemented optimizations:

### Temporal Smoothing

Reduces prediction fluctuations.

### Confidence Thresholding

Suppresses low-confidence predictions.

### Frame Buffer Management

Maintains temporal context.

### FPS Tracking

Monitors inference speed.

---

#  Project Structure

```text
code/

├── video_transformer_isl.py
├── realtime_isl_detector.py
├── train_isl_model.py
├── requirements.txt
└── README.md
```

### File Description

**video_transformer_isl.py**

Main architecture and training pipeline

**realtime_isl_detector.py**

Live inference and video processing

**train_isl_model.py**

Simplified training script

---

#  Outputs

After training:

```text
isl_video_transformer_model.pth
feature_distribution.png
confusion_matrix.png
classification_report.txt
training_curves.png
```

---

#  Performance Tips

Use GPU for training

~10× faster than CPU

Enable mixed precision

Improves memory efficiency

Reduce frame count for real-time speed

Tune batch size based on available VRAM

---

#  Troubleshooting

### CUDA Out of Memory

Reduce:

- Batch size
- Frame count

### Slow Training

Verify CUDA availability:

```python
torch.cuda.is_available()
```

### Poor Accuracy

Try:

- Increasing epochs
- Adjusting learning rate
- More data augmentation

### Real-time Lag

Reduce:

- Buffer size
- Frame count

---

#  Future Improvements

- Expand vocabulary
- Sentence-level ISL recognition
- Transformer-based temporal modeling
- Mobile deployment
- Quantization for edge devices
- Multi-person sign recognition

---

#  License

This project is released under the MIT License.

---

#  Acknowledgements

- VideoMAEv2
- PyTorch
- OpenCV
- HuggingFace
- Indian Sign Language datasets

---

If you use this work, consider giving the repository a STAR
