# ISL Video Transformer System

A comprehensive system for Indian Sign Language (ISL) recognition using Video Vision Transformers and LSTM networks.

## Features

- **Video Vision Transformer**: Uses VideoMAEv2-Huge for robust feature extraction from video sequences
- **LSTM Classifier**: Analyzes temporal sequences for accurate sign recognition
- **Feature Visualization**: Comprehensive analysis of extracted features with threshold analysis
- **Real-time Detection**: Live camera feed processing for real-time ISL recognition
- **Batch Processing**: Efficient training and evaluation on large datasets

## System Architecture

```
Video Input → VideoMAEv2-Huge (Feature Extraction) → LSTM (Sequence Analysis) → Classification
```

1. **Video Preprocessing**: Resizes videos to 224x224, extracts 16 frames per video
2. **Feature Extraction**: VideoMAEv2-Huge backbone extracts high-dimensional features
3. **Sequence Analysis**: Bidirectional LSTM processes temporal features
4. **Classification**: Final layer outputs predictions for ISL signs

## Dataset Structure

The system uses the following dataset structure (BASIC WORDS ONLY):
```
isl(diksha)/
├── dataset.csv                    # Video paths and labels
└── isl sign/
    └── basic_words/              # Common ISL words (USED FOR TRAINING)
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

**Note**: The system automatically filters to use only basic words, excluding alphabets (A-Z) and other categories.

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure your dataset is in the correct structure (see above)

## Usage

### 1. Training the Model

```bash
python train_isl_model.py
```

This will:
- Load and preprocess your ISL dataset
- Train the Video Transformer + LSTM model
- Generate feature visualizations and threshold analysis
- Save the trained model as `isl_video_transformer_model.pth`

### 2. Real-time Detection

```bash
# Use webcam
python realtime_isl_detector.py

# Use specific camera
python realtime_isl_detector.py --camera 1

# Save output video
python realtime_isl_detector.py --save --output my_detection.mp4
```

### 3. Process Video File

```bash
python realtime_isl_detector.py --video path/to/your/video.mp4 --save
```

## Model Configuration

### Key Parameters

- **Max Frames**: 16 frames per video (adjustable)
- **Frame Size**: 224x224 pixels
- **Batch Size**: 4 (adjust based on GPU memory)
- **Learning Rate**: 1e-4
- **Epochs**: 10 (adjustable)

### Architecture Details

- **VideoMAEv2-Huge Backbone**: Pre-trained on large video datasets with dual masking strategy
- **Feature Dimension**: 1024 (VideoMAEv2-Huge hidden size)
- **LSTM Hidden Size**: 512
- **LSTM Layers**: 2 (bidirectional)
- **Dropout**: 0.3

## Feature Analysis

The system provides comprehensive feature analysis:

1. **PCA Visualization**: 2D projection of high-dimensional features
2. **t-SNE Visualization**: Non-linear dimensionality reduction
3. **Feature Magnitude Distribution**: Analysis of feature strength
4. **Threshold Analysis**: Identifies optimal feature thresholds

## Performance Metrics

The system tracks:
- Training/Validation Loss
- Classification Accuracy
- Confusion Matrix
- Per-class Precision/Recall/F1-Score

## Real-time Features

- **Temporal Smoothing**: Reduces prediction jitter
- **Confidence Thresholding**: Filters low-confidence predictions
- **Frame Buffer Management**: Maintains temporal context
- **FPS Monitoring**: Real-time performance tracking

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or frame count
2. **Slow Training**: Ensure CUDA is available and properly configured
3. **Poor Accuracy**: Increase training epochs or adjust learning rate
4. **Real-time Lag**: Reduce frame buffer size or use smaller model

### Performance Tips

- Use GPU for training (10x faster than CPU)
- Adjust batch size based on available memory
- Use smaller frame counts for faster real-time processing
- Enable mixed precision training for better performance

## File Structure

```
code/
├── video_transformer_isl.py      # Main model and training code
├── realtime_isl_detector.py      # Real-time detection system
├── train_isl_model.py           # Simplified training script
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Model Outputs

After training, you'll get:
- `isl_video_transformer_model.pth`: Trained model weights
- `feature_distribution.png`: Feature visualization plots
- Training curves and evaluation metrics
- Confusion matrix and classification report

## Next Steps

1. **Train the model** on your ISL dataset
2. **Evaluate performance** using the provided metrics
3. **Fine-tune parameters** based on your specific needs
4. **Deploy real-time detection** for live ISL recognition

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify your dataset structure
3. Ensure all dependencies are installed
4. Check GPU memory and CUDA availability
