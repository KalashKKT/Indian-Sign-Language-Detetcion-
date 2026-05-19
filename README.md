ISL_Recognition
A comprehensive system for Indian Sign Language (ISL) recognition using Video Vision Transformers and LSTM networks.

Features
Video Vision Transformer: Uses VideoMAEv2-Huge for robust feature extraction from video sequences
LSTM Classifier: Analyzes temporal sequences for accurate sign recognition
Feature Visualization: Comprehensive analysis of extracted features with threshold analysis
Real-time Detection: Live camera feed processing for real-time ISL recognition
Batch Processing: Efficient training and evaluation on large datasets
System Architecture
Video Input → VideoMAEv2-Huge (Feature Extraction) → LSTM (Sequence Analysis) → Classification
Video Preprocessing: Resizes videos to 224x224, extracts 16 frames per video
Feature Extraction: VideoMAEv2-Huge backbone extracts high-dimensional features
Sequence Analysis: Bidirectional LSTM processes temporal features
Classification: Final layer outputs predictions for ISL signs
Dataset Structure
The system uses the following dataset structure (BASIC WORDS ONLY):

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
Note: The system automatically filters to use only basic words, excluding alphabets (A-Z) and other categories.

Installation
Install required packages:
pip install -r requirements.txt
Ensure your dataset is in the correct structure (see above)
Usage
1. Training the Model
python train_isl_model.py
This will:

Load and preprocess your ISL dataset
Train the Video Transformer + LSTM model
Generate feature visualizations and threshold analysis
Save the trained model as isl_video_transformer_model.pth
2. Real-time Detection
# Use webcam
python realtime_isl_detector.py

# Use specific camera
python realtime_isl_detector.py --camera 1

# Save output video
python realtime_isl_detector.py --save --output my_detection.mp4
3. Process Video File
python realtime_isl_detector.py --video path/to/your/video.mp4 --save
Model Configuration
Key Parameters
Max Frames: 16 frames per video (adjustable)
Frame Size: 224x224 pixels
Batch Size: 4 (adjust based on GPU memory)
Learning Rate: 1e-4
Epochs: 10 (adjustable)
Architecture Details
VideoMAEv2-Huge Backbone: Pre-trained on large video datasets with dual masking strategy
Feature Dimension: 1024 (VideoMAEv2-Huge hidden size)
LSTM Hidden Size: 512
LSTM Layers: 2 (bidirectional)
Dropout: 0.3
Feature Analysis
The system provides comprehensive feature analysis:

PCA Visualization: 2D projection of high-dimensional features
t-SNE Visualization: Non-linear dimensionality reduction
Feature Magnitude Distribution: Analysis of feature strength
Threshold Analysis: Identifies optimal feature thresholds
Performance Metrics
The system tracks:

Training/Validation Loss
Classification Accuracy
Confusion Matrix
Per-class Precision/Recall/F1-Score
Real-time Features
Temporal Smoothing: Reduces prediction jitter
Confidence Thresholding: Filters low-confidence predictions
Frame Buffer Management: Maintains temporal context
FPS Monitoring: Real-time performance tracking
Troubleshooting
Common Issues
CUDA Out of Memory: Reduce batch size or frame count
Slow Training: Ensure CUDA is available and properly configured
Poor Accuracy: Increase training epochs or adjust learning rate
Real-time Lag: Reduce frame buffer size or use smaller model
Performance Tips
Use GPU for training (10x faster than CPU)
Adjust batch size based on available memory
Use smaller frame counts for faster real-time processing
Enable mixed precision training for better performance
File Structure
code/
├── video_transformer_isl.py      # Main model and training code
├── realtime_isl_detector.py      # Real-time detection system
├── train_isl_model.py           # Simplified training script
├── requirements.txt             # Python dependencies
└── README.md                   # This file
Model Outputs
After training, you'll get:

isl_video_transformer_model.pth: Trained model weights
feature_distribution.png: Feature visualization plots
Training curves and evaluation metrics
Confusion matrix and classification report
Next Steps
Train the model on your ISL dataset
Evaluate performance using the provided metrics
Fine-tune parameters based on your specific needs
Deploy real-time detection for live ISL recognition
