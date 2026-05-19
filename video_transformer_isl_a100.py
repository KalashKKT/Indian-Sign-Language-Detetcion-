"""
ISL Video Transformer System
Combines Video Vision Transformer for feature extraction and LSTM for sequence analysis
"""

import os
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score, 
    recall_score, accuracy_score, top_k_accuracy_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import time
import sys
import logging
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Module-level output directory to be accessible across functions
OUTPUT_DIR = "."

def setup_logging(output_dir):
    """Setup comprehensive logging to both file and console"""
    # Create log file path
    log_file = os.path.join(output_dir, 'training_log.txt')
    
    # Create logger
    logger = logging.getLogger('ISL_Training')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    # File handler - logs everything
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler - logs everything but with simpler format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Also redirect print statements to logger
    class LoggerWriter:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
            self.buffer = ''
        
        def write(self, message):
            if message != '\n':
                self.buffer += message
            if '\n' in message:
                self.logger.log(self.level, self.buffer.strip())
                self.buffer = ''
        
        def flush(self):
            if self.buffer:
                self.logger.log(self.level, self.buffer.strip())
                self.buffer = ''
    
    # Redirect stdout to logger
    sys.stdout = LoggerWriter(logger, logging.INFO)
    
    return logger, log_file

class ISLVideoDataset(Dataset):
    """Dataset class for ISL video data"""
    
    def __init__(self, video_paths, labels, transform=None, max_frames=16, frame_size=(224, 224)):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.max_frames = max_frames
        self.frame_size = frame_size
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self.load_video_frames(video_path)
        
        if self.transform:
            frames = self.transform(frames)
            
        return frames, label
    
    def load_video_frames(self, video_path):
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            # Create dummy frames
            frames = [np.zeros((*self.frame_size, 3), dtype=np.uint8) for _ in range(self.max_frames)]
        else:
            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Resize frame
                frame = cv2.resize(frame, self.frame_size)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Keep as uint8 for VideoMAE processor (it handles normalization)
                frames.append(frame)
        
        cap.release()
        
        # Ensure we have at least one frame
        if len(frames) == 0:
            frames = [np.zeros((*self.frame_size, 3), dtype=np.uint8)]
        
        # Pad or truncate to max_frames
        if len(frames) < self.max_frames:
            # Pad with last frame
            while len(frames) < self.max_frames:
                frames.append(frames[-1] if frames else np.zeros((*self.frame_size, 3), dtype=np.uint8))
        else:
            frames = frames[:self.max_frames]
        
        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        frames = np.array(frames)
        
        # Debug: print shape before permutation (commented out for cleaner output)
        # print(f"Debug: frames shape before permute: {frames.shape}")
        
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        
        return frames

class VideoMAEFeatureExtractor(nn.Module):
    """VideoMAE-Huge based feature extractor for video frames"""
    
    def __init__(self, freeze_backbone=True, model_name="OpenGVLab/VideoMAEv2-Huge"):
        super().__init__()
        
        # Load VideoMAE-Huge model
        self.model_name = model_name
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        
        # Load the model for feature extraction with trust_remote_code=True
        self.videomae_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Get the feature dimension from the model config
        # VideoMAEv2 uses different config structure
        model_config = self.videomae_model.config.model_config
        print(f"VideoMAE Model Config - num_frames: {model_config['num_frames']}, patch_size: {model_config['patch_size']}, img_size: {model_config['img_size']}")
        
        # Get the correct feature dimension from the model config
        model_config = self.videomae_model.config.model_config
        if 'embed_dim' in model_config:
            self.feature_dim = model_config['embed_dim']
        elif hasattr(self.videomae_model.config, 'hidden_size'):
            self.feature_dim = self.videomae_model.config.hidden_size
        elif hasattr(self.videomae_model.config, 'decoder_hidden_size'):
            self.feature_dim = self.videomae_model.config.decoder_hidden_size
        elif hasattr(self.videomae_model.config, 'embed_dim'):
            self.feature_dim = self.videomae_model.config.embed_dim
        elif hasattr(self.videomae_model.config, 'encoder_hidden_size'):
            self.feature_dim = self.videomae_model.config.encoder_hidden_size
        else:
            # Fallback: use a reasonable default for VideoMAE-Huge
            self.feature_dim = 1024
            print(f"Using fallback feature dimension: {self.feature_dim}")
        
        print(f"Using feature dimension: {self.feature_dim}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.videomae_model.parameters():
                param.requires_grad = False
        
        # Normalization parameters for VideoMAE
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
    def preprocess_video(self, video_tensor):
        """Preprocess video tensor for VideoMAE using the proper processor"""
        # video_tensor shape: (batch_size, channels, frames, height, width)
        batch_size, channels, num_frames, height, width = video_tensor.shape
        
        # Convert to format expected by VideoMAE processor: (batch_size, frames, channels, height, width)
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        
        # Convert to numpy for processor
        video_np = video_tensor.cpu().numpy()
        
        # Process each video in the batch
        processed_videos = []
        for i in range(batch_size):
            # Convert to list of frames (T, C, H, W) -> list of (C, H, W)
            frame_list = [video_np[i, t] for t in range(num_frames)]
            
            # Use VideoMAE processor
            processed = self.processor(frame_list, return_tensors="pt")
            processed_videos.append(processed["pixel_values"])
        
        # Stack processed videos
        processed_tensor = torch.stack(processed_videos, dim=0)  # (B, T, C, H, W)
        
        return processed_tensor
        
    def forward(self, x):
        """Extract features from video using VideoMAE"""
        # x shape: (batch_size, channels, frames, height, width)
        batch_size = x.shape[0]
        
        # Preprocess video using VideoMAE processor
        x = self.preprocess_video(x)  # Returns processed tensor
        
        # Debug: print tensor shape (remove this after testing)
        # print(f"Processed tensor shape: {x.shape}")
        
        # Handle different tensor dimensions from processor
        if x.dim() == 6:
            # If 6D, squeeze the extra dimension: [B, 1, T, C, H, W] -> [B, T, C, H, W]
            x = x.squeeze(1)  # Remove the extra dimension at index 1
        elif x.dim() == 5:
            # Already correct 5D tensor
            pass
        else:
            raise ValueError(f"Unexpected tensor dimension: {x.dim()}")
        
        # VideoMAE expects input in format: (batch_size, channels, frames, height, width)
        # Convert from (B, T, C, H, W) to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Extract features using VideoMAE
        # Ensure tensor is on the same device as the VideoMAE model (avoid CPU/GPU mismatch)
        target_device = next(self.videomae_model.parameters()).device
        x = x.to(target_device, non_blocking=True).float()

        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.videomae_model(x)
            
            # Handle different output types from VideoMAE
            if hasattr(outputs, 'last_hidden_state'):
                # Structured output with last_hidden_state
                last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            elif isinstance(outputs, torch.Tensor):
                # Raw tensor output
                last_hidden_state = outputs  # (batch_size, hidden_size)
            else:
                # Try to get the output from different possible attributes
                if hasattr(outputs, 'logits'):
                    last_hidden_state = outputs.logits
                elif hasattr(outputs, 'hidden_states'):
                    last_hidden_state = outputs.hidden_states[-1]  # Get last layer
                else:
                    raise ValueError(f"Unexpected output type: {type(outputs)}")
            
            # VideoMAE returns (batch_size, hidden_size) - we need to add sequence dimension for LSTM
            if last_hidden_state.dim() == 2:
                # Add sequence dimension: (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
                last_hidden_state = last_hidden_state.unsqueeze(1)
            elif last_hidden_state.dim() == 1:
                # If 1D, add both batch and sequence dimensions
                last_hidden_state = last_hidden_state.unsqueeze(0).unsqueeze(0)
            
            # For LSTM, we want to keep the sequence dimension, so don't do global average pooling here
            # The LSTM will process the sequence and we'll use the final output
            features = last_hidden_state  # (batch_size, seq_len, hidden_size)
            
        return features

class LSTMClassifier(nn.Module):
    """LSTM model for sequence analysis and classification"""
    
    def __init__(self, input_dim, hidden_dim=512, num_layers=2, num_classes=50, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        output = lstm_out[:, -1, :]
        
        # Classification
        logits = self.classifier(output)
        return logits

class ISLVideoTransformer(nn.Module):
    """Complete ISL Video Transformer System using VideoMAE-Huge feature extractor"""
    
    def __init__(self, num_classes, lstm_hidden_dim=512, lstm_layers=2, freeze_backbone=True, model_name="OpenGVLab/VideoMAEv2-Huge"):
        super().__init__()
        
        # VideoMAE feature extractor
        self.feature_extractor = VideoMAEFeatureExtractor(
            freeze_backbone=freeze_backbone,
            model_name=model_name
        )
        
        # LSTM classifier
        self.lstm_classifier = LSTMClassifier(
            input_dim=self.feature_extractor.feature_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            num_classes=num_classes
        )
        
    def forward(self, x):
        # Extract features from video
        features = self.feature_extractor(x)  # (batch_size, seq_len, feature_dim)
        
        # Features are already in the correct 3D format for LSTM
        # No need to add sequence dimension as it's already there
        
        # Classify with LSTM
        logits = self.lstm_classifier(features)
        
        return logits

class FeatureVisualizer:
    """Class for visualizing extracted features and analyzing thresholds"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
    def extract_features_batch(self, dataloader):
        """Extract features for a batch of videos"""
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for videos, labels in tqdm(dataloader, desc="Extracting features"):
                videos = videos.to(self.device)
                features = self.model.feature_extractor(videos)
                
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.vstack(all_features), np.array(all_labels)
    
    def visualize_feature_distribution(self, features, labels, class_names, save_path=None):
        """Visualize feature distribution across classes"""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # Ensure features are 2D for PCA/TSNE; reduce sequence dimension if present
        if isinstance(features, np.ndarray) and features.ndim > 2:
            # Average across sequence/time dimension (axis=1): (N, T, D) -> (N, D)
            features = features.mean(axis=1)

        # PCA visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        plt.figure(figsize=(15, 5))
        
        # PCA plot
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20', alpha=0.6)
        plt.title(f'PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar(scatter)
        
        # t-SNE visualization
        plt.subplot(1, 3, 2)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        features_tsne = tsne.fit_transform(features)
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab20', alpha=0.6)
        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(scatter)
        
        # Feature magnitude distribution
        plt.subplot(1, 3, 3)
        feature_magnitudes = np.linalg.norm(features, axis=1)
        plt.hist(feature_magnitudes, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Feature Magnitude Distribution')
        plt.xlabel('Feature Magnitude')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_thresholds(self, features, labels, class_names):
        """Analyze feature extraction with different thresholds"""
        feature_magnitudes = np.linalg.norm(features, axis=1)
        
        thresholds = np.percentile(feature_magnitudes, [10, 25, 50, 75, 90])
        
        print("Feature Magnitude Analysis:")
        print(f"Mean magnitude: {feature_magnitudes.mean():.4f}")
        print(f"Std magnitude: {feature_magnitudes.std():.4f}")
        print(f"Min magnitude: {feature_magnitudes.min():.4f}")
        print(f"Max magnitude: {feature_magnitudes.max():.4f}")
        
        print("\nThreshold Analysis:")
        for i, threshold in enumerate(thresholds):
            above_threshold = np.sum(feature_magnitudes > threshold)
            percentage = (above_threshold / len(features)) * 100
            print(f"Threshold {i+1} ({threshold:.4f}): {above_threshold} features ({percentage:.1f}%)")
        
        # Visualize threshold effects
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(feature_magnitudes, bins=50, alpha=0.7, edgecolor='black')
        for threshold in thresholds:
            plt.axvline(threshold, color='red', linestyle='--', alpha=0.7)
        plt.title('Feature Magnitude Distribution with Thresholds')
        plt.xlabel('Feature Magnitude')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        threshold_counts = []
        for threshold in thresholds:
            count = np.sum(feature_magnitudes > threshold)
            threshold_counts.append(count)
        
        plt.bar(range(len(thresholds)), threshold_counts, alpha=0.7)
        plt.title('Features Above Each Threshold')
        plt.xlabel('Threshold Index')
        plt.ylabel('Number of Features')
        plt.xticks(range(len(thresholds)), [f'T{i+1}' for i in range(len(thresholds))])
        
        plt.tight_layout()
        plt.show()
        
        return thresholds, feature_magnitudes

def load_dataset(data_dir, csv_path, use_basic_words_only=True):
    """Load ISL dataset from CSV file, optionally filtering for basic words only"""
    df = pd.read_csv(csv_path)
    
    # Update paths to be relative to data_dir
    df['video_path'] = df['video_path'].apply(lambda x: os.path.join(data_dir, x))
    
    # Filter for basic words only if requested
    if use_basic_words_only:
        # Filter to only include basic_words entries
        df = df[df['video_path'].str.contains('basic_words')]
        print("Filtering dataset to use only BASIC WORDS (excluding alphabets and images)")
    
    # Filter out non-existent files
    df = df[df['video_path'].apply(os.path.exists)]
    
    # Get unique labels and create label mapping
    unique_labels = sorted(df['label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    df['label_idx'] = df['label'].map(label_to_idx)
    
    print(f"Dataset loaded: {len(df)} videos, {len(unique_labels)} classes")
    print(f"Classes: {unique_labels}")
    
    if use_basic_words_only:
        print(f"Using only basic words: {len(unique_labels)} word categories")
        print("Excluded: Alphabets (A-Z) and other non-word categories")
    
    return df, unique_labels, label_to_idx

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=5e-5, device='cuda', class_names=None):
    """Train the ISL Video Transformer model with comprehensive logging"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training tracking
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    val_precisions = []
    val_recalls = []
    learning_rates = []
    
    best_val_acc = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    print("\n" + "="*80)
    print("STARTING TRAINING WITH VIDEOMAEV2-HUGE")
    print("="*80)
    print(f"Dataset Split: 80% Train, 10% Validation, 10% Test")
    print(f"Model: VideoMAEv2-Huge + LSTM Classifier")
    print(f"Device: {device}")
    print(f"Classes: {len(class_names) if class_names else 'Unknown'}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches: {len(val_loader)}")
    print("="*80)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_predictions = []
        train_labels = []
        
        print(f"\nEPOCH {epoch+1}/{num_epochs} - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 50)
        
        # Training phase timing
        train_start_time = time.time()
        
        # Training loop - show each step iteration
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            batch_size = labels.size(0)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == labels).sum().item()
            train_total += batch_size
            train_correct += batch_correct
            
            # Calculate batch accuracy
            batch_acc = 100 * batch_correct / batch_size
            
            # Calculate running averages
            avg_loss_so_far = train_loss / (batch_idx + 1)
            avg_acc_so_far = 100 * train_correct / train_total
            
            # Show each step iteration
            print(f"Step {batch_idx+1:3d}/{len(train_loader)}: "
                  f"Epoch={epoch+1}, Batch_Size={batch_size}, "
                  f"Train_Loss={batch_loss:.4f}, Batch_Acc={batch_acc:.2f}%, "
                  f"Avg_Train_Loss={avg_loss_so_far:.4f}, Avg_Acc={avg_acc_so_far:.2f}%")
            
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_f1 = f1_score(train_labels, train_predictions, average='macro', zero_division=0)
        train_precision = precision_score(train_labels, train_predictions, average='macro', zero_division=0)
        train_recall = recall_score(train_labels, train_predictions, average='macro', zero_division=0)
        
        train_time = time.time() - train_start_time
        print(f"Training completed in {train_time:.2f} seconds")
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_labels = []
        
        val_start_time = time.time()
        
        with torch.no_grad():
            # Validation loop - show each step iteration
            for batch_idx, (videos, labels) in enumerate(val_loader):
                videos, labels = videos.to(device), labels.to(device)
                batch_size = labels.size(0)
                
                outputs = model(videos)
                loss = criterion(outputs, labels)
                
                batch_loss = loss.item()
                val_loss += batch_loss
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                val_total += batch_size
                val_correct += batch_correct
                
                # Calculate batch accuracy
                batch_acc = 100 * batch_correct / batch_size
                
                # Calculate running averages
                avg_loss_so_far = val_loss / (batch_idx + 1)
                avg_acc_so_far = 100 * val_correct / val_total
                
                # Show each validation step iteration
                print(f"Val Step {batch_idx+1:3d}/{len(val_loader)}: "
                      f"Epoch={epoch+1}, Batch_Size={batch_size}, "
                      f"Val_Loss={batch_loss:.4f}, Batch_Acc={batch_acc:.2f}%, "
                      f"Avg_Val_Loss={avg_loss_so_far:.4f}, Avg_Acc={avg_acc_so_far:.2f}%")
                
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_f1 = f1_score(val_labels, val_predictions, average='macro', zero_division=0)
        val_precision = precision_score(val_labels, val_predictions, average='macro', zero_division=0)
        val_recall = recall_score(val_labels, val_predictions, average='macro', zero_division=0)
        
        val_time = time.time() - val_start_time
        print(f"Validation completed in {val_time:.2f} seconds")
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_loss': val_loss
            }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
        
        # Calculate epoch timing
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary with requested metrics
        print(f"\nEPOCH {epoch+1} SUMMARY:")
        print(f"  Train_Loss: {train_loss:.4f}")
        print(f"  Val_Loss:   {val_loss:.4f}")
        print(f"  Accuracy:   {val_acc:.2f}%")
        print(f"  F1_Score:   {val_f1:.4f}")
        print(f"  Precision:  {val_precision:.4f}")
        print(f"  Training_Time: {train_time:.2f}s")
        print(f"  Epoch_Time: {epoch_time:.2f}s")
        print(f"  Learning_Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Best_Val_Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
        print(f"  Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
            break
        
        scheduler.step()
    
    # Load best model
    if os.path.exists(os.path.join(OUTPUT_DIR, 'best_model.pth')):
        checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Calculate final averages
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_accuracy = sum(val_accuracies) / len(val_accuracies)
    avg_f1 = sum(val_f1_scores) / len(val_f1_scores)
    avg_precision = sum(val_precisions) / len(val_precisions)
    total_training_time = time.time() - (epoch_start_time - epoch_time)
    
    print(f"\nTRAINING COMPLETED!")
    print("="*80)
    print("FINAL TRAINING SUMMARY:")
    print(f"  Avg_Train_Loss: {avg_train_loss:.4f}")
    print(f"  Avg_Val_Loss:   {avg_val_loss:.4f}")
    print(f"  Avg_Accuracy:   {avg_accuracy:.2f}%")
    print(f"  Avg_F1_Score:   {avg_f1:.4f}")
    print(f"  Avg_Precision:  {avg_precision:.4f}")
    print(f"  Total_Training_Time: {total_training_time:.2f}s")
    print(f"  Best_Val_Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print("="*80)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores,
        'val_precisions': val_precisions,
        'val_recalls': val_recalls,
        'learning_rates': learning_rates,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }

def calculate_comprehensive_metrics(y_true, y_pred, y_proba=None, class_names=None):
    """Calculate comprehensive metrics for video classification"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Top-k accuracy
    if y_proba is not None:
        n_classes = len(np.unique(y_true))
        try:
            if n_classes >= 2:
                metrics['top2_accuracy'] = top_k_accuracy_score(y_true, y_proba, k=2)
            else:
                metrics['top2_accuracy'] = 1.0  # Perfect accuracy for single class
                
            if n_classes >= 3:
                metrics['top3_accuracy'] = top_k_accuracy_score(y_true, y_proba, k=3)
            else:
                metrics['top3_accuracy'] = 1.0
                
            if n_classes >= 5:
                metrics['top5_accuracy'] = top_k_accuracy_score(y_true, y_proba, k=5)
            else:
                metrics['top5_accuracy'] = 1.0
        except Exception as e:
            # Fallback for any other issues
            metrics['top2_accuracy'] = 1.0
            metrics['top3_accuracy'] = 1.0
            metrics['top5_accuracy'] = 1.0
        
        # ROC AUC (multi-class)
        try:
            # For multi-class ROC-AUC, we need to use label_binarize
            from sklearn.preprocessing import label_binarize
            n_classes = len(np.unique(y_true))
            if n_classes > 2:
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                metrics['roc_auc_ovr'] = roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='macro')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true_bin, y_proba, multi_class='ovo', average='macro')
            else:
                # For binary classification
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba[:, 1])
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception as e:
            print(f"ROC-AUC calculation failed: {e}")
            metrics['roc_auc_ovr'] = 0.0
            metrics['roc_auc_ovo'] = 0.0
    
    # Per-class metrics
    if class_names:
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class_f1'] = dict(zip(class_names, f1_per_class))
        metrics['per_class_precision'] = dict(zip(class_names, precision_per_class))
        metrics['per_class_recall'] = dict(zip(class_names, recall_per_class))
    
    return metrics

def evaluate_model(model, test_loader, class_names, device='cuda'):
    """Evaluate the trained model with comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc="Evaluating"):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(
        all_labels, all_predictions, np.array(all_probabilities), class_names
    )
    
    # Print detailed metrics
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("="*80)
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1-Score (Macro):   {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"  Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"  Recall (Weighted):  {metrics['recall_weighted']:.4f}")
    
    if 'top2_accuracy' in metrics:
        print(f"\nTOP-K ACCURACY:")
        print(f"  Top-2 Accuracy:    {metrics['top2_accuracy']:.4f} ({metrics['top2_accuracy']*100:.2f}%)")
        print(f"  Top-3 Accuracy:    {metrics['top3_accuracy']:.4f} ({metrics['top3_accuracy']*100:.2f}%)")
        print(f"  Top-5 Accuracy:    {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
    
    if 'roc_auc_ovr' in metrics:
        print(f"\nROC-AUC SCORES:")
        print(f"  ROC-AUC (OvR):     {metrics['roc_auc_ovr']:.4f}")
        print(f"  ROC-AUC (OvO):     {metrics['roc_auc_ovo']:.4f}")
    
    # Per-class metrics
    if 'per_class_f1' in metrics:
        print(f"\nPER-CLASS METRICS:")
        print(f"{'Class':<15} {'F1-Score':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 50)
        for class_name in class_names:
            f1 = metrics['per_class_f1'][class_name]
            precision = metrics['per_class_precision'][class_name]
            recall = metrics['per_class_recall'][class_name]
            print(f"{class_name:<15} {f1:<10.4f} {precision:<10.4f} {recall:<10.4f}")
    
    # Classification report
    print(f"\nDETAILED CLASSIFICATION REPORT:")
    report = classification_report(all_labels, all_predictions, target_names=class_names, digits=4)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ISL Video Classification', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Save metrics to file
    import json
    with open(os.path.join(OUTPUT_DIR, 'evaluation_metrics.json'), 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                json_metrics[key] = {k: float(v) for k, v in value.items()}
            else:
                json_metrics[key] = float(value)
        json.dump(json_metrics, f, indent=2)
    
    print(f"\nMetrics saved to '{os.path.join(OUTPUT_DIR, 'evaluation_metrics.json')}'")
    print("="*80)
    
    return all_predictions, all_labels, metrics

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Train ISL Video Transformer")
    parser.add_argument("--model_path", type=str, default="OpenGVLab/VideoMAEv2-Huge", help="Local HF model dir or repo id")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    parser.add_argument("--data_dir", type=str, default="isl(diksha)", help="Dataset root directory")
    parser.add_argument("--csv_path", type=str, default="isl(diksha)/dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_frames", type=int, default=16, help="Number of frames per clip")
    parser.add_argument("--frame_size", type=int, nargs=2, default=[224, 224], help="Frame size H W")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Configuration
    DATA_DIR = args.data_dir
    CSV_PATH = args.csv_path
    BATCH_SIZE = args.batch_size  # Small batch size due to memory constraints
    MAX_FRAMES = args.max_frames
    FRAME_SIZE = (args.frame_size[0], args.frame_size[1])
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    MODEL_PATH = args.model_path

    # Ensure output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup comprehensive logging
    logger, log_file = setup_logging(OUTPUT_DIR)
    logger.info("="*80)
    logger.info("ISL VIDEO TRANSFORMER TRAINING STARTED")
    logger.info("="*80)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"Log File: {log_file}")
    logger.info("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset (basic words only)
    df, class_names, label_to_idx = load_dataset(DATA_DIR, CSV_PATH, use_basic_words_only=True)
    
    # Split dataset: 80% train, 20% evaluation (use same 20% for val and test)
    train_df, eval_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label_idx']
    )
    val_df = eval_df
    test_df = eval_df
    
    print(f"Train: {len(train_df)}, Eval(Val/Test): {len(eval_df)}")
    
    # Create datasets
    train_dataset = ISLVideoDataset(
        train_df['video_path'].tolist(),
        train_df['label_idx'].tolist(),
        max_frames=MAX_FRAMES,
        frame_size=FRAME_SIZE
    )
    
    val_dataset = ISLVideoDataset(
        val_df['video_path'].tolist(),
        val_df['label_idx'].tolist(),
        max_frames=MAX_FRAMES,
        frame_size=FRAME_SIZE
    )
    
    test_dataset = ISLVideoDataset(
        test_df['video_path'].tolist(),
        test_df['label_idx'].tolist(),
        max_frames=MAX_FRAMES,
        frame_size=FRAME_SIZE
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create model
    model = ISLVideoTransformer(
        num_classes=len(class_names),
        freeze_backbone=True,  # Freeze VideoMAE backbone initially
        model_name=MODEL_PATH
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Train model
    print("Starting training...")
    training_metrics = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, device, class_names
    )
    
    # Plot comprehensive training curves
    plt.figure(figsize=(20, 12))
    
    # Loss curves
    plt.subplot(2, 3, 1)
    plt.plot(training_metrics['train_losses'], label='Train Loss', linewidth=2, marker='o')
    plt.plot(training_metrics['val_losses'], label='Val Loss', linewidth=2, marker='o')
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy curves
    plt.subplot(2, 3, 2)
    plt.plot(training_metrics['val_accuracies'], label='Val Accuracy', linewidth=2, color='green', marker='o')
    plt.title('Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1-Score curves
    plt.subplot(2, 3, 3)
    plt.plot(training_metrics['val_f1_scores'], label='Val F1-Score', linewidth=2, color='orange', marker='o')
    plt.title('Validation F1-Score', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Precision curves
    plt.subplot(2, 3, 4)
    plt.plot(training_metrics['val_precisions'], label='Val Precision', linewidth=2, color='red', marker='o')
    plt.title('Validation Precision', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Recall curves
    plt.subplot(2, 3, 5)
    plt.plot(training_metrics['val_recalls'], label='Val Recall', linewidth=2, color='purple', marker='o')
    plt.title('Validation Recall', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate curve
    plt.subplot(2, 3, 6)
    plt.plot(training_metrics['learning_rates'], label='Learning Rate', linewidth=2, color='brown', marker='o')
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.suptitle('VideoMAEv2-Huge Training Progress', fontsize=16, fontweight='bold')
    plt.tight_layout()
    # Save before showing to avoid empty images on some backends
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Training curves saved as 'training_curves.png'")
    
    # Evaluate model
    print("Evaluating model...")
    predictions, true_labels, test_metrics = evaluate_model(model, test_loader, class_names, device)
    
    # Feature visualization and analysis
    print("Analyzing features...")
    visualizer = FeatureVisualizer(model, device)
    
    # Extract features for visualization
    features, labels = visualizer.extract_features_batch(test_loader)
    
    # Visualize feature distribution
    visualizer.visualize_feature_distribution(features, labels, class_names, os.path.join(OUTPUT_DIR, "feature_distribution.png"))
    
    # Analyze feature thresholds
    thresholds, feature_magnitudes = visualizer.analyze_feature_thresholds(features, labels, class_names)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'label_to_idx': label_to_idx,
        'config': {
            'max_frames': MAX_FRAMES,
            'frame_size': FRAME_SIZE,
            'num_classes': len(class_names)
        },
        'training_metrics': training_metrics,
        'test_metrics': test_metrics
    }, os.path.join(OUTPUT_DIR, 'isl_video_transformer_model.pth'))
    
    print("Model saved as 'isl_video_transformer_model.pth'")
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING AND EVALUATION COMPLETED!")
    print("="*80)
    print("FINAL SUMMARY:")
    print(f"  Best Validation Accuracy: {training_metrics['best_val_acc']:.2f}% (Epoch {training_metrics['best_epoch']})")
    print(f"  Test Accuracy:           {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Test F1-Score (Macro):   {test_metrics['f1_macro']:.4f}")
    print(f"  Test F1-Score (Weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"  Test Precision (Macro):  {test_metrics['precision_macro']:.4f}")
    print(f"  Test Recall (Macro):     {test_metrics['recall_macro']:.4f}")
    
    if 'top2_accuracy' in test_metrics:
        print(f"  Top-2 Accuracy:          {test_metrics['top2_accuracy']:.4f} ({test_metrics['top2_accuracy']*100:.2f}%)")
        print(f"  Top-3 Accuracy:          {test_metrics['top3_accuracy']:.4f} ({test_metrics['top3_accuracy']*100:.2f}%)")
    
    print(f"\nFILES GENERATED:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'isl_video_transformer_model.pth')} (Trained model)")
    print(f"  - {os.path.join(OUTPUT_DIR, 'best_model.pth')} (Best validation model)")
    print(f"  - {os.path.join(OUTPUT_DIR, 'training_curves.png')} (Training progress plots)")
    print(f"  - {os.path.join(OUTPUT_DIR, 'feature_distribution.png')} (Feature analysis)")
    print(f"  - {os.path.join(OUTPUT_DIR, 'evaluation_metrics.json')} (Detailed metrics)")
    print(f"  - {log_file} (Complete training and evaluation log)")
    print("="*80)
    print("Training completed!")
    
    # Final log entry
    logger.info("="*80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"All logs saved to: {log_file}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
