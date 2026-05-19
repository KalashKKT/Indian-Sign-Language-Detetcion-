"""
Real-time ISL Video Detection System
Processes live video feed and predicts ISL signs
"""

import cv2
import torch
import numpy as np
from collections import deque
import time
import os
from video_transformer_isl import ISLVideoTransformer, ISLVideoDataset, load_dataset
import warnings
warnings.filterwarnings('ignore')


def _normalize_device(device_preference: str) -> torch.device:
    """Normalize user-provided device string to a valid torch.device.

    Handles synonyms like 'gpu' -> 'cuda' and falls back gracefully.
    """
    if device_preference is None:
        device_preference = ''

    preference = str(device_preference).strip().lower()

    # Map common synonyms
    if preference in {"gpu", "cuda", "cuda:0"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA not available; falling back to CPU.")
            return torch.device("cpu")

    if preference in {"cpu", "cpu:0"}:
        return torch.device("cpu")

    # Fallback: try to construct device, otherwise default
    try:
        dev = torch.device(preference)
        if dev.type == "cuda" and not torch.cuda.is_available():
            print("CUDA not available; falling back to CPU.")
            return torch.device("cpu")
        return dev
    except Exception:
        # Unknown device string
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

class RealTimeISLDetector:
    """Real-time ISL sign detection system"""
    
    def __init__(self, model_path, device='cuda', max_frames=16, frame_size=(224, 224), 
                 confidence_threshold=0.5, smoothing_window=5, data_dir=None, csv_path=None):
        self.device = _normalize_device(device)
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.data_dir = data_dir
        self.csv_path = csv_path
        
        # Load model
        self.load_model(model_path)
        
        # Frame buffer for temporal analysis
        self.frame_buffer = deque(maxlen=max_frames)
        self.prediction_buffer = deque(maxlen=smoothing_window)
        
        # Initialize camera
        self.cap = None
        
    def load_model(self, model_path):
        """Load the trained ISL model"""
        print("Loading ISL model...")
        
        # Load checkpoint with a safe map_location to avoid legacy 'gpu' tag issues
        map_loc = 'cuda' if self.device.type == 'cuda' else 'cpu'
        checkpoint = torch.load(model_path, map_location=map_loc, weights_only=False)
        
        # Get class metadata (fallback if missing)
        if 'class_names' in checkpoint and 'label_to_idx' in checkpoint:
            self.class_names = checkpoint['class_names']
            self.label_to_idx = checkpoint['label_to_idx']
            self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        else:
            if self.data_dir is None or self.csv_path is None:
                raise KeyError("Checkpoint missing class metadata. Provide --data_dir and --csv so we can rebuild class mappings.")
            # Build from dataset
            df, self.class_names, self.label_to_idx = load_dataset(self.data_dir, self.csv_path, use_basic_words_only=True)
            self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
        config = checkpoint.get('config', {"max_frames": self.max_frames, "frame_size": self.frame_size, "num_classes": len(self.class_names)})
        
        # Create model
        self.model = ISLVideoTransformer(
            num_classes=len(self.class_names),
            freeze_backbone=True
        )
        
        # Load state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully! Classes: {self.class_names}")
        
    def preprocess_frame(self, frame):
        """Preprocess a single frame for the model"""
        # Resize frame
        frame = cv2.resize(frame, self.frame_size)
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def predict_from_buffer(self):
        """Make prediction from current frame buffer"""
        if len(self.frame_buffer) < self.max_frames:
            return None, 0.0
        
        # Convert frame buffer to tensor
        frames = list(self.frame_buffer)
        
        # Pad if necessary
        while len(frames) < self.max_frames:
            frames.append(frames[-1] if frames else np.zeros((*self.frame_size, 3), dtype=np.float32))
        
        # Convert to tensor: (T, H, W, C) -> (1, C, T, H, W)
        frames = np.array(frames)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
        frames = frames.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(frames)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.idx_to_label[predicted.item()]
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def smooth_predictions(self, prediction, confidence):
        """Apply temporal smoothing to predictions"""
        if confidence < self.confidence_threshold:
            prediction = "No Sign Detected"
        
        self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) < self.smoothing_window:
            return prediction, confidence
        
        # Get most common prediction in buffer
        predictions = [p[0] for p in self.prediction_buffer]
        avg_confidence = np.mean([p[1] for p in self.prediction_buffer])
        
        # Find most frequent prediction
        from collections import Counter
        prediction_counts = Counter(predictions)
        most_common = prediction_counts.most_common(1)[0]
        
        return most_common[0], avg_confidence
    
    def draw_info(self, frame, prediction, confidence, fps):
        """Draw prediction information on frame"""
        # Create overlay
        overlay = frame.copy()
        
        # Draw background rectangle
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        cv2.putText(frame, f"Prediction: {prediction}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.3f}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw frame buffer status
        buffer_status = f"Buffer: {len(self.frame_buffer)}/{self.max_frames}"
        cv2.putText(frame, buffer_status, (frame.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def start_detection(self, camera_index=0, save_video=False, output_path="isl_detection_output.mp4"):
        """Start real-time ISL detection"""
        print("Starting real-time ISL detection...")
        print("Press 'q' to quit, 's' to save current prediction")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Video writer for saving
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        # Detection variables
        current_prediction = "Initializing..."
        current_confidence = 0.0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end_time = time.time()
                    current_fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                self.frame_buffer.append(processed_frame)
                
                # Make prediction if buffer is full
                if len(self.frame_buffer) == self.max_frames:
                    prediction, confidence = self.predict_from_buffer()
                    if prediction:
                        current_prediction, current_confidence = self.smooth_predictions(prediction, confidence)
                
                # Draw information on frame
                frame = self.draw_info(frame, current_prediction, current_confidence, current_fps)
                
                # Display frame
                cv2.imshow('ISL Real-time Detection', frame)
                
                # Save video if enabled
                if save_video:
                    out.write(frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    print(f"Current prediction: {current_prediction} (confidence: {current_confidence:.3f})")
                elif key == ord('r'):
                    # Reset buffers
                    self.frame_buffer.clear()
                    self.prediction_buffer.clear()
                    print("Buffers reset")
        
        except KeyboardInterrupt:
            print("Detection stopped by user")
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            if save_video:
                out.release()
                print(f"Video saved as {output_path}")
    
    def detect_from_video_file(self, video_path, output_path=None, display=True):
        """Detect ISL signs from a video file"""
        print(f"Processing video file: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Video writer for output
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Reset buffers
        self.frame_buffer.clear()
        self.prediction_buffer.clear()
        
        frame_count = 0
        predictions = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                self.frame_buffer.append(processed_frame)
                
                # Make prediction if buffer is full
                current_prediction = "Processing..."
                current_confidence = 0.0
                
                if len(self.frame_buffer) == self.max_frames:
                    prediction, confidence = self.predict_from_buffer()
                    if prediction:
                        current_prediction, current_confidence = self.smooth_predictions(prediction, confidence)
                        predictions.append((frame_count, current_prediction, current_confidence))
                
                # Draw information on frame
                frame = self.draw_info(frame, current_prediction, current_confidence, fps)
                
                # Display frame if requested
                if display:
                    cv2.imshow('ISL Video Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Write output video
                if output_path:
                    out.write(frame)
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        except KeyboardInterrupt:
            print("Processing stopped by user")
        
        finally:
            cap.release()
            if output_path:
                out.release()
                print(f"Output video saved as {output_path}")
            if display:
                cv2.destroyAllWindows()
        
        return predictions

def main():
    """Main function for real-time detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time ISL Detection')
    parser.add_argument('--model', type=str, default='../runs/isl_video_transformer_model.pth',
                       help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--save', action='store_true',
                       help='Save output video')
    parser.add_argument('--output', type=str, default='isl_detection_output.mp4',
                       help='Output video path')
    parser.add_argument('--video', type=str, default=None,
                       help='Process video file instead of camera')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--data_dir', type=str, default='isl(diksha)',
                       help='Dataset root directory (used if model lacks class metadata)')
    parser.add_argument('--csv', type=str, default='isl(diksha)/dataset.csv',
                       help='Dataset CSV path (used if model lacks class metadata)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Please train the model first using video_transformer_isl.py")
        return
    
    # Create detector
    detector = RealTimeISLDetector(
        model_path=args.model,
        device=args.device,
        data_dir=args.data_dir,
        csv_path=args.csv
    )
    
    if args.video:
        # Process video file
        output_path = args.output if args.save else None
        predictions = detector.detect_from_video_file(
            video_path=args.video,
            output_path=output_path,
            display=True
        )
        
        # Print summary
        print("\nDetection Summary:")
        for frame_num, prediction, confidence in predictions:
            print(f"Frame {frame_num}: {prediction} (confidence: {confidence:.3f})")
    
    else:
        # Real-time camera detection
        detector.start_detection(
            camera_index=args.camera,
            save_video=args.save,
            output_path=args.output
        )

if __name__ == "__main__":
    import os
    main()
