"""
Training script for ISL Video Transformer
Simplified version for easy execution
"""

import os
import sys
import torch
import numpy as np
from video_transformer_isl import main as train_main

def check_requirements():
    """Check if all requirements are met"""
    print("Checking requirements...")
    
    # Check if dataset exists
    if not os.path.exists("isl(diksha)/dataset.csv"):
        print("❌ Dataset CSV not found!")
        print("Please ensure 'isl(diksha)/dataset.csv' exists")
        return False
    
    # Check if video directory exists
    if not os.path.exists("isl(diksha)/isl sign"):
        print("❌ Video directory not found!")
        print("Please ensure 'isl(diksha)/isl sign' directory exists")
        return False
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU (training will be slower)")
    
    print("✅ Requirements check passed!")
    return True

def main():
    """Main training function"""
    print("=" * 60)
    print("ISL Video Transformer Training")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements check failed. Please fix the issues above.")
        return
    
    print("\nStarting training...")
    print("This may take several hours depending on your hardware.")
    print("Press Ctrl+C to stop training at any time.\n")
    
    try:
        # Run training
        train_main()
        
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print("Model saved as: isl_video_transformer_model.pth")
        print("You can now use realtime_isl_detector.py for real-time detection")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        print("Please check the error message and try again")

if __name__ == "__main__":
    main()
