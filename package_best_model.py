"""
Utility to convert a training checkpoint (best_model.pth) into a packaged
checkpoint (isl_video_transformer_model.pth) that includes class metadata
required by realtime_isl_detector.py.
"""

import os
import argparse
import torch

from video_transformer_isl import ISLVideoTransformer, load_dataset


def main():
    parser = argparse.ArgumentParser(description="Package best_model.pth with class metadata")
    parser.add_argument("--best", type=str, default=os.path.join("..", "best_model.pth"),
                        help="Path to best_model.pth")
    parser.add_argument("--data_dir", type=str, default="isl(diksha)",
                        help="Dataset root directory used during training")
    parser.add_argument("--csv", type=str, default=os.path.join("isl(diksha)", "dataset.csv"),
                        help="Path to dataset CSV used during training")
    parser.add_argument("--out", type=str, default=os.path.join("..", "isl_video_transformer_model.pth"),
                        help="Output packaged checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to map the checkpoint to when loading")

    args = parser.parse_args()

    if not os.path.exists(args.best):
        raise FileNotFoundError(f"best_model.pth not found at: {args.best}")

    print(f"Loading checkpoint: {args.best}")
    checkpoint = torch.load(args.best, map_location=args.device)

    # Build class metadata from the dataset description
    print(f"Loading dataset metadata from: {args.csv}")
    df, class_names, label_to_idx = load_dataset(args.data_dir, args.csv, use_basic_words_only=True)
    num_classes = len(class_names)

    # Recreate model and load weights
    print(f"Recreating model with {num_classes} classes and loading weights...")
    model = ISLVideoTransformer(num_classes=num_classes, freeze_backbone=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    packaged = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "label_to_idx": label_to_idx,
        "config": {
            "max_frames": 16,
            "frame_size": (224, 224),
            "num_classes": num_classes,
        },
    }

    torch.save(packaged, args.out)
    print(f"Wrote packaged checkpoint: {args.out}")


if __name__ == "__main__":
    main()


