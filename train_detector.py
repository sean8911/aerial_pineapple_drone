"""Fine-tune a YOLO detector to count pineapples in aerial drone tiles."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

BASE_WEIGHTS = "yolov10s.pt"
EPOCHS = 200
IMAGE_SIZE = 640
OUTPUT_PROJECT = "."
OUTPUT_NAME = "assets"


def train(
    data_config: Path,
    base_weights: str = BASE_WEIGHTS,
    epochs: int = EPOCHS,
    image_size: int = IMAGE_SIZE,
) -> Path:
    """Train a detector and return the path to the best checkpoint."""
    if not data_config.is_file():
        raise FileNotFoundError(f"dataset config not found: {data_config}")

    model = YOLO(base_weights)
    results = model.train(
        data=str(data_config),
        epochs=epochs,
        imgsz=image_size,
        plots=True,
        project=OUTPUT_PROJECT,
        name=OUTPUT_NAME,
        exist_ok=True,
    )
    return Path(results.save_dir) / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_config", type=Path, help="path to the dataset data.yaml")
    parser.add_argument("--base-weights", default=BASE_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best = train(args.data_config, args.base_weights, args.epochs, args.image_size)
    print(f"best weights: {best}")


if __name__ == "__main__":
    main()
