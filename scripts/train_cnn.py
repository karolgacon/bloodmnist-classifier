#!/usr/bin/env python3
"""Train ResNet-18 (O3)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.train_cnn import train_cnn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-augment", action="store_true", help="Ablation: no augmentation")
    args = parser.parse_args()
    train_cnn(use_augmentation=not args.no_augment)


if __name__ == "__main__":
    main()
