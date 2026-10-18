#!/usr/bin/env python3
"""Train classical ML models (O2)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.train_sklearn import train_all_sklearn_models


def main() -> None:
    train_all_sklearn_models(tune=True)


if __name__ == "__main__":
    main()
