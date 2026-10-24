#!/usr/bin/env python3
"""Evaluate the frozen classical models on the TEST split (O2) — test day (D20) only.

Run once, after hyperparameters are frozen (D17) and scripts/train_sklearn.py
has produced models/sklearn_*.joblib. Never use these results for tuning.

Usage: python scripts/evaluate_sklearn_test.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.evaluate_sklearn import evaluate_sklearn_on_test


def main() -> None:
    evaluate_sklearn_on_test()


if __name__ == "__main__":
    main()
