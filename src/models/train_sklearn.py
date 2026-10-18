"""Classical ML training and tuning. Implemented by O2."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def train_all_sklearn_models(
    tune: bool = True,
    output_dir: Optional[Path] = None,
) -> dict:
    """Train LR, SVM, RF; save metrics JSON and best joblib."""
    raise NotImplementedError("O2: implement sklearn training")
