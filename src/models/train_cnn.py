"""CNN training loop. Implemented by O3."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def train_cnn(
    epochs: Optional[int] = None,
    use_augmentation: bool = True,
    output_weights: Optional[Path] = None,
) -> dict:
    """Train ResNet-18; early stopping on val loss."""
    raise NotImplementedError("O3: implement CNN training")


def predict_proba(image_tensor) -> dict[int, float]:
    """Inference for Streamlit: class index -> probability."""
    raise NotImplementedError("O3: implement predict_proba")
