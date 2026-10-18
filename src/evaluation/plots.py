"""Confusion matrix, ROC, learning curves. Implemented by O2/O3."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    normalize: bool = True,
) -> None:
    raise NotImplementedError("O2/O3: implement confusion matrix plot")
