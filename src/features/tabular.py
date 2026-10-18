"""Flatten / PCA features for sklearn. Implemented by O2."""
from __future__ import annotations

import numpy as np


def flatten_images(images: np.ndarray) -> np.ndarray:
    """(N, H, W, C) -> (N, H*W*C)."""
    n = images.shape[0]
    return images.reshape(n, -1).astype(np.float32)


def build_feature_matrix(
    images: np.ndarray,
    pca_components: int | None = None,
    fit_pca: bool = False,
):
    """O2: StandardScaler + optional PCA pipeline."""
    raise NotImplementedError("O2: implement tabular feature pipeline")
