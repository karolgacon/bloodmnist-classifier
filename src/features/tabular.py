"""Flatten / PCA features for sklearn. Implemented by O2.

All steps live in one sklearn Pipeline, so every fit (scaler, PCA) sees only the
data passed to .fit() — train, or the training folds inside CV (data_contract §2).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.config import RANDOM_SEED


def flatten_images(images: np.ndarray) -> np.ndarray:
    """(N, H, W, C) -> (N, H*W*C)."""
    n = images.shape[0]
    return images.reshape(n, -1).astype(np.float32)


def make_feature_pipeline(pca_components: Optional[int] = None, random_state: int = RANDOM_SEED):
    """Unfitted Pipeline: flatten (N, 28, 28, 3) uint8 -> StandardScaler -> optional PCA.

    Step names ("flatten", "scaler", "pca") are part of the API — tuning grids use
    e.g. "features__pca__n_components".
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    steps = [
        ("flatten", FunctionTransformer(flatten_images)),
        ("scaler", StandardScaler()),
    ]
    if pca_components:
        steps.append(("pca", PCA(n_components=pca_components, svd_solver="randomized", random_state=random_state)))
    return Pipeline(steps)


def build_feature_matrix(
    images: np.ndarray,
    pca_components: Optional[int] = None,
    pipeline=None,
):
    """StandardScaler + optional PCA features, returned as (X, fitted_pipeline).

    pipeline=None: fit a new pipeline on `images` (pass train images only).
    pipeline=<fitted>: transform only (val/test), e.g.
        X_train, feats = build_feature_matrix(train_images, pca_components=100)
        X_val, _ = build_feature_matrix(val_images, pipeline=feats)
    """
    if pipeline is None:
        pipeline = make_feature_pipeline(pca_components)
        return pipeline.fit_transform(images), pipeline
    return pipeline.transform(images), pipeline
