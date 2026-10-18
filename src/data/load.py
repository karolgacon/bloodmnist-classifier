"""Load BloodMNIST via official medmnist API. Implemented by O1."""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

SplitName = Literal["train", "val", "test"]


def get_medmnist_splits(download: bool = True):
    """Return train, val, test BloodMNIST dataset objects."""
    from medmnist import BloodMNIST

    train = BloodMNIST(split="train", download=download, as_rgb=True)
    val = BloodMNIST(split="val", download=download, as_rgb=True)
    test = BloodMNIST(split="test", download=download, as_rgb=True)
    return train, val, test


def load_numpy_split(
    split: SplitName,
    download: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Images uint8 (N, 28, 28, 3); labels int64 (N,)."""
    from medmnist import BloodMNIST

    ds = BloodMNIST(split=split, download=download, as_rgb=True)
    images = ds.imgs
    labels = ds.labels.squeeze().astype(np.int64)
    return images, labels
