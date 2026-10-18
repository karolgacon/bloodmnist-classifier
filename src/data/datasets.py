"""PyTorch datasets and transforms. Implemented by O1."""
from __future__ import annotations

from typing import Optional, Tuple

# O1: implement get_train_transform, get_val_test_transform, get_dataloaders


def get_train_transform():
    """Training transforms (augmentation + normalize)."""
    raise NotImplementedError("O1: implement train transforms")


def get_val_test_transform():
    """Validation/test transforms (normalize only)."""
    raise NotImplementedError("O1: implement val/test transforms")


def get_dataloaders(
    batch_size: int,
    num_workers: int = 0,
    download: bool = True,
) -> Tuple[object, object, object]:
    """Return train, val, test DataLoaders."""
    raise NotImplementedError("O1: implement DataLoaders")
