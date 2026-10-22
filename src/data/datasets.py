"""PyTorch datasets and transforms. Implemented by O1."""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import (
    CNN_BATCH_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    RANDOM_SEED,
    ROTATION_FILL,
)


def _label_to_int(label: np.ndarray) -> int:
    """MedMNIST label (1,) array -> int, so batches collate to int64 (B,)."""
    return int(label[0])


def _seed_worker(worker_id: int) -> None:
    """Seed numpy/random in DataLoader workers from the torch worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_transform(augment: bool = True):
    """Training transforms: augmentation (optional) + normalize.

    augment=False gives the same pipeline as val/test (ablation O3-04).
    """
    if not augment:
        return get_val_test_transform()
    return transforms.Compose(
        [
            transforms.RandomRotation(15, fill=ROTATION_FILL),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def get_val_test_transform():
    """Validation/test transforms (normalize only)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def get_datasets(download: bool = True, augment: bool = True):
    """Return train, val, test BloodMNIST datasets with transforms applied.

    Items are (image float32 (3, 28, 28), label int).
    """
    from medmnist import BloodMNIST

    def make(split: str, transform):
        return BloodMNIST(
            split=split,
            transform=transform,
            target_transform=_label_to_int,
            download=download,
            as_rgb=True,
        )

    train = make("train", get_train_transform(augment))
    val = make("val", get_val_test_transform())
    test = make("test", get_val_test_transform())
    return train, val, test


def get_dataloaders(
    batch_size: int = CNN_BATCH_SIZE,
    num_workers: int = 0,
    download: bool = True,
    augment: bool = True,
    seed: int = RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, val, test DataLoaders.

    Batches: images float32 (B, 3, 28, 28), labels int64 (B,).
    Only train is shuffled (seeded generator -> reproducible order).
    """
    train_ds, val_ds, test_ds = get_datasets(download=download, augment=augment)
    generator = torch.Generator().manual_seed(seed)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader
