"""Data pipeline tests (O1) — run from repo root: python -m pytest tests/"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_N = {"train": 11959, "val": 1712, "test": 3421}


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_numpy_split_shapes(split):
    from src.data.load import load_numpy_split

    images, labels = load_numpy_split(split)
    n = EXPECTED_N[split]
    assert images.shape == (n, 28, 28, 3) and images.dtype == np.uint8
    assert labels.shape == (n,) and labels.dtype == np.int64
    assert labels.min() == 0 and labels.max() == 7


def test_dataloader_batch_contract():
    from src.data.datasets import get_dataloaders

    for loader in get_dataloaders(batch_size=32):
        images, labels = next(iter(loader))
        assert images.shape == (32, 3, 28, 28) and images.dtype == torch.float32
        assert labels.shape == (32,) and labels.dtype == torch.int64


def test_val_test_deterministic():
    from src.data.datasets import get_datasets

    _, val, _ = get_datasets()
    assert torch.equal(val[0][0], val[0][0])


def test_augment_flag():
    from src.data.datasets import get_datasets

    train_aug, _, _ = get_datasets(augment=True)
    train_plain, _, _ = get_datasets(augment=False)
    assert torch.equal(train_plain[0][0], train_plain[0][0])
    torch.manual_seed(0)
    draws = [train_aug[0][0] for _ in range(5)]
    assert any(not torch.equal(draws[0], d) for d in draws[1:])


def test_train_shuffle_reproducible():
    from src.data.datasets import get_dataloaders

    first = next(iter(get_dataloaders(batch_size=64, seed=123)[0]))[1]
    second = next(iter(get_dataloaders(batch_size=64, seed=123)[0]))[1]
    assert torch.equal(first, second)


def test_normalized_train_stats():
    from torch.utils.data import DataLoader

    from src.data.datasets import get_datasets

    train, _, _ = get_datasets(augment=False)
    images = torch.cat([x for x, _ in DataLoader(train, batch_size=2048)])
    mean = images.mean(dim=(0, 2, 3))
    std = images.std(dim=(0, 2, 3))
    assert torch.allclose(mean, torch.zeros(3), atol=1e-3)
    assert torch.allclose(std, torch.ones(3), atol=1e-3)


def test_config_matches_stats_json():
    from src.config import NORMALIZE_MEAN, NORMALIZE_STD, ROTATION_FILL
    from src.data.eda import STATS_PATH

    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    assert stats["computed_on"] == "train"
    assert NORMALIZE_MEAN == pytest.approx(stats["mean_rgb"], abs=1e-4)
    assert NORMALIZE_STD == pytest.approx(stats["std_rgb"], abs=1e-4)
    assert list(ROTATION_FILL) == stats["background_rgb_uint8"]
