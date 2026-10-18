"""Smoke tests — run from repo root: python -m pytest tests/"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_import_config():
    from src.config import (
        CLASS_NAMES_PL,
        NUM_CLASSES,
        PROJECT_ROOT,
        RANDOM_SEED,
    )

    assert NUM_CLASSES == 8
    assert len(CLASS_NAMES_PL) == 8
    assert RANDOM_SEED == 42
    assert PROJECT_ROOT.name == "bloodmnist-classifier"


def test_import_data_load():
    from src.data.load import get_medmnist_splits, load_numpy_split

    assert callable(get_medmnist_splits)
    assert callable(load_numpy_split)
