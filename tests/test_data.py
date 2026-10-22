"""Data pipeline tests (O1) — run from repo root: python -m pytest tests/"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_config_matches_stats_json():
    from src.config import NORMALIZE_MEAN, NORMALIZE_STD, ROTATION_FILL
    from src.data.eda import STATS_PATH

    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    assert stats["computed_on"] == "train"
    assert NORMALIZE_MEAN == pytest.approx(stats["mean_rgb"], abs=1e-4)
    assert NORMALIZE_STD == pytest.approx(stats["std_rgb"], abs=1e-4)
    assert list(ROTATION_FILL) == stats["background_rgb_uint8"]
