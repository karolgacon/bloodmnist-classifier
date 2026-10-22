#!/usr/bin/env python3
"""Compute dataset stats and export EDA figures (O1)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.eda import STATS_PATH, write_dataset_stats


def main() -> None:
    stats = write_dataset_stats()
    print(f"Saved {STATS_PATH}")
    print(json.dumps({k: stats[k] for k in ("mean_rgb", "std_rgb", "background_rgb_uint8")}))


if __name__ == "__main__":
    main()
