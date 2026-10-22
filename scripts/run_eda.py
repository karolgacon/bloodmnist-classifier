#!/usr/bin/env python3
"""Compute dataset stats and export EDA figures (O1).

Usage: python scripts/run_eda.py [--lang pl|en]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.eda import STATS_PATH, run_eda, write_dataset_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=["pl", "en"], default="pl", help="figure language")
    args = parser.parse_args()

    stats = write_dataset_stats()
    print(f"Saved {STATS_PATH}")
    print(json.dumps({k: stats[k] for k in ("mean_rgb", "std_rgb", "background_rgb_uint8")}))

    for path in run_eda(lang=args.lang, stats=stats):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
