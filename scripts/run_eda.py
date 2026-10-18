#!/usr/bin/env python3
"""Run EDA and export figures (O1)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.eda import run_eda


def main() -> None:
    run_eda()


if __name__ == "__main__":
    main()
