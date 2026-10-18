#!/usr/bin/env python3
"""Merge metrics JSON into leaderboard.csv (O4)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR


def main() -> None:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "metrics_schema.json":
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "model_name" in data and "metrics" in data:
            row = {"model_name": data["model_name"], **data["metrics"]}
            rows.append(row)
    if not rows:
        print("No result JSON files yet.")
        return
    import pandas as pd

    out = RESULTS_DIR / "leaderboard.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
