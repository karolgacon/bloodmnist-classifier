#!/usr/bin/env python3
"""Error analysis gallery (O4)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    raise NotImplementedError("O4: implement error_analysis.py")


if __name__ == "__main__":
    main()
