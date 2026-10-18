"""EDA figure generation. Implemented by O1."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.config import FIGURES_DIR


def run_eda(output_dir: Optional[Path] = None) -> None:
    """Generate EDA figures into reports/figures/."""
    _ = output_dir or FIGURES_DIR
    raise NotImplementedError("O1: implement EDA plots")
