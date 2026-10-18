"""Shared evaluation metrics (O2 + O3)."""
from __future__ import annotations

from typing import Any

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict[str, Any]:
    """Accuracy, balanced accuracy, macro F1, macro AUC OvR, classification report."""
    raise NotImplementedError("O2/O3: implement compute_metrics")


def save_metrics_json(metrics: dict[str, Any], path: str) -> None:
    """Save metrics following reports/results/metrics_schema.json."""
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
