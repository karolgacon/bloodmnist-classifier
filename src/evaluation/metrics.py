"""Shared evaluation metrics (O2 + O3).

Output format follows reports/results/metrics_schema.json:

    record = build_metrics_record("logistic_regression", "test", hyperparams,
                                  compute_metrics(y_true, y_pred, y_proba))
    save_metrics_json(record, RESULTS_DIR / "sklearn_logistic_regression.json")
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.config import CLASS_NAMES_EN, NUM_CLASSES, RANDOM_SEED

REQUIRED_FIELDS = ("model_name", "split", "metrics", "hyperparameters", "seed")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict[str, Any]:
    """Accuracy, balanced accuracy, macro F1, macro AUC OvR, classification report.

    y_proba: (N, NUM_CLASSES) class probabilities; without it auc_macro_ovr is None.
    Returns {"metrics": {...}, "classification_report": {...}, "confusion_matrix": [[...]]}.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = list(range(NUM_CLASSES))

    auc = None
    if y_proba is not None:
        auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[CLASS_NAMES_EN[c] for c in labels],
        output_dict=True,
        zero_division=0,
    )
    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "auc_macro_ovr": auc,
        },
        "classification_report": _to_builtin(report),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def build_metrics_record(
    model_name: str,
    split: str,
    hyperparameters: dict[str, Any],
    evaluation: dict[str, Any],
    seed: int = RANDOM_SEED,
    **extra: Any,
) -> dict[str, Any]:
    """Wrap compute_metrics() output into a metrics_schema.json record.

    extra: additional top-level fields (e.g. cv_f1_macro), kept after the required ones.
    """
    record = {
        "model_name": model_name,
        "split": split,
        "seed": seed,
        "hyperparameters": _to_builtin(hyperparameters),
        **evaluation,
        **_to_builtin(extra),
    }
    missing = [k for k in REQUIRED_FIELDS if k not in record]
    if missing:
        raise ValueError(f"metrics record missing fields: {missing}")
    return record


def _to_builtin(obj: Any) -> Any:
    """numpy scalars/arrays -> plain Python, so json.dump never fails."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_metrics_json(metrics: dict[str, Any], path: str) -> None:
    """Save metrics following reports/results/metrics_schema.json."""
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(metrics), f, indent=2, ensure_ascii=False)
