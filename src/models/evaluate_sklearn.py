"""Test-set evaluation of the classical models — test day (D20) only. Implemented by O2.

Kept apart from training (data_contract §5): loads the models frozen by
src/models/train_sklearn.py and evaluates each exactly once on the test split.

Outputs:
    reports/results/sklearn_<model>.json           test metrics (metrics_schema format; leaderboard input)
    reports/figures/confusion_matrix_sklearn.png   CM of the best model on test
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from src.config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR
from src.models.train_sklearn import BASELINE_NAME, MODEL_NAMES, TUNING_PATH, evaluate_pipeline


def evaluate_sklearn_on_test(
    output_dir: Optional[Path] = None,
    models: Iterable[str] = (BASELINE_NAME, *MODEL_NAMES),
    download: bool = True,
) -> dict:
    """Evaluate the saved, frozen models on test. Returns {model_name: test metrics record}."""
    import joblib

    from src.data.load import load_numpy_split
    from src.evaluation.metrics import save_metrics_json
    from src.evaluation.plots import plot_confusion_matrix

    results_dir = output_dir / "results" if output_dir else RESULTS_DIR
    figures_dir = output_dir / "figures" if output_dir else FIGURES_DIR
    models_dir = output_dir / "models" if output_dir else MODELS_DIR
    tuning = json.loads((results_dir / TUNING_PATH.name).read_text(encoding="utf-8"))

    X_test, y_test = load_numpy_split("test", download=download)
    records = {}
    for name in models:
        pipeline = joblib.load(models_dir / f"sklearn_{name}.joblib")
        extra = {k: tuning[name][k] for k in ("cv_f1_macro_mean", "cv_f1_macro_std")} if name in tuning else {}
        record, y_pred = evaluate_pipeline(name, pipeline, X_test, y_test, "test", extra)
        records[name] = record
        save_metrics_json(record, str(results_dir / f"sklearn_{name}.json"))
        m = record["metrics"]
        print(f"{name:22s} test F1 {m['f1_macro']:.4f}  acc {m['accuracy']:.4f}  AUC {m['auc_macro_ovr']:.4f}")
        if name == tuning.get("best_model"):
            plot_confusion_matrix(y_test, y_pred, figures_dir / "confusion_matrix_sklearn.png")
    return records
