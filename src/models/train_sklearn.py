"""Classical ML training and tuning. Implemented by O2.

Uses the train and val splits only (data_contract §5). Hyperparameters are tuned
and the best model is selected with RandomizedSearchCV + StratifiedKFold on train;
val is only reported as a hold-out check. Test-set evaluation lives in
src/models/evaluate_sklearn.py (test day, D20).

Outputs:
    reports/results/sklearn_tuning.json            CV results + best params per model, best_model
    reports/results/val/sklearn_<model>.json       val metrics (metrics_schema format)
    reports/figures/sklearn_cm_<model>_val.png     confusion matrix on val
    reports/figures/sklearn_f1_macro_val.png       macro-F1 comparison on val
    models/sklearn_<model>.joblib, models/best_sklearn.joblib
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    SKLEARN_CV_FOLDS,
    SKLEARN_PCA_COMPONENTS,
    SKLEARN_RANDOM_SEARCH_ITER,
)

MODEL_NAMES = ("logistic_regression", "svm_rbf", "random_forest")
BASELINE_NAME = "majority_baseline"
MODEL_LABELS_PL = {
    BASELINE_NAME: "Baseline (klasa większościowa)",
    "logistic_regression": "Regresja logistyczna",
    "svm_rbf": "SVM RBF",
    "random_forest": "Las losowy",
}
BEST_MODEL_PATH = MODELS_DIR / "best_sklearn.joblib"
TUNING_PATH = RESULTS_DIR / "sklearn_tuning.json"


def _model_spec(name: str) -> dict:
    """Unfitted pipeline, search space and reported hyperparameters for one model.

    Every pipeline is features (src.features.tabular) -> model, so scaler/PCA are
    refit inside each CV fold.
    """
    from scipy.stats import loguniform
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    from src.features.tabular import make_feature_pipeline

    if name == BASELINE_NAME:
        model, pca, space = DummyClassifier(strategy="most_frequent"), None, {}
        report = ["strategy"]
    elif name == "logistic_regression":
        # Full 2352 pixels: PCA cost ~5 pp of val macro F1 and loses pixel-level coefficients.
        model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)
        pca = None
        space = {"model__C": loguniform(1e-4, 1e-1)}
        report = ["C", "class_weight", "max_iter"]
    elif name == "svm_rbf":
        # probability=False while tuning (5x faster); the final refit enables it for AUC / top-k.
        model = SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED)
        pca = SKLEARN_PCA_COMPONENTS
        space = {
            "features__pca__n_components": [50, 100, 150],
            "model__C": loguniform(0.5, 50),
            "model__gamma": loguniform(1e-4, 5e-3),
        }
        report = ["C", "gamma", "kernel", "class_weight"]
    elif name == "random_forest":
        # n_jobs=1: parallelism comes from the CV search instead.
        model = RandomForestClassifier(n_jobs=1, random_state=RANDOM_SEED)
        pca = None
        space = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 20, 40],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.05],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        }
        report = ["n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight"]
    else:
        raise ValueError(f"unknown model {name!r}; choose from {(BASELINE_NAME,) + MODEL_NAMES}")

    pipeline = Pipeline([("features", make_feature_pipeline(pca)), ("model", model)])
    return {"pipeline": pipeline, "space": space, "report": report}


def _hyperparameters(pipeline, report_keys: Iterable[str]) -> dict:
    params = pipeline.named_steps["model"].get_params()
    hp = {k: params[k] for k in report_keys}
    features = pipeline.named_steps["features"]
    hp["pca_components"] = features.named_steps["pca"].n_components if "pca" in features.named_steps else None
    return hp


def _tune(name: str, X: np.ndarray, y: np.ndarray, n_iter: int, cv_folds: int, n_jobs: int) -> tuple:
    """RandomizedSearchCV on train, scoring macro F1. Returns (best unfitted pipeline, cv summary)."""
    from sklearn.base import clone
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    spec = _model_spec(name)
    search = RandomizedSearchCV(
        spec["pipeline"],
        spec["space"],
        n_iter=n_iter,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=n_jobs,
        random_state=RANDOM_SEED,
        refit=False,
    )
    search.fit(X, y)
    res = search.cv_results_
    i = search.best_index_
    summary = {
        "cv_f1_macro_mean": float(res["mean_test_score"][i]),
        "cv_f1_macro_std": float(res["std_test_score"][i]),
        "best_params": search.best_params_,
        "candidates": [
            {"params": p, "cv_f1_macro_mean": float(m), "cv_f1_macro_std": float(s)}
            for p, m, s in sorted(
                zip(res["params"], res["mean_test_score"], res["std_test_score"]), key=lambda t: -t[1]
            )
        ],
    }
    return clone(spec["pipeline"]).set_params(**search.best_params_), summary


def _cross_validate(name: str, X: np.ndarray, y: np.ndarray, cv_folds: int, n_jobs: int) -> tuple:
    """No tuning: default hyperparameters, CV score only (for --no-tune and the baseline)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    pipeline = _model_spec(name)["pipeline"]
    scores = cross_val_score(
        pipeline, X, y, scoring="f1_macro", n_jobs=n_jobs,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED),
    )
    return pipeline, {"cv_f1_macro_mean": float(scores.mean()), "cv_f1_macro_std": float(scores.std()), "best_params": {}}


def evaluate_pipeline(name: str, pipeline, X: np.ndarray, y: np.ndarray, split: str, extra: dict) -> tuple:
    """Returns (metrics record, y_pred).

    y_pred = argmax of predict_proba, not .predict(): for SVC the two can disagree
    (Platt scaling), and metrics, CM figure and Streamlit top-k must use the same rule.
    """
    from src.evaluation.metrics import build_metrics_record, compute_metrics

    proba = pipeline.predict_proba(X)
    y_pred = proba.argmax(axis=1)
    evaluation = compute_metrics(y, y_pred, proba)
    record = build_metrics_record(
        name, split, _hyperparameters(pipeline, _model_spec(name)["report"]), evaluation, **extra
    )
    return record, y_pred


def _save_tuning(tuning: dict, path: Path = TUNING_PATH) -> dict:
    """Merge into the existing tuning JSON (a --models subset keeps the other entries); returns the merged dict."""
    from src.evaluation.metrics import _to_builtin

    merged = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    merged.update(_to_builtin(tuning))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return merged


def train_all_sklearn_models(
    tune: bool = True,
    output_dir: Optional[Path] = None,
    models: Iterable[str] = MODEL_NAMES,
    n_iter: int = SKLEARN_RANDOM_SEARCH_ITER,
    cv_folds: int = SKLEARN_CV_FOLDS,
    n_jobs: int = -1,
    download: bool = True,
) -> dict:
    """Train LR, SVM, RF; save metrics JSON and best joblib.

    Tuning, CV and selection of the best model (CV macro F1) on train only; val
    metrics are reported, not used for decisions. Never loads the test split.
    output_dir: root for results/ figures/ models/ (default: repo layout from config).
    Returns {model_name: val metrics record} for the models trained in this run.
    """
    import joblib

    from src.data.load import load_numpy_split
    from src.evaluation.metrics import save_metrics_json
    from src.evaluation.plots import plot_confusion_matrix, plot_metric_comparison

    results_dir = output_dir / "results" if output_dir else RESULTS_DIR
    figures_dir = output_dir / "figures" if output_dir else FIGURES_DIR
    models_dir = output_dir / "models" if output_dir else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_numpy_split("train", download=download)
    X_val, y_val = load_numpy_split("val", download=download)

    records, tuning = {}, {}
    for name in (BASELINE_NAME, *models):
        start = time.time()
        if tune and name != BASELINE_NAME:
            pipeline, summary = _tune(name, X_train, y_train, n_iter, cv_folds, n_jobs)
        else:
            pipeline, summary = _cross_validate(name, X_train, y_train, cv_folds, n_jobs)
        if name == "svm_rbf":
            pipeline.set_params(model__probability=True)
        pipeline.fit(X_train, y_train)
        summary["seconds"] = round(time.time() - start, 1)
        summary["tuned"] = tune and name != BASELINE_NAME
        tuning[name] = summary

        record, y_pred = evaluate_pipeline(
            name, pipeline, X_val, y_val, "val",
            {"cv_f1_macro_mean": summary["cv_f1_macro_mean"], "cv_f1_macro_std": summary["cv_f1_macro_std"]},
        )
        records[name] = record
        save_metrics_json(record, str(results_dir / "val" / f"sklearn_{name}.json"))
        joblib.dump(pipeline, models_dir / f"sklearn_{name}.joblib")
        plot_confusion_matrix(y_val, y_pred, figures_dir / f"sklearn_cm_{name}_val.png")
        m = record["metrics"]
        print(
            f"{name:22s} cv F1 {summary['cv_f1_macro_mean']:.4f} ± {summary['cv_f1_macro_std']:.4f} | "
            f"val F1 {m['f1_macro']:.4f}  bal.acc {m['balanced_accuracy']:.4f}  "
            f"AUC {m['auc_macro_ovr']:.4f} | {summary['seconds']:.0f}s",
            flush=True,
        )

    # Best model over every trained model on disk (not just this run's --models subset), by CV on train.
    merged = _save_tuning(tuning, results_dir / TUNING_PATH.name)
    trained = [n for n in MODEL_NAMES if n in merged and (models_dir / f"sklearn_{n}.joblib").exists()]
    best = max(trained, key=lambda n: merged[n]["cv_f1_macro_mean"])
    shutil.copyfile(models_dir / f"sklearn_{best}.joblib", models_dir / BEST_MODEL_PATH.name)
    _save_tuning({"best_model": best, "selected_by": "cv_f1_macro_mean (train)"}, results_dir / TUNING_PATH.name)
    print(f"Best sklearn model (CV macro F1): {best} -> {models_dir / BEST_MODEL_PATH.name}")

    val_f1 = {}
    for name in (BASELINE_NAME, *MODEL_NAMES):
        path = results_dir / "val" / f"sklearn_{name}.json"
        if path.exists():
            val_f1[MODEL_LABELS_PL[name]] = json.loads(path.read_text(encoding="utf-8"))["metrics"]["f1_macro"]
    plot_metric_comparison(
        val_f1,
        figures_dir / "sklearn_f1_macro_val.png",
        title="Modele klasyczne — macro F1 na zbiorze walidacyjnym",
    )
    return records


def load_best_sklearn(path: Optional[Path] = None):
    """Fitted Pipeline for O4 (SHAP, Streamlit): .predict_proba(images (N, 28, 28, 3) uint8)."""
    import joblib

    return joblib.load(path or BEST_MODEL_PATH)
