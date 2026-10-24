"""Classical ML pipeline tests (O2) — run from repo root: python -m pytest tests/

Synthetic images only, so these run in seconds and never touch the test split.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import NUM_CLASSES, RESULTS_DIR


def _fake_images(n=80, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % NUM_CLASSES
    images = rng.integers(0, 256, size=(n, 28, 28, 3), dtype=np.uint8)
    images[..., 0] = (labels * 30)[:, None, None]  # class signal in the R channel
    return images, labels.astype(np.int64)


def test_feature_pipeline_shapes_and_fit_on_train_only():
    from src.features.tabular import build_feature_matrix

    train, _ = _fake_images(80, seed=0)
    val, _ = _fake_images(40, seed=1)
    X_train, feats = build_feature_matrix(train, pca_components=10)
    mean_before = feats.named_steps["scaler"].mean_.copy()
    X_val, same = build_feature_matrix(val, pipeline=feats)

    assert X_train.shape == (80, 10) and X_val.shape == (40, 10)
    assert same is feats
    np.testing.assert_array_equal(feats.named_steps["scaler"].mean_, mean_before)


def test_feature_pipeline_without_pca():
    from src.features.tabular import build_feature_matrix

    X, feats = build_feature_matrix(_fake_images(16)[0])
    assert X.shape == (16, 28 * 28 * 3)
    assert "pca" not in feats.named_steps
    assert np.allclose(X.mean(axis=0), 0, atol=1e-4)


def test_compute_metrics_perfect_and_schema():
    from src.evaluation.metrics import REQUIRED_FIELDS, build_metrics_record, compute_metrics

    y = np.arange(40) % NUM_CLASSES
    proba = np.eye(NUM_CLASSES)[y]
    ev = compute_metrics(y, y, proba)
    assert ev["metrics"] == {"accuracy": 1.0, "balanced_accuracy": 1.0, "f1_macro": 1.0, "auc_macro_ovr": 1.0}
    assert np.array_equal(np.diag(ev["confusion_matrix"]), np.bincount(y))
    assert "basophil" in ev["classification_report"]

    record = build_metrics_record("dummy", "val", {"C": np.float64(1.0)}, ev)
    schema = json.loads((RESULTS_DIR / "metrics_schema.json").read_text(encoding="utf-8"))
    assert set(schema["required_fields"]) <= set(record) and set(REQUIRED_FIELDS) <= set(record)
    json.dumps(record)  # plain Python types only


def test_compute_metrics_without_proba():
    from src.evaluation.metrics import compute_metrics

    y = np.arange(16) % NUM_CLASSES
    ev = compute_metrics(y, np.zeros_like(y))
    assert ev["metrics"]["auc_macro_ovr"] is None
    assert ev["metrics"]["accuracy"] == pytest.approx(2 / 16)
    assert ev["metrics"]["balanced_accuracy"] == pytest.approx(1 / NUM_CLASSES)


def test_plots_write_png(tmp_path):
    from src.evaluation.plots import plot_confusion_matrix, plot_metric_comparison

    y = np.arange(40) % NUM_CLASSES
    cm = plot_confusion_matrix(y, np.roll(y, 1), tmp_path / "cm.png")
    bars = plot_metric_comparison({"a": 0.8, "b": 0.6}, tmp_path / "f1.png", errors={"a": 0.01})
    assert cm.stat().st_size > 0 and bars.stat().st_size > 0


@pytest.mark.parametrize("name", ["majority_baseline", "logistic_regression", "svm_rbf", "random_forest"])
def test_model_specs_fit_predict_proba(name):
    from src.models.train_sklearn import _model_spec

    spec = _model_spec(name)
    pipeline = spec["pipeline"]
    if name == "svm_rbf":
        pipeline.set_params(model__probability=True, features__pca__n_components=10)
    images, labels = _fake_images(80)
    pipeline.fit(images, labels)
    proba = pipeline.predict_proba(images[:5])
    assert proba.shape == (5, NUM_CLASSES)
    assert np.allclose(proba.sum(axis=1), 1)
    # every searched parameter exists on the pipeline
    valid = pipeline.get_params()
    assert all(key in valid for key in spec["space"])


def test_evaluate_prediction_matches_record():
    """CM figure is drawn from the returned y_pred — it must be the one behind the JSON metrics."""
    from sklearn.metrics import confusion_matrix

    from src.models.train_sklearn import _model_spec, evaluate_pipeline

    pipeline = _model_spec("svm_rbf")["pipeline"]
    pipeline.set_params(model__probability=True, features__pca__n_components=10)
    images, labels = _fake_images(80)
    pipeline.fit(images, labels)
    record, y_pred = evaluate_pipeline("svm_rbf", pipeline, images, labels, "val", {})
    assert np.array_equal(np.argmax(pipeline.predict_proba(images), axis=1), y_pred)
    assert confusion_matrix(labels, y_pred, labels=range(NUM_CLASSES)).tolist() == record["confusion_matrix"]


def test_training_end_to_end_never_loads_test(tmp_path, monkeypatch):
    """Full train -> select -> test-day flow on synthetic data, written to tmp_path.

    Training must never request the test split (data_contract §5); a --models
    subset run must not replace a better best_sklearn.joblib from an earlier run.
    """
    import joblib

    import src.data.load
    from src.models.evaluate_sklearn import evaluate_sklearn_on_test
    from src.models.train_sklearn import train_all_sklearn_models

    requested = []

    def fake_split(split, download=True):
        requested.append(split)
        return _fake_images(240 if split == "train" else 48, seed={"train": 0, "val": 1, "test": 2}[split])

    monkeypatch.setattr(src.data.load, "load_numpy_split", fake_split)
    train_all_sklearn_models(tune=False, output_dir=tmp_path, n_jobs=1, cv_folds=2)
    assert "test" not in requested

    tuning = json.loads((tmp_path / "results" / "sklearn_tuning.json").read_text(encoding="utf-8"))
    best = tuning["best_model"]
    assert best == max(("logistic_regression", "svm_rbf", "random_forest"), key=lambda n: tuning[n]["cv_f1_macro_mean"])
    for name in ("majority_baseline", "logistic_regression", "svm_rbf", "random_forest"):
        assert (tmp_path / "results" / "val" / f"sklearn_{name}.json").exists()
    assert (tmp_path / "figures" / "sklearn_f1_macro_val.png").exists()

    # retrain one non-best model only: best_model must stay the overall CV winner
    other = next(n for n in ("logistic_regression", "svm_rbf", "random_forest") if n != best)
    train_all_sklearn_models(tune=False, output_dir=tmp_path, models=[other], n_jobs=1, cv_folds=2)
    tuning = json.loads((tmp_path / "results" / "sklearn_tuning.json").read_text(encoding="utf-8"))
    assert tuning["best_model"] == best
    best_pipe = joblib.load(tmp_path / "models" / "best_sklearn.joblib")
    assert type(best_pipe.named_steps["model"]) is type(
        joblib.load(tmp_path / "models" / f"sklearn_{best}.joblib").named_steps["model"]
    )
    assert "test" not in requested

    records = evaluate_sklearn_on_test(output_dir=tmp_path)
    assert requested.count("test") == 1
    assert all(r["split"] == "test" for r in records.values())
    assert (tmp_path / "results" / f"sklearn_{best}.json").exists()
    assert (tmp_path / "figures" / "confusion_matrix_sklearn.png").exists()


def test_training_code_has_no_test_split():
    """data_contract §5: the test split never appears in training/tuning code."""
    for path in ("src/models/train_sklearn.py", "src/features/tabular.py", "scripts/train_sklearn.py"):
        source = (ROOT / path).read_text(encoding="utf-8")
        assert '"test"' not in source and "'test'" not in source, path
