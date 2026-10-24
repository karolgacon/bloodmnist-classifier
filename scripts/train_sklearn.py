#!/usr/bin/env python3
"""Train classical ML models (O2) — train and val splits only.

Usage:
    python scripts/train_sklearn.py                      # tune + train all, metrics on val
    python scripts/train_sklearn.py --models svm_rbf     # subset
    python scripts/train_sklearn.py --no-tune            # default hyperparameters (quick check)

Test set (D20 only): python scripts/evaluate_sklearn_test.py
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SKLEARN_CV_FOLDS, SKLEARN_RANDOM_SEARCH_ITER
from src.models.train_sklearn import MODEL_NAMES, train_all_sklearn_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--no-tune", action="store_true", help="skip RandomizedSearchCV")
    parser.add_argument("--n-iter", type=int, default=SKLEARN_RANDOM_SEARCH_ITER)
    parser.add_argument("--cv-folds", type=int, default=SKLEARN_CV_FOLDS)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    train_all_sklearn_models(
        tune=not args.no_tune,
        models=args.models,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
