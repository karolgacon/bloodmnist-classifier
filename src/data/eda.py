"""EDA: dataset statistics and figure generation. Implemented by O1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import FIGURES_DIR, NUM_CLASSES, RESULTS_DIR
from src.data.load import load_numpy_split

STATS_PATH = RESULTS_DIR / "dataset_stats.json"


def _border_pixels(images: np.ndarray) -> np.ndarray:
    """All pixels on the 1-px frame of each image, shape (M, 3)."""
    frame = np.concatenate(
        [images[:, 0, :], images[:, -1, :], images[:, 1:-1, 0], images[:, 1:-1, -1]],
        axis=1,
    )
    return frame.reshape(-1, 3)


def compute_dataset_stats(download: bool = True) -> dict:
    """Normalization stats (train only) and class counts for every split.

    mean/std are per-channel over all train pixels scaled to [0, 1] (as after
    ToTensor). background_rgb_uint8 is the median train border colour, used as
    the rotation fill in augmentation.
    """
    import medmnist

    train_images, _ = load_numpy_split("train", download=download)
    pixels = train_images.reshape(-1, 3).astype(np.float64) / 255.0

    class_counts = {}
    n_samples = {}
    for split in ("train", "val", "test"):
        _, labels = load_numpy_split(split, download=download)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        class_counts[split] = {str(c): int(n) for c, n in enumerate(counts)}
        n_samples[split] = int(labels.shape[0])

    return {
        "medmnist_version": medmnist.__version__,
        "computed_on": "train",
        "mean_rgb": [round(float(v), 6) for v in pixels.mean(axis=0)],
        "std_rgb": [round(float(v), 6) for v in pixels.std(axis=0)],
        "background_rgb_uint8": [
            int(v) for v in np.median(_border_pixels(train_images), axis=0).round()
        ],
        "n_samples": n_samples,
        "class_counts": class_counts,
    }


def write_dataset_stats(path: Optional[Path] = None, download: bool = True) -> dict:
    """Compute stats and save them as JSON (default reports/results/dataset_stats.json)."""
    path = path or STATS_PATH
    stats = compute_dataset_stats(download=download)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def run_eda(output_dir: Optional[Path] = None) -> None:
    """Generate EDA figures into reports/figures/."""
    _ = output_dir or FIGURES_DIR
    raise NotImplementedError("O1: implement EDA plots")
