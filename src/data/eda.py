"""EDA: dataset statistics and figure generation. Implemented by O1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import FIGURES_DIR, NUM_CLASSES, RANDOM_SEED, RESULTS_DIR
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


# --- Figures -----------------------------------------------------------------

DPI = 200
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BAR_COLOR = "#2a78d6"
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
CHANNEL_COLORS = ("#e34948", "#008300", "#2a78d6")

LABELS = {
    "pl": {
        "class_dist_title": "Rozkład klas — zbiór treningowy",
        "n_images": "Liczba obrazów",
        "grid_title": "Przykładowe obrazy każdej klasy — zbiór treningowy",
        "rgb_title": "Rozkład intensywności pikseli — zbiór treningowy",
        "intensity": "Intensywność piksela (0–255)",
        "density": "Gęstość",
        "channels": ("Kanał R", "Kanał G", "Kanał B"),
        "mean": "średnia",
        "proto_title": "Średni obraz klasy — zbiór treningowy",
        "split_title": "Udział klas w oficjalnych podziałach",
        "share": "Udział w podziale [%]",
        "splits": {"train": "treningowy", "val": "walidacyjny", "test": "testowy"},
    },
    "en": {
        "class_dist_title": "Class distribution — training set",
        "n_images": "Number of images",
        "grid_title": "Sample images per class — training set",
        "rgb_title": "Pixel intensity distribution — training set",
        "intensity": "Pixel intensity (0–255)",
        "density": "Density",
        "channels": ("R channel", "G channel", "B channel"),
        "mean": "mean",
        "proto_title": "Mean image per class — training set",
        "split_title": "Class share in the official splits",
        "share": "Share of split [%]",
        "splits": {"train": "train", "val": "validation", "test": "test"},
    },
}


def _style_axes(ax, grid_axis: str = "y") -> None:
    """Recessive chrome: hairline grid, no top/right spines."""
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelcolor=INK)


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def plot_class_distribution(labels: np.ndarray, names: dict, txt: dict, path: Path) -> Path:
    """Horizontal bars of train counts per class, sorted, with count and % labels."""
    import matplotlib.pyplot as plt

    counts = np.bincount(labels, minlength=NUM_CLASSES)
    order = np.argsort(counts)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        [names[c] for c in order], counts[order], color=BAR_COLOR, height=0.7
    )
    for bar, c in zip(bars, order):
        ax.text(
            bar.get_width() + total * 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{counts[c]}  ({counts[c] / total:.1%})",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax.set_xlabel(txt["n_images"], color=INK)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_title(txt["class_dist_title"], color=INK, loc="left")
    _style_axes(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return _save(fig, path)


def plot_sample_grid(
    images: np.ndarray, labels: np.ndarray, names: dict, txt: dict, path: Path, per_class: int = 8
) -> Path:
    """Rows = classes, columns = random train images (seeded)."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(RANDOM_SEED)
    cell, label_w, title_h = 0.9, 2.0, 0.5
    width = per_class * cell + label_w
    height = NUM_CLASSES * cell + title_h
    fig, axes = plt.subplots(NUM_CLASSES, per_class, figsize=(width, height))
    fig.subplots_adjust(
        left=label_w / width, right=0.995, bottom=0.005, top=1 - title_h / height,
        wspace=0.05, hspace=0.05,
    )
    for c in range(NUM_CLASSES):
        idx = rng.choice(np.flatnonzero(labels == c), size=per_class, replace=False)
        for j, i in enumerate(idx):
            ax = axes[c, j]
            ax.imshow(images[i], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[c, 0].set_ylabel(
            names[c], rotation=0, ha="right", va="center", fontsize=9, color=INK
        )
    fig.suptitle(txt["grid_title"], color=INK, x=0.02, y=1 - 0.1 / height, ha="left", va="top")
    return _save(fig, path)


def plot_rgb_histograms(images: np.ndarray, txt: dict, path: Path, y_max: float = 0.04) -> Path:
    """One panel per channel (small multiples) with the train mean marked.

    The y-axis is capped at y_max so the saturated 255 bin of the R channel
    (~15% of pixels, bright background) doesn't flatten the rest; clipped bins
    are labelled with their value.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ch, ax in enumerate(axes):
        values = images[..., ch].ravel()
        density = np.bincount(values, minlength=256) / values.size
        ax.fill_between(np.arange(256), density, step="mid", color=CHANNEL_COLORS[ch], alpha=0.25, linewidth=0)
        ax.step(np.arange(256), density, where="mid", color=CHANNEL_COLORS[ch], linewidth=1.5)
        for v in np.flatnonzero(density > y_max):
            ax.text(
                v - 3, y_max * 0.97, f"{v}: {density[v]:.1%} ↑",
                ha="right", va="top", fontsize=8, color=INK_SECONDARY,
            )
        mean = values.mean()
        ax.axvline(mean, ymax=0.85, color=INK_SECONDARY, linestyle="--", linewidth=1)
        ax.text(
            mean - 4, y_max * 0.55, f"{txt['mean']} = {mean:.0f}",
            ha="right", fontsize=8, color=INK_SECONDARY,
        )
        ax.set_ylim(0, y_max)
        ax.set_title(txt["channels"][ch], color=INK, fontsize=10)
        ax.set_xlim(0, 255)
        ax.set_xlabel(txt["intensity"], color=INK, fontsize=9)
        _style_axes(ax)
    axes[0].set_ylabel(txt["density"], color=INK)
    fig.suptitle(txt["rgb_title"], color=INK, x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, path)


def plot_class_prototypes(images: np.ndarray, labels: np.ndarray, names: dict, txt: dict, path: Path) -> Path:
    """Pixel-wise mean image of each class."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(8, 4.9))
    for c, ax in enumerate(axes.ravel()):
        proto = images[labels == c].mean(axis=0).round().astype(np.uint8)
        ax.imshow(proto, interpolation="nearest")
        ax.set_title(names[c], fontsize=9, color=INK)
        ax.axis("off")
    fig.suptitle(txt["proto_title"], color=INK, x=0.02, ha="left")
    fig.tight_layout()
    return _save(fig, path)


def plot_split_consistency(class_counts: dict, names: dict, txt: dict, path: Path) -> Path:
    """Grouped bars: % of each class within train / val / test."""
    import matplotlib.pyplot as plt

    x = np.arange(NUM_CLASSES)
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, split in enumerate(("train", "val", "test")):
        counts = np.array([class_counts[split][str(c)] for c in range(NUM_CLASSES)])
        share = counts / counts.sum() * 100
        ax.bar(
            x + (k - 1) * width, share, width=width * 0.92,
            color=SPLIT_COLORS[split], label=f"{txt['splits'][split]} (n={counts.sum()})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([names[c] for c in range(NUM_CLASSES)], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(txt["share"], color=INK)
    ax.set_title(txt["split_title"], color=INK, loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    _style_axes(ax)
    return _save(fig, path)


def run_eda(
    output_dir: Optional[Path] = None,
    lang: str = "pl",
    stats: Optional[dict] = None,
    download: bool = True,
) -> list:
    """Generate the 5 EDA figures into reports/figures/ (train images only).

    lang: "pl" (default, report) or "en". stats: output of compute_dataset_stats()
    (recomputed if None; needed for the split-consistency figure).
    """
    import matplotlib

    matplotlib.use("Agg")
    from src.config import CLASS_NAMES_EN, CLASS_NAMES_PL

    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    txt = LABELS[lang]
    names = CLASS_NAMES_PL if lang == "pl" else CLASS_NAMES_EN
    stats = stats or compute_dataset_stats(download=download)
    images, labels = load_numpy_split("train", download=download)

    return [
        plot_class_distribution(labels, names, txt, output_dir / "eda_class_distribution.png"),
        plot_sample_grid(images, labels, names, txt, output_dir / "eda_sample_grid.png"),
        plot_rgb_histograms(images, txt, output_dir / "eda_rgb_histogram.png"),
        plot_class_prototypes(images, labels, names, txt, output_dir / "eda_class_prototypes.png"),
        plot_split_consistency(stats["class_counts"], names, txt, output_dir / "eda_split_consistency.png"),
    ]
