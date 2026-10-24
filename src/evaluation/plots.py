"""Confusion matrix and model comparison plots (O2, shared with O3).

Same visual style as the EDA figures (src/data/eda.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.config import CLASS_NAMES_PL, NUM_CLASSES
from src.data.eda import BAR_COLOR, INK, INK_SECONDARY, _save, _style_axes


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    normalize: bool = True,
    class_names: Optional[dict] = None,
    title: Optional[str] = None,
) -> Path:
    """Confusion matrix heatmap (rows = true class, columns = predicted).

    normalize=True: cells show % of the true class (row) — recall on the diagonal —
    with raw counts underneath; the colour scale is the row share in both modes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    names = class_names or CLASS_NAMES_PL
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), labels=labels)
    row_share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    for i in labels:
        for j in labels:
            if cm[i, j] == 0:
                continue
            text = f"{row_share[i, j]:.0%}\n({cm[i, j]})" if normalize else str(cm[i, j])
            ax.text(
                j, i, text, ha="center", va="center", fontsize=7,
                color="white" if row_share[i, j] > 0.5 else INK,
            )
    ticks = [names[c] for c in labels]
    ax.set_xticks(labels)
    ax.set_xticklabels(ticks, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(labels)
    ax.set_yticklabels(ticks, fontsize=9)
    ax.set_xlabel("Klasa przewidziana", color=INK)
    ax.set_ylabel("Klasa prawdziwa", color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, labelcolor=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Udział klasy prawdziwej", color=INK_SECONDARY)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    if title:
        ax.set_title(title, color=INK, loc="left")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return _save(fig, Path(output_path))


def plot_metric_comparison(
    scores: dict[str, float],
    output_path: Path,
    errors: Optional[dict[str, float]] = None,
    xlabel: str = "Macro F1",
    title: Optional[str] = None,
) -> Path:
    """Horizontal bars, one per model, sorted best-on-top; optional ± error (e.g. CV std)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(scores, key=scores.get)
    values = [scores[m] for m in order]
    xerr = [errors.get(m, 0.0) for m in order] if errors else None

    fig, ax = plt.subplots(figsize=(7, 0.55 * len(order) + 1.2))
    ax.barh(
        order, values, xerr=xerr, color=BAR_COLOR, height=0.6,
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 3},
    )
    for y, v in enumerate(values):
        ax.text(v + 0.015, y, f"{v:.3f}", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(xlabel, color=INK)
    if title:
        ax.set_title(title, color=INK, loc="left")
    _style_axes(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return _save(fig, Path(output_path))
