"""Duration-density plots per label, stacked or mirrored around a shared baseline."""

from collections import deque
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .utils import get_label_colors, get_ax, get_fig_ax, resolve


def plot_density(durations, color, ax=None):
    """KDE plus rug plot of durations (in seconds), drawn in milliseconds."""
    ax = get_ax(ax)
    plt.sca(ax)
    sns.kdeplot(
        durations * 1000,
        color=color,
        # bw_adjust=1/2,
        fill=True,
        lw=0,
        alpha=1,
        cut=1,
        common_norm=False,
    )
    sns.rugplot(durations * 1000, color="k", height=0.25, linewidth=1)
    sns.despine(left=True)
    ax.set_ymargin(0)
    plt.yticks([])
    plt.ylabel("")
    plt.xlabel("")


def show_density(
    data: Optional[pd.DataFrame] = None,
    *,
    label: Any = "label",
    duration: Any = "duration",
    height_ratio: float = 0.5,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, np.ndarray]:
    """One duration-density panel per label for a single bird, below an "all" panel.

    Args:
        data: DataFrame to resolve `label`/`duration` from, when those are
            column names. Omit and pass array-likes directly to use this
            standalone (no fixed dataframe schema required).
        label: Column name (resolved against `data`) or array-like of labels.
        duration: Column name (resolved against `data`) or array-like of durations
            in seconds.
        height_ratio: Figure height per label, in inches (minimum height 3).
        figsize: Optional explicit figure size.

    Returns:
        (fig, axs), with one axes per panel.
    """
    labels = np.asarray(resolve(data, label))
    durations = np.asarray(resolve(data, duration))
    unique_labels = sorted(set(labels), key=str)
    label_colors = get_label_colors(unique_labels)

    if figsize is None:
        figsize = (5, max(len(unique_labels) * height_ratio, 3))
    fig, axs = plt.subplots(nrows=len(unique_labels) + 1, figsize=figsize, sharex=True)

    plot_density(durations, ".8", ax=axs[0])
    axs[0].set_ylabel("all", rotation=0, fontweight="bold", va="center", ha="left")

    for i, lbl in enumerate(unique_labels):
        subset = durations[labels == lbl]
        color = label_colors[lbl]
        plot_density(subset, color, ax=axs[i + 1])
        axs[i + 1].set_ylabel(
            lbl, rotation=0, fontweight="bold", va="center", ha="left", fontsize=8
        )

    axs[-1].set_xlabel("duration (ms)")
    fig.supylabel("label", fontsize=8)
    plt.tight_layout()
    return fig, axs


def _bhattacharyya(mean1: float, std1: float, mean2: float, std2: float) -> float:
    """Bhattacharyya coefficient for two univariate Gaussians, in [0, 1]."""
    s = std1**2 + std2**2
    return float(
        np.sqrt(2 * std1 * std2 / s) * np.exp(-0.25 * (mean1 - mean2) ** 2 / s)
    )


def _assign_sides(means: np.ndarray, stds: np.ndarray, threshold: float) -> np.ndarray:
    """BFS 2-coloring of the overlap graph: returns +1 (above) or -1 (below) per label.

    Overlapping pairs (Bhattacharyya coefficient > threshold) are placed on opposite
    sides. For non-bipartite graphs (odd overlap cycles) the assignment is best-effort.
    """
    n = len(means)
    adj = [
        [
            _bhattacharyya(means[i], stds[i], means[j], stds[j]) > threshold
            for j in range(n)
        ]
        for i in range(n)
    ]
    colors = [-1] * n
    for start in range(n):
        if colors[start] != -1:
            continue
        colors[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in range(n):
                if adj[node][neighbor]:
                    if colors[neighbor] == -1:
                        colors[neighbor] = 1 - colors[node]
                        queue.append(neighbor)
    return np.where(np.array(colors) == 0, 1, -1)


def plot_bilateral_densities(
    data: Optional[pd.DataFrame] = None,
    *,
    label: Any = "label",
    duration: Any = "duration",
    ax: Optional[Axes] = None,
    overlap_threshold: float = 0.3,
    bw_adjust: float = 1.0,
    cut: float = 3,
    alpha: float = 0.3,
    lw: float = 1,
    show_xticklabels: bool = False,
    show_centered_axis: bool = False,
    normalize_height: bool = False,
    show_labels: bool = False,
    colors: Optional[dict] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Axes]:
    """Draw per-label duration KDEs on a shared axis, flipping overlapping ones below.

    Overlap is measured with the Bhattacharyya coefficient on Gaussian fits to each
    label's durations. The resulting overlap graph is 2-colored via BFS so that
    overlapping distributions end up on opposite sides of the baseline. The
    KDEs match `sns.kdeplot` (`scipy.stats.gaussian_kde`). The y-axis is
    symmetric and the baseline sits at y=0.

    Args:
        data: DataFrame to resolve `label`/`duration` from, when those are
            column names. Omit and pass array-likes directly to use this
            standalone (no fixed dataframe schema required).
        label: Column name (resolved against `data`) or array-like of labels.
        duration: Column name (resolved against `data`) or array-like of durations.
        ax: Axes to draw on; a new figure is created when omitted.
        overlap_threshold: Bhattacharyya coefficient above which two distributions
            are considered overlapping (0 = never flip, 1 = always flip).
        bw_adjust: Multiplier for the automatic (Scott) KDE bandwidth.
        cut: How many bandwidths the density extends past the data.
        alpha: Fill opacity.
        lw: Outline linewidth.
        show_xticklabels: Show x-tick labels and the "duration" axis label.
        show_centered_axis: Use the bottom spine as the baseline at y=0, instead of
            drawing a separate horizontal line.
        normalize_height: Scale all KDEs to the same peak height and hide y-ticks.
        show_labels: Annotate each label's name at its KDE peak.
        colors: Mapping from label to color; defaults to `get_label_colors`.
        figsize: Figure size when `ax` is None.

    Returns:
        (fig, ax).
    """
    labels = np.asarray(resolve(data, label))
    durations = np.asarray(resolve(data, duration))
    unique_labels = sorted(set(labels), key=str)
    label_colors = colors if colors is not None else get_label_colors(unique_labels)

    groups = {lbl: durations[labels == lbl] for lbl in unique_labels}

    means = np.array([d.mean() for d in groups.values()])
    stds = np.array([d.std() for d in groups.values()])
    sides = _assign_sides(means, stds, overlap_threshold)

    fig, ax = get_fig_ax(ax, figsize or (6, 2))
    plt.sca(ax)

    # One KDE per label, as in `sns.kdeplot`: no density for fewer than two
    # values or zero variance; the grid extends `cut` bandwidths past the data.
    curves = {}
    for lbl in unique_labels:
        d = groups[lbl][np.isfinite(groups[lbl])]
        if len(d) < 2 or np.var(d) == 0:
            continue
        kde = gaussian_kde(d)
        kde.set_bandwidth(kde.factor * bw_adjust)
        bw = np.sqrt(kde.covariance.squeeze())
        x = np.linspace(d.min() - bw * cut, d.max() + bw * cut, 200)
        curves[lbl] = (x, kde(x))
    y_max = max((y.max() for _, y in curves.values()), default=0.0)

    for side, lbl in zip(sides, unique_labels):
        if lbl not in curves:
            continue
        x, y = curves[lbl]
        if normalize_height:
            y = y * y_max / y.max()
        y = side * y
        color = label_colors[lbl]
        ax.fill_between(
            x,
            0,
            y,
            facecolor=to_rgba(color, alpha),
            edgecolor=to_rgba(color, 1),
            linewidth=lw,
            label=lbl,
        )

        if show_labels:
            peak = np.argmax(np.abs(y))
            ax.annotate(
                lbl,
                xy=(x[peak], y[peak]),
                xytext=(0, 2 * side),
                textcoords="offset points",
                ha="center",
                va="bottom" if side == 1 else "top",
                fontsize=8,
                color=color,
                clip_on=False,
            )

    if not ax.get_ylabel():
        ax.set_ylabel("Density")
    sns.despine(ax=ax)
    if show_labels:
        y_margin = 1.8
    elif normalize_height:
        y_margin = 1.2
    else:
        y_margin = 1.05
    ax.set_ylim(-y_max * y_margin, y_max * y_margin)
    if normalize_height:
        ax.set_yticks([])
    else:
        # Symmetric ticks; flipped densities are labelled with positive values
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}"))

    if show_centered_axis:
        ax.spines["bottom"].set_position(("data", 0))
    else:
        ax.axhline(0, color="k", lw=0.8, zorder=1)

    if show_xticklabels:
        ax.set_xlabel("duration")
    else:
        ax.tick_params(axis="x", bottom=True, labelbottom=False)
        ax.set_xlabel("")

    return fig, ax
