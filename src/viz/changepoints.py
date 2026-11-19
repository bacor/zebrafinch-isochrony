"""Plots of a changepoint-detection result, used for bird identification.

There is a per-segment timeline, the changepoint score signal, and a figure
stacking both with a shared color scale for the peak prominences. The
functions take plain arrays (positions, scores, breakpoints, prominences), so
they work with any changepoint-detection method.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from matplotlib.transforms import blended_transform_factory

from .timeline import plot_slice
from .palette import cmap_slice
from .utils import get_fig_ax, resolve

# Confidence colors: plasma without its light tail.
CONFIDENCE_CMAP = cmap_slice("plasma", 0.0, 0.75)

# A checkmark drawn as a path, since "✓" is missing from the Arial font we use.
_CHECKMARK = Path(
    vertices=[(-0.45, 0.05), (-0.1, -0.45), (0.5, 0.5)],
    codes=[Path.MOVETO, Path.LINETO, Path.LINETO],
)


def plot_changepoint_timeline(
    breakpoints,
    segment_confidence,
    n_items: int,
    *,
    ax: Optional[Axes] = None,
    cmap: Optional[Colormap] = None,
    norm: Optional[Normalize] = None,
    true_breakpoints=None,
    tolerance: Optional[int] = None,
) -> Axes:
    """Timeline of segments, one colored+labeled span per segment (see `plot_slice`).

    Args:
        breakpoints: 0-based indices where a new segment starts.
        segment_confidence: Confidence per segment (length `len(breakpoints) + 1`),
            colors each span.
        n_items: Total sequence length, to size the first/last segment's span.
        ax: Axes to draw on; a new figure is created when omitted.
        cmap: Defaults to `plasma` with its light tail dropped (`CONFIDENCE_CMAP`),
            since a bright-yellow endpoint reads poorly as "high confidence."
        norm: Value range the colors are scaled against; defaults to
            `Normalize(0, segment_confidence.max())`. Pass the same `norm` used
            elsewhere (e.g. by `plot_changepoint_scores`) to keep colors comparable
            across panels.
        true_breakpoints: Optional known-true breakpoints, e.g. for validating against
            ground truth. Drawn as a row above the timeline: a black checkmark where a
            breakpoint lies within `tolerance`, and a red cross where none does.
        tolerance: How close (in the same index units as `breakpoints`) counts as a
            match for `true_breakpoints`. Required when `true_breakpoints` is given.

    Returns:
        The axes drawn on.
    """
    if true_breakpoints is not None and tolerance is None:
        raise ValueError("pass `tolerance` when giving `true_breakpoints`")

    breakpoints = np.asarray(breakpoints)
    segment_confidence = np.asarray(segment_confidence)
    if cmap is None:
        cmap = CONFIDENCE_CMAP
    if norm is None:
        norm = plt.Normalize(0, segment_confidence.max())

    _, ax = get_fig_ax(ax)
    segment_bounds = np.concatenate([[0], breakpoints, [n_items]])

    plt.sca(ax)
    for i, conf in enumerate(segment_confidence):
        start, end = segment_bounds[i], segment_bounds[i + 1]
        plot_slice(
            start,
            end - start,
            y=0,
            color=cmap(norm(conf)),
            label=str(i + 1),
            text_dx_pt=2,
            bar_height=0.6,
        )

    if true_breakpoints is not None:
        true_breakpoints = np.asarray(true_breakpoints)
        matched = np.array(
            [np.any(np.abs(breakpoints - tb) <= tolerance) for tb in true_breakpoints]
        )
        dot_y = 0.75
        ax.scatter(
            true_breakpoints[matched],
            np.full(matched.sum(), dot_y),
            marker=_CHECKMARK,
            s=25,
            facecolors="none",
            edgecolors="black",
            linewidths=1.3,
            zorder=4,
            clip_on=False,
        )
        ax.scatter(
            true_breakpoints[~matched],
            np.full((~matched).sum(), dot_y),
            color="crimson",
            marker="x",
            s=30,
            linewidths=1.5,
            zorder=4,
            clip_on=False,
        )
        row_label = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            -0.01, 0, "pred", transform=row_label, ha="right", va="center", fontsize=8
        )
        ax.text(
            -0.01,
            dot_y,
            "true",
            transform=row_label,
            ha="right",
            va="center",
            fontsize=8,
        )
        ax.set_ylim(-1, 1.3)
    else:
        ax.set_ylim(-1, 1)
    ax.axis("off")

    return ax


def plot_changepoint_scores(
    breakpoints,
    positions,
    scores,
    prominences,
    *,
    ax: Optional[Axes] = None,
    cmap: Optional[Colormap] = None,
    norm: Optional[Normalize] = None,
    xlabel: str = "sequence index",
    ylabel: str = "changepoint score",
) -> Tuple[Axes, ScalarMappable]:
    """The changepoint score signal, with detected breakpoints marked by confidence.

    Args:
        breakpoints: 0-based indices where a new segment starts.
        positions: Candidate split points the score signal was evaluated at
            (as returned alongside `scores` by a changepoint-scoring function).
        scores: Changepoint score at each of `positions`.
        prominences: Confidence per breakpoint (same length/order as `breakpoints`),
            colors the markers/boundary lines.
        ax: Axes to draw on; a new figure is created when omitted.
        cmap: Defaults to `plasma` with its light tail dropped (`CONFIDENCE_CMAP`).
        norm: Value range the colors are scaled against; defaults to
            `Normalize(0, prominences.max())`. Pass the same `norm` used elsewhere
            (e.g. by `plot_changepoint_timeline`) to keep colors comparable across panels.
        xlabel, ylabel: Axis text.

    Returns:
        (ax, mappable), where `mappable` can be passed to `fig.colorbar`.
    """
    breakpoints = np.asarray(breakpoints)
    positions = np.asarray(positions)
    scores = np.asarray(scores)
    prominences = np.asarray(prominences)
    if cmap is None:
        cmap = CONFIDENCE_CMAP
    if norm is None:
        norm = plt.Normalize(0, prominences.max())

    _, ax = get_fig_ax(ax)
    breakpoint_scores = scores[breakpoints - positions[0]]

    ax.plot(positions, scores, lw=0.5, color="0.4")
    sc = ax.scatter(
        breakpoints,
        breakpoint_scores,
        c=prominences,
        cmap=cmap,
        norm=norm,
        s=20,
        zorder=3,
    )
    for bp, conf in zip(breakpoints, prominences):
        ax.axvline(bp, color=cmap(norm(conf)), lw=0.6, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)

    return ax, sc


def _segment_confidence(
    breakpoints: np.ndarray, per_boundary_value: np.ndarray
) -> np.ndarray:
    """Per-segment confidence: the weaker of its two bounding changepoints' value.

    Segment i is bounded by `breakpoints[i-1]` (left) and `breakpoints[i]`
    (right); the first/last segments only have one bounding changepoint.
    Low confidence means at least one edge of that segment was only weakly
    detected.
    """
    n_segments = len(breakpoints) + 1
    left = np.concatenate([[np.inf], per_boundary_value])
    right = np.concatenate([per_boundary_value, [np.inf]])
    return np.minimum(left, right)[:n_segments]


def plot_changepoints(
    data=None,
    n_items: Optional[int] = None,
    *,
    breakpoints="breakpoints",
    positions="positions",
    scores="scores",
    prominences="prominences",
    true_breakpoints=None,
    tolerance: Optional[int] = None,
    cmap: Optional[Colormap] = None,
    figsize: Tuple[float, float] = (8, 3.5),
    xlabel: str = "sequence index",
    ylabel: str = "changepoint score",
    colorbar_label: str = "peak prominence",
) -> Tuple[Figure, Axes, Axes]:
    """Changepoint timeline stacked above the score signal, with a shared colorbar.

    The two panels (`plot_changepoint_timeline`, `plot_changepoint_scores`) share an
    x-axis and a color scale for the peak prominences; a segment is colored by
    the weaker of its two bounding peaks. `breakpoints`, `positions`, `scores`,
    `prominences` and `true_breakpoints` each take either an array directly, or (seaborn-style) a key
    into `data`, so the result of `predict_bird_ids` can be passed as `data`:
    `plot_changepoints(result, n_chunks, true_breakpoints="true_breakpoints", tolerance=5)`.

    See `plot_changepoint_timeline`/`plot_changepoint_scores` for what each of those
    arrays means; `n_items`/`true_breakpoints`/`tolerance` go to
    `plot_changepoint_timeline`, `xlabel`/`ylabel` to `plot_changepoint_scores`.
    `colorbar_label` labels the shared colorbar.

    Returns:
        (fig, ax_timeline, ax_scores).
    """
    breakpoints = np.asarray(resolve(data, breakpoints))
    positions = np.asarray(resolve(data, positions))
    scores = np.asarray(resolve(data, scores))
    prominences = np.asarray(resolve(data, prominences))
    segment_confidence = _segment_confidence(breakpoints, prominences)
    if true_breakpoints is not None:
        true_breakpoints = resolve(data, true_breakpoints)

    if cmap is None:
        cmap = CONFIDENCE_CMAP
    norm = plt.Normalize(0, prominences.max())

    # A dedicated colorbar column (not fig.colorbar(..., ax=ax_scores)) so ax_timeline
    # and ax_scores share an identical width by construction: colorbar(ax=...) only
    # shrinks the axes it's given to make room, and sharex alone only locks x-limits,
    # not pixel width, so the two panels would otherwise end up misaligned.
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        2,
        2,
        height_ratios=[1, 5],
        width_ratios=[30, 1],
        hspace=0.05,
        wspace=0.05,
        figure=fig,
    )
    ax_timeline = fig.add_subplot(gs[0, 0])
    ax_scores = fig.add_subplot(gs[1, 0], sharex=ax_timeline)
    cax = fig.add_subplot(gs[1, 1])

    plot_changepoint_timeline(
        breakpoints,
        segment_confidence,
        n_items,
        ax=ax_timeline,
        cmap=cmap,
        norm=norm,
        true_breakpoints=true_breakpoints,
        tolerance=tolerance,
    )
    _, sc = plot_changepoint_scores(
        breakpoints,
        positions,
        scores,
        prominences,
        ax=ax_scores,
        cmap=cmap,
        norm=norm,
        xlabel=xlabel,
        ylabel=ylabel,
    )
    ax_scores.set_xlim(0, n_items)

    fig.colorbar(sc, cax=cax, label=colorbar_label)

    return fig, ax_timeline, ax_scores
