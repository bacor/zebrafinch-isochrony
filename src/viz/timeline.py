"""Timeline plots of chunks: one row per chunk, one colored bar per unit."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .utils import get_label_colors, best_text_color, get_ax, get_fig_ax, resolve


def plot_slice(
    start: float,
    duration: float,
    y: float,
    color: str,
    label: str,
    left_dash: bool = True,
    bar_height: float = 0.6,
    text_dx_pt: float = 2,
    text_dy_pt: float = 0,
) -> None:
    """Draw one labeled bar from `start` to `start + duration` on the current axes."""
    ax = get_ax()
    y0 = y - bar_height / 2

    rect = mpatches.Rectangle(
        (start, y0),
        duration,
        bar_height,
        facecolor=color,
        edgecolor="none",
        snap=True,
    )
    ax.add_patch(rect)

    if left_dash:
        ax.vlines(
            start,
            y0,
            y0 + bar_height,
            colors="w",
            linewidth=1,
            linestyles=":",
            zorder=rect.get_zorder() + 1,
        )

    if label:
        ax.annotate(
            label,
            xy=(start, y),
            xytext=(text_dx_pt, text_dy_pt),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=best_text_color(color),
            clip_on=True,
        )


def setup_label_colors(
    data: Optional[pd.DataFrame] = None,
    *,
    label: Any = "label",
    colors: Optional[dict] = None,
) -> dict:
    """Return `colors` if given, else a color per label from `get_label_colors`."""
    if colors is not None:
        return colors
    return get_label_colors(resolve(data, label))


def _select_chunks(
    events: pd.DataFrame,
    max_chunks: int,
    gap_size: int = 2,
    rows: Optional[Sequence] = None,
):
    """Return (chunks, y_positions, gap_y), with `gap_size` empty rows when truncated.

    `rows` overrides which chunks get a row (default: those in `events`).
    """
    chunks = list(rows if rows is not None else events["chunk_idx"].unique())[::-1]
    if len(chunks) <= max_chunks:
        return chunks, list(range(len(chunks))), None
    mid = round(max_chunks / 2)
    chunks = chunks[:mid] + chunks[-mid:]
    # Bottom half at 0..mid-1, top half shifted up by gap_size to create empty rows.
    y_positions = list(range(mid)) + list(range(mid + gap_size, mid + gap_size + mid))
    gap_y = mid + (gap_size - 1) / 2  # centre of the empty band
    return chunks, y_positions, gap_y


def _draw_gap_line(ax: Axes, gap_y: float) -> None:
    ax.axhline(gap_y, color="k", lw=0.8, linestyle=(0, (4, 3)), zorder=0)


def _draw_chunk_bars(
    events: pd.DataFrame,
    chunks: list,
    y_positions: list,
    label_colors: dict,
    ax: Axes,
    align_beginnings: bool,
    bar_height: float,
    show_inner_labels: bool,
) -> Dict[str, List[float]]:
    """Draw all chunk bars onto `ax`; return {label: [center_x of each unit]}."""
    plt.sca(ax)
    all_positions: Dict[str, List[float]] = {}

    for y, chunk_idx in zip(y_positions, chunks):
        chunk_df = events[events["chunk_idx"] == chunk_idx]
        if chunk_df.empty:
            continue
        chunk_start = chunk_df.iloc[0]["start"] if align_beginnings else 0
        prev_end = -1000.0
        for _, row in chunk_df.iterrows():
            start = row["start"] - chunk_start
            color = label_colors[row["label"]]
            plot_slice(
                start,
                row["duration"],
                y,
                color,
                label=row["label"] if show_inner_labels else "",
                bar_height=bar_height,
                left_dash=show_inner_labels and abs(start - prev_end) < 1e-6,
            )
            all_positions.setdefault(row["label"], []).append(
                start + row["duration"] / 2
            )
            prev_end = start + row["duration"]

    ax.autoscale_view()
    return all_positions


def _build_event_df(
    data: Optional[pd.DataFrame], chunk: Any, start: Any, duration: Any, label: Any
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chunk_idx": resolve(data, chunk),
            "start": resolve(data, start),
            "duration": resolve(data, duration),
            "label": resolve(data, label),
        }
    )


def plot_timeline(
    data: Optional[pd.DataFrame] = None,
    *,
    chunk: Any = "chunk_idx",
    start: Any = "start",
    duration: Any = "duration",
    label: Any = "label",
    align_beginnings: bool = False,
    ax: Optional[Axes] = None,
    bar_height: float = 0.6,
    height_ratio: float = 0.25,
    width_ratio: float = 0.6,
    figsize: Optional[Tuple[float, float]] = None,
    max_chunks: int = 15,
    colors: Optional[dict] = None,
    rows: Optional[Sequence] = None,
) -> Tuple[Figure, Axes]:
    """Plot a timeline of chunks, with each unit's label written inside its bar.

    Args:
        data: DataFrame to resolve `chunk`/`start`/`duration`/`label`
            from, when those are column names. Omit and pass array-likes
            directly to use this standalone (no fixed dataframe schema
            required).
        chunk, start, duration, label: Column name (resolved against
            `data`) or array-like, one value per unit.
        align_beginnings: Whether to align chunk starts to time 0.
        ax: Axes to draw on; a new figure is created when omitted.
        bar_height: Height of each bar in the timeline.
        height_ratio: Scale factor for auto figure height (unused when ax is given).
        width_ratio: Scale factor for auto figure width (unused when ax is given).
        figsize: Explicit figure size (unused when ax is given).
        max_chunks: Maximum chunks to display (keeps head and tail when exceeded).
        colors: Mapping from label to color; defaults to `get_label_colors`.
        rows: Chunk ids to give a row, in order (default: the chunks in
            the data). Pass the same `rows` to several timelines to line
            up their chunks; chunks without units get an empty row.

    Returns:
        (fig, ax).
    """
    events = _build_event_df(data, chunk, start, duration, label)
    chunks, y_positions, gap_y = _select_chunks(events, max_chunks, rows=rows)
    label_colors = setup_label_colors(events, colors=colors)

    if ax is None and figsize is None:
        figsize = (len(label_colors) * width_ratio, len(chunks) * height_ratio)
    fig, ax = get_fig_ax(ax, figsize)

    _draw_chunk_bars(
        events,
        chunks,
        y_positions,
        label_colors,
        ax,
        align_beginnings,
        bar_height,
        show_inner_labels=True,
    )
    if gap_y is not None:
        _draw_gap_line(ax, gap_y)

    sns.despine(ax=ax)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(chunks)
    ax.set_xlabel("time after start of chunk (ms)")
    ax.axvline(0, c="0.8", lw=1, zorder=-1)
    ax.set_ylabel("chunk number")

    return fig, ax


def plot_compact_timeline(
    data: Optional[pd.DataFrame] = None,
    *,
    chunk: Any = "chunk_idx",
    start: Any = "start",
    duration: Any = "duration",
    label: Any = "label",
    align_beginnings: bool = True,
    ax: Optional[Axes] = None,
    bar_height: float = 0.6,
    height_ratio: float = 0.25,
    width_ratio: float = 0.6,
    figsize: Optional[Tuple[float, float]] = None,
    max_chunks: int = 10,
    label_pad_pt: float = 3,
    show_labels: bool = True,
    colors: Optional[dict] = None,
    rows: Optional[Sequence] = None,
) -> Tuple[Figure, Axes]:
    """Plot a compact timeline with each label written once above the axes.

    Labels are not drawn inside bars; instead each unique label is annotated
    above the axes, centered at the mean position of its units across the shown
    chunks. Only two y-ticks (first and last chunk number) are shown.

    Args:
        data: DataFrame to resolve `chunk`/`start`/`duration`/`label`
            from, when those are column names. Omit and pass array-likes
            directly to use this standalone (no fixed dataframe schema
            required).
        chunk, start, duration, label: Column name (resolved against
            `data`) or array-like, one value per unit.
        align_beginnings: Whether to align chunk starts to time 0 (default True).
        ax: Axes to draw on; a new figure is created when omitted.
        bar_height: Height of each bar.
        height_ratio: Scale factor for auto figure height (unused when ax is given).
        width_ratio: Scale factor for auto figure width (unused when ax is given).
        figsize: Explicit figure size (unused when ax is given).
        max_chunks: Maximum chunks to display (keeps head and tail when exceeded).
        label_pad_pt: Vertical gap (points) between the top spine and above-axis labels.
        show_labels: Whether to draw the labels above the axes.
        colors: Mapping from label to color; defaults to `get_label_colors`.
        rows: Chunk ids to give a row, in order (default: the chunks in
            the data). Pass the same `rows` to several timelines to line
            up their chunks; chunks without units get an empty row.

    Returns:
        (fig, ax).
    """
    events = _build_event_df(data, chunk, start, duration, label)
    chunks, y_positions, gap_y = _select_chunks(events, max_chunks, rows=rows)
    label_colors = setup_label_colors(events, colors=colors)

    if ax is None and figsize is None:
        figsize = (len(label_colors) * width_ratio, len(chunks) * height_ratio)
    fig, ax = get_fig_ax(ax, figsize)

    top_positions = _draw_chunk_bars(
        events,
        chunks,
        y_positions,
        label_colors,
        ax,
        align_beginnings,
        bar_height,
        show_inner_labels=False,
    )
    if gap_y is not None:
        _draw_gap_line(ax, gap_y)

    if show_labels:
        pad_frac = label_pad_pt / 72 / fig.get_size_inches()[1]
        for lbl, cx_list in top_positions.items():
            cx = sum(cx_list) / len(cx_list)
            ax.text(
                cx,
                1.0 + pad_frac,
                lbl,
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
                clip_on=False,
                transform=ax.get_xaxis_transform(),
            )

    sns.despine(ax=ax)
    ax.set_yticks([y_positions[0], y_positions[-1]])
    ax.set_yticklabels([chunks[0], chunks[-1]])
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("chunk")

    return fig, ax
