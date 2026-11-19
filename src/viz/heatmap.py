"""Heatmap with a separate colormap and value range per column."""

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure

from .palette import (
    DEFAULT_PALETTE,
    ColumnSpec,
    PaletteSpec,
    cmap_slice,
    diverging_from_white,
    resolve_cmap,
    resolve_item_cmaps,
    sample_column_cmaps,
    white_to_color_cmap,
)
from .utils import best_text_color, get_fig_ax

__all__ = [
    "ColumnSpec",
    "PaletteSpec",
    "DEFAULT_PALETTE",
    "white_to_color_cmap",
    "diverging_from_white",
    "cmap_slice",
    "resolve_cmap",
    "sample_column_cmaps",
    "per_column_heatmap",
]


def per_column_heatmap(
    data: pd.DataFrame,
    *,
    columns: Optional[Dict[str, dict]] = None,
    cmap: Optional[PaletteSpec] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fmt: str = ".2f",
    annot: bool = True,
    text_kws: Optional[dict] = None,
    cbar: bool = True,
    cbar_location: str = "bottom",
    cbar_pad: float = 0.06,
    cbar_height: float = 0.06,
    cbar_xpad: float = 0.08,
    column_title: bool = False,
    column_title_pad: float = 0.03,
    column_title_rotation: float = 0,
    column_title_kws: Optional[dict] = None,
    ax: Optional[Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    linewidths: float = 0.5,
    linecolor: str = "white",
) -> Tuple[Figure, Axes, List[Axes]]:
    """Draw a heatmap where every column has its own colormap and range.

    Unlike `sns.heatmap`, colors are not comparable across columns: each column is
    normalized to its own `vmin`/`vmax` and mapped through its own colormap, which
    suits columns holding metrics on different scales.

    Args:
        data: DataFrame to plot; index becomes row labels, columns become column labels.
        columns: Optional per-column overrides, e.g.
            `{"inconsistency": dict(cmap="Reds", vmin=0, fmt=".3f", label="Inconsistency (ms)")}`.
            Each value may set `cmap`, `vmin`, `vmax`, `fmt`, `label` (the displayed
            column name), `diverging` and/or `reverse` (see `resolve_item_cmaps`).
            Unset keys fall back to the global arguments below, or the raw column
            name for `label`.
        cmap: Global default colormap/palette (name, `Colormap`, or a sequence of
            colors) for columns without their own `cmap`. One color is sampled per
            column and ramped from white to that color. Defaults to `DEFAULT_PALETTE`.
        vmin: Global default lower bound; defaults to each column's minimum.
        vmax: Global default upper bound; defaults to each column's maximum.
        fmt: Global default format spec for cell annotations.
        annot: Whether to write the value in each cell.
        text_kws: Extra kwargs passed to `ax.text` for annotations.
        cbar: Whether to draw a small colorbar under/above each column.
        cbar_location: `"bottom"` or `"top"`.
        cbar_pad: Gap between the heatmap and the per-column colorbars, in
            axes fraction of the heatmap height.
        cbar_height: Height of the per-column colorbars, in axes fraction of
            the heatmap height.
        cbar_xpad: Horizontal gap on each side of a per-column colorbar, as a
            fraction of that column's width, so neighboring colorbars don't touch.
        column_title: If True, replace the bottom x-tick labels with a bold title row
            above the heatmap (above the top colorbar, if `cbar_location="top"`).
        column_title_pad: Gap between the heatmap (or its top colorbar) and
            the title row, in axes fraction of the heatmap height.
        column_title_rotation: Rotation of the title-row text, in degrees.
        column_title_kws: Extra kwargs passed to `ax.text` for the title row.
        ax: Axes to draw the heatmap into. A new figure/axes is created if omitted.
        figsize: Figure size, used only when `ax` is None.
        linewidths: Width of the grid lines between cells.
        linecolor: Color of the grid lines between cells.

    Returns:
        (fig, ax, cbar_axes), where `cbar_axes` lists the per-column colorbar axes
        (empty if `cbar=False`).
    """
    column_names = list(data.columns)
    n_rows, n_cols = data.shape
    columns = columns or {}

    def _column_override(col: str, key: str):
        return columns.get(col, {}).get(key)

    col_cmaps = resolve_item_cmaps(column_names, cmap=cmap, overrides=columns)

    col_vmin = {
        col: (
            v
            if (v := _column_override(col, "vmin")) is not None
            else (vmin if vmin is not None else data[col].min())
        )
        for col in column_names
    }
    col_vmax = {
        col: (
            v
            if (v := _column_override(col, "vmax")) is not None
            else (vmax if vmax is not None else data[col].max())
        )
        for col in column_names
    }
    norms = {
        col: Normalize(vmin=col_vmin[col], vmax=col_vmax[col]) for col in column_names
    }
    col_fmt = {
        col: (v if (v := _column_override(col, "fmt")) is not None else fmt)
        for col in column_names
    }
    col_labels = {
        col: (v if (v := _column_override(col, "label")) is not None else col)
        for col in column_names
    }

    if ax is None and figsize is None:
        figsize = (n_cols * 1.2, n_rows * 0.6)
    fig, ax = get_fig_ax(ax, figsize)

    text_kws = {"ha": "center", "va": "center", "fontsize": 9, **(text_kws or {})}

    for j, col in enumerate(column_names):
        norm = norms[col]
        cmap = col_cmaps[col]
        for i in range(n_rows):
            value = data.iloc[i, j]
            color = cmap(norm(value)) if pd.notna(value) else (1, 1, 1, 1)
            ax.add_patch(
                plt.Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=color,
                    edgecolor=linecolor,
                    linewidth=linewidths,
                )
            )
            if annot and pd.notna(value):
                text_color = best_text_color(color)
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    format(value, col_fmt[col]),
                    color=text_color,
                    **text_kws,
                )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    if column_title:
        ax.set_xticks([])
    else:
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_xticklabels([col_labels[col] for col in column_names])
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(data.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar_axes: List[Axes] = []
    if cbar:
        cbar_axes = _add_column_colorbars(
            ax,
            column_names,
            norms,
            col_cmaps,
            location=cbar_location,
            pad=cbar_pad,
            height=cbar_height,
            xpad=cbar_xpad,
        )

    if column_title:
        title_y = 1 + column_title_pad
        if cbar and cbar_location == "top":
            # Leave room for the colorbar itself plus its tick labels above it.
            title_y += cbar_pad + cbar_height + column_title_pad
        title_kws = {
            "ha": "left",
            "va": "bottom",
            "fontweight": "bold",
            "fontsize": 9,
            "rotation": column_title_rotation,
            **(column_title_kws or {}),
        }
        for j, col in enumerate(column_names):
            ax.text(
                j,
                title_y,
                col_labels[col],
                transform=ax.get_xaxis_transform(),
                **title_kws,
            )

    return fig, ax, cbar_axes


def _add_column_colorbars(
    ax: Axes,
    columns: Sequence[str],
    norms: Dict[str, Normalize],
    cmaps: Dict[str, Colormap],
    *,
    location: str,
    pad: float,
    height: float,
    xpad: float = 0.0,
) -> List[Axes]:
    """Add one thin horizontal colorbar per column, aligned to that column's width."""
    n_cols = len(columns)
    col_width = 1 / n_cols
    x_inset = col_width * xpad
    cbar_axes = []
    for j, col in enumerate(columns):
        sub_ax = ax.inset_axes(
            (
                j * col_width + x_inset,
                -pad - height if location == "bottom" else 1 + pad,
                col_width - 2 * x_inset,
                height,
            ),
            transform=ax.transAxes,
        )
        fig = ax.figure
        norm = norms[col]
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmaps[col])
        fig.colorbar(mappable, cax=sub_ax, orientation="horizontal")
        sub_ax.xaxis.set_ticks_position(location)
        sub_ax.xaxis.set_label_position(location)
        sub_ax.set_xticks([norm.vmin, norm.vmax])
        sub_ax.tick_params(labelsize=7, length=2)
        tick_labels = sub_ax.get_xticklabels()
        tick_labels[0].set_ha("left")
        tick_labels[-1].set_ha("right")
        cbar_axes.append(sub_ax)
    return cbar_axes
