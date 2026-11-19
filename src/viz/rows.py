"""Figures with one row of panels per condition, grouped into titled sections.

A layout is a list of row specs: `('header', title, None)` opens a
section, followed by one `('data', row_label, row)` per row in it, where
`row` is whatever the caller's plotting code needs. Header rows are kept
as (thin) axes so the sections and their row spans can be read off the
layout itself.
"""

from typing import Callable, Sequence

import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


def flatten_sections(sections: Sequence[dict], row_label: Callable) -> list:
    """Row specs for `sections`, labeling each data row by `row_label(row)`.

    Each section is a dict with keys 'title' and 'rows'.
    """
    row_specs = []
    for section in sections:
        row_specs.append(("header", section["title"], None))
        for row in section["rows"]:
            row_specs.append(("data", row_label(row), row))
    return row_specs


def row_height_ratios(row_specs, header_ratio=0.4, data_ratio=1.0):
    """Per-row `height_ratios` for `plt.subplots`, with shorter header rows.

    Use a small `header_ratio` (e.g. 0.12) when the section titles are
    drawn in the margin by `annotate_row_groups`.
    """
    return [
        header_ratio if kind == "header" else data_ratio for kind, _, _ in row_specs
    ]


def annotate_row_groups(fig, axes, row_specs, x=0.015, fontsize=8, **line_kws):
    """Draw section titles rotated in the left margin, with dividers between sections.

    Call after `fig.tight_layout()`, so the axes positions are final. `x`
    is a figure fraction and should stay within [0, 1]: a title outside the
    canvas widens figures saved with `bbox_inches='tight'`.
    """
    line_kws = {"color": ".7", "lw": 0.8, **line_kws}
    header_idxs = [i for i, (kind, _, _) in enumerate(row_specs) if kind == "header"]
    for j, h in enumerate(header_idxs):
        title = row_specs[h][1]
        start = h + 1
        end = (
            (header_idxs[j + 1] - 1) if j + 1 < len(header_idxs) else len(row_specs) - 1
        )
        top = axes[start].get_position().y1
        bottom = axes[end].get_position().y0
        fig.text(
            x,
            (top + bottom) / 2,
            title,
            rotation=90,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            transform=fig.transFigure,
        )
        if j > 0:
            header_pos = axes[h].get_position()
            sep_y = (header_pos.y0 + header_pos.y1) / 2
            fig.add_artist(
                Line2D(
                    [0.0, 1.0], [sep_y, sep_y], transform=fig.transFigure, **line_kws
                )
            )


def clean_density_yaxis(ax, n_ticks=5, color="0.7"):
    """A minimal, light y-axis for a small density panel.

    Integer ticks from 0 to the current top y-limit, labeled only at the
    bottom and top, and no axis label. Call after the data and any
    `set_ylim` are in place.
    """
    top = ax.get_ylim()[1]
    locator = mticker.MaxNLocator(nbins=n_ticks - 1, integer=True, min_n_ticks=2)
    ticks = [t for t in locator.tick_values(0, top) if 0 <= t <= top] or [0]

    ax.set_yticks(ticks)
    labels = (
        ["0"]
        if len(ticks) == 1
        else ["0", *([""] * (len(ticks) - 2)), f"{ticks[-1]:.0f}"]
    )
    ax.set_yticklabels(labels)
    ax.set_ylabel(None)

    ax.spines["left"].set_color(color)
    ax.tick_params(axis="y", colors=color)
