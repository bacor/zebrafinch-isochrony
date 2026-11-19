"""Nested bars with a separate colormap per bar.

Draws several named horizontal bars stacked top-to-bottom, each shaded from
light to dark along its own colormap to show a sequence of nested fractions
(e.g. ">= k" fractions for increasing k, where a higher k's fraction is
already a subset of a lower k's, so plain stacking would double-count them).
Per-bar colors are resolved by `palette.resolve_item_cmaps`.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from .palette import PaletteSpec, resolve_item_cmaps

__all__ = ["nested_bar", "stacked_bars"]


def nested_bar(
    ax: Axes,
    fractions: Optional[Sequence[float]],
    *,
    cmap: Colormap,
    y: float = 0,
    height: float = 0.8,
    shade_range: Tuple[float, float] = (0.45, 0.95),
    bg_color="0.92",
    text: Optional[str] = None,
    text_kws: Optional[dict] = None,
) -> None:
    """Draw one horizontal nested bar centered at row `y`.

    Draws a full-width background rectangle in `bg_color`, then one
    rectangle per entry of `fractions`, in order, each spanning
    `[0, fractions[i]]` and shaded from light to dark along `cmap`
    (evenly spaced across `shade_range`).

    Fractions are not summed: each is drawn from 0, so a later (smaller)
    fraction appears nested inside an earlier one.

    `fractions=None` (or empty) draws only the background, with `text`
    (if given) centered on it, e.g. for a bar without data.
    """
    text_kws = {
        "ha": "center",
        "va": "center",
        "fontsize": 6,
        "style": "italic",
        "color": ".5",
        **(text_kws or {}),
    }
    ax.barh(y, 1, height=height, color=bg_color, zorder=0)
    if fractions is None or len(fractions) == 0:
        if text:
            ax.text(0.5, y, text, **text_kws)
        return
    shades = np.linspace(*shade_range, len(fractions))
    for frac, shade in zip(fractions, shades):
        ax.barh(y, frac, height=height, color=cmap(shade), zorder=1)


def stacked_bars(
    ax: Axes,
    data: Dict[str, Optional[Sequence[float]]],
    *,
    bars: Optional[Dict[str, dict]] = None,
    cmap: Optional[PaletteSpec] = None,
    shade_range: Tuple[float, float] = (0.45, 0.95),
    gap: float = 0.2,
    pad: float = 0.0,
    bg_color="0.92",
    text: Optional[Dict[str, str]] = None,
    text_kws: Optional[dict] = None,
    xlim: Tuple[float, float] = (0, 1),
) -> Dict[str, Colormap]:
    """Draw one `nested_bar` per entry of `data`, stacked top-to-bottom.

    Each key of `data` becomes one row, occupying the unit-height band
    `[i, i + 1]` (`i` = row index); `gap` (a fraction of that unit
    band, default 0.2) is left as whitespace between bars, split evenly
    above and below each one.

    `pad` (in row units, default 0) adds blank margin above the first
    and below the last bar, without changing the size of `ax`, e.g. to
    keep a row aligned with other columns in a combined figure.

    Colors are resolved as in `palette.resolve_item_cmaps`:
    `bars={name: dict(cmap=...)}` sets a bar's colormap or color, and
    other bars get a color sampled from `cmap` (or
    `palette.DEFAULT_PALETTE`).

    `data[name]` may be `None` or empty to draw a placeholder band,
    with `text[name]` (if given) centered on it.

    Returns the resolved `{name: Colormap}` mapping, e.g. to build a
    legend from the same colors afterwards.
    """
    names = list(data.keys())
    cmaps = resolve_item_cmaps(names, cmap=cmap, overrides=bars)
    ax.set_xlim(*xlim)
    ax.set_ylim(len(names) + pad, -pad)
    ax.set_yticks([])
    height = 1 - gap
    for i, name in enumerate(names):
        nested_bar(
            ax,
            data[name],
            cmap=cmaps[name],
            y=i + 0.5,
            height=height,
            shade_range=shade_range,
            bg_color=bg_color,
            text=(text or {}).get(name),
            text_kws=text_kws,
        )
    return cmaps
