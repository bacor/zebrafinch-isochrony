"""Colormap helpers for plots with one colormap per named item.

Used by `heatmap.per_column_heatmap` (one colormap per column) and
`stacked_bar.stacked_bars` (one colormap per bar), via `resolve_item_cmaps`.
"""

from typing import Dict, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Colormap, LinearSegmentedColormap

ColumnSpec = Union[str, Colormap]
# A colormap/name to sample colors from, or an explicit palette (sequence of colors).
PaletteSpec = Union[str, Colormap, Sequence]

DEFAULT_PALETTE = sns.color_palette("hls", 8)


def white_to_color_cmap(color, name: str = "white_to_color") -> Colormap:
    """Build a colormap that ramps from white to `color`."""
    return LinearSegmentedColormap.from_list(name, ["white", color])


def diverging_from_white(cmap: Colormap) -> Colormap:
    """Turn a white-to-color colormap into white-color-white, peaking at the midpoint.

    Takes the color at `cmap(1.0)` (its far end) as the peak color, so this
    works on any colormap, including one sampled automatically.
    """
    color = cmap(1.0)
    return LinearSegmentedColormap.from_list(
        f"{cmap.name}_diverging", ["white", color, "white"]
    )


def cmap_slice(
    cmap: ColumnSpec,
    vmin: float = 0.0,
    vmax: float = 1.0,
    n: int = 256,
    name: Optional[str] = None,
) -> Colormap:
    """Return a colormap covering only the `[vmin, vmax]` slice of `cmap`.

    E.g. `cmap_slice("viridis", 0.0, 0.2)` gives a full 0-1 colormap whose
    colors are drawn only from the first 20% of viridis.
    """
    base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    colors = base(np.linspace(vmin, vmax, n))
    return LinearSegmentedColormap.from_list(
        name or f"{base.name}_{vmin:.2f}_{vmax:.2f}", colors
    )


def resolve_cmap(spec: Union[str, Colormap, Sequence]) -> Colormap:
    """Resolve `spec` to a `Colormap`.

    `spec` may be a `Colormap` (returned as-is), the name of a
    registered colormap (e.g. `"viridis"`, resolved via `plt.get_cmap`),
    or a plain color (name, hex string, or RGB(A) tuple), which is turned
    into a white-to-color colormap with `white_to_color_cmap`.
    """
    if isinstance(spec, Colormap):
        return spec
    if isinstance(spec, str):
        try:
            return plt.get_cmap(spec)
        except ValueError:
            return white_to_color_cmap(spec)
    return white_to_color_cmap(spec)


def sample_column_cmaps(
    source: PaletteSpec, columns: Sequence[str]
) -> Dict[str, Colormap]:
    """Pick one color per column from `source` and ramp each from white to it.

    `source` may be a colormap (name or `Colormap`), sampled at equally
    spaced points, or an explicit palette (e.g. a `seaborn` palette or any
    sequence of colors), from which colors are picked at equally spaced
    indices (cycling if there are more columns than colors).
    """
    n = len(columns)
    if isinstance(source, (str, Colormap)):
        cmap = plt.get_cmap(source) if isinstance(source, str) else source
        positions = np.linspace(0, 1, n) if n > 1 else np.array([0.5])
        colors = [cmap(pos) for pos in positions]
    else:
        palette = list(source)
        m = len(palette)
        indices = (
            np.arange(n) % m
            if n > m
            else np.round(np.linspace(0, m - 1, n)).astype(int)
        )
        colors = [palette[i] for i in indices]
    return {
        col: white_to_color_cmap(color, name=f"white_to_{col}")
        for col, color in zip(columns, colors)
    }


def resolve_item_cmaps(
    items: Sequence[str],
    *,
    cmap: Optional[PaletteSpec] = None,
    overrides: Optional[Dict[str, dict]] = None,
) -> Dict[str, Colormap]:
    """Resolve one colormap per item, with per-item overrides.

    Each item gets a colormap from `overrides[item]["cmap"]` if set (see
    `resolve_cmap`); otherwise one color per item is auto-sampled from `cmap` (or
    `DEFAULT_PALETTE`) via `sample_column_cmaps`. Then `overrides[item]["diverging"]`
    (see `diverging_from_white`) and `overrides[item]["reverse"]` are applied, in
    that order, so they also work on an auto-sampled colormap.

    `overrides` is the caller's `columns`/`bars` dict; other keys it holds
    (e.g. `vmin`, `label`) are ignored here.
    """
    overrides = overrides or {}
    auto_cmaps = sample_column_cmaps(
        cmap if cmap is not None else DEFAULT_PALETTE, items
    )
    resolved: Dict[str, Colormap] = {}
    for item in items:
        spec = overrides.get(item, {})
        override = spec.get("cmap")
        resolved[item] = (
            resolve_cmap(override) if override is not None else auto_cmaps[item]
        )
        if spec.get("diverging"):
            resolved[item] = diverging_from_white(resolved[item])
        if spec.get("reverse"):
            resolved[item] = resolved[item].reversed()
    return resolved
