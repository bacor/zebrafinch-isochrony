"""Matplotlib styling and figure helpers."""

from __future__ import annotations

from numbers import Number
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent.parent
FIGURES_DIR = ROOT_DIR / "figures"
DEFAULT_STYLE = CUR_DIR / "arial.mplstyle"

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_ax(ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Return the provided Axes or the current Axes."""
    return ax if ax is not None else plt.gca()


def resolve(data: Optional[Any], value: Any) -> Any:
    """Resolve a seaborn-style plotting input: a column name, or array-like.

    If `value` is a string and `data` is given, returns `data[value]`
    (a column lookup). Otherwise returns `value` unchanged, on the
    assumption it is already array-like. This lets every plotting function
    accept either `data=df, label="label"` or `label=my_array` directly,
    without requiring callers to conform to a fixed dataframe schema.
    """
    if isinstance(value, str) and data is not None:
        return data[value]
    return value


def get_fig_ax(
    ax: Optional[plt.Axes] = None, figsize: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Return (fig, ax), creating a new figure/axes when `ax` is not given."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def normalize_single_color(value: Any) -> Any:
    """Normalize RGB/RGBA tuples/lists to an unambiguous single color."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if (
        isinstance(value, (list, tuple))
        and len(value) in (3, 4)
        and all(isinstance(c, Number) for c in value)
    ):
        return mcolors.to_hex(value, keep_alpha=len(value) == 4)
    return value


class WithDefaults:
    """Mixin for classes holding style defaults in `*_DEFAULTS` class-level dicts."""

    def update_defaults(
        self,
        updater: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        max_depth: int = 10,
    ) -> None:
        """Apply `updater` (or `self.defaults_updater`) to every `*_DEFAULTS` dict."""
        if updater is None:
            if not hasattr(self, "defaults_updater"):
                raise ValueError(
                    "Either provide an updater function or define a defaults_updater method"
                )
            else:
                updater = self.defaults_updater

        # Recursively apply the updates also to nested dicts
        def update_recursively(defaults, max_depth=max_depth):
            if not isinstance(defaults, dict):
                raise ValueError("defaults must be a dict")

            defaults = updater(defaults)
            if not isinstance(defaults, dict):
                raise ValueError("defaults_updater must return a dict")

            if max_depth > 0:
                for k, v in defaults.items():
                    if isinstance(v, dict):
                        defaults[k] = update_recursively(
                            v.copy(), max_depth=max_depth - 1
                        )
            else:
                raise RecursionError("Max recursion depth reached in update_defaults")
            return defaults

        defaults_names = [k for k in vars(self.__class__) if k.endswith("_DEFAULTS")]
        for default_name in defaults_names:
            defaults = getattr(self, default_name).copy()
            defaults = update_recursively(defaults)
            setattr(self, default_name, defaults)

    def defaults(self, base: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
        """Merge `overrides` into `base`, recursing into nested dicts."""

        def merge(left, right):
            merged = dict(left)
            for key, value in right.items():
                if (
                    key in merged
                    and isinstance(merged[key], dict)
                    and isinstance(value, dict)
                ):
                    merged[key] = merge(merged[key], value)
                else:
                    merged[key] = value
            return merged

        return merge(base, overrides)


def savefig(name: Union[str, Path], dpi: int = 300, **kws: Any) -> None:
    """Save the current figure as PDF and PNG in `figures/`, overwriting.

    Args:
        name: File stem or path relative to `figures/` (extension is added).
        dpi: Resolution for the PNG output.
        **kws: Extra arguments forwarded to `plt.savefig`.
    """
    base_path = (FIGURES_DIR / Path(name)).with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(base_path.with_suffix(".pdf"), **kws)
    plt.savefig(base_path.with_suffix(".png"), dpi=dpi, **kws)


def subplot_title(
    index: int,
    title: str,
    ha: str = "left",
    x: float = 0,
    fontsize: int = 10,
    fontweight: str = "bold",
    ax: Optional[plt.Axes] = None,
    **title_kws: Any,
) -> plt.Text:
    """Set a subplot title prefixed with a letter (A., B., ...).

    Args:
        index: Subplot index used to pick the letter.
        title: Text to show after the prefix.
        ha, x, fontsize, fontweight: Passed through to `ax.set_title`.
        ax: Axes to title; defaults to the current axes.
        **title_kws: Additional keyword arguments forwarded to `ax.set_title`.
    """
    ax = get_ax(ax)
    return ax.set_title(
        f"{ALPHABET[index]}. {title}",
        ha=ha,
        x=x,
        fontsize=fontsize,
        fontweight=fontweight,
        **title_kws,
    )


def set_mpl_style(style_path: Union[str, Path, None] = None) -> None:
    """Apply the repo's default Matplotlib style (Arial) or a custom one."""
    path = Path(style_path) if style_path is not None else DEFAULT_STYLE
    plt.style.use(path)


def best_text_color(bg_color) -> str:
    """'white' on backgrounds with a relative luminance below 0.253, else 'black'.

    The WCAG contrast crossover is at 0.18; the higher threshold favours white text.
    """
    return "white" if sns.utils.relative_luminance(bg_color) < 0.253 else "black"


def ylabel_annot(label: str, y: float, x: float = 0, **kwargs) -> None:
    """Draw a rotated y-axis label in figure coordinates.

    Uses `fig.text` so the position is unaffected by `tight_layout()`.
    Pass `ax=` or `fig=` to target a specific figure; all other kwargs
    are forwarded to `fig.text`.
    """
    ax = kwargs.pop("ax", None)
    fig = kwargs.pop("fig", None)
    if ax is not None:
        fig = ax.get_figure()
    elif fig is None:
        fig = plt.gcf()
    kwargs.setdefault("fontsize", 9)
    kwargs.setdefault("fontweight", "bold")
    fig.text(x, y, label, rotation=90, va="center", ha="left", **kwargs)


def get_label_colors(labels, palette=None, tab20_threshold=10, husl_threshold=20):
    """Map each unique label to a color.

    Uses the seaborn default palette for up to `tab20_threshold` labels,
    `"tab20"` up to `husl_threshold`, and `"husl"` beyond that, unless
    `palette` is given explicitly.
    """
    unique_labels = sorted(set(labels), key=lambda label: str(label))
    if palette is None:
        n = len(unique_labels)
        if n > husl_threshold:
            palette = "husl"
        elif n > tab20_threshold:
            palette = "tab20"
    colors = sns.color_palette(palette=palette, n_colors=len(unique_labels))
    label_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
    return label_colors
