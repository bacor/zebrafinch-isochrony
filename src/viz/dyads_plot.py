"""Phase plots of dyads: scatter, Gaussian contours and transitions per dyad cluster.

A dyad is a pair of successive durations (first, second). `DyadsPlotter` groups the
dyads of one bird by label into `Cluster`s and draws them, plus the transitions
between clusters within chunks and the isochrony criteria.
"""

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx

from .utils import get_label_colors, get_ax, normalize_single_color, WithDefaults


def _principal_axes(
    cov: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Eigenvalues (descending) and eigenvectors (as columns) of a covariance matrix."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def ellipse_from_cov(
    mu: NDArray[np.floating],
    cov: NDArray[np.floating],
    n_std: float = 1.0,
    **patch_kwargs: Any,
) -> Ellipse:
    """Ellipse patch for the `n_std` contour of a 2D Gaussian (`mu`, `cov`)."""
    vals, vecs = _principal_axes(cov)

    # Width/height of ellipse are 2 * n_std * sqrt(eigenvalues)
    width, height = 2 * n_std * np.sqrt(vals)

    # Angle of major axis (in degrees)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    return Ellipse(xy=mu, width=width, height=height, angle=angle, **patch_kwargs)


def plot_pc_variance_lines(
    mu: NDArray[np.floating],
    cov: NDArray[np.floating],
    ax: plt.Axes,
    n_std: float = 1.0,
    **line_kwargs: Any,
) -> None:
    """Plot both principal axes of a 2D Gaussian as lines through its mean `mu`.

    Each line spans +/- `n_std` standard deviations along its principal direction;
    `line_kwargs` are passed to `ax.plot`.
    """
    vals, vecs = _principal_axes(cov)
    for i in range(2):
        std_i = np.sqrt(vals[i])
        d = n_std * std_i * vecs[:, i]

        p1 = mu - d
        p2 = mu + d

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **line_kwargs)


def plot_gaussian(
    mu: NDArray[np.floating],
    cov: NDArray[np.floating],
    n_stds: Sequence[int] = (1, 2, 3, 4),
    alpha: float = 1.0,
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Draw filled `n_stds` contours of a 2D Gaussian, fading outwards."""
    ax = get_ax(ax)
    for n_std in n_stds:
        ell = ellipse_from_cov(
            mu, cov, n_std=n_std, fill=True, alpha=alpha * n_std**-1.5, facecolor=color
        )
        ax.add_patch(ell)
        ax.plot([], [])
    return mu, cov


def collapse_labels(
    labels: NDArray[Any], keep: Iterable[Any], other_label: Any = "other"
) -> NDArray[Any]:
    """Replace any label not in `keep` with `other_label`."""
    keep = set(keep)
    return np.array([label if label in keep else other_label for label in labels])


def collapse_rare_labels(
    labels: NDArray[Any], min_percentile: float, other_label: Any = "other"
) -> NDArray[Any]:
    """Replace labels with a share below `min_percentile`% by `other_label`."""
    counts = Counter(labels)
    total = sum(counts.values())
    keep = [l for l, count in counts.items() if count / total >= min_percentile / 100]
    return collapse_labels(labels, keep, other_label)


def collapse_to_top_n_labels(
    labels: NDArray[Any], n: int, other_label: Any = "other"
) -> NDArray[Any]:
    """Keep only the `n` most frequent labels; replace the rest with `other_label`."""
    keep = [l for l, _ in Counter(labels).most_common(n)]
    return collapse_labels(labels, keep, other_label)


def sample_chunks(
    chunks: NDArray[Any], max_chunks: int, random_state: Optional[int] = None
) -> NDArray[Any]:
    """Randomly sample up to `max_chunks` unique chunk ids (all of them if fewer)."""
    unique_chunks = np.unique(chunks)
    if len(unique_chunks) <= max_chunks:
        return unique_chunks
    rng = np.random.default_rng(random_state)
    return rng.choice(unique_chunks, size=max_chunks, replace=False)


def linear_width(
    count: float,
    count_min: float,
    count_max: float,
    w_min: float = 0.2,
    w_max: float = 3.5,
) -> float:
    """Line width for `count`, linear from `w_min` at 0 to `w_max` at `count_max`.

    `count_min` is only used to return `w_max` when all counts are equal.
    """
    if count_max <= count_min:
        return w_max
    return w_min + (w_max - w_min) * (count / count_max)


def repel_direction(
    positions: NDArray[np.floating], index: int, span_x: float, span_y: float
) -> NDArray[np.floating]:
    """Unit vector pointing `positions[index]` away from the other positions.

    Contributions are weighted by inverse distance, after scaling the axes by
    `span_x` and `span_y`.
    """
    base = positions[index]
    v = np.zeros(2, dtype=float)
    for j, other in enumerate(positions):
        if index == j:
            continue
        delta = base - other
        delta_norm = np.array([delta[0] / span_x, delta[1] / span_y])
        dist2 = np.dot(delta_norm, delta_norm) + 1e-6
        v += delta_norm / dist2
    if np.allclose(v, 0):
        v = np.array([1.0, 0.0])
    return v / np.linalg.norm(v)


def get_repel_offset(
    ax: plt.Axes,
    positions: Sequence[Sequence[float]],
    index: int,
    offset_pts: float,
) -> List[float]:
    """Label offset of `offset_pts` points away from the other positions, in points."""
    positions = np.asarray(positions, dtype=float)
    span_x = max(positions[:, 0].max() - positions[:, 0].min(), 1.0)
    span_y = max(positions[:, 1].max() - positions[:, 1].min(), 1.0)
    mean = positions[index]
    v = repel_direction(positions, index, span_x, span_y)
    mean_px = ax.transData.transform(mean)
    dir_px = (
        ax.transData.transform(mean + np.array([v[0] * span_x, v[1] * span_y]))
        - mean_px
    )
    if np.allclose(dir_px, 0):
        dir_px = np.array([1.0, 0.0])
    dir_px = dir_px / np.linalg.norm(dir_px)
    return (dir_px * offset_pts).tolist()


def _shade_between_diagonals(
    ax: plt.Axes,
    lower: Tuple[float, float],
    upper: Tuple[float, float],
    color: Any,
    alpha: float,
    fill: bool,
    lw: float,
    ls: Any,
) -> plt.Axes:
    """Shade between two lines (if `fill`) and draw both as borders (if `lw > 0`).

    `lower` and `upper` are `(intercept, slope)` pairs of lines
    `y = intercept + slope * x`. Axis limits are left unchanged.
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if fill:
        far = 10 * max(abs(v) for v in (*xlim, *ylim))
        xs = np.array([-far, far])
        (b_lo, m_lo), (b_hi, m_hi) = lower, upper
        ax.fill(
            [*xs, *xs[::-1]],
            [*(b_lo + m_lo * xs), *(b_hi + m_hi * xs[::-1])],
            color=color,
            alpha=alpha,
            lw=0,
            zorder=0,
        )
    if lw > 0:
        for b, m in (lower, upper):
            ax.axline((0, b), slope=m, color=color, lw=lw, ls=ls, zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


def plot_relative_criterion(
    rho: float,
    ax: Optional[plt.Axes] = None,
    color: Any = "tab:blue",
    alpha: float = 0.08,
    fill: bool = True,
    lw: float = 0,
    ls: Any = "-",
) -> plt.Axes:
    """Shade the relative criterion `|x / (x + y) - 0.5| < rho` on a phase plot.

    With x the first and y the second duration, this is a wedge from the origin.
    """
    lo = (0.5 - rho) / (0.5 + rho)
    return _shade_between_diagonals(
        get_ax(ax), (0, lo), (0, 1 / lo), color, alpha, fill, lw, ls
    )


def plot_absolute_criterion(
    tau: float,
    ax: Optional[plt.Axes] = None,
    color: Any = "tab:orange",
    alpha: float = 0.08,
    fill: bool = True,
    lw: float = 0,
    ls: Any = "-",
) -> plt.Axes:
    """Shade the absolute criterion `|x - y| < tau` on a phase plot.

    With x the first and y the second duration, this is a band around the diagonal.
    """
    return _shade_between_diagonals(
        get_ax(ax), (-tau, 1), (tau, 1), color, alpha, fill, lw, ls
    )


class Cluster(WithDefaults):
    """The dyads of one label in a phase plot, with their mean and covariance.

    The `*_DEFAULTS` dicts hold style defaults per plotting method; string values
    such as `"cluster_color"` are replaced by this cluster's attributes, and
    `relative_zorder` is added to the cluster's `zorder`.
    """

    ## DEFAULTS

    SCATTER_DEFAULTS = {
        "color": "cluster_color",
        "marker": "cluster_marker",
        "alpha": "cluster_alpha",
        "s": 15,
        "edgecolor": "w",
        "lw": 0.5,
        "label": "cluster_label",
        "relative_zorder": 2,
    }

    GAUSSIAN_DEFAULTS = {
        "alpha": 0.25,
        "fill": True,
        "n_stds": [1],
        "facecolor": "cluster_color",
        "relative_zorder": 0,
    }

    PC_DEFAULTS = {
        "relative_zorder": 1,
        "n_std": 1,
        "color": "w",
        "lw": 1,
    }

    MEAN_DEFAULTS = {
        "marker": "^",
        "color": "cluster_color",
        "edgecolor": "w",
        "lw": 1,
        "s": 80,
        "relative_zorder": 3,
    }

    CLUSTER_LABEL_DEFAULTS = {
        "textcoords": "offset points",
        "color": "cluster_color",
        "ha": "center",
        "va": "center",
        "fontsize": 9,
        "relative_zorder": 4,
        "bbox": {
            "boxstyle": "round, pad=0.25",
            "facecolor": "#fff",
            "edgecolor": "cluster_color",
            "lw": 0.5,
            "alpha": 1,
        },
        "arrowprops": {
            "arrowstyle": "-",
            "linestyle": (0, (0.01, 2.5)),
            "lw": 1.5,
            "color": "black",
            "shrinkA": 0,
            "shrinkB": 0,
        },
    }

    ## BASIC METHODS

    def __init__(
        self,
        points: NDArray[np.floating],
        label: Optional[str] = None,
        color: Optional[str] = None,
        marker: str = "o",
        alpha: Optional[float] = None,
        zorder: int = 10,
    ) -> None:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be an (N,2) array")
        self.points = np.asarray(points)
        self.label = label
        self.color = color
        self.marker = marker
        self.alpha = alpha
        self.z = zorder
        self.legend_handles = []

        # Covariance: ddof=1 is the unbiased sample covariance; ddof=0 is the MLE for Gaussian.
        self.mu = points.mean(axis=0)
        self.cov = np.cov(points.T, ddof=0)

        # Update defaults
        self.update_defaults()

    def defaults_updater(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in cluster attributes and an absolute `zorder` in one defaults dict."""
        for key, value in defaults.items():
            if value == "cluster_color":
                defaults[key] = self.color
            if value == "cluster_label":
                defaults[key] = self.label
            if value == "cluster_marker":
                defaults[key] = self.marker
            if value == "cluster_alpha":
                defaults[key] = self.alpha

        # Set zorder relative to cluster zorder
        if "relative_zorder" in defaults:
            defaults["zorder"] = self.z + defaults["relative_zorder"]
            del defaults["relative_zorder"]

        return defaults

    @property
    def xs(self) -> NDArray[np.floating]:
        return self.points[:, 0]

    @property
    def ys(self) -> NDArray[np.floating]:
        return self.points[:, 1]

    ## PLOTTING METHODS

    def scatter(self, ax: Optional[plt.Axes] = None, **kwargs: Any) -> None:
        """Scatter cluster points; kwargs override SCATTER_DEFAULTS."""
        kwargs = self.defaults(self.SCATTER_DEFAULTS, **kwargs)
        if kwargs.get("marker") == "x":
            # Unfilled marker: matplotlib warns and ignores edgecolor/lw for it
            kwargs.pop("edgecolor", None)
            kwargs.pop("lw", None)
        get_ax(ax).scatter(self.xs, self.ys, **kwargs)
        self.register_legend_handle(self.get_legend_handle(), "scatter")

    def gaussian(self, ax: Optional[plt.Axes] = None, **kwargs: Any) -> None:
        """Draw Gaussian contours; kwargs override GAUSSIAN_DEFAULTS."""
        kwargs = self.defaults(self.GAUSSIAN_DEFAULTS, **kwargs)
        alpha = kwargs.pop("alpha", 0.25)
        n_stds = kwargs.pop("n_stds", [1, 2])
        ax = get_ax(ax)
        for n_std in n_stds:
            ell = ellipse_from_cov(
                self.mu,
                self.cov,
                n_std=n_std,
                alpha=alpha * n_std**-1.5,
                **kwargs,
            )
            ax.add_patch(ell)
        ax.autoscale_view()
        self.register_legend_handle(self.get_gaussian_legend_handle(), "gaussian")

    def mean(self, ax: Optional[plt.Axes] = None, **kwargs: Any) -> None:
        """Draw cluster mean marker; kwargs override MEAN_DEFAULTS."""
        kwargs = self.defaults(self.MEAN_DEFAULTS, **kwargs)
        get_ax(ax).scatter(self.mu[0], self.mu[1], **kwargs)
        self.register_legend_handle(self.get_mean_legend_handle(), "means")

    def principal_components(
        self, ax: Optional[plt.Axes] = None, **line_kwargs: Any
    ) -> None:
        """Draw principal component lines; line_kwargs override PC_DEFAULTS."""
        line_kwargs = self.defaults(self.PC_DEFAULTS, **line_kwargs)
        return plot_pc_variance_lines(self.mu, self.cov, ax=get_ax(ax), **line_kwargs)

    def cluster_label(self, ax: Optional[plt.Axes] = None, **kwargs: Any) -> None:
        """Annotate the cluster label at its mean, shifted by `offset` (in points)."""
        if self.label is None:
            return
        offset = kwargs.pop("offset", (0, 0))
        kwargs = self.defaults(self.CLUSTER_LABEL_DEFAULTS, **kwargs)
        ax = get_ax(ax)
        ax.annotate(self.label, self.mu, xytext=offset, **kwargs)

    ## LEGENDS

    def get_legend_handle(self) -> Line2D:
        """Legend handle for the scatter markers."""
        return Line2D(
            [],
            [],
            marker=self.marker,
            linestyle="None",
            markersize=8,
            markerfacecolor=self.color,
            markeredgecolor="w",
            label=self.label,
        )

    def get_mean_legend_handle(self) -> Line2D:
        """Legend handle for the mean marker."""
        return Line2D(
            [],
            [],
            marker=self.MEAN_DEFAULTS.get("marker", "^"),
            linestyle="None",
            markersize=8,
            markerfacecolor=self.color,
            markeredgecolor=self.MEAN_DEFAULTS.get("edgecolor", "w"),
            label=self.label,
        )

    def get_gaussian_legend_handle(self) -> Ellipse:
        """Legend handle for the Gaussian contour."""
        return Ellipse(
            (0, 0),
            width=0.35,
            height=0.25,
            facecolor=self.color,
            alpha=0.3,
            edgecolor="none",
            label="1 std",
        )

    def register_legend_handle(self, handle: Artist, handle_type: str) -> None:
        """Store a legend handle, skipping one with the same type and label."""
        handle_label = handle.get_label()
        for entry in self.legend_handles:
            if (
                entry["type"] == handle_type
                and entry["handle"].get_label() == handle_label
            ):
                return
        self.legend_handles.append({"handle": handle, "type": handle_type})


class DyadsPlotter(WithDefaults):
    """Phase plot of dyads, with one `Cluster` per label.

    Rare labels can be merged into an "other" cluster (`min_label_percentile` or
    `top_n_labels`), which is drawn in grey and left out of the transition network.
    `max_chunks` randomly subsamples chunks to keep dense plots readable.
    """

    ## DEFAULTS

    TRAJECTORY_DEFAULTS = {
        "color": "0.95",
        "lw": 0.5,
        "relative_zorder": -2,
    }

    NETWORK_DEFAULTS = {
        "min_count": 1,
        "w_min": 0.2,
        "w_max": 1.5,
        "width_fn": linear_width,
        "color": "0.2",
        "edge_color": "0.2",
        "selfloop_pts": 20,
        "self_loops": True,
        "straight": False,
        "arrows": True,
        "arrowsize": 10,
        "connectionstyle": "arc3,rad=-0.10",
        "relative_zorder": 10,
        "show_legend": True,
    }

    CONSTELLATION_DEFAULTS = {
        "self_loops": False,
        "arrows": False,
        "straight": False,
    }

    CLUSTER_LABEL_DEFAULTS = {
        "relative_zorder": 11,
        "offset": 40,
    }

    ISOCHRONY_DEFAULTS = {
        "color": "0.8",
        "lw": 1,
        "zorder": 0,
    }

    OTHER_STYLE = {
        "color": "0.8",
        "marker": "x",
        "alpha": 0.5,
    }

    ## BASIC METHODS

    def __init__(
        self,
        points: NDArray[np.floating],
        labels: Sequence[Any],
        chunks: Sequence[Any],
        color=None,
        min_label_percentile: Optional[float] = None,
        top_n_labels: Optional[int] = None,
        other_label: Any = "other",
        max_chunks: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if min_label_percentile is not None and top_n_labels is not None:
            raise ValueError(
                "Pass at most one of min_label_percentile and top_n_labels"
            )

        points = np.asarray(points)
        labels = np.asarray(labels)
        chunks = np.asarray(chunks)

        if max_chunks is not None:
            sampled_chunks = sample_chunks(chunks, max_chunks, random_state)
            mask = np.isin(chunks, sampled_chunks)
            points, labels, chunks = points[mask], labels[mask], chunks[mask]

        self.other_label = None
        if min_label_percentile is not None:
            labels = collapse_rare_labels(labels, min_label_percentile, other_label)
            self.other_label = other_label
        elif top_n_labels is not None:
            labels = collapse_to_top_n_labels(labels, top_n_labels, other_label)
            self.other_label = other_label

        self.labels = labels
        self.points = points
        self.chunks = chunks
        self.unique_labels = sorted(np.unique(labels))
        self.unique_chunks = sorted(np.unique(chunks))
        self.z = 5
        self.legend_handles = []

        if not color:
            self.colors = get_label_colors(self.unique_labels)
        else:
            self.colors = {label: color for label in self.unique_labels}
        if self.other_label is not None and self.other_label in self.colors:
            self.colors[self.other_label] = self.OTHER_STYLE["color"]

        self.clusters = {}
        for label in self.unique_labels:
            pts = self.points[self.labels == label]
            col = self.colors[label]
            if label == self.other_label:
                marker = self.OTHER_STYLE["marker"]
                alpha = self.OTHER_STYLE.get("alpha")
            else:
                marker, alpha = "o", None
            self.clusters[label] = Cluster(
                pts, label=label, color=col, marker=marker, alpha=alpha, zorder=self.z
            )

        ignore_clusters = [self.other_label] if self.other_label is not None else None
        self.graph = self.build_transition_network(ignore_clusters=ignore_clusters)
        self.update_defaults()

    def defaults_updater(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Turn `relative_zorder` into an absolute `zorder` in one defaults dict."""
        if "relative_zorder" in defaults:
            defaults["zorder"] = self.z + defaults["relative_zorder"]
            del defaults["relative_zorder"]

        return defaults

    def build_transition_network(
        self, ignore_clusters: Optional[Iterable[Any]] = None
    ) -> nx.DiGraph:
        """Directed graph of clusters, with transition counts as edge attribute `count`."""
        graph = nx.DiGraph()
        for label, cluster in self.clusters.items():
            graph.add_node(label, pos=cluster.mu, color=cluster.color)

        # Count cluster transitions and store as edge attributes
        transitions = Counter()
        for chunk in self.unique_chunks:
            chunk_mask = self.chunks == chunk
            chunk_labels = self.labels[chunk_mask]
            transitions.update(zip(chunk_labels[:-1], chunk_labels[1:]))
        for (src, dst), count in transitions.items():
            graph.add_edge(src, dst, count=count)

        # Remove ignored clusters from graph
        if ignore_clusters is not None:
            for label in ignore_clusters:
                if label in graph:
                    graph.remove_node(label)

        return graph

    def register_legend_handle(
        self, handle_type: str, index: int = -1, **line_kwargs: Any
    ) -> None:
        """Store a `Line2D` legend handle at `index`, skipping duplicates."""
        handle = Line2D([], [], **line_kwargs)
        handle_label = handle.get_label()
        if any(
            h["handle"].get_label() == handle_label and h["type"] == handle_type
            for h in self.legend_handles
        ):
            return
        entry = {"handle": handle, "type": handle_type}
        if index < 0 or index >= len(self.legend_handles):
            self.legend_handles.append(entry)
        else:
            self.legend_handles.insert(index, entry)

    ## AXES SETUP

    def patdur_axes(
        self,
        ax: Optional[plt.Axes] = None,
        xlabel: str = "pattern",
        ylabel: str = "dyad duration",
        xlim: Optional[Tuple[float, float]] = None,
        show_isochrony: bool = True,
        isochrony_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set up axes for rhythm ratio vs. dyad duration, marking 0.5 (isochrony)."""
        ax = get_ax(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(*xlim)
        sns.despine(ax=ax)
        if show_isochrony:
            isochrony_kwargs = isochrony_kwargs or {}
            kwargs = self.defaults(self.ISOCHRONY_DEFAULTS, **isochrony_kwargs)
            ax.axvline(0.5, **kwargs)

    def phase_axes(
        self,
        ax: Optional[plt.Axes] = None,
        show_isochrony: bool = True,
        isochrony_kwargs: Optional[Dict[str, Any]] = None,
        equal: bool = False,
        square: bool = False,
        xlabel: str = "first duration",
        ylabel: str = "second duration",
    ) -> None:
        """Set up phase-plot axes, optionally drawing the diagonal (isochrony)."""
        ax = get_ax(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
        if equal:
            ax.set_aspect("equal", adjustable="box")
        if square:
            lim = (0, max(ax.get_xlim()[1], ax.get_ylim()[1]))
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal", adjustable="box")
        if show_isochrony:
            isochrony_kwargs = isochrony_kwargs or {}
            kwargs = self.defaults(self.ISOCHRONY_DEFAULTS, **isochrony_kwargs)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.axline((0, 0), slope=1, **kwargs)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    def legend(
        self,
        ax: Optional[plt.Axes] = None,
        below: bool = False,
        **legend_kwargs: Any,
    ) -> None:
        """Assemble and draw legend from registered handles.

        Args:
            below: Place the legend centred below the axes instead of inside it.
            legend_kwargs: Forwarded to `ax.legend`.
        """
        # Collect all handles
        scatter_handles = []
        means_handles = []
        gaussian_handles = []
        for cluster in self.clusters.values():
            for entry in cluster.legend_handles:
                if entry["type"] == "scatter":
                    scatter_handles.append(entry["handle"])
                elif entry["type"] == "means":
                    means_handles.append(entry["handle"])
                elif entry["type"] == "gaussian":
                    gaussian_handles.append(entry["handle"])

        # Use an exemplar to get properties of the handles
        exemplar = next(iter(self.clusters.values()))
        if means_handles and scatter_handles:
            means_handle = exemplar.get_mean_legend_handle()
            means_handle.set_markerfacecolor("0.2")
            means_handle.set_label("mean")
            means_handles = [means_handle]

        # Gaussian handles: never show in color, but add scatter handles
        # if no color handles (scatter/means) exist yet
        if gaussian_handles and not scatter_handles and not means_handles:
            scatter_handles = [
                cluster.get_legend_handle() for cluster in self.clusters.values()
            ]
        if gaussian_handles:
            gaussian_handle = exemplar.get_gaussian_legend_handle()
            gaussian_handle.set_facecolor("0.2")
            gaussian_handle.set_label("1 std")
            gaussian_handles = [gaussian_handle]

        handles = scatter_handles + means_handles + gaussian_handles
        handles += [entry["handle"] for entry in self.legend_handles]
        legend_kwargs.setdefault("ncols", min(3, max(1, math.ceil(len(handles) / 6))))

        if below:
            legend_kwargs.setdefault("loc", "upper left")
            legend_kwargs.setdefault("bbox_to_anchor", (0, -0.2))
            legend_kwargs.setdefault("frameon", False)

        get_ax(ax).legend(handles=handles, **legend_kwargs)

    ## PLOTTING METHODS

    def scatter(self, **kwargs: Any) -> None:
        """Scatter all clusters; kwargs forwarded to Cluster.scatter."""
        for cluster in self.clusters.values():
            cluster.scatter(**kwargs)

    def gaussian(self, **kwargs: Any) -> None:
        """Draw Gaussian contours for all clusters except the "other" cluster."""
        for label, cluster in self.clusters.items():
            if label == self.other_label:
                continue
            cluster.gaussian(**kwargs)

    def principal_components(self, **kwargs: Any) -> None:
        """Draw principal components for all clusters."""
        for cluster in self.clusters.values():
            cluster.principal_components(**kwargs)

    def trajectories(self, ax: Optional[plt.Axes] = None, **kwargs: Any) -> None:
        """Connect successive dyads per chunk; kwargs override TRAJECTORY_DEFAULTS."""
        ax = get_ax(ax)
        kwargs = self.defaults(self.TRAJECTORY_DEFAULTS, **kwargs)
        for chunk in self.unique_chunks:
            chunk_mask = self.chunks == chunk
            pts = self.points[chunk_mask]
            ax.plot(pts[:, 0], pts[:, 1], **kwargs)

    def means(self, **kwargs: Any) -> None:
        """Draw mean markers for all clusters except the "other" cluster."""
        for label, cluster in self.clusters.items():
            if label == self.other_label:
                continue
            cluster.mean(**kwargs)

    def transition_network(self, **kwargs: Any) -> None:
        """Draw transitions between cluster means; kwargs override NETWORK_DEFAULTS.

        Edge widths scale with the transition count (see `linear_width`); edges
        below `min_count` are dropped.
        """
        raw_kwargs = dict(kwargs)
        # Get plotting parameters
        kwargs = self.defaults(self.NETWORK_DEFAULTS, **kwargs)
        min_count = kwargs.pop("min_count")
        w_min = kwargs.pop("w_min")
        w_max = kwargs.pop("w_max")
        width_fn = kwargs.pop("width_fn")
        selfloop_pts = kwargs.pop("selfloop_pts")
        self_loops = kwargs.pop("self_loops")
        straight = kwargs.pop("straight")
        show_legend = kwargs.pop("show_legend")
        zorder = kwargs.pop("zorder")
        ax = get_ax(kwargs.pop("ax", None))
        # `color` and `edge_color` both set the edge color; `color` wins.
        color, edge_color = kwargs.pop("color"), kwargs.pop("edge_color")
        if "color" in raw_kwargs:
            edge_color = normalize_single_color(color)
        elif "edge_color" not in raw_kwargs:
            edge_color = color
        kwargs["edge_color"] = edge_color

        if straight and "connectionstyle" not in raw_kwargs:
            kwargs["connectionstyle"] = "arc3,rad=0.0"
        # NetworkX draws LineCollection when arrows=False; connectionstyle is ignored
        # and emits a warning in that mode, so only pass it for arrow patches.
        if not kwargs.get("arrows", False):
            kwargs.pop("connectionstyle", None)

        # Collect edges with a minimum transition count
        edges = [
            (u, v)
            for u, v in self.graph.edges()
            if self.graph[u][v].get("count", 0) >= min_count and (self_loops or u != v)
        ]
        if len(edges) == 0:
            return

        edge_counts = [self.graph[u][v].get("count", 0) for u, v in edges]
        if "width" in kwargs:
            edge_widths = kwargs["width"]
        else:
            get_width = lambda count: width_fn(
                count, min(edge_counts), max(edge_counts), w_min=w_min, w_max=w_max
            )
            edge_widths = [get_width(count) for count in edge_counts]
            kwargs["width"] = edge_widths

            # Add legend handles only when counts actually vary (otherwise uninformative)
            if show_legend:
                legend_counts = np.round(
                    np.linspace(min(edge_counts), max(edge_counts), 3)
                )
                unique_legend_counts = np.unique(legend_counts)
                if len(unique_legend_counts) > 1:
                    for count in unique_legend_counts:
                        self.register_legend_handle(
                            "transition_network",
                            color=edge_color,
                            lw=get_width(count),
                            label=str(int(count)),
                        )

        width_by_edge = {edge: w for edge, w in zip(edges, edge_widths)}
        self_edges = [e for e in edges if e[0] == e[1]]
        non_self_edges = [e for e in edges if e[0] != e[1]]

        edges = None
        if non_self_edges:
            draw_kwargs = dict(kwargs)
            draw_kwargs["width"] = [width_by_edge[e] for e in non_self_edges]
            edges = nx.draw_networkx_edges(
                self.graph,
                nx.get_node_attributes(self.graph, "pos"),
                edgelist=non_self_edges,
                ax=ax,
                **draw_kwargs,
            )

        if self_edges:
            for u, _ in self_edges:
                x, y = self.graph.nodes[u]["pos"]
                p = ax.transData.transform((x, y))
                inv = ax.transData.inverted()
                dx = inv.transform((p[0] + selfloop_pts, p[1]))[0] - x
                dy = inv.transform((p[0], p[1] + selfloop_pts))[1] - y
                loop = Ellipse(
                    (x + dx, y + dy),
                    width=2 * dx,
                    height=2 * dy,
                    fill=False,
                    lw=width_by_edge.get((u, u), 1.0),
                    color=edge_color,
                    zorder=zorder,
                )
                ax.add_patch(loop)

        # Set z order
        if edges is not None:
            if isinstance(edges, list):
                for art in edges:
                    art.set_zorder(zorder)
            else:
                edges.set_zorder(zorder)

        # Unhide axes
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    def constellation(self, **kwargs: Any) -> None:
        """Draw a transition constellation without self-loops; no arrows by default."""
        kwargs = self.defaults(self.CONSTELLATION_DEFAULTS, **kwargs)
        self.transition_network(**kwargs)

    def cluster_labels(
        self, ax: Optional[plt.Axes] = None, ax_margin=0.05, **kwargs: Any
    ) -> None:
        """Draw cluster labels, `offset` points apart (see `get_repel_offset`)."""
        ax = get_ax(ax)
        kwargs = self.defaults(self.CLUSTER_LABEL_DEFAULTS, **kwargs)
        offset = kwargs.pop("offset", 0)
        clusters = [c for c in self.clusters.values() if c.label is not None]
        means = [cluster.mu for cluster in clusters]
        for i, cluster in enumerate(clusters):
            if offset == 0:
                offset_pts = (0, 0)
            else:
                offset_pts = get_repel_offset(ax, means, i, offset)
            cluster.cluster_label(ax=ax, offset=offset_pts, **kwargs)
        ax.margins(x=ax_margin, y=ax_margin)

    def plot(self, ax: Optional[plt.Axes] = None, min_count=1) -> None:
        """Default plot: scatter, trajectories, Gaussians, transitions and legend."""
        ax = get_ax(ax)
        self.scatter(ax=ax)
        self.trajectories(ax=ax)
        self.gaussian(ax=ax)
        self.transition_network(ax=ax, min_count=min_count)
        self.legend()
