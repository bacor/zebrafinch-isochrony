"""Pooled distance between groups of durations, used to measure differentiation.

The distance between two centroids, standardized by the pooled covariance,
is Cohen's d (unsigned) in one dimension and the Mahalanobis distance in k
dimensions. Isolated labels (1D, durations) and isolated clusters (2D,
dyads) therefore use the same statistic.

`pooled_distance` computes it for a single pair;
`nearest_neighbour_pooled_distance` computes it for all pairs at once,
which is fast enough to rerun for every shuffle of a permutation test. The
tests check the 1D case against pingouin's Cohen's d.
"""

from __future__ import annotations

import warnings
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd


def _as_points(x: np.ndarray) -> np.ndarray:
    """(n,) -> (n, 1); (n, k) unchanged."""
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


# Variances below (VAR_RTOL * mean)^2 are rounding error on a constant sample.
VAR_RTOL = 1e-9


def _drop_rounding_variance(cov: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Zero the variance (and covariances) of axes whose variance is only rounding error."""
    negligible = np.diagonal(cov) <= (VAR_RTOL * mean) ** 2
    cov = cov.copy()
    cov[negligible, :] = 0
    cov[:, negligible] = 0
    return cov


def pooled_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized distance between two samples' centroids, pooling their covariance.

    `a`/`b` are (n,) (e.g. plain durations) or (n, k) arrays (e.g. dyad
    (duration_1, duration_2) pairs). k=1 is (unsigned) Cohen's d; k>1 is the
    Mahalanobis distance between centroids. NaN if either sample has fewer
    than 2 points, or the pooled covariance is singular (e.g. both samples
    have zero variance along the same axis).
    """
    a, b = _as_points(a), _as_points(b)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan
    cov_a = _drop_rounding_variance(
        np.atleast_2d(np.cov(a, rowvar=False)), a.mean(axis=0)
    )
    cov_b = _drop_rounding_variance(
        np.atleast_2d(np.cov(b, rowvar=False)), b.mean(axis=0)
    )
    pooled_cov = ((n_a - 1) * cov_a + (n_b - 1) * cov_b) / (n_a + n_b - 2)
    diff = a.mean(axis=0) - b.mean(axis=0)
    if _is_singular(pooled_cov):
        return np.nan
    return float(np.sqrt(diff @ np.linalg.inv(pooled_cov) @ diff))


def _is_singular(cov: np.ndarray) -> bool:
    """Whether a (pooled) covariance matrix is singular, judged scale-free.

    The determinant of the covariance matrix depends on the duration scale,
    so the test uses the determinant of the correlation matrix instead.
    """
    var = np.diagonal(cov)
    if np.any(var <= 0) or not np.all(np.isfinite(var)):
        return True
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.sqrt(np.outer(var, var))
        det_corr = np.linalg.det(corr)
    return not np.isfinite(det_corr) or abs(det_corr) < 1e-8


def nearest_neighbour_pooled_distance(
    means: np.ndarray,
    covs: np.ndarray,
    ns: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """`pooled_distance`, for every pair among many groups at once.

    `means` is (L, k), `covs` is (L, k, k) and `ns` is (L,): one mean,
    covariance and count per group (e.g. per label or per cluster). Returns
    the distance of the most isolated group to its nearest neighbour
    (`most_isolated_d`) and the number of groups whose nearest neighbour is
    further than `threshold` (`n_isolated`).

    As in `pooled_distance`, pairs where a group has fewer than 2 points or
    the pooled covariance is singular (see `_is_singular`) are ignored.
    """
    means = np.atleast_2d(means)
    L, k = means.shape
    if L == 0:
        return {"most_isolated_d": np.nan, "n_isolated": 0}

    valid = ns >= 2
    n_i, n_j = ns[:, None], ns[None, :]
    denom = n_i + n_j - 2
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pooled_cov = (
            (n_i - 1)[:, :, None, None] * covs[:, None]
            + (n_j - 1)[:, :, None, None] * covs[None, :]
        ) / denom[:, :, None, None]
        # Same singularity test as _is_singular, for all pairs at once.
        var = np.diagonal(pooled_cov, axis1=-2, axis2=-1)
        var_ok = np.all(var > 0, axis=-1)
        scale = np.sqrt(var[..., :, None] * var[..., None, :])
        corr = pooled_cov / scale
        det_corr = np.linalg.det(corr)
        singular = ~var_ok | ~np.isfinite(det_corr) | (np.abs(det_corr) < 1e-8)
        safe_cov = np.where(singular[:, :, None, None], np.eye(k), pooled_cov)
        inv_cov = np.linalg.inv(safe_cov)
        diff = means[:, None, :] - means[None, :, :]
        d2 = np.einsum("ijk,ijkl,ijl->ij", diff, inv_cov, diff)
        d = np.sqrt(np.clip(d2, 0, None))

    d[singular] = np.nan
    d[~valid, :] = np.nan
    d[:, ~valid] = np.nan
    np.fill_diagonal(d, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        nearest = np.nanmin(d, axis=1)
        most_isolated = np.nanmax(nearest)

    return {
        "most_isolated_d": float(most_isolated),
        "n_isolated": int(np.nansum(nearest > threshold)),
    }


def reduce_moments(
    df: pd.DataFrame, index: list[str], value_cols: list[str]
) -> pd.DataFrame:
    """Per-`index`-group count, mean and covariance of `value_cols`.

    `value_cols` is `['duration']` for labels or `['duration_1', 'duration_2']`
    for dyads. Covariance (ddof=1) is computed from deviations around each
    group's mean; variances that are only rounding error (see `VAR_RTOL`)
    are set to zero, as in `pooled_distance`.

    Returns a DataFrame indexed by `index` with columns `n`, `mean_{c}` for
    each `c` in `value_cols`, and `cov_{a}_{b}` for each unordered pair
    (including `a == b`, the variances).
    """
    work = df[index + value_cols].copy()
    pairs = list(combinations_with_replacement(value_cols, 2))
    g = work.groupby(index)
    centered = {c: work[c] - g[c].transform("mean") for c in value_cols}
    for a, b in pairs:
        work[f"_prod_{a}_{b}"] = centered[a] * centered[b]

    g = work.groupby(index)
    n = g.size()
    out = pd.DataFrame({"n": n})
    for c in value_cols:
        out[f"mean_{c}"] = g[c].mean()
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for a, b in pairs:
            out[f"cov_{a}_{b}"] = g[f"_prod_{a}_{b}"].sum() / (n - 1)
    for c in value_cols:
        negligible = out[f"cov_{c}_{c}"] <= (VAR_RTOL * out[f"mean_{c}"]) ** 2
        for a, b in pairs:
            if c in (a, b):
                out.loc[negligible, f"cov_{a}_{b}"] = 0.0
    return out


def isolation_stats(
    df: pd.DataFrame,
    by: list[str],
    value_cols: list[str],
    threshold: float,
    label_col: str = "label",
) -> pd.DataFrame:
    """Isolation statistics per `by`-group (e.g. per bird).

    Within each group, the units (or dyads) of every `label_col` value form
    one cluster in `len(value_cols)` dimensions; see
    `nearest_neighbour_pooled_distance` for the returned statistics.
    """
    moments = reduce_moments(df, by + [label_col], value_cols)
    means = moments[[f"mean_{c}" for c in value_cols]].to_numpy()
    ns = moments["n"].to_numpy()
    k = len(value_cols)
    covs = np.zeros((len(moments), k, k))
    for (i, j), (a, b) in zip(
        combinations_with_replacement(range(k), 2),
        combinations_with_replacement(value_cols, 2),
    ):
        covs[:, i, j] = covs[:, j, i] = moments[f"cov_{a}_{b}"].to_numpy()

    rows = []
    for key, idx in moments.groupby(level=by).indices.items():
        key = key if isinstance(key, tuple) else (key,)
        stats = nearest_neighbour_pooled_distance(
            means[idx], covs[idx], ns[idx], threshold
        )
        rows.append({**dict(zip(by, key)), **stats})
    return pd.DataFrame(rows).set_index(by)
