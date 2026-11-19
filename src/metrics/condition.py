"""Per-bird and per-condition summary metrics.

Every metric is first reduced to one value per bird, from that bird's
label-, cluster- and bird-level metrics (`label.py`, `cluster.py`, `bird.py`). A
condition's value is then the mean over its birds, where a boolean per-bird
metric becomes the percentage of birds for which it holds. Permutation
draws go through the same two functions, so a reported value and its tested
value are always the same number.
"""

from __future__ import annotations

import pandas as pd

from .dyad import threshold50
from .thresholds import MIN_CLUSTER_SHARE, RHO, TAU

# Metrics tested against the label-shuffle null, and the side of the null
# where real values are expected: 'low' (below) or 'high' (above).
TESTS = {
    "inconsistency": "low",
    "isolated_label": "high",
    "cluster_inconsistency": "low",
    "isolated_cluster": "high",
    "most_isochronous": "high",
    "median_isochrony": "high",
    "least_isochronous": "high",
}


def bird_metrics(
    label: pd.DataFrame, cluster: pd.DataFrame, bird: pd.DataFrame, by: list[str]
) -> pd.DataFrame:
    """One row per bird, from label-, cluster- and bird-level metrics.

    `label` and `cluster` are indexed by `by + ['label']` and `bird` by `by`,
    as returned by `metrics.analyze`. Besides descriptive columns, it holds:

    - inconsistency: median over the bird's labels of the within-label SD (ms).
    - cluster_inconsistency: the same for clusters, where a cluster's SD
      is the mean of the SDs of its first and second durations.
    - isolated_label / isolated_cluster: whether the bird has at least one
      isolated label / cluster.
    - most_isochronous / median_isochrony / least_isochronous: the lowest,
      median and highest non-isochrony |r - 0.5| of the centroids of the
      bird's frequent clusters (see `MIN_CLUSTER_SHARE`).
    - repetitions: percentage of dyads that repeat a label (e.g. A-A).
    """
    per_label = label.groupby(level=by)
    per_cluster = cluster.groupby(level=by)
    cluster_sd = cluster[["centroid_x_std", "centroid_y_std"]].mean(axis=1)
    share = cluster["count"] / per_cluster["count"].transform("sum")
    frequent = cluster.loc[
        share >= MIN_CLUSTER_SHARE, "centroid_non_isochrony"
    ].groupby(level=by)
    repeated = cluster["count"].where(cluster["label_x"] == cluster["label_y"], 0)
    return pd.DataFrame(
        {
            "n_labels": bird["n_labels"],
            "units_per_label": per_label["count"].median(),
            "dur_mean": per_label["dur_mean"].median(),
            "inconsistency": per_label["dur_std"].median(),
            "n_isolated_labels": bird["n_isolated_labels"],
            "isolated_label": bird["n_isolated_labels"] > 0,
            "cluster_inconsistency": cluster_sd.groupby(level=by).median(),
            "isolated_cluster": bird["n_isolated_clusters"] > 0,
            "most_isochronous": frequent.min(),
            "median_isochrony": frequent.median(),
            "least_isochronous": frequent.max(),
            "repetitions": repeated.groupby(level=by).sum()
            / per_cluster["count"].sum()
            * 100,
        }
    )


def condition_metrics(birds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean over birds per `by`-group; boolean columns become % of birds.

    Also adds `n_birds`.
    """
    flags = birds.select_dtypes(bool).columns
    out = birds.astype({c: float for c in flags}).groupby(level=by).mean()
    out[flags] *= 100
    out["n_birds"] = birds.groupby(level=by).size()
    return out


def isochrony_metrics(dyads: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Label-free isochrony metrics per `by`-group, from single dyads.

    `dyads` holds `dyad.dyad_scores` for every dyad, indexed by `by` plus
    `bird_id`. Per bird: the mean non-isochrony, the mean asymmetry (ms)
    and the percentage of isochronous dyads under the relative (`RHO`) and
    absolute (`TAU`) criterion, then averaged over birds. `rho50`/`tau50`
    are the thresholds at which half of the dyads are isochronous
    (`dyad.threshold50`).
    """
    birds = (
        pd.DataFrame(
            {
                "non_isochrony": dyads["non_isochrony"],
                "asymmetry": dyads["asymmetry"],
                "perc_rel_isochronous": dyads["non_isochrony"] < RHO,
                "perc_abs_isochronous": dyads["asymmetry"] < TAU,
            }
        )
        .groupby(level=by + ["bird_id"])
        .mean()
    )
    out = condition_metrics(birds, by)
    out[["perc_rel_isochronous", "perc_abs_isochronous"]] *= 100
    grouped = dyads.groupby(level=by)
    out["rho50"] = grouped["non_isochrony"].apply(threshold50)
    out["tau50"] = grouped["asymmetry"].apply(threshold50)
    return out.drop(columns="n_birds")
