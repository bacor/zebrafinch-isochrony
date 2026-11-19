"""Per-cluster metrics: all of a bird's dyads with one label (e.g. 'a-b').

Dyad labels join the labels of their two units with a dash; a cluster is a
repetition when both are equal (`label_x == label_y`).
"""

from __future__ import annotations

import pandas as pd


def analyze_clusters(dyads: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Size, centroid and rhythm of each `by + ['label']` cluster of `dyads`.

    `dyads` comes from `rhythm_types.build_dyads`. The centroid is
    (`centroid_x_mean`, `centroid_y_mean`), with SDs `centroid_x_std` and
    `centroid_y_std`. These cover only the units in this context (e.g. A
    when followed by B), so they can differ from the label-level statistics.
    `centroid_asymmetry` is |x - y|, `centroid_rhythm_ratio` is
    r = x / (x + y) and `centroid_non_isochrony` is |r - 0.5| (which is
    proportional to nPVI), all of the centroid. `label_x` and `label_y` are
    the labels of the first and second unit.
    """
    clusters = dyads.groupby(by + ["label"])
    x, y = clusters["duration_1"], clusters["duration_2"]
    stats = pd.DataFrame(
        {
            "count": clusters.size(),
            "centroid_x_mean": x.mean(),
            "centroid_x_std": x.std(),
            "centroid_y_mean": y.mean(),
            "centroid_y_std": y.std(),
        }
    )
    parts = stats.index.get_level_values("label").str.split("-", n=1)
    stats.insert(0, "label_x", parts.str[0])
    stats.insert(1, "label_y", parts.str[1])
    mean_x, mean_y = stats["centroid_x_mean"], stats["centroid_y_mean"]
    stats["centroid_asymmetry"] = (mean_x - mean_y).abs()
    stats["centroid_rhythm_ratio"] = mean_x / (mean_x + mean_y)
    stats["centroid_non_isochrony"] = (stats["centroid_rhythm_ratio"] - 0.5).abs()
    return stats
