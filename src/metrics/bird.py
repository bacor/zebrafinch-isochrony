"""Per-bird metrics: counts, chunk lengths and durations, and isolation."""

from __future__ import annotations

import pandas as pd

from .distance import isolation_stats
from .thresholds import ISOLATION_D


def analyze_birds(
    units: pd.DataFrame, dyads: pd.DataFrame, by: list[str]
) -> pd.DataFrame:
    """Descriptive and isolation metrics, one row per `by`-group (bird).

    `dyads` comes from `rhythm_types.build_dyads`. Descriptive columns:
    `n_labels`, `n_units`, `n_dyads`, `n_chunks`, units per chunk
    (`chunk_len_*`) and chunk duration from first onset to last offset
    (`chunk_dur_*`).

    Isolation (`distance.isolation_stats`): `n_isolated_labels` counts the
    labels whose nearest other label is further than `ISOLATION_D` (Cohen's
    d), and `most_isolated_label_d` is the largest nearest-label distance.
    `n_isolated_clusters`/`most_isolated_cluster_d` are the same for
    clusters, using the Mahalanobis distance between their dyads.
    """
    chunks = units.assign(end=units["start"] + units["duration"]).groupby(
        by + ["chunk_idx"]
    )
    chunk_len = chunks.size().groupby(level=by)
    chunk_dur = (chunks["end"].max() - chunks["start"].min()).groupby(level=by)
    birds = pd.DataFrame(
        {
            "n_labels": units.groupby(by)["label"].nunique(),
            "n_units": units.groupby(by).size(),
            "n_dyads": dyads.groupby(by).size(),
            "n_chunks": chunk_len.size(),
            "chunk_len_mean": chunk_len.mean(),
            "chunk_len_min": chunk_len.min(),
            "chunk_len_max": chunk_len.max(),
            "chunk_dur_mean": chunk_dur.mean(),
            "chunk_dur_std": chunk_dur.std(),
            "chunk_dur_min": chunk_dur.min(),
            "chunk_dur_max": chunk_dur.max(),
        }
    )
    labels = isolation_stats(units, by, ["duration"], threshold=ISOLATION_D)
    clusters = isolation_stats(
        dyads, by, ["duration_1", "duration_2"], threshold=ISOLATION_D
    )
    return birds.join(
        [
            labels.rename(
                columns={
                    "n_isolated": "n_isolated_labels",
                    "most_isolated_d": "most_isolated_label_d",
                }
            ),
            clusters.rename(
                columns={
                    "n_isolated": "n_isolated_clusters",
                    "most_isolated_d": "most_isolated_cluster_d",
                }
            ),
        ]
    )
