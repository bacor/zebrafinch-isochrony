"""Per-label metrics: the durations of all of a bird's units with one label."""

from __future__ import annotations

import pandas as pd


def analyze_labels(units: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Count and duration statistics per `by + ['label']` group.

    Columns: `count`, `dur_mean`, `dur_std`, `dur_min`, `dur_max` and the
    quartiles `dur_q1`, `dur_q2`, `dur_q3`.
    """
    durations = units.groupby(by + ["label"])["duration"]
    stats = durations.agg(
        count="size", dur_mean="mean", dur_std="std", dur_min="min", dur_max="max"
    )
    quartiles = durations.quantile([0.25, 0.5, 0.75]).unstack()
    stats[["dur_q1", "dur_q2", "dur_q3"]] = quartiles.to_numpy()
    return stats
