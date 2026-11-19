"""Label-, cluster-, bird- and condition-level metrics (see README.md)."""

from __future__ import annotations

import pandas as pd

from .bird import analyze_birds
from .cluster import analyze_clusters
from .label import analyze_labels

try:
    from ..rhythm_types import build_dyads
except ImportError:
    # Imported as a top-level package (``src/`` on sys.path, as in the notebooks)
    from rhythm_types import build_dyads


def analyze(
    units: pd.DataFrame, by: list[str], label_col: str = "label"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label-, cluster- and bird-level metrics of `units`, labeled by `label_col`.

    `by` identifies a bird (e.g. `['bird_id']`); `units` needs the columns
    `by`, `chunk_idx`, `start`, `duration` and `label_col`. Units without a
    label, and dyads containing one, are left out. Returns
    `(label, cluster, bird)`.
    """
    units = units.assign(label=units[label_col])
    dyads = build_dyads(units, by + ["chunk_idx"])
    dyads = dyads[dyads["label"].notna()]
    units = units[units["label"].notna()]
    return (
        analyze_labels(units, by),
        analyze_clusters(dyads, by),
        analyze_birds(units, dyads, by),
    )
