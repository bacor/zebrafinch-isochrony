"""Isochrony of single dyads: pairs of successive durations (x, y) within a chunk.

Everything here is independent of labels, unlike `cluster.py`, which
describes all same-label dyads together. Dyads are built from units by
`rhythm_types.build_dyads`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dyad_scores(dyads: pd.DataFrame) -> pd.DataFrame:
    """Isochrony of every dyad, from its own two durations.

    `dyads` has `duration_1` and `duration_2` columns. Returns `asymmetry`
    (|x - y|, in ms) and `non_isochrony` (|r - 0.5| with r = x / (x + y)).

    Both are rounded to 9 decimals: durations annotated in whole
    milliseconds often give exact ties with a threshold (e.g. |x - y| = 20
    ms), which floating-point noise from unit conversion would otherwise
    put on either side of it.
    """
    d1, d2 = dyads["duration_1"], dyads["duration_2"]
    return pd.DataFrame(
        {
            "asymmetry": (d1 - d2).abs().round(9),
            "non_isochrony": (d1 / (d1 + d2) - 0.5).abs().round(9),
        }
    )


def threshold50(values: pd.Series) -> float:
    """Threshold at which half of the dyads meet an isochrony criterion.

    `values` holds one condition's criterion value for every dyad (the
    `asymmetry` or `non_isochrony` of `dyad_scores`, giving tau50 in ms
    or rho50), with `bird_id` in its index. Returns the smallest observed
    value t such that, averaged over birds (each bird weighted equally), at
    least half of the dyads have a value <= t.
    """
    bird = values.index.get_level_values("bird_id")
    weights = 1 / values.groupby(bird).transform("size")
    return float(
        np.quantile(
            values.to_numpy(), 0.5, weights=weights.to_numpy(), method="inverted_cdf"
        )
    )
