"""Bird identification: predict the bird of each chunk by changepoint detection.

The generic changepoint detection lives in `src.identify`, which is reusable
on its own; this module wires it up to this study's units columns
(`bird_id`, `chunk_idx`) and bird ID naming.
"""

import numpy as np
import pandas as pd

try:
    from ..identify import detect_changepoints, sequences_from_units
except ImportError:
    # Notebooks put src/ on sys.path and import this as `pipeline.X`.
    from identify import detect_changepoints, sequences_from_units


def predict_bird_ids(
    units: pd.DataFrame,
    *,
    half_window: int,
    min_gap: int,
    n_individuals: int | None = None,
    id_prefix: str = "pred-",
) -> dict:
    """Predict the bird of each chunk by detecting (number of birds - 1) changepoints.

    If `bird_id` is annotated, it gives the number of birds and the true
    boundaries, which are returned for evaluation; the predicted boundaries
    never use it. Without `bird_id` (RT), pass `n_individuals` instead.

    Args:
        units: A units DataFrame with `bird_id`, `chunk_idx`, `duration`
            columns (see `sequences_from_units`).
        half_window, min_gap: See `detect_changepoints`.
        n_individuals: Number of birds; defaults to the number of annotated
            `bird_id` values.
        id_prefix: Prefix of the predicted bird IDs, e.g. "bvs-pred-"
            gives "bvs-pred-001".

    Returns:
        `detect_changepoints`'s dict, plus:
            n_chunks: number of chunks.
            true_breakpoints: annotated boundaries, in the same chunk order
                as `breakpoints` (empty without `bird_id`).
            bird_id_pred: predicted bird ID for every row of `units`.
    """
    if n_individuals is None:
        n_individuals = units["bird_id"].nunique()
        if n_individuals == 0:
            raise ValueError(
                "units['bird_id'] is entirely missing; pass `n_individuals` explicitly"
            )
    sequences = sequences_from_units(units)

    result = detect_changepoints(
        sequences,
        n_changepoints=n_individuals - 1,
        half_window=half_window,
        min_gap=min_gap,
    )

    group_keys = (
        units[["bird_id", "chunk_idx"]]
        .drop_duplicates()
        .sort_values(["bird_id", "chunk_idx"])
    )
    true_breakpoints = (
        np.nonzero(np.diff(pd.factorize(group_keys["bird_id"].to_numpy())[0]) != 0)[0]
        + 1
    )

    group_keys["bird_id_pred"] = [
        f"{id_prefix}{label + 1:03d}" for label in result["labels"]
    ]
    bird_id_pred = (
        units[["bird_id", "chunk_idx"]]
        .merge(group_keys, on=["bird_id", "chunk_idx"], how="left")["bird_id_pred"]
        .to_numpy()
    )

    result["n_chunks"] = len(sequences)
    result["true_breakpoints"] = true_breakpoints
    result["bird_id_pred"] = bird_id_pred

    return result
