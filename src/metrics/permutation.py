"""Permutation tests against the shuffled baseline.

The shuffled baseline shuffles the labels within each chunk. Durations and
their order are preserved, but labels no longer map to durations.
`shuffle_labels_within_chunk` produces one shuffle; `permutation_test`
compares an observed statistic to its values under many shuffles.
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs


def shuffle_labels_within_chunk(
    units: pd.DataFrame,
    rng: np.random.Generator,
    by: list[str],
    label_col: str = "label",
) -> pd.DataFrame:
    """Shuffle `label_col` within each chunk (`by` group); other columns are kept."""
    shuffled = units.copy()
    shuffled[label_col] = units.groupby(by)[label_col].transform(
        lambda s: rng.permutation(s.values)
    )
    return shuffled


def _summarize_draws(observed: pd.DataFrame, draws: np.ndarray, n: int) -> pd.DataFrame:
    """Summarize `draws` (shape `(n, *observed.shape)`); see `permutation_test`."""
    obs = observed.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        null_mean = np.nanmean(draws, axis=0)
        null_std = np.nanstd(draws, axis=0, ddof=1)
    valid = ~np.isnan(null_mean)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(valid, (obs - null_mean) / null_std, np.nan)
        p = np.where(valid, (np.sum(draws <= obs, axis=0) + 1) / (n + 1), np.nan)
        p_high = np.where(valid, (np.sum(draws >= obs, axis=0) + 1) / (n + 1), np.nan)

    out = {}
    for j, col in enumerate(observed.columns):
        out[f"{col}_null_mean"] = null_mean[:, j]
        out[f"{col}_null_std"] = null_std[:, j]
        out[f"{col}_z"] = z[:, j]
        out[f"{col}_p"] = p[:, j]
        out[f"{col}_p_high"] = p_high[:, j]
    return pd.DataFrame(out, index=observed.index)


def _report_progress(
    i: int,
    n: int,
    start: float,
    progress_every: int | None,
    progress_label: str,
    n_jobs: int,
) -> None:
    if progress_every and (i + 1) % progress_every == 0:
        elapsed = time.time() - start
        remaining = elapsed / (i + 1) * (n - i - 1)
        workers = f", {effective_n_jobs(n_jobs)} workers" if n_jobs != 1 else ""
        print(
            f"{progress_label}{i + 1}/{n} draws, {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining{workers}",
            flush=True,
        )


def _draws(
    draw_null: Callable[[np.random.Generator], pd.DataFrame],
    n: int,
    seed: int | None,
    n_jobs: int,
    progress_every: int | None,
    progress_label: str,
) -> Iterator[pd.DataFrame]:
    """Yield `n` draws of `draw_null`, reporting progress.

    Every draw gets its own generator spawned from `seed`, so the draws do
    not depend on `n_jobs` (with `n_jobs=1`, joblib runs them in-process).
    """
    start = time.time()
    seeds = np.random.SeedSequence(seed).spawn(n)
    jobs = (delayed(draw_null)(np.random.default_rng(s)) for s in seeds)
    for i, draw in enumerate(Parallel(n_jobs=n_jobs, return_as="generator")(jobs)):
        _report_progress(i, n, start, progress_every, progress_label, n_jobs)
        yield draw


def permutation_test(
    observed: pd.DataFrame,
    draw_null: Callable[[np.random.Generator], pd.DataFrame],
    n: int = 500,
    seed: int | None = None,
    n_jobs: int = 1,
    progress_every: int | None = None,
    progress_label: str = "",
) -> pd.DataFrame:
    """Compare `observed` to `n` draws of `draw_null(rng)`.

    Args:
        observed: Observed values, e.g. one row per condition.
        draw_null: Returns one draw of the shuffled baseline from a generator,
            as a DataFrame whose index/columns are a superset of `observed`'s
            (extra rows are ignored; a missing row becomes an all-NaN draw).
        n_jobs: Number of parallel workers; see `_draws`.
        progress_every: Print progress every that many draws (prefixed by
            `progress_label`).

    Returns:
        A DataFrame indexed like `observed` with, per column `col`:
        `{col}_null_mean`, `{col}_null_std`: mean and SD over the draws.
        `{col}_z`: (observed - null_mean) / null_std.
        `{col}_p`: one-sided p-value (k + 1) / (n + 1), with k the number of
        draws <= observed; for values expected to be low (e.g. inconsistency).
        `{col}_p_high`: the same with draws >= observed, for values expected
        to be high (e.g. differentiation).
    """
    draws = np.full((n, *observed.shape), np.nan)
    for i, draw in enumerate(
        _draws(draw_null, n, seed, n_jobs, progress_every, progress_label)
    ):
        draw = draw.reindex(observed.index)
        draws[i] = draw[observed.columns].to_numpy()
    return _summarize_draws(observed, draws, n)
