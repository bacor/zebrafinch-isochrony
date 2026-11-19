"""Find a known number of changepoints in a sequence of items, using DTW.

Used for bird identification, where the items are chunks (variable-length
duration sequences) and a changepoint is a boundary between birds. Items are
compared pairwise with DTW, which handles variable length and internal
alignment. DTW is the standard normalized distance of the dtw-python
package (its default symmetric2 step pattern, normalized by n + m). Each
candidate split point is scored by comparing DTW distances *across* the split to DTW distances *within* each side
(`changepoint_scores`): items on the same side of a real changepoint are
more alike than items on opposite sides, so the score peaks there.

`detect_changepoints` is the main entry point. It returns the
`n_changepoints` most prominent local maxima of the score signal, so there
is no threshold to pick. The functions below follow the steps in order. `half_window` and `min_gap` have no defaults
because they depend on how many items make up one regime (one bird).
"""

import numpy as np
import pandas as pd
from dtw import dtw
from numba import njit
from scipy.signal import find_peaks


def sequences_from_units(
    units: pd.DataFrame,
    duration_col: str = "duration",
    chunk_col: str = "chunk_idx",
    bird_col: str = "bird_id",
) -> list[np.ndarray]:
    """One duration array per chunk of `units`, ordered by bird, then chunk.

    Chunk indices are only unique within a bird, so chunks are grouped by
    bird and chunk. Missing bird IDs (as in RT) form a single group.
    """
    group_cols = [col for col in (bird_col, chunk_col) if col in units.columns]
    return [
        group[duration_col].to_numpy(dtype=np.float64)
        for _, group in units.groupby(group_cols, sort=True, dropna=False)
    ]


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized DTW distance between two duration sequences (dtw-python defaults)."""
    return dtw(a, b, distance_only=True).normalizedDistance


def banded_dtw(sequences: list[np.ndarray], band: int) -> np.ndarray:
    """DTW distance from each sequence i to sequences i+1 .. i+band.

    This is a band of the sequence-by-sequence distance matrix (only nearby
    sequences are compared), not a Sakoe-Chiba band within one alignment.

    Returns an array of shape (n, band); entry `[i, k]` holds the distance
    between sequences i and i+k+1, or -1.0 where i+k+1 is out of range.
    """
    n = len(sequences)
    out = np.full((n, band), -1.0)
    for i in range(n):
        for k in range(min(band, n - i - 1)):
            out[i, k] = dtw_distance(sequences[i], sequences[i + k + 1])
    return out


@njit
def cross_within_score(D: np.ndarray, half_window: int) -> np.ndarray:
    """Cross-vs-within score at every split point, from a banded distance matrix.

    `D` must have been built with band >= 2 * half_window, with
    `D[i, k]` = distance(item_i, item_{i+k+1}). The score at split point t
    is the mean cross distance minus the mean of the two within-side means
    (see `changepoint_scores`). Returns one score per split point t from
    `half_window` to `n - half_window` (inclusive), so t equals `half_window`
    plus the array index.
    """
    n = D.shape[0]
    lo, hi = half_window, n - half_window + 1
    scores = np.empty(hi - lo, dtype=np.float64)

    for idx in range(hi - lo):
        t = lo + idx
        cross_sum, cross_n = 0.0, 0
        for p in range(t - half_window, t):
            for q in range(t, t + half_window):
                cross_sum += D[p, q - p - 1]
                cross_n += 1

        left_sum, left_n = 0.0, 0
        for p in range(t - half_window, t):
            for q in range(p + 1, t):
                left_sum += D[p, q - p - 1]
                left_n += 1

        right_sum, right_n = 0.0, 0
        for p in range(t, t + half_window):
            for q in range(p + 1, t + half_window):
                right_sum += D[p, q - p - 1]
                right_n += 1

        cross_mean = cross_sum / cross_n
        left_mean = left_sum / left_n if left_n > 0 else 0.0
        right_mean = right_sum / right_n if right_n > 0 else 0.0
        scores[idx] = cross_mean - 0.5 * (left_mean + right_mean)

    return scores


def changepoint_scores(
    sequences: list[np.ndarray], half_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-vs-within DTW changepoint score at every candidate split point.

    At split point t, compares DTW distances between the `half_window`
    items before t and the `half_window` after it (cross) against DTW
    distances within each side (within). A real changepoint shows cross >>
    within; inside one regime they're similar since neighbouring items are
    all renditions of the same underlying process.

    Returns:
        `(positions, scores)`: positions are candidate split points t (the
        split falls between item t-1 and item t); `scores[i]` is the
        cross-minus-within statistic at `positions[i]`. Higher means more
        likely a real changepoint.
    """
    n = len(sequences)
    D = banded_dtw(sequences, band=2 * half_window)

    positions = np.arange(half_window, n - half_window + 1)
    scores = cross_within_score(D, half_window)
    return positions, scores


def detect_changepoints(
    sequences: list[np.ndarray],
    n_changepoints: int,
    *,
    half_window: int,
    min_gap: int,
) -> dict:
    """Find a known number of changepoints in a sequence of items.

    Ranks every local maximum of `changepoint_scores` by prominence
    (`scipy.signal.find_peaks`, subject to `min_gap`) and keeps the
    `n_changepoints` strongest.

    Args:
        n_changepoints: How many changepoints to find (e.g. the number of
            birds minus one).
        half_window: Items compared on each side of a candidate split (see
            `changepoint_scores`). Too small is noisy; too large blurs
            changepoints closer together than `half_window`.
        min_gap: Minimum spacing between changepoints: roughly the
            shortest plausible number of items in one regime.

    Returns:
        A dict with keys:
            labels: per-item 0-based segment id.
            n_changepoints: `len(breakpoints)` (equals the input).
            breakpoints: 0-based item indices where a new segment starts.
            positions, scores: the changepoint score signal (`changepoint_scores`).
            prominences: each breakpoint's peak prominence.
            params: hyperparameters this call was made with.

    Raises:
        ValueError: If fewer than `n_changepoints` peaks lie at least `min_gap` apart.
    """
    n = len(sequences)
    positions, scores = changepoint_scores(sequences, half_window)

    peaks, props = find_peaks(scores, prominence=0, distance=min_gap)
    if len(peaks) < n_changepoints:
        raise ValueError(
            f"only {len(peaks)} candidate changepoints available (min_gap={min_gap}), "
            f"fewer than the {n_changepoints} requested; try a smaller min_gap."
        )
    top = np.sort(np.argsort(props["prominences"])[::-1][:n_changepoints])

    breakpoints = positions[peaks[top]]
    prominences = props["prominences"][top]

    labels = labels_from_breakpoints(n, breakpoints)

    return {
        "labels": labels,
        "n_changepoints": len(breakpoints),
        "breakpoints": breakpoints,
        "positions": positions,
        "scores": scores,
        "prominences": prominences,
        "params": {
            "n_changepoints": n_changepoints,
            "half_window": half_window,
            "min_gap": min_gap,
        },
    }


def labels_from_breakpoints(n: int, breakpoints: np.ndarray) -> np.ndarray:
    """Map each item index (0-based) to a 0-based segment id."""
    return np.searchsorted(breakpoints, np.arange(n), side="right")
