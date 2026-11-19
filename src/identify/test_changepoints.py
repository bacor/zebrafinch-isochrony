import numpy as np
import pandas as pd
import pytest

from identify.changepoints import (
    banded_dtw,
    changepoint_scores,
    cross_within_score,
    detect_changepoints,
    dtw_distance,
    labels_from_breakpoints,
    sequences_from_units,
)

N_BIRDS = 5
CHUNKS_PER_BIRD = 80


def _synthetic_birds(seed=0):
    """Chunks of N_BIRDS birds, CHUNKS_PER_BIRD each, in bird order.

    Each bird has its own fixed template of 12 durations (with its own tempo);
    a chunk is a noisy window of 4-8 successive template durations. Birds are
    so distinct that changepoint detection finds the boundaries exactly.
    """
    rng = np.random.default_rng(seed)
    sequences = []
    for bird in range(N_BIRDS):
        template = rng.uniform(60, 250, size=12) + bird * 40
        for _ in range(CHUNKS_PER_BIRD):
            length = rng.integers(4, 9)
            start = rng.integers(0, len(template) - length + 1)
            window = template[start : start + length] + rng.normal(0, 2.0, size=length)
            sequences.append(np.maximum(window, 1.0))
    return sequences


def _banded_from_full(D_full: np.ndarray, band: int) -> np.ndarray:
    """The banded layout of `banded_dtw`, from a full n x n distance matrix."""
    n = D_full.shape[0]
    D = np.full((n, band), -1.0)
    for i in range(n):
        for k in range(band):
            j = i + k + 1
            if j < n:
                D[i, k] = D_full[i, j]
    return D


# --- DTW ---


def test_dtw_distance_is_normalized_symmetric2_distance():
    # dtw-python's default step pattern symmetric2 counts the start cell once
    # and a diagonal step twice; on the diagonal path that is
    # |0 - 1| + 2 * |0 - 1| = 3, and normalizedDistance divides by n + m = 4.
    assert dtw_distance(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.75


def test_banded_dtw_matches_dtw_distance_and_masks_out_of_range():
    sequences = [
        np.array([100.0, 150.0]),
        np.array([90.0, 200.0]),
        np.array([80.0, 120.0]),
    ]
    D = banded_dtw(sequences, band=2)

    assert D[0, 0] == dtw_distance(sequences[0], sequences[1])
    assert D[0, 1] == dtw_distance(sequences[0], sequences[2])
    assert D[1, 0] == dtw_distance(sequences[1], sequences[2])
    assert D[1, 1] == -1.0  # sequence index 3 doesn't exist
    assert D[2, 0] == -1.0
    assert D[2, 1] == -1.0


# --- Changepoint score ---


def test_cross_within_score_by_hand():
    # The first split point is t = half_window = 2, with items 0, 1 on the left
    # and 2, 3 on the right:
    # cross = mean(D02, D03, D12, D13) = (2 + 3 + 4 + 5) / 4 = 3.5
    # within = (D01 + D23) / 2 = (1 + 2) / 2 = 1.5, so the score is 3.5 - 1.5 = 2.
    D_full = np.array(
        [
            [0, 1, 2, 3, 9],
            [1, 0, 4, 5, 9],
            [2, 4, 0, 2, 9],
            [3, 5, 2, 0, 9],
            [9, 9, 9, 9, 0],
        ],
        dtype=np.float64,
    )
    scores = cross_within_score(_banded_from_full(D_full, band=4), half_window=2)
    assert scores[0] == 2.0


def test_cross_within_score_constant_without_regime_change():
    # With D[i, j] = |i - j| and half_window 5, every split point looks the same:
    # cross pairs have mean gap 5, the 10 pairs within 5 items have mean gap
    # (4*1 + 3*2 + 2*3 + 1*4) / 10 = 2, so the score is 5 - 2 = 3 everywhere.
    n = 20
    D_full = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(np.float64)
    scores = cross_within_score(_banded_from_full(D_full, band=10), half_window=5)
    np.testing.assert_allclose(scores, 3.0)


def test_cross_within_score_peaks_at_a_regime_change():
    # Not DTW-derived: the score works on any banded distance matrix.
    n = 20
    change_at = 10
    D_full = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            same_block = (i < change_at) == (j < change_at)
            D_full[i, j] = abs(i - j) * (1.0 if same_block else 20.0)

    half_window = 5
    scores = cross_within_score(_banded_from_full(D_full, band=10), half_window)
    assert half_window + np.argmax(scores) == change_at


def test_changepoint_scores_at_a_block_boundary():
    # 30 items [100, 100] then 30 items [200, 200]. Within a block DTW is 0;
    # across blocks it is d = (100 + 2 * 100) / 4 = 75 (see the DTW test above).
    # t = 30: all cross pairs differ, no within pair does: score d = 75.
    # t = 29: item 29 sits on the right but belongs to block A. Cross: 5 x 4 of
    # 25 pairs differ, mean 0.8 d. Right: 4 of 10 pairs differ, mean 0.4 d;
    # left: 0. Score 0.8 d - 0.4 d / 2 = 0.6 d = 45.
    # |t - 30| >= 5: both windows lie inside one block, score 0.
    sequences = [np.array([100.0, 100.0])] * 30 + [np.array([200.0, 200.0])] * 30
    positions, scores = changepoint_scores(sequences, half_window=5)

    # The first split point with 5 items on its left is t = 5.
    assert positions[0] == 5
    assert np.all(np.diff(positions) == 1)
    score_at = dict(zip(positions, scores))
    assert score_at[30] == pytest.approx(75.0)
    assert score_at[29] == pytest.approx(45.0)
    assert score_at[31] == pytest.approx(45.0)
    assert all(score_at[t] == 0.0 for t in positions if abs(t - 30) >= 5)


# --- Changepoint detection ---


def test_detect_changepoints_recovers_synthetic_boundaries():
    sequences = _synthetic_birds()
    true_breaks = [80, 160, 240, 320]  # a new bird every CHUNKS_PER_BIRD chunks

    result = detect_changepoints(
        sequences, n_changepoints=4, half_window=15, min_gap=20
    )

    np.testing.assert_array_equal(result["breakpoints"], true_breaks)
    np.testing.assert_array_equal(
        result["labels"], np.repeat(np.arange(N_BIRDS), CHUNKS_PER_BIRD)
    )
    assert result["n_changepoints"] == 4


def test_detect_changepoints_raises_when_not_enough_candidates():
    # Identical items give a flat score without any peak.
    sequences = [np.array([1.0, 2.0])] * 10
    with pytest.raises(ValueError, match="fewer than the 3 requested"):
        detect_changepoints(sequences, n_changepoints=3, half_window=2, min_gap=2)


def test_labels_from_breakpoints():
    labels = labels_from_breakpoints(5, np.array([2, 4]))
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2])


# --- Sequences from units ---


def test_sequences_from_units_groups_by_chunk_idx_in_order():
    units = pd.DataFrame(
        {
            "duration": [1.0, 2.0, 3.0, 4.0, 5.0],
            "chunk_idx": [2, 2, 1, 1, 1],
        }
    )

    sequences = sequences_from_units(units)

    # sorted by chunk_idx (1 before 2), not by row order
    assert len(sequences) == 2
    np.testing.assert_array_equal(sequences[0], [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(sequences[1], [1.0, 2.0])


def test_sequences_from_units_disambiguates_per_bird_local_chunk_idx():
    # chunk_idx is only unique within a bird (as in NS), so grouping by
    # chunk_idx alone would merge chunk 1 of bird A with chunk 1 of bird B.
    units = pd.DataFrame(
        {
            "duration": [1.0, 2.0, 3.0, 4.0],
            "chunk_idx": [1, 1, 1, 1],
            "bird_id": ["A", "A", "B", "B"],
        }
    )

    sequences = sequences_from_units(units)

    assert len(sequences) == 2
    np.testing.assert_array_equal(sequences[0], [1.0, 2.0])
    np.testing.assert_array_equal(sequences[1], [3.0, 4.0])
