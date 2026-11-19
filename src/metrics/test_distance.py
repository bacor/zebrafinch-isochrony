import numpy as np
import pandas as pd
import pingouin
import pytest

from metrics.distance import (
    isolation_stats,
    nearest_neighbour_pooled_distance,
    pooled_distance,
    reduce_moments,
)

# Four points on the corners of a 2 x 2 square: mean (1, 1), covariance (4/3) * I.
_SQUARE = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])

# Constant durations of 100 and 200 ms, computed from onsets in seconds as
# (k + 0.1 - k) * 1000, so their variance is rounding noise (~1e-26), not 0.
_NOISY_100 = np.array([(k + 0.1 - k) * 1000 for k in range(1, 6)])
_NOISY_200 = np.array([(k + 0.2 - k) * 1000 for k in range(1, 6)])


def _moments(groups):
    """Means, covariances and counts of a list of (n, k) samples."""
    means = np.array([g.mean(axis=0) for g in groups])
    covs = np.array([np.atleast_2d(np.cov(g, rowvar=False)) for g in groups])
    ns = np.array([len(g) for g in groups])
    return means, covs, ns


# --- pooled_distance: Cohen's d in 1D, Mahalanobis distance in k > 1 dimensions ---


def test_pooled_distance_1d():
    # Both samples have variance 4/3 and means 1 and 3: d = 2 / sqrt(4/3) = sqrt(3)
    a = np.array([0.0, 0.0, 2.0, 2.0])
    b = np.array([2.0, 2.0, 4.0, 4.0])
    assert pooled_distance(a, b) == pytest.approx(np.sqrt(3))
    assert pooled_distance(b, a) == pytest.approx(np.sqrt(3))


def test_pooled_distance_1d_matches_cohens_d():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 30), rng.normal(3, 1.5, 25)
    expected = abs(pingouin.compute_effsize(a, b, eftype="cohen"))
    assert pooled_distance(a, b) == pytest.approx(expected)


def test_pooled_distance_2d():
    # Pooled covariance (4/3) * I and centroid difference (3, 4) of length 5:
    # d = 5 / sqrt(4/3)
    assert pooled_distance(_SQUARE, _SQUARE + [3, 4]) == pytest.approx(
        5 / np.sqrt(4 / 3)
    )


def test_pooled_distance_is_scale_free():
    # The same samples in seconds: the covariance determinant is only ~1e-12,
    # but the distance is unchanged.
    a, b = _SQUARE / 1000, (_SQUARE + [3, 4]) / 1000
    assert pooled_distance(a, b) == pytest.approx(5 / np.sqrt(4 / 3))


def test_pooled_distance_singleton_returns_nan():
    assert np.isnan(pooled_distance(np.array([1.0]), np.array([1.0, 2.0, 3.0])))


def test_pooled_distance_1d_zero_variance_returns_nan():
    assert np.isnan(pooled_distance(np.array([1.0, 1.0]), np.array([1.0, 1.0])))


def test_pooled_distance_rounding_variance_counts_as_zero():
    assert np.isnan(pooled_distance(_NOISY_100, _NOISY_200))


def test_pooled_distance_2d_zero_covariance_returns_nan():
    a = np.array([[1.0, 1.0], [1.0, 1.0]])
    b = np.array([[5.0, 5.0], [5.0, 5.0]])
    assert np.isnan(pooled_distance(a, b))


def test_pooled_distance_2d_parallel_pairs_return_nan():
    # Two points per sample lie on a line; both lines run along (1, 1), so the
    # pooled covariance has rank 1.
    a = np.array([[800.0, 200.0], [850.0, 250.0]])
    b = np.array([[600.0, 900.0], [620.0, 920.0]])
    assert np.isnan(pooled_distance(a, b))


# --- nearest_neighbour_pooled_distance: all pairs at once ---


@pytest.mark.parametrize(
    "threshold, n_isolated",
    [(1.9, 3), (3, 1), (4, 0)],  # a nearest distance equal to threshold is not isolated
)
def test_nearest_neighbour_isolation_threshold(threshold, n_isolated):
    # Unit variance and means 0, 2 and 6: d is 2, 4 and 6 for the three pairs,
    # so the nearest-neighbour distances are 2, 2 and 4.
    means = np.array([[0.0], [2.0], [6.0]])
    covs = np.ones((3, 1, 1))
    ns = np.array([5, 5, 5])
    result = nearest_neighbour_pooled_distance(means, covs, ns, threshold)
    assert result["most_isolated_d"] == pytest.approx(4.0)
    assert result["n_isolated"] == n_isolated


@pytest.mark.parametrize(
    "means, most_isolated_d, n_isolated",
    [
        ([0.0, 1.0, 10.0], 1.0, 0),  # the n=1 group would be the most isolated (d=9)
        ([0.0, 10.0, 9.0], 10.0, 2),  # it would be the second group's nearest (d=1)
    ],
)
def test_nearest_neighbour_singleton_group_excluded(means, most_isolated_d, n_isolated):
    # The third group has n=1, so it has no distance to any group. With unit
    # variances, the pooled variance of every pair is 1 and d is the difference.
    covs = np.ones((3, 1, 1))
    ns = np.array([5, 5, 1])
    result = nearest_neighbour_pooled_distance(
        np.array(means)[:, None], covs, ns, threshold=3
    )
    assert result["most_isolated_d"] == pytest.approx(most_isolated_d)
    assert result["n_isolated"] == n_isolated


def test_nearest_neighbour_is_scale_free():
    # The 2D example of test_pooled_distance_2d, in seconds
    means, covs, ns = _moments([_SQUARE / 1000, (_SQUARE + [3, 4]) / 1000])
    result = nearest_neighbour_pooled_distance(means, covs, ns, threshold=1)
    assert result["most_isolated_d"] == pytest.approx(5 / np.sqrt(4 / 3))


def test_nearest_neighbour_empty_is_nan():
    result = nearest_neighbour_pooled_distance(
        np.empty((0, 1)), np.empty((0, 1, 1)), np.empty((0,)), threshold=1
    )
    assert np.isnan(result["most_isolated_d"])
    assert result["n_isolated"] == 0


def test_nearest_neighbour_ignores_singular_pairs():
    # a and b are the parallel pairs of test_pooled_distance_2d_parallel_pairs_return_nan;
    # their pair is skipped, while both still have a distance to c.
    groups = [
        np.array([[800.0, 200.0], [850.0, 250.0]]),
        np.array([[600.0, 900.0], [620.0, 920.0]]),
        _SQUARE + [700, 500],
    ]
    result = nearest_neighbour_pooled_distance(*_moments(groups), threshold=1)
    assert np.isfinite(result["most_isolated_d"])


def test_nearest_neighbour_both_singular_is_nan():
    # Both groups are exact repetitions: zero (singular) pooled covariance
    means = np.array([[1.0, 1.0], [5.0, 5.0]])
    covs = np.zeros((2, 2, 2))
    ns = np.array([3, 3])
    result = nearest_neighbour_pooled_distance(means, covs, ns, threshold=1)
    assert np.isnan(result["most_isolated_d"])


# --- reduce_moments and isolation_stats ---


def test_reduce_moments_1d():
    df = pd.DataFrame(
        {
            "bird_id": ["b1", "b1", "b1", "b2"],
            "label": ["a", "a", "b", "a"],
            "duration": [10.0, 12.0, 100.0, 5.0],
        }
    )
    result = reduce_moments(df, index=["bird_id", "label"], value_cols=["duration"])
    row = result.loc[("b1", "a")]
    assert row["n"] == 2
    assert row["mean_duration"] == pytest.approx(11.0)
    assert row["cov_duration_duration"] == pytest.approx(2.0)  # (1 + 1) / 1


def test_reduce_moments_2d():
    # Deviations from the means (12, 24): x -2, 0, 2; y -4, -2, 6. Divided by n - 1 = 2:
    # var x = 8 / 2, var y = 56 / 2, cov = (8 + 0 + 12) / 2
    df = pd.DataFrame(
        {
            "bird_id": ["b1", "b1", "b1"],
            "label": ["a-b"] * 3,
            "duration_1": [10.0, 12.0, 14.0],
            "duration_2": [20.0, 22.0, 30.0],
        }
    )
    result = reduce_moments(
        df, index=["bird_id", "label"], value_cols=["duration_1", "duration_2"]
    )
    row = result.loc[("b1", "a-b")]
    assert row["cov_duration_1_duration_1"] == pytest.approx(4.0)
    assert row["cov_duration_2_duration_2"] == pytest.approx(28.0)
    assert row["cov_duration_1_duration_2"] == pytest.approx(10.0)


def test_reduce_moments_rounding_variance_counts_as_zero():
    df = pd.DataFrame({"bird_id": "b1", "label": "a", "duration": _NOISY_100})
    result = reduce_moments(df, index=["bird_id", "label"], value_cols=["duration"])
    assert result.loc[("b1", "a"), "cov_duration_duration"] == 0.0


@pytest.mark.parametrize("k", [1, 2])
def test_isolation_stats_matches_pooled_distance(k):
    # Five random labels compared with a plain double loop over pooled_distance
    rng = np.random.default_rng(1)
    groups = [
        rng.normal(loc=i * 8.0, scale=1.0, size=(rng.integers(3, 8), k))
        for i in range(5)
    ]
    cols = [f"duration_{c}" for c in range(k)]
    df = pd.concat(
        pd.DataFrame(g, columns=cols).assign(bird_id="b1", label=f"l{i}")
        for i, g in enumerate(groups)
    )
    d = np.array([[pooled_distance(g, h) for h in groups] for g in groups])
    np.fill_diagonal(d, np.nan)
    nearest = np.nanmin(d, axis=1)
    threshold = np.median(nearest)

    result = isolation_stats(df, by=["bird_id"], value_cols=cols, threshold=threshold)
    assert result.loc["b1", "most_isolated_d"] == pytest.approx(nearest.max())
    assert result.loc["b1", "n_isolated"] == np.sum(nearest > threshold)


def test_isolation_stats_single_label_bird_is_nan():
    df = pd.DataFrame(
        {"bird_id": ["b1", "b1"], "label": ["a", "a"], "duration": [1.0, 2.0]}
    )
    result = isolation_stats(df, by=["bird_id"], value_cols=["duration"], threshold=3)
    assert np.isnan(result.loc["b1", "most_isolated_d"])


def test_isolation_stats_independent_per_by_group():
    # Each bird's labels are compared only with its own. Pooled variance is 4/3 in
    # all cases (see _SQUARE); b1's labels are 2 apart (d = sqrt(3) < 3), b2's are
    # 100 apart.
    base = np.array([0.0, 0.0, 2.0, 2.0])
    df = pd.DataFrame(
        {
            "bird_id": ["b1"] * 8 + ["b2"] * 8,
            "label": (["a"] * 4 + ["b"] * 4) * 2,
            "duration": np.concatenate([base, base + 2, base, base + 100]),
        }
    )
    result = isolation_stats(df, by=["bird_id"], value_cols=["duration"], threshold=3)
    assert result.loc["b1", "n_isolated"] == 0
    assert result.loc["b2", "n_isolated"] == 2


def test_isolation_stats_2d_multi_column_by():
    df = pd.DataFrame(
        {
            "dataset_id": "ds1",
            "bird_id": "b1",
            "label": ["a-a"] * 4 + ["b-b"] * 4,
            "duration_1": np.concatenate([_SQUARE[:, 0], _SQUARE[:, 0] + 3]),
            "duration_2": np.concatenate([_SQUARE[:, 1], _SQUARE[:, 1] + 4]),
        }
    )
    result = isolation_stats(
        df,
        by=["dataset_id", "bird_id"],
        value_cols=["duration_1", "duration_2"],
        threshold=3,
    )
    assert result.index.names == ["dataset_id", "bird_id"]
    # As in test_pooled_distance_2d
    assert result.loc[("ds1", "b1"), "most_isolated_d"] == pytest.approx(
        5 / np.sqrt(4 / 3)
    )
