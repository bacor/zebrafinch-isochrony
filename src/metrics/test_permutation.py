import numpy as np
import pandas as pd
import pytest

from metrics.permutation import permutation_test, shuffle_labels_within_chunk

BY = ["bird_id", "chunk_idx"]


def _units(chunks: list[list[str]]) -> pd.DataFrame:
    """One bird; each chunk a list of labels, with durations 1, 2, 3, ..."""
    rows = [
        dict(bird_id="b1", chunk_idx=c, start=float(i), label=label)
        for c, labels in enumerate(chunks)
        for i, label in enumerate(labels)
    ]
    return pd.DataFrame(rows).assign(duration=lambda df: np.arange(1.0, len(df) + 1))


def _frame(values: dict[str, float]) -> pd.DataFrame:
    """A one-column ('stat') table indexed by label."""
    return pd.DataFrame(
        {"stat": list(values.values())},
        index=pd.Index(list(values), name="label"),
    )


# --- shuffle_labels_within_chunk ---


def test_shuffle_keeps_each_chunks_labels_and_all_durations():
    units = _units([["a", "b", "c"], ["x", "y"]])
    rng = np.random.default_rng(0)
    for _ in range(20):
        shuffled = shuffle_labels_within_chunk(units, rng, by=BY)
        assert sorted(shuffled["label"].iloc[:3]) == ["a", "b", "c"]
        assert sorted(shuffled["label"].iloc[3:]) == ["x", "y"]
        pd.testing.assert_frame_equal(
            shuffled.drop(columns="label"), units.drop(columns="label")
        )


def test_shuffle_changes_the_order():
    # A chunk of 4 labels keeps its order with probability 1/24 per shuffle
    units = _units([["a", "b", "c", "d"]])
    orders = {
        tuple(shuffle_labels_within_chunk(units, np.random.default_rng(s), by=BY).label)
        for s in range(20)
    }
    assert len(orders) > 1


def test_shuffle_is_seed_reproducible():
    units = _units([["a", "b", "c", "d", "e", "f"]])
    a = shuffle_labels_within_chunk(units, np.random.default_rng(42), by=BY)
    b = shuffle_labels_within_chunk(units, np.random.default_rng(42), by=BY)
    c = shuffle_labels_within_chunk(units, np.random.default_rng(43), by=BY)
    assert a["label"].tolist() == b["label"].tolist()
    assert a["label"].tolist() != c["label"].tolist()


# --- permutation_test ---


def test_permutation_test_summary_of_known_draws():
    # Draws 1, 2, 3, 4 (with n_jobs=1 they are made in order): mean 2.5, SD
    # sqrt((2.25 + 0.25 + 0.25 + 2.25) / 3) = sqrt(5 / 3). The observed 2 is tied
    # with one draw, which counts on both sides: 2 draws <= 2, 3 draws >= 2.
    draws = iter([1.0, 2.0, 3.0, 4.0])
    result = permutation_test(
        _frame({"a": 2.0}), lambda rng: _frame({"a": next(draws)}), n=4, seed=0
    )
    row = result.loc["a"]
    assert row["stat_null_mean"] == pytest.approx(2.5)
    assert row["stat_null_std"] == pytest.approx(np.sqrt(5 / 3))
    assert row["stat_z"] == pytest.approx(-0.5 / np.sqrt(5 / 3))
    assert row["stat_p"] == pytest.approx((2 + 1) / (4 + 1))
    assert row["stat_p_high"] == pytest.approx((3 + 1) / (4 + 1))


@pytest.mark.parametrize(
    "observed, p, p_high",
    [(-1000.0, 1 / 51, 1.0), (1000.0, 1.0, 1 / 51)],
)
def test_permutation_test_observed_beyond_every_draw(observed, p, p_high):
    # No draw on the observed side: p is (0 + 1) / (50 + 1), never 0
    result = permutation_test(
        _frame({"a": observed}),
        lambda rng: _frame({"a": rng.normal(0, 1)}),
        n=50,
        seed=0,
    )
    assert result.loc["a", "stat_p"] == pytest.approx(p)
    assert result.loc["a", "stat_p_high"] == pytest.approx(p_high)


def test_permutation_test_nan_row_stays_nan():
    result = permutation_test(
        _frame({"a": np.nan, "b": 5.0}),
        lambda rng: _frame({"a": np.nan, "b": rng.normal(0, 1)}),
        n=50,
        seed=0,
    )
    assert np.isnan(result.loc["a", "stat_z"])
    assert np.isnan(result.loc["a", "stat_p"])
    assert not np.isnan(result.loc["b", "stat_z"])


def test_permutation_test_aligns_draws_with_observed_rows():
    # Draws hold an extra row c (ignored) and miss row b (all NaN)
    result = permutation_test(
        _frame({"a": 1.0, "b": 1.0}),
        lambda rng: _frame({"c": 5.0, "a": 2.0}),
        n=10,
        seed=0,
    )
    assert result.index.tolist() == ["a", "b"]
    assert result.loc["a", "stat_null_mean"] == 2.0
    assert np.isnan(result.loc["b", "stat_null_mean"])


def test_permutation_test_does_not_depend_on_n_jobs():
    def draw_null(rng):
        return _frame({"a": rng.normal(0, 1)})

    observed = _frame({"a": 1.0})
    sequential = permutation_test(observed, draw_null, n=30, seed=0, n_jobs=1)
    parallel = permutation_test(observed, draw_null, n=30, seed=0, n_jobs=2)
    pd.testing.assert_frame_equal(sequential, parallel)
