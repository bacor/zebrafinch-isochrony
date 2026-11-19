import numpy as np
import pandas as pd
import pytest

from metrics.bird import analyze_birds
from rhythm_types import build_dyads


def _analyze(units):
    return analyze_birds(
        units, build_dyads(units, ["bird_id", "chunk_idx"]), by=["bird_id"]
    )


def _units(durations, labels, chunk_idx=None, bird_id="b1"):
    """Units played back to back within each chunk."""
    chunk_idx = chunk_idx or [0] * len(durations)
    df = pd.DataFrame(
        {
            "bird_id": bird_id,
            "chunk_idx": chunk_idx,
            "label": labels,
            "duration": durations,
        }
    )
    df["start"] = df.groupby("chunk_idx")["duration"].cumsum() - df["duration"]
    return df


def test_analyze_birds_counts_and_chunks():
    # Chunk 0: 100 + 105 + 200 = 405 ms, 3 units; chunk 1: 100 + 200 = 300 ms, 2 units
    birds = _analyze(
        _units(
            [100.0, 105.0, 200.0, 100.0, 200.0],
            ["a", "a", "b", "a", "b"],
            chunk_idx=[0, 0, 0, 1, 1],
        )
    )
    row = birds.loc["b1"]
    assert (row["n_labels"], row["n_units"], row["n_chunks"], row["n_dyads"]) == (
        2,
        5,
        2,
        3,
    )
    assert (row["chunk_len_min"], row["chunk_len_max"]) == (2, 3)
    assert row["chunk_len_mean"] == 2.5
    assert (row["chunk_dur_min"], row["chunk_dur_max"]) == (300.0, 405.0)
    assert row["chunk_dur_mean"] == 352.5
    # SD (ddof=1) of two values is their difference / sqrt(2): 105 / sqrt(2)
    assert row["chunk_dur_std"] == pytest.approx(105 / np.sqrt(2))


def test_analyze_birds_label_isolation():
    # a: mean 102.5, var 12.5; b: mean 205, var 50; pooled var (12.5 + 50) / 2.
    # Each label's nearest neighbour is the other, at
    # d = (205 - 102.5) / sqrt(31.25) = 18.34 > ISOLATION_D, so both are isolated.
    birds = _analyze(_units([100.0, 105.0, 200.0, 210.0], ["a", "a", "b", "b"]))
    assert birds.loc["b1", "most_isolated_label_d"] == pytest.approx(
        102.5 / np.sqrt(31.25)
    )
    assert birds.loc["b1", "n_isolated_labels"] == 2


def test_analyze_birds_cluster_isolation():
    # Three a-b chunks (dyads near (50, 150)) and three b-a chunks (near
    # (150, 50)): two clusters ~100 ms apart with SDs of ~1 ms, so both are
    # far beyond ISOLATION_D from each other.
    a = [50.0, 52.0, 51.0]
    b = [150.0, 149.0, 152.0]
    durations = [d for i in range(3) for d in (a[i], b[i])]
    durations += [d for i in range(3) for d in (b[i], a[i])]
    units = _units(
        durations,
        ["a", "b"] * 3 + ["b", "a"] * 3,
        chunk_idx=[c for c in range(6) for _ in range(2)],
    )
    birds = _analyze(units)
    assert birds.loc["b1", "n_isolated_clusters"] == 2
    assert birds.loc["b1", "most_isolated_cluster_d"] > 10


def test_analyze_birds_single_label_is_not_isolated():
    birds = _analyze(_units([1.0, 2.0, 3.0], ["a", "a", "a"]))
    assert np.isnan(birds.loc["b1", "most_isolated_label_d"])
    assert birds.loc["b1", "n_isolated_labels"] == 0
