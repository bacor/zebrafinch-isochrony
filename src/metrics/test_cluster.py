import numpy as np
import pandas as pd
import pytest

from metrics.cluster import analyze_clusters


def _dyads(duration_1, duration_2, label, bird_id="b1"):
    return pd.DataFrame(
        {
            "bird_id": bird_id,
            "label": label,
            "duration_1": duration_1,
            "duration_2": duration_2,
        }
    )


def test_analyze_clusters_groups_by_bird_and_label():
    dyads = pd.concat(
        [
            _dyads([100.0, 100.0, 50.0], [100.0, 100.0, 150.0], ["a-a", "a-a", "a-b"]),
            _dyads([80.0, 80.0], [60.0, 60.0], ["c-c", "c-c"], bird_id="b2"),
        ]
    )
    stats = analyze_clusters(dyads, by=["bird_id"])
    assert set(stats.index) == {("b1", "a-a"), ("b1", "a-b"), ("b2", "c-c")}
    assert stats.loc[("b1", "a-a"), "count"] == 2


def test_analyze_clusters_centroid():
    stats = analyze_clusters(
        _dyads([100.0, 80.0], [100.0, 120.0], ["a-b", "a-b"]), by=["bird_id"]
    )
    row = stats.loc[("b1", "a-b")]
    assert (row["centroid_x_mean"], row["centroid_y_mean"]) == (90.0, 110.0)
    # Both components deviate 10 from their mean: SD (ddof=1) = sqrt(2 * 10**2 / 1)
    assert row["centroid_x_std"] == pytest.approx(np.sqrt(200))
    assert row["centroid_y_std"] == pytest.approx(np.sqrt(200))


def test_analyze_clusters_rhythm_of_the_centroid():
    stats = analyze_clusters(
        _dyads([50.0, 100.0], [150.0, 100.0], ["a-b", "c-c"]), by=["bird_id"]
    )
    assert stats.loc[("b1", "a-b"), "centroid_asymmetry"] == pytest.approx(100.0)
    assert stats.loc[("b1", "a-b"), "centroid_rhythm_ratio"] == pytest.approx(0.25)
    assert stats.loc[("b1", "a-b"), "centroid_non_isochrony"] == pytest.approx(0.25)
    assert stats.loc[("b1", "c-c"), "centroid_non_isochrony"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "label, expected",
    [
        ("a-b", ("a", "b")),
        ("ab-bc", ("ab", "bc")),
    ],
)
def test_analyze_clusters_splits_label_on_dash(label, expected):
    stats = analyze_clusters(_dyads([1.0], [1.0], [label]), by=["bird_id"])
    row = stats.loc[("b1", label)]
    assert (row["label_x"], row["label_y"]) == expected
