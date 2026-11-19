import numpy as np
import pandas as pd
import pytest

from metrics.condition import bird_metrics, condition_metrics, isochrony_metrics

BY = ["cond", "bird_id"]


def _index(rows, names):
    return pd.MultiIndex.from_tuples(rows, names=names)


def _tables():
    """Two birds in one condition: b1 has labels a, b, c; b2 has labels a, b."""
    label = pd.DataFrame(
        {
            "count": [10, 20, 30, 5, 5],
            "dur_mean": [100.0, 200.0, 300.0, 50.0, 60.0],
            "dur_std": [1.0, 2.0, 9.0, 4.0, 6.0],
        },
        index=_index(
            [
                ("c", "b1", "a"),
                ("c", "b1", "b"),
                ("c", "b1", "c"),
                ("c", "b2", "a"),
                ("c", "b2", "b"),
            ],
            BY + ["label"],
        ),
    )
    # b1: a-b is 50% of dyads, b-b 48%, b-a only 2% (infrequent); b2: one cluster
    cluster = pd.DataFrame(
        {
            "label_x": ["a", "b", "b", "a"],
            "label_y": ["b", "b", "a", "b"],
            "count": [25, 24, 1, 9],
            "centroid_x_std": [1.0, 3.0, np.nan, 2.0],
            "centroid_y_std": [3.0, 5.0, np.nan, 2.0],
            "centroid_non_isochrony": [0.2, 0.0, 0.4, 0.1],
        },
        index=_index(
            [
                ("c", "b1", "a-b"),
                ("c", "b1", "b-b"),
                ("c", "b1", "b-a"),
                ("c", "b2", "a-b"),
            ],
            BY + ["label"],
        ),
    )
    bird = pd.DataFrame(
        {
            "n_labels": [3, 2],
            "n_isolated_labels": [2, 0],
            "n_isolated_clusters": [1, 0],
        },
        index=_index([("c", "b1"), ("c", "b2")], BY),
    )
    return label, cluster, bird


def test_bird_metrics_takes_medians_over_labels_and_clusters():
    birds = bird_metrics(*_tables(), by=BY)
    assert birds.loc[("c", "b1"), "inconsistency"] == 2.0
    assert birds.loc[("c", "b2"), "inconsistency"] == 5.0
    assert birds.loc[("c", "b1"), "dur_mean"] == 200.0
    # Cluster SDs are means of both components: b1 has 2 and 4 (b-a has no SD)
    assert birds.loc[("c", "b1"), "cluster_inconsistency"] == 3.0


def test_bird_metrics_isolation_flags():
    birds = bird_metrics(*_tables(), by=BY)
    assert birds["isolated_label"].tolist() == [True, False]
    assert birds["isolated_cluster"].tolist() == [True, False]
    assert birds["n_isolated_labels"].tolist() == [2, 0]


def test_bird_metrics_isochrony_ignores_infrequent_clusters():
    birds = bird_metrics(*_tables(), by=BY)
    # b-a (2% of b1's dyads, score 0.4) is dropped
    assert birds.loc[("c", "b1"), "least_isochronous"] == 0.2
    assert birds.loc[("c", "b1"), "most_isochronous"] == 0.0
    assert birds.loc[("c", "b1"), "median_isochrony"] == pytest.approx(0.1)


def test_bird_metrics_keeps_a_cluster_at_exactly_the_minimum_share():
    label, cluster, bird = _tables()
    cluster["count"] = [10, 9, 1, 9]  # b1's b-a is now 1 / 20 = MIN_CLUSTER_SHARE
    birds = bird_metrics(label, cluster, bird, by=BY)
    assert birds.loc[("c", "b1"), "least_isochronous"] == 0.4


def test_bird_metrics_repetitions_are_a_share_of_dyads():
    birds = bird_metrics(*_tables(), by=BY)
    assert birds.loc[("c", "b1"), "repetitions"] == pytest.approx(48.0)
    assert birds.loc[("c", "b2"), "repetitions"] == 0.0


def test_condition_metrics_averages_birds_and_turns_flags_into_percentages():
    cond = condition_metrics(bird_metrics(*_tables(), by=BY), by=["cond"])
    assert cond.loc["c", "inconsistency"] == pytest.approx(3.5)
    assert cond.loc["c", "isolated_label"] == pytest.approx(50.0)
    assert cond.loc["c", "n_birds"] == 2


def test_isochrony_metrics():
    # b1 has 2 dyads and b2 has 4, so a bird-weighted mean differs from a pooled one.
    # b2's first dyad lies exactly on both criteria (0.05 = RHO, 20 ms = TAU),
    # which do not count as isochronous (strict <).
    dyads = pd.DataFrame(
        {
            "non_isochrony": [0.0, 0.2, 0.05, 0.3, 0.3, 0.01],
            "asymmetry": [0.0, 40.0, 20.0, 60.0, 60.0, 10.0],
        },
        index=_index([("c", "b1")] * 2 + [("c", "b2")] * 4, BY),
    )
    cond = isochrony_metrics(dyads, by=["cond"])
    # Per bird: non_isochrony 0.1 and 0.66 / 4; asymmetry 20 and 150 / 4
    assert cond.loc["c", "non_isochrony"] == pytest.approx((0.1 + 0.165) / 2)
    assert cond.loc["c", "asymmetry"] == pytest.approx((20.0 + 37.5) / 2)
    # Isochronous: b1 1 of 2 (0.0), b2 1 of 4 (0.01 / 10 ms)
    assert cond.loc["c", "perc_rel_isochronous"] == pytest.approx((50 + 25) / 2)
    assert cond.loc["c", "perc_abs_isochronous"] == pytest.approx((50 + 25) / 2)
    # Mean share of dyads <= t: at 0.01 it is (1/2 + 1/4) / 2 < 0.5, at 0.05 it
    # is (1/2 + 2/4) / 2 = 0.5. Likewise tau50 is reached at 20 ms.
    assert cond.loc["c", "rho50"] == pytest.approx(0.05)
    assert cond.loc["c", "tau50"] == pytest.approx(20.0)
