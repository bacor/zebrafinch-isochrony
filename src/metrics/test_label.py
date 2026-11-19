import pandas as pd
import pytest

from metrics.label import analyze_labels


def test_analyze_labels_groups_by_bird_and_label():
    units = pd.DataFrame(
        {
            "bird_id": ["b1", "b1", "b1", "b1", "b2"],
            "label": ["a", "a", "a", "b", "a"],
            "duration": [100.0, 110.0, 120.0, 50.0, 80.0],
        }
    )
    stats = analyze_labels(units, by=["bird_id"])
    assert set(stats.index) == {("b1", "a"), ("b1", "b"), ("b2", "a")}
    row = stats.loc[("b1", "a")]
    assert row["count"] == 3
    assert row["dur_mean"] == pytest.approx(110.0)
    assert row["dur_std"] == pytest.approx(10.0)
    assert (row["dur_min"], row["dur_q2"], row["dur_max"]) == (100.0, 110.0, 120.0)
    assert (row["dur_q1"], row["dur_q3"]) == (105.0, 115.0)


def test_analyze_labels_single_unit_has_no_sd():
    units = pd.DataFrame({"bird_id": ["b1"], "label": ["a"], "duration": [100.0]})
    assert pd.isna(analyze_labels(units, by=["bird_id"]).loc[("b1", "a"), "dur_std"])


def test_analyze_labels_multi_column_by():
    units = pd.DataFrame(
        {
            "dataset_id": "ds",
            "bird_id": ["b1", "b1"],
            "label": ["a", "a"],
            "duration": [100.0, 110.0],
        }
    )
    stats = analyze_labels(units, by=["dataset_id", "bird_id"])
    assert stats.index.names == ["dataset_id", "bird_id", "label"]
    assert stats.loc[("ds", "b1", "a"), "count"] == 2
