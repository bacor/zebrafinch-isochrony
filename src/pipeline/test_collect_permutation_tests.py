import pandas as pd
import pytest

from metrics.condition import TESTS
from pipeline.collect_permutation_tests import _to_long


def test_tests_match_hypotheses():
    # Real labels should be more consistent than shuffled ones (low), and
    # more isolated and less isochronous (high).
    assert TESTS == {
        "inconsistency": "low",
        "isolated_label": "high",
        "cluster_inconsistency": "low",
        "isolated_cluster": "high",
        "most_isochronous": "high",
        "median_isochrony": "high",
        "least_isochronous": "high",
    }


@pytest.mark.parametrize("metric", list(TESTS))
def test_to_long_takes_p_of_the_metric_direction(metric):
    # Distinct p-values in both tails, so taking the wrong one is visible.
    result = pd.DataFrame(
        {
            col: [value]
            for m in TESTS
            for col, value in {
                m: 1.0,
                f"{m}_null_mean": 2.0,
                f"{m}_null_std": 0.5,
                f"{m}_z": -2.0,
                f"{m}_p": 0.01,
                f"{m}_p_high": 0.99,
            }.items()
        },
        index=pd.Index(["cond"], name="condition"),
    )
    long = _to_long(result, pd.Series([7], index=result.index)).set_index("metric")
    row = long.loc[metric]
    assert row["direction"] == TESTS[metric]
    assert row["p"] == (0.01 if TESTS[metric] == "low" else 0.99)
    assert row["n_birds"] == 7
    assert len(long) == len(TESTS)
