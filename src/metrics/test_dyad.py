import numpy as np
import pandas as pd
import pytest

from metrics.dyad import dyad_scores, threshold50


def test_dyad_scores():
    dyads = pd.DataFrame(
        {"duration_1": [100.0, 60.0, 30.0], "duration_2": [100.0, 140.0, 10.0]}
    )
    scores = dyad_scores(dyads)
    assert scores["asymmetry"].tolist() == pytest.approx([0.0, 80.0, 20.0])
    assert scores["non_isochrony"].tolist() == pytest.approx([0.0, 0.2, 0.25])


def test_dyad_scores_round_off_floating_point_noise():
    # Durations from onsets in seconds carry noise: (1.1 - 1.0) * 1000 is
    # 100.00000000000009 and (1.22 - 1.1) * 1000 is 119.99999999999989, so |x - y|
    # is just below 20 ms. 45 / (45 + 55) - 0.5 is -0.04999999999999999.
    dyads = pd.DataFrame(
        {
            "duration_1": [(1.1 - 1.0) * 1000, 45.0],
            "duration_2": [(1.22 - 1.1) * 1000, 55.0],
        }
    )
    scores = dyad_scores(dyads)
    assert scores.loc[0, "asymmetry"] == 20.0
    assert scores.loc[1, "non_isochrony"] == 0.05


def _by_bird(values: dict[str, list[float]]) -> pd.Series:
    """One criterion value per dyad, indexed by bird_id."""
    return pd.Series(
        [v for vs in values.values() for v in vs],
        index=pd.Index([b for b, vs in values.items() for _ in vs], name="bird_id"),
    )


def test_threshold50_single_bird_is_the_median():
    assert threshold50(_by_bird({"b1": [5.0, 20.0, 20.0, 80.0]})) == 20.0


def test_threshold50_is_where_the_mean_curve_crosses_half():
    # Below 10ms: b1 has 100% of its dyads, b2 has 0% -> mean 0.5 at 10ms, even though
    # b2's own median (50ms) is far higher.
    assert threshold50(_by_bird({"b1": [10.0], "b2": [50.0, 90.0]})) == 10.0


def test_threshold50_weights_birds_equally():
    # Pooled, the median of these 11 dyads is 10ms. Per bird, the share at 10ms is
    # 0.9 for b1 and 0 for b2 (mean 0.45); the mean first reaches .5 at b2's 90ms.
    assert threshold50(_by_bird({"b1": [10.0] * 9 + [200.0], "b2": [90.0]})) == 90.0


def test_threshold50_matches_a_brute_force_sweep():
    rng = np.random.default_rng(0)
    values = _by_bird(
        {f"b{b}": list(rng.uniform(0, 100, size=rng.integers(5, 40))) for b in range(4)}
    )
    birds = values.index.unique()

    def mean_share_up_to(t: float) -> float:
        return (values <= t).groupby(level="bird_id").mean().reindex(birds).mean()

    expected = next(t for t in sorted(values) if mean_share_up_to(t) >= 0.5)
    assert threshold50(values) == expected
