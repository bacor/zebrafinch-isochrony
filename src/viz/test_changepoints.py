import numpy as np

from viz.changepoints import _segment_confidence


def test_segment_confidence_takes_weaker_bounding_peak():
    breakpoints = np.array([10, 20, 30])
    per_boundary_value = np.array([5.0, 1.0, 3.0])

    conf = _segment_confidence(breakpoints, per_boundary_value)

    np.testing.assert_array_equal(conf, [5.0, 1.0, 1.0, 3.0])
