import numpy as np
import pandas as pd
import pytest

from identify.test_changepoints import CHUNKS_PER_BIRD, _synthetic_birds
from pipeline.predict_bird_ids import predict_bird_ids


def _synthetic_units(with_bird_id=True):
    """The chunks of `_synthetic_birds` as a units DataFrame.

    With `with_bird_id`, birds are 'bird-0' .. 'bird-4' with chunk_idx counted
    per bird (as in NS). Without, bird_id is missing and chunk_idx is global
    (as in RT).
    """
    rows = []
    for i, seq in enumerate(_synthetic_birds()):
        bird = i // CHUNKS_PER_BIRD
        bird_id = f"bird-{bird}" if with_bird_id else np.nan
        chunk_idx = i % CHUNKS_PER_BIRD if with_bird_id else i
        rows += [
            {"bird_id": bird_id, "chunk_idx": chunk_idx, "duration": d} for d in seq
        ]
    return pd.DataFrame(rows)


def test_predict_bird_ids_recovers_birds_per_row():
    # Shuffled rows: predictions must still line up with each row's chunk.
    units = _synthetic_units().sample(frac=1, random_state=0)

    result = predict_bird_ids(units, half_window=15, min_gap=20, id_prefix="test-pred-")

    # 5 birds of 80 chunks each: a new bird starts every 80 chunks.
    assert result["n_chunks"] == 400
    np.testing.assert_array_equal(result["true_breakpoints"], [80, 160, 240, 320])
    np.testing.assert_array_equal(result["breakpoints"], [80, 160, 240, 320])
    # Birds are found in order, so bird-k gets the 1-based id k + 1.
    expected = [f"test-pred-{int(b[-1]) + 1:03d}" for b in units["bird_id"]]
    assert list(result["bird_id_pred"]) == expected


def test_predict_bird_ids_with_explicit_n_individuals_for_unknown_bird_id():
    units = _synthetic_units(with_bird_id=False)

    result = predict_bird_ids(
        units, half_window=15, min_gap=20, n_individuals=5, id_prefix="rt-pred-"
    )

    # All rows share the missing bird_id, so chunks are told apart by chunk_idx.
    assert result["n_chunks"] == 400
    assert len(result["true_breakpoints"]) == 0  # no ground truth available
    expected = [f"rt-pred-{c // CHUNKS_PER_BIRD + 1:03d}" for c in units["chunk_idx"]]
    assert list(result["bird_id_pred"]) == expected


def test_predict_bird_ids_raises_without_n_individuals_when_bird_id_missing():
    units = pd.DataFrame(
        {"bird_id": [np.nan, np.nan], "chunk_idx": [0, 1], "duration": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="n_individuals"):
        predict_bird_ids(units, half_window=15, min_gap=20)
