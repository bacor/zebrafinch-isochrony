import numpy as np
import pandas as pd
import pytest

from rhythm_types import (
    aggregate_units,
    build_dyads,
    drop_short_chunks,
    get_bound_units_from_chunk,
    get_dur_units,
    get_ioi_units,
    is_increasing,
)

# Units a, b, c separated by gaps ab (0.5 - 1.0) and bc (2.0 - 3.0).
CHUNK = {
    "start": [0.0, 1.0, 3.0],
    "end": [0.5, 2.0, 4.0],
    "label": ["a", "b", "c"],
}

# Five units where c and d touch (c ends at 10.0 where d starts), so there is
# no gap cd: the gaps are ab, bc and de.
TOUCHING_CHUNK = {
    "start": [0.0, 4.5, 6.0, 10.0, 11.5],
    "end": [3.0, 5.5, 10.0, 11.0, 14.0],
    "label": ["a", "b", "c", "d", "e"],
}

DYAD_BY = ["dataset_id", "rhythm_type", "bird_id", "chunk_idx"]


def _bound_units(chunk=CHUNK, chunk_idx=7):
    return get_bound_units_from_chunk(
        pd.DataFrame(chunk),
        start_col="start",
        end_col="end",
        label_col="label",
        bird_id="bird-1",
        chunk_idx=chunk_idx,
    )


def _make_dyad_units(rows: list[dict]) -> pd.DataFrame:
    defaults = dict(dataset_id="ds", rhythm_type="syll_dur")
    return pd.DataFrame([{**defaults, **row} for row in rows])


# --- Bound units ---


def test_is_increasing():
    assert is_increasing(pd.Series([0.0, 1.0, 2.0]))
    assert not is_increasing(pd.Series([0.0, 0.0, 1.0]))
    assert not is_increasing(pd.Series([2.0, 1.0, 3.0]))


def test_get_bound_units_from_chunk():
    units = _bound_units()

    def row(start, duration, is_gap, label):
        return dict(
            start=start,
            duration=duration,
            is_gap=is_gap,
            label=label,
            chunk_idx=7,
            bird_id="bird-1",
        )

    expected = pd.DataFrame(
        [
            row(0.0, 0.5, False, "a"),
            row(0.5, 0.5, True, "ab"),
            row(1.0, 1.0, False, "b"),
            row(2.0, 1.0, True, "bc"),
            row(3.0, 1.0, False, "c"),
        ]
    )
    pd.testing.assert_frame_equal(units, expected)


def test_touching_units_get_no_gap():
    labels = _bound_units(TOUCHING_CHUNK)["label"].tolist()
    assert labels == ["a", "ab", "b", "bc", "c", "d", "de", "e"]


def test_missing_unit_label_propagates_to_gap_label():
    chunk = {"start": [0.0, 1.0], "end": [0.5, 1.5], "label": [np.nan, "b"]}
    assert pd.isna(_bound_units(chunk).loc[1, "label"])


def test_get_bound_units_from_chunk_raises_on_empty_chunk():
    empty = {"start": [], "end": [], "label": []}
    with pytest.raises(ValueError, match="bird bird-1, chunk 3"):
        _bound_units(empty, chunk_idx=3)


def test_get_bound_units_from_chunk_raises_on_unsorted_starts():
    unsorted = {"start": [1.0, 0.0], "end": [1.5, 0.5], "label": ["a", "b"]}
    with pytest.raises(ValueError, match="strictly increasing"):
        _bound_units(unsorted)


# --- Durations and IOIs ---


def test_dur_and_gap_dur_units():
    bound = _bound_units()
    units = get_dur_units(bound)
    gaps = get_dur_units(bound, gaps=True)

    assert "is_gap" not in units and "is_gap" not in gaps
    assert units["label"].tolist() == ["a", "b", "c"]
    assert units["duration"].tolist() == [0.5, 1.0, 1.0]
    assert gaps["label"].tolist() == ["ab", "bc"]
    assert gaps["duration"].tolist() == [0.5, 1.0]


def test_ioi_units_and_gap_ioi_units():
    bound = _bound_units()
    unit_ioi = get_ioi_units(bound)
    gap_ioi = get_ioi_units(bound, gaps=True)

    # The final IOI ends at chunk end (4.0).
    assert "is_gap" not in unit_ioi
    assert unit_ioi["start"].tolist() == [0.0, 1.0, 3.0]
    assert unit_ioi["duration"].tolist() == [1.0, 2.0, 1.0]
    assert unit_ioi["label"].tolist() == ["ab", "bc", "c!"]

    assert gap_ioi["start"].tolist() == [0.5, 2.0]
    assert gap_ioi["duration"].tolist() == [1.5, 2.0]
    assert gap_ioi["label"].tolist() == ["ab", "bc"]


def test_ioi_labels_when_units_touch():
    # Gap IOIs join the first letters of both gap labels: ab -> bc gives 'ab',
    # bc -> de gives 'bd', and the final gap de -> unit e gives 'de'.
    bound = _bound_units(TOUCHING_CHUNK)
    assert get_ioi_units(bound)["label"].tolist() == ["ab", "bc", "cd", "de", "e!"]
    assert get_ioi_units(bound, gaps=True)["label"].tolist() == ["ab", "bd", "de"]


def test_ioi_units_with_single_unit():
    bound = _bound_units({"start": [0.0], "end": [0.5], "label": ["a"]})

    ioi = get_ioi_units(bound)

    assert list(ioi.columns) == ["start", "duration", "label", "chunk_idx", "bird_id"]
    assert len(ioi) == 0


# --- Chunks ---


def test_drop_short_chunks_drops_single_row_groups():
    slices = pd.DataFrame(
        [
            {"bird_id": "bird-1", "chunk_idx": 1, "duration": 0.1},
            {"bird_id": "bird-1", "chunk_idx": 1, "duration": 0.2},
            {"bird_id": "bird-1", "chunk_idx": 2, "duration": 0.3},
            {"bird_id": "bird-2", "chunk_idx": 1, "duration": 0.4},
            {"bird_id": "bird-2", "chunk_idx": 1, "duration": 0.5},
        ]
    )

    filtered = drop_short_chunks(slices)

    assert filtered["duration"].tolist() == [0.1, 0.2, 0.4, 0.5]


def test_drop_short_chunks_custom_grouping():
    slices = pd.DataFrame(
        [
            {"chunk_idx": 1, "duration": 0.1},
            {"chunk_idx": 1, "duration": 0.2},
            {"chunk_idx": 2, "duration": 0.3},
        ]
    )

    filtered = drop_short_chunks(slices, by=["chunk_idx"])

    assert filtered["chunk_idx"].tolist() == [1, 1]


def test_aggregate_units_skips_empty_and_drops_short_chunks():
    def chunks():
        for chunk_idx, chunk in [
            (1, CHUNK),
            (2, {"start": [], "end": [], "label": []}),  # empty: skipped
            (3, {"start": [0.0], "end": [0.5], "label": ["a"]}),  # 1 unit: dropped
        ]:
            yield dict(
                chunk_df=pd.DataFrame(chunk),
                start_col="start",
                end_col="end",
                label_col="label",
                bird_id="bird-1",
                chunk_idx=chunk_idx,
                min_gap=1e-10,
            )

    syll = aggregate_units(chunks(), level="syll")
    elem = aggregate_units(chunks(), level="elem")

    # Only syllables have gaps between them.
    assert list(syll) == ["syll_bound", "syll_dur", "syll_ioi", "gap_dur", "gap_ioi"]
    assert list(elem) == ["elem_bound", "elem_dur", "elem_ioi"]
    # CHUNK has 3 units and 2 gaps.
    lengths = {
        "syll_bound": 5,
        "syll_dur": 3,
        "syll_ioi": 3,
        "gap_dur": 2,
        "gap_ioi": 2,
    }
    for rhythm_type, units in syll.items():
        assert units["chunk_idx"].tolist() == [1] * lengths[rhythm_type]


# --- Dyads ---


def test_build_dyads_pairs_successive_units_per_chunk():
    units = _make_dyad_units(
        [
            dict(
                bird_id="b1",
                chunk_idx=0,
                start=10.0,
                duration=20.0,
                label="b",
                label_pred="y",
            ),
            dict(
                bird_id="b1",
                chunk_idx=0,
                start=0.0,
                duration=10.0,
                label="a",
                label_pred="x",
            ),
            dict(
                bird_id="b1",
                chunk_idx=0,
                start=30.0,
                duration=30.0,
                label="c",
                label_pred=None,
            ),
            dict(
                bird_id="b1",
                chunk_idx=1,
                start=0.0,
                duration=5.0,
                label="a",
                label_pred="x",
            ),
            dict(
                bird_id="b1",
                chunk_idx=1,
                start=5.0,
                duration=15.0,
                label="b",
                label_pred="y",
            ),
        ]
    )
    dyads = build_dyads(units, DYAD_BY, labels=["label", "label_pred", "label_shuffle"])
    assert list(dyads["duration_1"]) == [10.0, 20.0, 5.0]
    assert list(dyads["duration_2"]) == [20.0, 30.0, 15.0]
    assert list(dyads["label"]) == ["a-b", "b-c", "a-b"]
    assert dyads["label_pred"].isna().tolist() == [False, True, False]
    assert "label_shuffle" not in dyads
