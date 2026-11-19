"""Rhythm types, label types and datasets, and the conditions they define."""

from typing import Literal, TypeAlias

RhythmType: TypeAlias = Literal[
    "syll_ioi",
    "syll_dur",
    "syll_bound",
    "elem_ioi",
    "elem_dur",
    "elem_bound",
    "gap_ioi",
    "gap_dur",
]
LabelType: TypeAlias = Literal["annot", "pred", "shuffle"]

# Every rhythm type, in display order. A rhythm type is a level (syllables,
# elements, or the gaps between syllables) and a unit type: unit durations,
# inter-onset intervals, or all intervals between successive boundaries
# (units and gaps alike). Gap boundaries would be the syllable boundaries,
# so there is no 'gap_bound'.
RHYTHM_TYPES: dict[RhythmType, dict] = {
    "syll_ioi": dict(level="syll", unit_type="ioi", label="Syllable IOIs"),
    "syll_dur": dict(level="syll", unit_type="dur", label="Syllable durations"),
    "syll_bound": dict(level="syll", unit_type="bound", label="Syllable boundaries"),
    "elem_ioi": dict(level="elem", unit_type="ioi", label="Element IOIs"),
    "elem_dur": dict(level="elem", unit_type="dur", label="Element durations"),
    "elem_bound": dict(level="elem", unit_type="bound", label="Element boundaries"),
    "gap_ioi": dict(level="gap", unit_type="ioi", label="Gap IOIs"),
    "gap_dur": dict(level="gap", unit_type="dur", label="Gap durations"),
}

# The column each label type is read from. 'shuffle' is one fixed
# within-chunk shuffle stored at preprocessing time, used to illustrate the
# null in figures; it is shuffled from 'label' where there is one (bvs, ns)
# and from 'label_pred' otherwise.
LABEL_TYPES: dict[LabelType, str] = {
    "annot": "label",
    "pred": "label_pred",
    "shuffle": "label_shuffle",
}

# Per dataset: its rhythm types, label types, and the column identifying
# birds. rt has no known identity, so birds are the predicted 'bird_id_pred'.
DATASETS: dict[str, dict] = {
    "bvs": dict(
        rhythm_types=list(RHYTHM_TYPES),
        label_types=["annot", "pred", "shuffle"],
        bird_id_col="bird_id",
    ),
    "ns": dict(
        rhythm_types=["syll_ioi", "syll_dur", "syll_bound", "gap_ioi", "gap_dur"],
        label_types=["annot", "pred", "shuffle"],
        bird_id_col="bird_id",
    ),
    "rt": dict(
        rhythm_types=["syll_ioi"],
        label_types=["pred", "shuffle"],
        bird_id_col="bird_id_pred",
    ),
    "bwps": dict(
        rhythm_types=["syll_ioi"],
        label_types=["pred", "shuffle"],
        bird_id_col="bird_id",
    ),
    "bwps-clean": dict(
        rhythm_types=["syll_ioi"],
        label_types=["pred", "shuffle"],
        bird_id_col="bird_id",
    ),
}

# Every condition of the study: a dataset, rhythm type and label type.
conditions: list[dict] = [
    dict(dataset_id=dataset_id, rhythm_type=rhythm_type, label_type=label_type)
    for dataset_id, spec in DATASETS.items()
    for rhythm_type in spec["rhythm_types"]
    for label_type in spec["label_types"]
]
