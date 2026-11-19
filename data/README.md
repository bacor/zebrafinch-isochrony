# Rhythm types, units, dyads

## Datasets

- `ns`: Norton & Scharff (2016). Only contains syllable-level annotations.
- `bvs`: Burchardt, Varkevisser & Spierings (2025). Three birds were
  recorded twice, as tutor and (younger) as tutee. The rhythm files keep both;
  the tutee recordings have a `-tutee` suffix on `bird_id` and are left out of
  all analyses, which use the 50 remaining birds.
- `rt`, `bwps`, `bwps-clean`: syllable IOIs only, with predicted labels
  (and, for `rt`, predicted bird identity).

Which rhythm types and label types each dataset has is set in
`src/constants.py` (`DATASETS`).

## Rhythm types

Each recording is split into **chunks**: uninterrupted stretches of song,
delimited as in the source data. In `bvs` every chunk is one motif; in the
other datasets a chunk need not align with motif boundaries.

The annotations are on- and offsets of syllables or elements. Within a chunk,
the time between two of them is a gap, so every chunk is a sequence of
syllables (or elements) and gaps. Note that they need not alternate: inside
a syllable, elements follow each other without a gap.

A **rhythm type** combines a **level** (`syll`, `elem`, or `gap`: the gaps
between syllables) with a **unit type**:

- **dur**: durations
- **ioi**: the time between onsets of successive syllables (elements, gaps).
  The final IOI of a chunk ends at the end of the chunk: we ignore the gap
  separating chunks.
- **bound**: all intervals between successive boundaries (on- and offsets),
  i.e. the durations of both syllables (elements) and gaps.

This gives eight rhythm types: `syll_dur`, `syll_ioi`, `syll_bound`,
`elem_dur`, `elem_ioi`, `elem_bound`, `gap_dur` and `gap_ioi` (gap boundaries
would be the syllable boundaries). Each duration in a rhythm type is a
**unit**.

The units of each rhythm type are stored as
`[dataset_id]/rhythms/[dataset_id]-[rhythm_type].csv`, with columns:

1. `start`: the start time of the unit in seconds
2. `duration`: the duration of the unit in seconds
3. `is_gap` (`*_bound` only): whether the unit is a gap
4. `label`: label of the unit (e.g., `a` for a syllable and `ab` for the gap
   between `a` and `b`)
5. `chunk_idx`: an index identifying the chunk of that bird.
6. `bird_id`: a unique id for the bird.

followed by `bird_id_pred`, `label_pred` and `label_shuffle`: predicted bird
identity, predicted labels, and one within-chunk shuffle of the labels.

## Dyads

A dyad is a pair of successive units within a chunk. Dyads are not stored:
`src/data.py`'s `load_dyads`/`load_all_dyads` build them from the units
(`src/rhythm_types.py`'s `build_dyads`), joining the labels of both units
with a dash (e.g. `a-b`).

## Metrics

Metrics computed by `src/pipeline/collect_metrics.py` (see `src/metrics/`)
are written to `[dataset_id]/metrics/`, one file per level and label column:
`[dataset_id]-{label,cluster,bird}-metrics[-using_label_pred|-using_label_shuffle].csv`.
Condition-level metrics are in `metrics.csv`, and the permutation tests
(`src/pipeline/collect_permutation_tests.py`) in `permutation-tests.csv`.
