"""Building the units of each rhythm type from chunks, and dyads from units.

A unit is one duration in a rhythm type (see `constants.RHYTHM_TYPES`).
From a sequence of annotated syllables or elements, this module builds:
- bound: all intervals between successive boundaries, i.e. the units and the
  gaps between them, in time order (`is_gap` tells them apart)
- dur: unit durations
- ioi: time between onsets of successive units (the final IOI ends at
  chunk end)
- gap_dur, gap_ioi: the same for the gaps between units

A units DataFrame has columns: 'start', 'duration', 'label', 'chunk_idx',
and 'bird_id' (bound units also have 'is_gap').

The preprocess-*.ipynb notebooks use these to build the CSVs in `data/`.
To load them, or the dyads built from them, see `data.py`.
"""

from typing import Iterable, Literal, Sequence, TypedDict
import numpy as np
import pandas as pd

try:
    from .constants import RhythmType
except ImportError:
    from constants import RhythmType


class ChunkArgs(TypedDict):
    """Arguments needed to build bound units for a single chunk."""

    chunk_df: pd.DataFrame
    start_col: str
    end_col: str
    label_col: str
    bird_id: str | int
    chunk_idx: int
    min_gap: float


def is_increasing(column: pd.Series) -> bool:
    """Return True when a series is strictly increasing."""
    return column.diff().iloc[1:].gt(0).all()


def _strip(label):
    return label.strip() if isinstance(label, str) else label


def get_bound_units_from_chunk(
    chunk_df: pd.DataFrame,
    start_col: str,
    end_col: str,
    label_col: str,
    bird_id: str | int,
    chunk_idx: int,
    min_gap: float = 1e-10,
) -> pd.DataFrame:
    """Convert a chunk-level DataFrame into bound units: units and gaps in time order.

    Args:
        chunk_df: DataFrame containing chunk timing information.
        start_col: Column name containing unit start times.
        end_col: Column name containing unit end times.
        label_col: Column name containing unit labels (missing values propagate).
        bird_id: Identifier to assign to all units from this chunk.
        chunk_idx: Chunk index to assign to all units.
        min_gap: Minimum absolute gap to include between units.

    Returns:
        DataFrame with units and gaps in time order. A gap's label is NaN when
        either neighbouring unit's label is missing.
    """

    if len(chunk_df) == 0:
        raise ValueError(
            f"chunk_df is empty for bird {bird_id}, chunk {chunk_idx} "
            "(no annotated units) -- check the source annotation file"
        )

    if not is_increasing(chunk_df[start_col]):
        raise ValueError("Start times should be strictly increasing.")

    units = []
    rows = [row for _, row in chunk_df.iterrows()]
    for i, row in enumerate(rows):
        start = row[start_col]
        end = row[end_col]
        label = row[label_col]

        # Add a unit
        units.append(
            dict(
                start=start,
                duration=end - start,
                is_gap=False,
                label=_strip(label),
            )
        )

        # Add a gap
        if i < len(rows) - 1:
            next_unit = rows[i + 1]
            next_start = next_unit[start_col]
            next_label = next_unit[label_col]
            if abs(end - next_start) > min_gap:
                # The gap between a and b is labeled ab
                gap_label = (
                    np.nan
                    if pd.isna(label) or pd.isna(next_label)
                    else f"{_strip(label)}{_strip(next_label)}"
                )
                units.append(
                    dict(
                        start=end,
                        duration=next_start - end,
                        is_gap=True,
                        label=gap_label,
                    )
                )

    # Turn into a dataframe, clean up, and return
    units = pd.DataFrame(units)
    units["chunk_idx"] = int(chunk_idx)
    units["bird_id"] = bird_id
    return units


def get_dur_units(bound: pd.DataFrame, gaps: bool = False) -> pd.DataFrame:
    """Unit durations (or gap durations, with `gaps=True`) from bound units."""
    selected = bound.loc[bound["is_gap"] == gaps, :]
    return selected.drop(columns="is_gap").reset_index(drop=True)


def get_ioi_label(cur_unit, next_unit, gaps, is_last):
    """Label of the IOI starting at `cur_unit`; NaN if a label it needs is missing.

    Unit IOIs join both labels ('ab'), the final unit IOI of a chunk gets a
    '!' suffix ('a!'), and gap IOIs join the first letters of both labels.
    """
    cur_label = cur_unit["label"]
    next_label = next_unit["label"]
    if pd.isna(cur_label) or ((gaps or not is_last) and pd.isna(next_label)):
        return np.nan
    if gaps:
        return f"{cur_label[0]}{next_label[0]}"
    elif is_last:
        return f"{cur_label}!"
    else:
        return f"{cur_label}{next_label}"


def get_ioi_units(bound: pd.DataFrame, gaps: bool = False) -> pd.DataFrame:
    """Inter-onset intervals of the units (or gaps, with `gaps=True`) in bound units.

    Returns an empty DataFrame when there are fewer than two units (gaps).
    """
    indices = np.where(bound["is_gap"] == gaps)[0]
    columns = bound.columns.drop("is_gap")

    if len(indices) <= 1:
        kind = "gap" if gaps else "unit"
        print(f"Cannot extract {kind} IOIs from {len(indices)} {kind}(s).")
        return pd.DataFrame(columns=columns)

    # Collect all IOIs
    iois = []
    for cur_idx, next_idx in zip(indices, indices[1:]):
        cur_unit = bound.iloc[cur_idx, :]
        next_unit = bound.iloc[next_idx, :]
        ioi = cur_unit.copy()
        ioi["duration"] = next_unit["start"] - cur_unit["start"]
        ioi["label"] = get_ioi_label(cur_unit, next_unit, gaps, False)
        iois.append(ioi)

    # Add final IOI
    last_unit = bound.iloc[indices[-1], :]
    final_unit = bound.iloc[-1, :]
    end = final_unit["start"] + final_unit["duration"]
    ioi = last_unit.copy()
    ioi["duration"] = end - last_unit["start"]
    ioi["label"] = get_ioi_label(last_unit, final_unit, gaps, True)
    iois.append(ioi)

    # Return as clean dataframe
    return pd.DataFrame(iois)[columns].reset_index(drop=True)


def drop_short_chunks(
    units: pd.DataFrame,
    min_units: int = 2,
    by: list[str] = ["bird_id", "chunk_idx"],
) -> pd.DataFrame:
    """Drop chunks with fewer than `min_units` units, as these yield no dyads.

    Use `by=["chunk_idx"]` when bird identity is not yet known (RT).
    """
    counts = units.groupby(by)[by[0]].transform("size")
    return units[counts >= min_units].reset_index(drop=True)


def aggregate_units(
    chunks_iterator: Iterable[ChunkArgs],
    level: Literal["syll", "elem"],
) -> dict[RhythmType, pd.DataFrame]:
    """The units of every rhythm type at `level`, for a collection of chunks.

    Args:
        chunks_iterator: Iterable of `ChunkArgs` dictionaries for
            `get_bound_units_from_chunk`. Empty chunks are skipped.
        level: The level of the annotated units ('syll' or 'elem').

    Returns:
        Dictionary with concatenated, `drop_short_chunks`-filtered
        DataFrames for `{level}_bound`, `{level}_dur` and
        `{level}_ioi`. For syllables also `gap_dur` and `gap_ioi`:
        the gap level consists of the gaps between syllables.
    """
    results = {f"{level}_bound": [], f"{level}_dur": [], f"{level}_ioi": []}
    if level == "syll":
        results.update(gap_dur=[], gap_ioi=[])

    for kwargs in chunks_iterator:
        if len(kwargs["chunk_df"]) == 0:
            print(
                f"Skipping bird {kwargs['bird_id']}, chunk {kwargs['chunk_idx']}: no annotated units"
            )
            continue
        bound = get_bound_units_from_chunk(**kwargs)
        results[f"{level}_bound"].append(bound)
        results[f"{level}_dur"].append(get_dur_units(bound))
        results[f"{level}_ioi"].append(get_ioi_units(bound))
        if level == "syll":
            results["gap_dur"].append(get_dur_units(bound, gaps=True))
            results["gap_ioi"].append(get_ioi_units(bound, gaps=True))

    return {
        # Chunks too short for a rhythm type give empty tables; leave them out.
        rhythm_type: drop_short_chunks(pd.concat([u for u in units if len(u)]))
        for rhythm_type, units in results.items()
    }


def build_dyads(
    units: pd.DataFrame, by: list[str], labels: Sequence[str] = ("label",)
) -> pd.DataFrame:
    """Every dyad (pair of successive units) within each `by`-group (one chunk).

    Returns columns `by + ['start', 'duration_1', 'duration_2']`, plus each
    column in `labels` that `units` has, joined with a dash (e.g. 'a-b';
    NaN if either label is missing). One row per dyad; a chunk of n units
    gives n - 1 dyads.
    """
    labels = [col for col in labels if col in units]
    df = units.sort_values(by + ["start"])
    following = df.groupby(by, sort=False)[["duration", *labels]].shift(-1)
    dyads = df[by + ["start"]].assign(
        duration_1=df["duration"], duration_2=following["duration"]
    )
    for col in labels:
        both = df[col].notna() & following[col].notna()
        dyads[col] = (df[col].astype(str) + "-" + following[col].astype(str)).where(
            both
        )
    return dyads.dropna(subset=["duration_2"]).reset_index(drop=True)
