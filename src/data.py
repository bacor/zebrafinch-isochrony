"""Loading and saving the units CSVs in `data/`, and building dyads from them.

Each rhythm type of a dataset has one file under `data/<dataset_id>/rhythms/`,
e.g. `data/bvs/rhythms/bvs-syll_ioi.csv`. The algorithms that build these
files live in `rhythm_types.py`.
"""

from pathlib import Path
from typing import List, Mapping, Optional
import pandas as pd

try:
    from .constants import DATASETS, LABEL_TYPES, RhythmType, conditions
    from .rhythm_types import build_dyads
except ImportError:
    from constants import DATASETS, LABEL_TYPES, RhythmType, conditions
    from rhythm_types import build_dyads

CUR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUR_DIR.parent
DATA_DIR = ROOT_DIR / "data"

LABEL_COLUMNS = tuple(LABEL_TYPES.values())


def units_path(
    dataset_id: str, rhythm_type: RhythmType, data_dir: Path = DATA_DIR
) -> Path:
    """Path to the units CSV of one dataset and rhythm type."""
    return data_dir / dataset_id / "rhythms" / f"{dataset_id}-{rhythm_type}.csv"


def save_units(
    units: pd.DataFrame,
    rhythm_type: RhythmType,
    dataset_id: str,
    data_dir: Path = DATA_DIR,
) -> Path:
    """Save units to their standardized CSV path and return the filepath."""
    path = units_path(dataset_id, rhythm_type, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    units.to_csv(path, index=0)
    return path


def load_units(
    dataset_id: str,
    rhythm_type: RhythmType,
    data_dir: Path = DATA_DIR,
    use_milliseconds: bool = True,
) -> pd.DataFrame:
    """Load the units of one dataset and rhythm type, as saved by `save_units`.

    With `use_milliseconds` (default), 'start' and 'duration' are converted
    from seconds to milliseconds.
    """
    units = pd.read_csv(units_path(dataset_id, rhythm_type, data_dir))
    if use_milliseconds:
        units[["start", "duration"]] *= 1000
    return units


def load_all_units(
    conds: List[Mapping[str, str]] = conditions,
    data_dir: Path = DATA_DIR,
    use_milliseconds: bool = True,
    bird_id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Load and concatenate the units of every dataset and rhythm type in `conds`.

    Bird identity is read from `bird_id_col` (default: each dataset's own,
    see `constants.DATASETS`) and always stored as `bird_id`. Rhythm types
    where that column is entirely missing are skipped, since bird-level
    metrics can't be computed for them. Each row is also tagged with its
    `dataset_id` and `rhythm_type`.

    The label types in `conds` are not used: every label column is kept,
    and filtering by label type is left to the metrics collection code.
    """
    dfs = []
    for dataset_id, rhythm_type in dict.fromkeys(
        (c["dataset_id"], c["rhythm_type"]) for c in conds
    ):
        df = load_units(
            dataset_id,
            rhythm_type,
            data_dir=data_dir,
            use_milliseconds=use_milliseconds,
        )
        col = bird_id_col or DATASETS[dataset_id]["bird_id_col"]
        if col != "bird_id":
            df = df.assign(bird_id=df[col])
        if df["bird_id"].isna().all():
            print(f"Skipping {dataset_id}-{rhythm_type}: no {col} data")
            continue
        df = df.assign(dataset_id=dataset_id, rhythm_type=rhythm_type)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_dyads(
    dataset_id: str,
    rhythm_type: RhythmType,
    bird_id_col: Optional[str] = None,
    data_dir: Path = DATA_DIR,
    use_milliseconds: bool = True,
) -> pd.DataFrame:
    """All dyads (pairs of successive units in a chunk) of one dataset and rhythm type.

    One row per dyad, with `bird_id_col` (default: the dataset's own, see
    `constants.DATASETS`), `chunk_idx`, `start`, `duration_1`, `duration_2`
    and every column in `LABEL_COLUMNS`, joined with a dash (e.g. 'a-b';
    NaN where the units have no such label); see `rhythm_types.build_dyads`.
    """
    units = load_units(
        dataset_id, rhythm_type, data_dir=data_dir, use_milliseconds=use_milliseconds
    )
    bird_id_col = bird_id_col or DATASETS[dataset_id]["bird_id_col"]
    dyads = build_dyads(units, [bird_id_col, "chunk_idx"], labels=LABEL_COLUMNS)
    return dyads.reindex(
        columns=[*dyads.columns, *(c for c in LABEL_COLUMNS if c not in dyads)]
    )


def load_all_dyads(
    conds: List[Mapping[str, str]] = conditions,
    bird_id_col: Optional[str] = None,
) -> pd.DataFrame:
    """The dyads of every dataset and rhythm type in `conds` (see `load_all_units`).

    Bird identity is stored as `bird_id`; each dyad is tagged with its
    `dataset_id` and `rhythm_type`.
    """
    units = load_all_units(conds=conds, bird_id_col=bird_id_col)
    return build_dyads(
        units,
        ["dataset_id", "rhythm_type", "bird_id", "chunk_idx"],
        labels=LABEL_COLUMNS,
    )
