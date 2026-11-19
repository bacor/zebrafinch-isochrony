"""Compute label, cluster, bird and condition metrics and write them to disk.

The metrics themselves live in `src.metrics`; this module wires them up to
this study's conditions, data loading and output paths.

```bash
poetry run python -m src.pipeline.collect_metrics
```
"""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

try:
    from ..constants import LABEL_TYPES, RHYTHM_TYPES, conditions
    from ..data import DATA_DIR, load_all_dyads, load_all_units
    from ..metrics import analyze
    from ..metrics.condition import bird_metrics, condition_metrics, isochrony_metrics
    from ..metrics.dyad import dyad_scores
except ImportError:
    # Notebooks put src/ on sys.path and import this as `pipeline.X`.
    from constants import LABEL_TYPES, RHYTHM_TYPES, conditions
    from data import DATA_DIR, load_all_dyads, load_all_units
    from metrics import analyze
    from metrics.condition import bird_metrics, condition_metrics, isochrony_metrics
    from metrics.dyad import dyad_scores

INDEX_BIRD = ["dataset_id", "rhythm_type", "bird_id"]
INDEX_CONDITION = ["dataset_id", "rhythm_type"]
INDEX_CHUNK = INDEX_BIRD + ["chunk_idx"]


def with_label_type(units: pd.DataFrame, label_type: str) -> pd.DataFrame:
    """The units of the conditions (in `constants.conditions`) with this label type."""
    keys = {
        (c["dataset_id"], c["rhythm_type"])
        for c in conditions
        if c["label_type"] == label_type
    }
    return units[
        [key in keys for key in zip(units["dataset_id"], units["rhythm_type"])]
    ]


def metrics_path(
    dataset_id: str, level: str, label_col: str = "label", data_dir: Path = DATA_DIR
) -> Path:
    """Path to one dataset's metrics CSV. `level` is 'bird', 'cluster', or 'label'.

    The default `label_col` ('label') gets no suffix; any other column
    (e.g. 'label_pred', 'label_shuffle') gets `-using_{label_col}`.
    """
    suffix = "" if label_col == "label" else f"-using_{label_col}"
    return (
        data_dir / dataset_id / "metrics" / f"{dataset_id}-{level}-metrics{suffix}.csv"
    )


def with_rhythm_types(df: pd.DataFrame, index: List[str]) -> pd.DataFrame:
    """Add rhythm-type metadata (`constants.RHYTHM_TYPES`) to `df`, keeping its index.

    The rhythm type's `label` becomes the column `condition`, and its
    position in `RHYTHM_TYPES` the column `order`.
    """
    meta = (
        pd.DataFrame.from_dict(RHYTHM_TYPES, orient="index")
        .rename(columns={"label": "condition"})
        .assign(order=range(len(RHYTHM_TYPES)))
    )
    return (
        df.reset_index()
        .merge(meta, left_on="rhythm_type", right_index=True)
        .set_index(index)
    )


def _is_tutee(df: pd.DataFrame) -> pd.Series:
    return df.index.get_level_values("bird_id").str.endswith("-tutee")


def load_metrics(
    exclude_tutees: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bird, cluster and label metrics of every dataset and label type on disk.

    Rows get a `label_type` index level right after 'rhythm_type' (see
    `constants.LABEL_TYPES`), and rhythm-type metadata as columns. Tutees
    are excluded by default.
    """
    tables = {"bird": [], "cluster": [], "label": []}
    for dataset_id in sorted({c["dataset_id"] for c in conditions}):
        for label_type, label_col in LABEL_TYPES.items():
            if not metrics_path(dataset_id, "bird", label_col).exists():
                continue
            for level, frames in tables.items():
                index = INDEX_BIRD if level == "bird" else INDEX_BIRD + ["label"]
                df = pd.read_csv(
                    metrics_path(dataset_id, level, label_col), index_col=index
                )
                if exclude_tutees:
                    df = df[~_is_tutee(df)]
                df = (
                    with_rhythm_types(df, index)
                    .reset_index()
                    .assign(label_type=label_type)
                )
                frames.append(
                    df.set_index(
                        INDEX_CONDITION + ["label_type"] + index[len(INDEX_CONDITION) :]
                    )
                )
    return (
        pd.concat(tables["bird"]),
        pd.concat(tables["cluster"]),
        pd.concat(tables["label"]),
    )


def load_bird_metrics(exclude_tutees: bool = True) -> pd.DataFrame:
    """`metrics.condition.bird_metrics` for every condition, label type and bird."""
    bird, cluster, label = load_metrics(exclude_tutees=exclude_tutees)
    return bird_metrics(
        label, cluster, bird, by=INDEX_CONDITION + ["label_type", "bird_id"]
    )


def load_dyad_scores(exclude_tutees: bool = True) -> pd.DataFrame:
    """Every dyad with its isochrony scores (`metrics.dyad.dyad_scores`).

    Indexed by `INDEX_BIRD`. Dyads don't depend on labels, so there is no
    label type. Tutees are excluded by default.
    """
    dyads = load_all_dyads()[INDEX_BIRD + ["duration_1", "duration_2"]]
    dyads = dyads.join(dyad_scores(dyads)).set_index(INDEX_BIRD).sort_index()
    if exclude_tutees:
        dyads = dyads[~_is_tutee(dyads)]
    return dyads


def collect_condition_metrics(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Condition-level metrics per label type, written to `metrics.csv` in `data_dir`.

    The `metrics.condition.condition_metrics` of `load_bird_metrics`, joined
    with the label-free `metrics.condition.isochrony_metrics` (the same for
    every label type).
    """
    by = INDEX_CONDITION + ["label_type"]
    metrics = condition_metrics(load_bird_metrics(), by=by)
    isochrony = isochrony_metrics(load_dyad_scores(), by=INDEX_CONDITION)
    metrics = (
        metrics.reset_index()
        .merge(isochrony.reset_index(), on=INDEX_CONDITION)
        .set_index(by)
        .sort_index()
    )
    metrics.to_csv(data_dir / "metrics.csv")
    print(f"Saved condition metrics -> {data_dir / 'metrics.csv'}")
    return metrics


def collect_all(data_dir: Path = DATA_DIR) -> None:
    """Write the label, cluster, bird and condition metrics of every label type.

    Output: `data/<dataset_id>/metrics/<dataset_id>-{label,cluster,bird}-metrics[-using_<label_col>].csv`,
    then `collect_condition_metrics`.
    """
    units = load_all_units()
    for label_type, label_col in LABEL_TYPES.items():
        tables = dict(
            zip(
                ["label", "cluster", "bird"],
                analyze(with_label_type(units, label_type), INDEX_BIRD, label_col),
            )
        )
        for level, table in tables.items():
            for dataset_id, group in table.groupby(level="dataset_id"):
                path = metrics_path(dataset_id, level, label_col, data_dir)
                path.parent.mkdir(parents=True, exist_ok=True)
                group.to_csv(path)
                print(f"Saved {level} metrics -> {path}")
    collect_condition_metrics(data_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Compute all metrics and write them to data/."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    collect_all(args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
