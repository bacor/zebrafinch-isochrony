"""Permutation-test condition-level metrics against the shuffled baseline.

For every condition with annotated or predicted labels, the metrics in
`metrics.condition.TESTS` are computed from the units (`metrics.analyze`,
then `metrics.condition`) and compared with the same computation on labels
shuffled within each chunk (`metrics.permutation`). Each draw shuffles once
and recomputes every metric from it. The result is one long table,
`data/permutation-tests.csv`.

Kept separate from `collect_metrics.py` since thousands of draws take
tens of minutes rather than seconds.
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from statsmodels.stats.multitest import multipletests

try:
    from ..constants import LABEL_TYPES
    from ..data import DATA_DIR, load_all_units
    from ..metrics import analyze
    from ..metrics.condition import TESTS, bird_metrics, condition_metrics
    from ..metrics.permutation import permutation_test, shuffle_labels_within_chunk
    from .collect_metrics import (
        INDEX_BIRD,
        INDEX_CONDITION,
        INDEX_CHUNK,
        with_label_type,
    )
except ImportError:
    # See collect_metrics.py for why: notebooks bare-import this with only
    # `src/` on sys.path, making relative imports above fail.
    from constants import LABEL_TYPES
    from data import DATA_DIR, load_all_units
    from metrics import analyze
    from metrics.condition import TESTS, bird_metrics, condition_metrics
    from metrics.permutation import permutation_test, shuffle_labels_within_chunk
    from pipeline.collect_metrics import (
        INDEX_BIRD,
        INDEX_CONDITION,
        INDEX_CHUNK,
        with_label_type,
    )

INDEX_TEST = INDEX_CONDITION + ["label_type", "metric"]
DEFAULT_N_PERMUTATIONS = 2000


def permtest_path(data_dir: Path = DATA_DIR) -> Path:
    """Path of the permutation-test table."""
    return data_dir / "permutation-tests.csv"


def load_permtest(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the permutation-test table (default: `permtest_path`)."""
    return pd.read_csv(path or permtest_path(), index_col=INDEX_TEST)


def condition_level(units: pd.DataFrame) -> pd.DataFrame:
    """The condition-level metrics of `units`, labeled by their 'label' column."""
    birds = bird_metrics(*analyze(units, INDEX_BIRD), by=INDEX_BIRD)
    return condition_metrics(birds, by=INDEX_CONDITION)


def _to_long(result: pd.DataFrame, n_birds: pd.Series) -> pd.DataFrame:
    """One row per condition and metric, with the p-value of the metric's own tail."""
    rows = []
    for metric, direction in TESTS.items():
        p_col = f"{metric}_p" if direction == "low" else f"{metric}_p_high"
        rows.append(
            pd.DataFrame(
                {
                    "metric": metric,
                    "direction": direction,
                    "value": result[metric],
                    "null_mean": result[f"{metric}_null_mean"],
                    "null_std": result[f"{metric}_null_std"],
                    "z": result[f"{metric}_z"],
                    "p": result[p_col],
                    "n_birds": n_birds,
                }
            )
        )
    return pd.concat(rows)


def permutation_test_conditions(
    units: pd.DataFrame,
    label_col: str,
    n: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Permutation-test `TESTS` for every condition in `units`, with labels `label_col`.

    Tutees are left out.
    """
    units = units[units[label_col].notna() & ~units["bird_id"].str.endswith("-tutee")]
    units = units.assign(label=units[label_col])

    observed = condition_level(units)
    result = permutation_test(
        observed[list(TESTS)],
        lambda rng: condition_level(
            shuffle_labels_within_chunk(units, rng, by=INDEX_CHUNK)
        ),
        n=n,
        seed=seed,
        n_jobs=n_jobs,
        progress_every=max(1, n // 20),
        progress_label=f"[{label_col}] ",
    )
    return _to_long(observed[list(TESTS)].join(result), observed["n_birds"])


def collect_all(
    output: Optional[Path] = None,
    n: int = DEFAULT_N_PERMUTATIONS,
    seed: int = 0,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Test annotated and predicted labels in every condition and write the table.

    `output` defaults to `permtest_path`. `p_bonf` is Bonferroni corrected
    per metric, across all conditions; with m tests per metric it can only
    drop below 0.05 when `n` is at least about 20 * m (e.g. ~600 draws for
    29 conditions). `n_jobs` is passed to joblib (-1: all CPUs, -2: all but one).
    """

    units = load_all_units()
    tables = []
    for label_type in ["annot", "pred"]:
        label_col = LABEL_TYPES[label_type]
        table = permutation_test_conditions(
            with_label_type(units, label_type), label_col, n=n, seed=seed, n_jobs=n_jobs
        )
        tables.append(table.assign(label_type=label_type))

    result = pd.concat(tables).reset_index().set_index(INDEX_TEST).sort_index()
    result["p_bonf"] = result.groupby("metric")["p"].transform(
        lambda p: multipletests(p, method="bonferroni")[1]
    )
    result["n_draws"] = n
    output = output or permtest_path()
    result.to_csv(output)
    print(f"Saved permutation tests -> {output}")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Permutation-test condition-level metrics and write data/permutation-tests.csv."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: data/permutation-tests.csv).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help="Number of permutation draws.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for permutation draws (-1: all CPUs, -2: all but one).",
    )
    args = parser.parse_args(argv)
    collect_all(args.output, n=args.n, seed=args.seed, n_jobs=args.n_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
