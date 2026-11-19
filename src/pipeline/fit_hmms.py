"""Label prediction: fit one HMM per bird for every dataset and rhythm type.

The generic HMM fitting lives in `src.hmm`, which is reusable on its own;
this module wires it up to this study's data loading and shared storage.
"""

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from ..constants import DATASETS, conditions
    from ..data import load_units
    from ..utils import get_shared_data_directory
    from ..hmm import fit, save_fit
except ImportError:
    # Notebooks put src/ on sys.path and import this as `pipeline.X`.
    from constants import DATASETS, conditions
    from data import load_units
    from utils import get_shared_data_directory
    from hmm import fit, save_fit


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message.rstrip()}\n")


def _bird_sequences(
    dataset_id: str, rhythm_type: str, bird_id_col: str = "bird_id"
) -> Dict[str, list]:
    """Durations of one dataset and rhythm type, as bird_id -> one array per chunk.

    `bird_id_col` identifies the bird; use 'bird_id_pred' for RT, where only
    the identity predicted by `pipeline.predict_bird_ids` is available.
    """
    units = load_units(dataset_id, rhythm_type)
    bird_data = {}
    for bird_id in units[bird_id_col].dropna().unique():
        bird_units = units[units[bird_id_col] == bird_id]
        bird_data[str(bird_id)] = [
            group["duration"].values for _, group in bird_units.groupby("chunk_idx")
        ]
    return bird_data


def _fit_and_save_bird(
    bird_id: str,
    sequences: list,
    kwargs: Dict[str, Any],
    output_dir: Path,
) -> str:
    """Fit one bird's HMM, save it, and return a log message.

    hmmlearn's warnings (from both `logging` and `warnings`) go to run.log.
    """
    log_path = output_dir / "run.log"

    def _log_warning(message, category, filename, lineno, file=None, line=None):
        _append_log(log_path, f"warning bird={bird_id} {category.__name__}: {message}")

    hmm_logger = logging.getLogger("hmmlearn")
    log_handler = logging.FileHandler(log_path, encoding="utf-8")
    log_handler.setFormatter(
        logging.Formatter(
            f"[%(asctime)s] warning bird={bird_id} %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    hmm_logger.addHandler(log_handler)
    try:
        with warnings.catch_warnings():
            warnings.showwarning = _log_warning
            result = fit(sequences, **kwargs)
    except Exception as e:
        return f"error bird={bird_id} error={e}"
    finally:
        hmm_logger.removeHandler(log_handler)
        log_handler.close()
    save_fit(result, output_dir / f"{bird_id}.joblib")

    return (
        f"done bird={bird_id} n_states={result['n_states']} "
        f"k_initial={result['k_initial']} n_chunks={len(result['labels'])}"
    )


def _compute_all_birds(
    bird_data: Sequence[Tuple[str, list]],
    kwargs: Dict[str, Any],
    output_dir: Path,
    n_jobs: int,
) -> List[str]:
    if n_jobs == 1:
        return [
            _fit_and_save_bird(bird_id, sequences, kwargs, output_dir)
            for bird_id, sequences in tqdm(bird_data, desc="birds", unit="bird")
        ]
    return cast(
        List[str],
        Parallel(n_jobs=n_jobs)(
            delayed(_fit_and_save_bird)(bird_id, sequences, kwargs, output_dir)
            for bird_id, sequences in bird_data
        ),
    )


def get_output_dir(dataset_id: str, rhythm_type: str) -> Path:
    """Directory with one `<bird_id>.joblib` fit per bird, in the shared data directory.

    Laid out as `hmm/<dataset_id>/<rhythm_type>/`.
    """
    shared_data = Path(get_shared_data_directory())
    return shared_data / "hmm" / dataset_id / rhythm_type


def run(
    dataset_id: str,
    rhythm_type: str,
    refresh: bool = False,
    n_jobs: int = 1,
    bird_id_col: Optional[str] = None,
    k_min: int = 1,
    k_max: int = 12,
    select_n_iter: int = 100,
    window: int = 2,
    transmat_prior: float = 1.0,
    seed: int = 0,
) -> None:
    """Fit and save an HMM for every bird of one dataset and rhythm type.

    Birds that already have a fit are skipped unless `refresh`. Progress and
    hmmlearn warnings go to `run.log` in `get_output_dir`. `bird_id_col`
    defaults to the dataset's own (see `constants.DATASETS`); `n_jobs` is
    passed to joblib (-1: all CPUs, -2: all but one). The remaining arguments
    are passed to `hmm.fit`.
    """
    bird_id_col = bird_id_col or DATASETS[dataset_id]["bird_id_col"]
    output_dir = get_output_dir(dataset_id, rhythm_type)
    log_path = output_dir / "run.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs: Dict[str, Any] = dict(
        k_min=k_min,
        k_max=k_max,
        select_n_iter=select_n_iter,
        window=window,
        transmat_prior=transmat_prior,
        seed=seed,
    )
    _append_log(
        log_path,
        f"start dataset={dataset_id} rhythm_type={rhythm_type} "
        f"bird_id_col={bird_id_col} {' '.join(f'{k}={v}' for k, v in kwargs.items())} n_jobs={n_jobs}",
    )

    bird_sequences = _bird_sequences(dataset_id, rhythm_type, bird_id_col)
    if not bird_sequences:
        message = f"skip: no {bird_id_col} data for {dataset_id}-{rhythm_type}"
        print(message)
        _append_log(log_path, message)
        return

    bird_data = list(bird_sequences.items())

    # Skip birds that already have a fit, so an interrupted run can resume.
    if not refresh:
        bird_data = [
            (bird_id, seqs)
            for bird_id, seqs in bird_data
            if not (output_dir / f"{bird_id}.joblib").exists()
        ]
    if not bird_data:
        _append_log(log_path, "nothing to do, all birds already fit")
        return

    messages = _compute_all_birds(bird_data, kwargs, output_dir, n_jobs)
    for message in messages:
        if message.startswith("error"):
            print(message)
        _append_log(log_path, message)


def run_all(conds: Sequence[Mapping[str, str]] = conditions, **kwargs: Any) -> None:
    """`run` for every dataset and rhythm type in `conds`, with the same `kwargs`."""
    for dataset_id, rhythm_type in dict.fromkeys(
        (c["dataset_id"], c["rhythm_type"]) for c in conds
    ):
        print(f"Running HMM: dataset={dataset_id} rhythm_type={rhythm_type}")
        run(dataset_id, rhythm_type, **kwargs)


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    cleaned = [item.strip() for item in value.split(",") if item.strip()]
    return cleaned or None


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Fit a per-bird HMM to syllable/gap durations.",
        epilog="With no dataset_id/rhythm_type, runs every condition in constants.conditions.",
    )
    parser.add_argument("dataset_id", nargs="?", default=None)
    parser.add_argument("rhythm_type", nargs="?", default=None)
    parser.add_argument(
        "--datasets",
        help="Comma-separated dataset IDs to restrict the batch run to "
        "(only valid without dataset_id/rhythm_type; default: every dataset in constants.conditions).",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of birds fitted in parallel (-1: all CPUs, -2: all but one).",
    )
    parser.add_argument(
        "--bird-id-col",
        default=None,
        help="Units column identifying the individual to group by (default: "
        "the dataset's own, see constants.DATASETS; 'bird_id_pred' for rt).",
    )
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--select-n-iter", type=int, default=100)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--transmat-prior", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    positional = [args.dataset_id, args.rhythm_type]
    if any(positional) and not all(positional):
        parser.error(
            "dataset_id and rhythm_type must be given together, or both omitted to run every condition."
        )

    shared_kws = dict(
        refresh=args.refresh,
        n_jobs=args.n_jobs,
        bird_id_col=args.bird_id_col,
        k_min=args.k_min,
        k_max=args.k_max,
        select_n_iter=args.select_n_iter,
        window=args.window,
        transmat_prior=args.transmat_prior,
        seed=args.seed,
    )

    if all(positional):
        if args.datasets:
            parser.error(
                "--datasets can't be combined with a single dataset_id/rhythm_type."
            )
        run(dataset_id=args.dataset_id, rhythm_type=args.rhythm_type, **shared_kws)
    else:
        conds = conditions
        if args.datasets:
            wanted = set(_split_csv(args.datasets))
            conds = [c for c in conds if c["dataset_id"] in wanted]
        run_all(conds, **shared_kws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
