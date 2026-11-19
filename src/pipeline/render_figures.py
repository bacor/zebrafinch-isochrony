"""Render and save the per-bird figures of every dataset and rhythm type.

Unlike `src.viz`, this module is specific to this study's data layout
and datasets (see `src.data`), and to the website's figure output. The
plotting itself lives in `src.viz` and is reusable on its own.
"""

import argparse
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, TypeVar, Literal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    from ..constants import DATASETS, RHYTHM_TYPES
    from ..data import DATA_DIR, load_dyads, load_units, units_path
    from ..viz.dyads_plot import DyadsPlotter
    from ..viz.utils import set_mpl_style
    from ..viz.density import show_density
    from ..viz.timeline import plot_timeline
except ImportError:
    # Notebooks put src/ on sys.path and import this as `pipeline.X`.
    from constants import DATASETS, RHYTHM_TYPES
    from data import DATA_DIR, load_dyads, load_units, units_path
    from viz.dyads_plot import DyadsPlotter
    from viz.utils import set_mpl_style
    from viz.density import show_density
    from viz.timeline import plot_timeline

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "figures"
# The website lives in the separate zebrafinch-isochrony-astro repository, cloned next to this one
ASTRO_FIGURES_DIR = (
    ROOT_DIR.parent / "zebrafinch-isochrony-astro" / "public" / "figures"
)
DATASET_IDS = list(DATASETS)
RHYTHM_TYPE_IDS = list(RHYTHM_TYPES)

## Helpers


def iter_rhythm_types(
    rhythm_types: Sequence[str] = RHYTHM_TYPE_IDS,
    dataset_ids: Sequence[str] = DATASET_IDS,
    data_dir: Path = DATA_DIR,
) -> Iterator[Tuple[str, str]]:
    """Yield the (dataset_id, rhythm_type) pairs that have a units file."""
    for dataset_id in dataset_ids:
        for rhythm_type in rhythm_types:
            if units_path(dataset_id, rhythm_type, data_dir).exists():
                yield dataset_id, rhythm_type


T = TypeVar("T")


def filter_list(
    items: Sequence[T],
    include: Optional[Sequence[T]] = None,
    exclude: Optional[Sequence[T]] = None,
    limit: Optional[int] = None,
    drop_na: bool = True,
) -> List[T]:
    """Items in `include` and not in `exclude`, at most `limit` of them.

    With `drop_na`, None and NaN items are dropped.
    """
    kept = [
        item
        for item in items
        if not (drop_na and pd.isna(item))
        and (include is None or item in include)
        and (exclude is None or item not in exclude)
    ]
    return kept[:limit]


def get_output_paths(
    dir_path: Path, filename: str, extensions=["png"], refresh: bool = False
) -> Tuple[bool, List[Path]]:
    """Create `dir_path`; return whether to render and the output path per extension."""
    dir_path.mkdir(parents=True, exist_ok=True)
    paths = [dir_path / f"{filename}.{ext}" for ext in extensions]
    should_render = refresh or np.any([not path.exists() for path in paths])
    return should_render, paths


def save_figure(paths: Sequence[Path], refresh: bool = False, **kwargs) -> None:
    """Save the current figure to `paths` (overwrite only if `refresh`) and close it."""
    plt.tight_layout()
    for path in paths:
        if refresh or not path.exists():
            options = dict(kwargs)
            if path.suffix in (".png", ".jpg"):
                options["dpi"] = 150
            if path.suffix == ".png":
                options["transparent"] = True
            plt.savefig(path, **options)
    plt.close()


def _label_column(
    df: pd.DataFrame, label_col: str = "label", fallback_col: str = "label_pred"
) -> str:
    """Pick `label_col`, or `fallback_col` when `label_col` is entirely empty.

    BWPS and BWPS-clean only have predicted labels ('label_pred').
    """
    if (
        df[label_col].isna().all()
        and fallback_col in df.columns
        and not df[fallback_col].isna().all()
    ):
        return fallback_col
    return label_col


def get_square_axis_limit(step: int = 50, offset: float = 0.1) -> int:
    """Square axis limit: the current extent rounded up to a multiple of `step`."""
    axis_range = max(plt.xlim()[1], plt.ylim()[1])
    limit = step
    threshold = 1 - offset
    while axis_range > limit * threshold:
        limit += step
    return limit


def get_centered_axis_limits(
    center: float = 0.5,
    step: float = 0.1,
    min_value: float = 0.0,
    max_value: float = 1.0,
    min_delta: float = 0.2,
) -> tuple[float, float]:
    """Symmetric x-axis limits around `center`, snapped to `step`."""
    current_min, current_max = plt.xlim()
    required_delta = max(abs(current_min - center), abs(current_max - center))
    delta = step * np.ceil(required_delta / step)
    delta = max(delta, min_delta)
    delta = min(delta, max_value - center, center - min_value)
    return center - delta, center + delta


## Timelines


def render_timelines(
    rhythm_types: Sequence[str] = RHYTHM_TYPE_IDS,
    include_birds: Optional[Sequence[str]] = None,
    exclude_birds: Optional[Sequence[str]] = None,
    dataset_ids: Sequence[str] = DATASET_IDS,
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    refresh: bool = True,
    limit: Optional[int] = None,
    bird_id_col: Optional[str] = None,
) -> None:
    """Render one timeline per bird and level (syllables or elements) to disk.

    Args:
        rhythm_types: Only the levels of the durations among these are rendered.
        dataset_ids: Dataset identifiers to process.
        data_dir: Root directory containing the input units.
        output_dir: Root directory for generated figures.
        refresh: When False, skip writing files that already exist.
        include_birds: Optional list of bird IDs to include.
        exclude_birds: Optional list of bird IDs to exclude.
        limit: Optional max count of birds to render.
        bird_id_col: Units column identifying the individual to group by.
            Defaults to each dataset's own (see `constants.DATASETS`).
    """
    set_mpl_style()

    durations = [r for r in rhythm_types if r in ("syll_dur", "elem_dur")]
    for dataset_id, rhythm_type in iter_rhythm_types(durations, dataset_ids, data_dir):
        level = RHYTHM_TYPES[rhythm_type]["level"]
        id_col = bird_id_col or DATASETS[dataset_id]["bird_id_col"]
        units = load_units(
            dataset_id, rhythm_type, data_dir=data_dir, use_milliseconds=False
        )
        birds = units[id_col].unique()
        birds = filter_list(
            birds, include=include_birds, exclude=exclude_birds, limit=limit
        )

        # Generate figures
        for bird_id in birds:
            dir = output_dir / dataset_id / str(bird_id) / "timeline"
            output_fn = f"{bird_id}-{level}-timeline"
            should_render, paths = get_output_paths(dir, output_fn, refresh=refresh)
            if should_render:
                bird = units[units[id_col] == bird_id]
                width_ratio = 5 if level == "syll" else 1
                plot_timeline(
                    bird,
                    label=_label_column(bird),
                    align_beginnings=True,
                    width_ratio=width_ratio,
                    figsize=(6, 4),
                )
                save_figure(paths, refresh)


def render_scatterplot(
    type: Literal["phase", "patdur"],
    rhythm_types: Sequence[str] = RHYTHM_TYPE_IDS,
    dataset_ids: Sequence[str] = DATASET_IDS,
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    refresh: bool = True,
    include_birds: Optional[Sequence[str]] = None,
    exclude_birds: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    bird_id_col: Optional[str] = None,
) -> None:
    """Render one dyad plot per bird and rhythm type to disk.

    `type` is "phase" (first against second duration) or "patdur" (rhythm
    ratio against dyad duration). The other arguments are as in
    `render_timelines`, except that all of `rhythm_types` are plotted.
    """
    set_mpl_style()

    selected_birds: dict[str, List[str]] = {}
    for dataset_id, rhythm_type in iter_rhythm_types(
        rhythm_types, dataset_ids, data_dir
    ):
        id_col = bird_id_col or DATASETS[dataset_id]["bird_id_col"]
        dyads = load_dyads(
            dataset_id, rhythm_type, bird_id_col=id_col, data_dir=data_dir
        )

        # Filter bird ids
        birds_in_rhythm = dyads[id_col].unique()
        if dataset_id not in selected_birds:
            selected_birds[dataset_id] = filter_list(
                birds_in_rhythm,
                include=include_birds,
                exclude=exclude_birds,
                limit=limit,
            )
        birds_in_rhythm_set = set(birds_in_rhythm)
        birds = [
            bird for bird in selected_birds[dataset_id] if bird in birds_in_rhythm_set
        ]
        if not birds:
            continue

        for bird_id in birds:
            dir = output_dir / dataset_id / str(bird_id) / type
            output_fn = f"{bird_id}-{rhythm_type}-{type}-plot"
            should_render, paths = get_output_paths(dir, output_fn, refresh=refresh)
            if should_render:
                bird = dyads[dyads[id_col] == bird_id]
                d1, d2 = bird["duration_1"].to_numpy(), bird["duration_2"].to_numpy()
                if type == "phase":
                    points = np.column_stack((d1, d2))
                elif type == "patdur":
                    points = np.column_stack((d1 / (d1 + d2), d1 + d2))

                labels = bird[_label_column(bird)]
                if labels.isna().all():
                    # No labels available at all: treat all points as one cluster.
                    labels = pd.Series("all", index=labels.index)

                plot = DyadsPlotter(
                    points,
                    list(labels),
                    bird["chunk_idx"],
                    max_chunks=200,
                    min_label_percentile=2,
                    random_state=0,
                )
                plot.plot(min_count=5)

                if type == "phase":
                    plot.phase_axes(
                        xlabel="first duration (ms)", ylabel="second duration (ms)"
                    )
                    axis_limit = get_square_axis_limit(step=50, offset=0.1)
                    plt.xlim(0, axis_limit)
                    plt.ylim(0, axis_limit)

                elif type == "patdur":
                    plot.patdur_axes(xlabel="pattern", ylabel="dyad duration (ms)")

                    # Set axes limits; even around .5 for x
                    axis_limit = get_square_axis_limit(step=50, offset=0.1)
                    plt.ylim(0, axis_limit)
                    x_min, x_max = get_centered_axis_limits(
                        center=0.5,
                        step=0.1,
                        min_delta=0.3,
                    )
                    plt.xlim(x_min, x_max)

                save_figure(paths, refresh)


def render_densities(
    rhythm_types: Sequence[str] = RHYTHM_TYPE_IDS,
    include_birds: Optional[Sequence[str]] = None,
    exclude_birds: Optional[Sequence[str]] = None,
    dataset_ids: Sequence[str] = DATASET_IDS,
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    refresh: bool = True,
    limit: Optional[int] = None,
    bird_id_col: Optional[str] = None,
) -> None:
    """Render one duration density plot per bird and rhythm type to disk.

    Arguments as in `render_timelines`, except that all of `rhythm_types`
    are plotted.
    """
    set_mpl_style()

    for dataset_id, rhythm_type in iter_rhythm_types(
        rhythm_types, dataset_ids, data_dir
    ):
        id_col = bird_id_col or DATASETS[dataset_id]["bird_id_col"]
        df = load_units(
            dataset_id, rhythm_type, data_dir=data_dir, use_milliseconds=False
        )
        birds = df[id_col].unique()
        birds = filter_list(
            birds, include=include_birds, exclude=exclude_birds, limit=limit
        )

        # Generate figures
        for bird_id in birds:
            dir = output_dir / dataset_id / str(bird_id) / "density"
            output_fn = f"{bird_id}-{rhythm_type}-density"
            should_render, paths = get_output_paths(dir, output_fn, refresh=refresh)
            if should_render:
                bird = df[df[id_col] == bird_id]
                show_density(bird, label=_label_column(bird))
                save_figure(paths, refresh)


## CLI


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    cleaned = [item.strip() for item in value.split(",") if item.strip()]
    return cleaned or None


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser for `main`."""
    parser = argparse.ArgumentParser(description="Render zebrafinch figures.")
    parser.add_argument(
        "--plot",
        choices=["all", "timeline", "phase", "patdur", "density"],
        default="all",
        help="Plot type to render.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force regeneration even if output files already exist.",
    )
    parser.add_argument(
        "--include-birds",
        default=None,
        help="Comma-separated bird IDs to include (e.g., bvs-140,bvs-129).",
    )
    parser.add_argument(
        "--exclude-birds",
        default=None,
        help="Comma-separated bird IDs to exclude.",
    )
    parser.add_argument(
        "--datasets",
        help=f"Comma-separated dataset IDs (default: {','.join(DATASET_IDS)}).",
    )
    parser.add_argument(
        "--rhythm-types",
        help=f"Comma-separated rhythm types (default: {','.join(RHYTHM_TYPE_IDS)}).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ASTRO_FIGURES_DIR),
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max count of birds to render per dataset.",
    )
    parser.add_argument(
        "--bird-id-col",
        default=None,
        help="Units column identifying the individual to group by (default: "
        "the dataset's own, see constants.DATASETS; 'bird_id_pred' for rt).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    include_birds = _split_csv(args.include_birds)
    exclude_birds = _split_csv(args.exclude_birds)
    datasets = _split_csv(args.datasets) or DATASET_IDS
    rhythm_types = _split_csv(args.rhythm_types) or RHYTHM_TYPE_IDS
    output_dir = Path(args.output_dir)

    shared_kws = dict(
        rhythm_types=rhythm_types,
        include_birds=include_birds,
        exclude_birds=exclude_birds,
        dataset_ids=datasets,
        output_dir=output_dir,
        refresh=args.refresh,
        limit=args.limit,
        bird_id_col=args.bird_id_col,
    )

    if args.plot in ("all", "timeline"):
        render_timelines(**shared_kws)
    if args.plot in ("all", "phase"):
        render_scatterplot("phase", **shared_kws)
    if args.plot in ("all", "patdur"):
        render_scatterplot("patdur", **shared_kws)
    if args.plot in ("all", "density"):
        render_densities(**shared_kws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
