# Consistent but not isochronous: zebra finch rhythm in four datasets

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24370596.svg)](https://doi.org/10.5281/zenodo.24370596)

Code for Cornelissen, Maiolini, Spierings & Honing, *Consistent but not
isochronous: zebra finch rhythm in four datasets* (in preparation).

We reanalyze zebra finch song from four published datasets:

| id | source | data | birds | labels |
|---|---|---|---|---|
| `bvs` | Burchardt, Varkevisser & Spierings (2025) | [Zenodo](https://doi.org/10.5281/zenodo.14779410) | 50 | annotated |
| `ns` | Norton & Scharff (2016) | [Zenodo](https://doi.org/10.5281/zenodo.17842767) | 15 | annotated |
| `rt` | Roeske et al. (2020) | [Mendeley Data](https://doi.org/10.17632/zhb728dc4z.1) | 93 | predicted (HMM) |
| `bwps`, `bwps-clean` | Bertolo et al. (2026) | [OSF](https://doi.org/10.17605/OSF.IO/FN2MX) | 61 | predicted (HMM) |

All four datasets are licensed under CC BY 4.0; full references are below.

## Layout

- `src/`: all analysis code, with unit tests next to the modules
  (`test_*.py`).
  - `rhythm_types.py`: derives the eight rhythm types from syllable and
    element annotations.
  - `metrics/`: consistency, differentiation and isochrony metrics, and the
    label-shuffling permutation tests (see `src/metrics/README.md`).
  - `hmm/`: predicting syllable labels from durations.
  - `identify/`: predicting bird identity (RT dataset).
  - `pipeline/`: scripts that run the above on all datasets.
  - `viz/`: plotting (see `src/viz/README.md`).
- `notebooks/`: preprocessing (`preprocess-*`), figures (`fig*`) and
  statistics (`statistics.ipynb`).
- `data/`: derived data; see `data/README.md`.
- `figures/`: the figures in the paper.

## Data

Not all data is part of this repository. The source annotations belong to
the original authors, and most derived data is not included either. This
repository contains the condition-level results (`data/metrics.csv`,
`data/permutation-tests.csv`), bird-level metrics for all datasets, and the
unit-level rhythm data of the publicly available NS dataset. The NS
annotations are also available from Norton & Scharff (2016). The other
derived data is available from us on request.

As a result, most notebooks will not run from this repository alone: only
`statistics.ipynb` runs fully. All notebooks are committed with their
outputs, so the figures and the code that makes them can still be
inspected.

## Running the analyses

Install the dependencies with [Poetry](https://python-poetry.org/)
(Python 3.11):

```bash
poetry install
poetry run pytest src
```

Preprocessing reads the source data from the directories named in a `.env`
file in the repository root:

```bash
NS_DIR=/path/to/norton-scharff
BVS_DIR=/path/to/burchardt-varkevisser-spierings
RT_DIR=/path/to/roeske
BWPS_DIR=/path/to/bertolo
SHARED_DATA_DIR=/path/to/storage   # fitted HMMs are written here
```

The full pipeline runs in this order:

1. `notebooks/preprocess-{ns,bvs,rt,bwps}.ipynb`: write the rhythm files to
   `data/<dataset>/rhythms/`.
2. `poetry run python -m src.pipeline.fit_hmms`: fit an HMM per bird to
   predict labels, then run `notebooks/preprocess-augment.ipynb` and the label
   cells of the preprocessing notebooks to add the predicted labels.
   `notebooks/preprocess-bwps-clean.ipynb` derives `bwps-clean` from `bwps`.
3. `poetry run python -m src.pipeline.collect_metrics`: writes
   `data/<dataset>/metrics/` and `data/metrics.csv`.
4. `poetry run python -m src.pipeline.collect_permutation_tests`: writes
   `data/permutation-tests.csv` (2000 shuffles; takes a while).
5. The `fig*` notebooks and `statistics.ipynb`.

## About the code

The analyses were first developed and implemented by the authors, with 
assistance by Claude Code (Anthropic), under close supervision and extensive 
unit testing. The code in `src/` was afterwards cleaned up and consolidated 
by Claude Code, touching most of the code to make the coding style, naming,
docstrings and documentation consistent.

## Citation

Please cite the paper (reference to follow) and this archived release:

> Cornelissen, B., Maiolini, M., Spierings, M., & Honing, H. (2026).
> Consistent but not isochronous: zebra finch rhythm in four datasets (code
> and data). Zenodo. https://doi.org/10.5281/zenodo.24370596

## License

The code is licensed under the MIT License (`LICENSE`). The derived data in
`data/` is licensed under CC BY 4.0 (`data/LICENSE`), like the source
datasets it is derived from.

## References

Datasets:

- Burchardt, L. S., Varkevisser, J. M., & Spierings, M. J. (2025). Dataset
  for "Zebra finch tutees not only share the melody but also the rhythm of
  their tutor's song". Zenodo. https://doi.org/10.5281/zenodo.14779410
- Norton, P., & Scharff, C. (2025). Dataset for "Bird Song Metronomics".
  Zenodo. https://doi.org/10.5281/zenodo.17842767
- Roeske, T. C., & Tchernichovski, O. (2020). Birdsong rhythm: Onsets/intervals
  of a zebra finch song corpus and 3 thrush nightingale song corpora
  (Version 1). Mendeley Data. https://doi.org/10.17632/zhb728dc4z.1
- Bertolo, M., Weiss, M. W., Peretz, I., & Sakata, J. T. (2025). Repository of
  primary data and analysis code for "Humans, but not zebra finches, produce
  spontaneously isochronous song through the lifespan". OSF.
  https://doi.org/10.17605/OSF.IO/FN2MX

Papers:

- Bertolo, M., Weiss, M. W., Peretz, I., & Sakata, J. T. (2026). Humans, but
  not zebra finches, produce spontaneously isochronous song through the
  lifespan. *iScience*, 29, 116751.
  https://doi.org/10.1016/j.isci.2026.116751
- Burchardt, L. S., Varkevisser, J. M., & Spierings, M. J. (2025). Zebra
  finch tutees not only share the melody but also the rhythm of their tutor's
  song. *Scientific Reports*, 15, 35573.
  https://doi.org/10.1038/s41598-025-22811-8
- Norton, P., & Scharff, C. (2016). "Bird Song Metronomics": Isochronous
  organization of zebra finch song rhythm. *Frontiers in Neuroscience*, 10,
  309. https://doi.org/10.3389/fnins.2016.00309
- Roeske, T. C., Tchernichovski, O., Poeppel, D., & Jacoby, N. (2020).
  Categorical rhythms are shared between songbirds and humans. *Current
  Biology*, 30, 3544–3555. https://doi.org/10.1016/j.cub.2020.06.072