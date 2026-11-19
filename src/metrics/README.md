# metrics

Metrics of rhythmic data, computed from a units DataFrame (one row per
duration, with `chunk_idx`, `start`, `duration` and a label column):

```python
from metrics import analyze
label, cluster, bird = analyze(units, by=['bird_id'], label_col='label')
```

`analyze` drops unlabeled units, builds the dyads (`rhythm_types.build_dyads`, in
`src/`) and
returns one table per level:

| level | grouping | module |
|---|---|---|
| label | all of a bird's units with one label (e.g. `'a'`) | `label.py` |
| cluster | all of a bird's dyads with one label (e.g. `'a-b'`) | `cluster.py` |
| bird | one bird | `bird.py` |
| condition | all birds of one dataset and rhythm type | `condition.py` |

A *dyad* is a pair of successive durations (x, y) within a chunk. All
thresholds are in `thresholds.py`. The study-specific pipeline
(`src/pipeline/collect_metrics.py`) writes the label, cluster and bird tables
to `data/<dataset_id>/metrics/` and the condition table to
`data/metrics.csv`:

```bash
poetry run python -m src.pipeline.collect_metrics
```

Tests: `poetry run pytest src/metrics/`

## Label level — `analyze_labels(units, by)`

`count`, `dur_mean`, `dur_std`, `dur_min`, `dur_max`, and the quartiles
`dur_q1`, `dur_q2`, `dur_q3`.

## Cluster level — `analyze_clusters(dyads, by)`

| column | definition |
|---|---|
| `label_x`, `label_y` | the labels of the two units; a repetition has `label_x == label_y` |
| `count` | number of dyads |
| `centroid_x_mean`, `centroid_y_mean` | the centroid |
| `centroid_x_std`, `centroid_y_std` | SDs of the first and second durations. These cover only the units in this context (e.g. A when followed by B), so they can differ from the label level |
| `centroid_asymmetry` | \|x − y\| of the centroid (ms) |
| `centroid_rhythm_ratio` | r = x / (x + y) of the centroid |
| `centroid_non_isochrony` | \|r − 0.5\| of the centroid (proportional to nPVI) |

## Bird level — `analyze_birds(units, dyads, by)`

Descriptive: `n_labels`, `n_units`, `n_dyads`, `n_chunks`, units per
chunk (`chunk_len_mean/min/max`) and chunk duration from first onset to last
offset (`chunk_dur_mean/std/min/max`).

Isolation (`distance.isolation_stats`): `n_isolated_labels` counts the
labels whose nearest other label is further than `ISOLATION_D` (Cohen's d),
and `most_isolated_label_d` is the largest nearest-neighbour distance.
`n_isolated_clusters` and `most_isolated_cluster_d` are the same for
clusters, using the Mahalanobis distance between their dyads.

## Condition level — `condition.py`

Every reported metric is first reduced to one value per bird
(`bird_metrics(label, cluster, bird, by)`). A condition's value is the mean
over its birds (`condition_metrics`), where a boolean per-bird metric
becomes the percentage of birds. The notebooks, `data/metrics.csv` and every
permutation draw use these functions, so reported and tested values are the
same number.

| per bird | definition |
|---|---|
| `inconsistency` | median over labels of `dur_std` (ms) |
| `cluster_inconsistency` | median over clusters of the mean of `centroid_x_std` and `centroid_y_std` (ms) |
| `isolated_label` / `isolated_cluster` | at least one isolated label / cluster |
| `most_isochronous` / `median_isochrony` / `least_isochronous` | min / median / max `centroid_non_isochrony` over clusters holding ≥ `MIN_CLUSTER_SHARE` of the bird's dyads |
| `n_labels`, `units_per_label`, `dur_mean`, `n_isolated_labels`, `repetitions` | descriptive (medians over labels; `repetitions` is the % of dyads like A-A) |

`isochrony_metrics(dyads, by)` describes single dyads, independent of
labels (`dyad.dyad_scores`): `non_isochrony` (mean \|r − 0.5\|), `asymmetry`
(mean \|x − y\| in ms),
`perc_rel_isochronous` (% of dyads with \|r − 0.5\| < `RHO`), `perc_abs_isochronous`
(% with \|x − y\| < `TAU` ms), and `rho50`/`tau50`, the thresholds at which
half of the dyads are isochronous (`dyad.threshold50`).

## Distances — `distance.py`

Label and cluster isolation use the distance between two samples' centroids
standardized by their pooled covariance: Cohen's d in 1D, the Mahalanobis
distance in 2D. `isolation_stats(df, by, value_cols, threshold)` computes,
per `by`-group, each group's distance to its nearest neighbour, as one
L×L×k×k broadcast (`nearest_neighbour_pooled_distance`) since it runs once
per permutation draw. `pooled_distance(a, b)` is the readable single-pair
version, used in tests.

## Permutation tests — `permutation.py`

The null shuffles which label attaches to which duration within each
chunk (`shuffle_labels_within_chunk`), preserving each chunk's labels and
durations but destroying their pairing. `permutation_test(observed,
draw_null, n, seed, n_jobs)` adds `{col}_null_mean`, `_null_std`, `_z`,
`_p` (`P(null <= observed)`) and `_p_high` (`P(null >= observed)`), with
p ≥ 1/(n + 1).

The tested metrics and their direction are `condition.TESTS`.
`src/pipeline/collect_permutation_tests.py` reruns `analyze` and the
condition metrics on every draw and writes `data/permutation-tests.csv`:
one row per condition, label type and metric, with the observed value, null
mean and SD, z, the p-value of the metric's own direction, and `p_bonf`
(Bonferroni per metric, across conditions and label types).
`notebooks/statistics.ipynb` shows it per metric:

```bash
poetry run python -m src.pipeline.collect_permutation_tests --n 2000 --n-jobs 4
```

The `preprocess-*.ipynb` notebooks also store one fixed shuffle per
condition as a `'label_shuffle'` column (from `'label'` for bvs and ns, from
`'label_pred'` otherwise). The pipeline treats it like any other label
column; it only illustrates the null in figures.
