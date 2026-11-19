# viz

Standalone plotting toolkit for rhythm data (durations, labels, chunks).
No study-specific code lives here: no dataset IDs, no fixed directory layout,
no fixed dataframe schema. For this study's batch figure export (walks
`data/<dataset_id>/...` and writes PNGs for the website), see `src/pipeline/render_figures.py`
instead — that's the only consumer of `viz` that's specific to this study.

## Naming convention

Tracks whether a function owns its figure, not the verb:

| prefix | takes `ax=`? | owns its `Figure`? | returns |
|---|---|---|---|
| `plot_*` | yes | no (draws into given/current axes) | `ax` or `(fig, ax)` |
| `show_*` | no | yes, always builds one | `(fig, ax)` / `(fig, axs)` |

`draw_*`/`render_*` names belong to `src.pipeline` (batch export), not here.

## `data=` + role params

Functions that need more than one aligned column (e.g. `label` and `duration`)
accept `data=None` plus per-role keyword params. Each role param is either a
column name (resolved against `data`) or an array-like passed directly —
so callers aren't locked into a fixed dataframe schema:

```python
show_density(bird_df)                                    # bird_df has "label"/"duration" columns
show_density(bird_df, label="condition", duration="dur")  # different column names
show_density(label=my_labels, duration=my_durations)      # no dataframe at all
```

The resolution itself is `utils.resolve(data, value)`.

## Modules

- **`dyads_plot.py`** — `Cluster` (one label's point cloud: scatter, Gaussian
  contour, mean, PC lines) and `DyadsPlotter` (multi-cluster scatter +
  trajectories + transition network + phase/patdur axes + legend), built from
  bare `points`/`labels`/`chunks` arrays, not a dataframe. Also
  `plot_relative_criterion` (wedge) and `plot_absolute_criterion` (band):
  the isochrony criteria shaded on a phase plot, for given `rho`/`tau`.
- **`density.py`** — `show_density` (one KDE+rug panel per label, stacked),
  `plot_bilateral_densities` (overlap-aware: flips overlapping label KDEs to
  opposite sides of a shared baseline via a Bhattacharyya-coefficient graph
  2-coloring), `plot_density` (single KDE+rug panel primitive).
- **`timeline.py`** — `show_timeline` (per-chunk bars, one row per chunk),
  `plot_compact_timeline` (axes-level variant with labels above the top row
  instead of inside each bar), `plot_slice` (single labeled bar primitive),
  `setup_label_colors`.
- **`heatmap.py`** — `per_column_heatmap` (each column gets its own colormap
  and value range — unlike `sns.heatmap`, colors aren't comparable across
  columns).
- **`stacked_bar.py`** — `stacked_bars` (named horizontal bars stacked
  top-to-bottom, each shaded light-to-dark along its own colormap to show
  nested — not summed — fractions, e.g. cumulative ">= k" counts), plus the
  single-bar primitive `nested_bar`. Per-bar colors are resolved the same
  way as `per_column_heatmap`'s columns.
- **`palette.py`** — colormap-per-named-item resolution shared by
  `heatmap.py` and `stacked_bar.py`: `resolve_item_cmaps` (per-item
  overrides falling back to one color auto-sampled from a shared palette),
  `resolve_cmap` (colormap, colormap name, or plain color → `Colormap`),
  plus `cmap_slice`/`white_to_color_cmap`/`sample_column_cmaps`/`diverging_from_white`.
- **`rows.py`** — figures with one row of panels per condition, grouped
  into titled sections: `flatten_sections` (sections → row specs),
  `row_height_ratios`, `annotate_row_groups` (section titles in the left
  margin, dividers between sections), `clean_density_yaxis`.
- **`utils.py`** — shared helpers: `get_ax`/`get_fig_ax` (reuse-or-create
  axes/figure), `resolve` (the `data=` role-param resolution), `get_label_colors`
  (palette tiering: default ≤10 labels, `"tab20"` ≤20, `"husl"` beyond),
  `set_mpl_style`, `WithDefaults` (mixin for `*_DEFAULTS` dict merging, used
  by `Cluster`/`DyadsPlotter`), plus small figure-annotation helpers
  (`subplot_title`, `ylabel_annot`, `savefig`, `best_text_color`).
