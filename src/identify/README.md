# identify

Bird identification: find the boundaries between birds in a long sequence of
chunks, when it is not known which chunk came from which bird (as in RT).
Nothing here is specific to this study's datasets; the study-specific wrapper
is `src/pipeline/predict_bird_ids.py`.

```python
from identify import detect_changepoints, sequences_from_units

sequences = sequences_from_units(units)  # one duration array per chunk
result = detect_changepoints(sequences, n_changepoints=92, half_window=75, min_gap=75)
result["labels"]  # 0-based bird index per chunk
```

## Method

1. **Distances.** Chunks are compared pairwise with dynamic time warping
   (DTW): the standard normalized distance of the
   [dtw-python](https://dynamictimewarping.github.io/) package, with its
   default settings. Only chunks at most `2 * half_window` apart are
   compared (`banded_dtw`).
2. **Score.** At every candidate boundary, the score is the mean distance
   between the `half_window` chunks on either side, minus the mean distance
   within each side (`cross_within_score`). Chunks of the same bird are more
   alike than chunks of different birds, so the score peaks at a boundary.
3. **Boundaries.** The `n_changepoints` most prominent peaks of the score,
   at least `min_gap` chunks apart (`scipy.signal.find_peaks`), are the
   boundaries.

`viz/changepoints.py` plots the result: the score signal with the detected
boundaries, colored by peak prominence.

`half_window` and `min_gap` have no defaults: they depend on how many chunks
a bird typically has. On the datasets where bird identity is known, the
method assigns 90% (BWPS) to over 99% (NS) of chunks to the right bird.
