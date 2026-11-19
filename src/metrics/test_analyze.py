import pandas as pd

from metrics import analyze


def test_analyze_uses_label_col_and_skips_unlabeled_units():
    # Chunk 0 is x-y-x; chunk 1 is x-?-y. The unlabeled unit in chunk 1 breaks
    # both of its dyads, so x and y there must not form an x-y dyad.
    units = pd.DataFrame(
        {
            "bird_id": "b1",
            "chunk_idx": [0, 0, 0, 1, 1, 1],
            "start": [0.0, 100.0, 200.0, 0.0, 100.0, 200.0],
            "duration": [100.0] * 6,
            "label": ["a", "b", "a", "a", "b", "a"],
            "label_pred": ["x", "y", "x", "x", None, "y"],
        }
    )
    label, cluster, bird = analyze(units, ["bird_id"], label_col="label_pred")
    assert set(label.index.get_level_values("label")) == {"x", "y"}
    assert cluster["count"].to_dict() == {("b1", "x-y"): 1, ("b1", "y-x"): 1}
    assert (bird.loc["b1", "n_units"], bird.loc["b1", "n_dyads"]) == (5, 2)
