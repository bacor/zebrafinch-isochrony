import numpy as np
import pytest

from hmm.utils import concat_sequences, evaluate, split_sequences


def test_concat_and_split_sequences_round_trip():
    sequences = [np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0, 5.0, 6.0])]
    X, lengths = concat_sequences(sequences)

    assert X.shape == (6, 1)
    assert lengths == [2, 1, 3]
    split = split_sequences(X.ravel(), lengths)
    assert [s.tolist() for s in split] == [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]


def test_evaluate_perfect_agreement_after_relabeling():
    # Predicted labels use a different vocabulary but correspond 1:1 to
    # true labels; alignment should recover a perfect match.
    labels_true = np.array(["a", "b", "a", "b", "c"])
    labels_pred = np.array(["x", "y", "x", "y", "z"])
    scores = evaluate(labels_true, labels_pred)

    assert scores["label_mapping"] == {"x": "a", "y": "b", "z": "c"}
    assert scores["accuracy"] == pytest.approx(1.0)
    assert scores["homogeneity"] == pytest.approx(1.0)
    assert scores["completeness"] == pytest.approx(1.0)
    assert scores["v_measure"] == pytest.approx(1.0)
    assert scores["adjusted_rand_score"] == pytest.approx(1.0)
    assert scores["precision"] == pytest.approx(1.0)
    assert scores["recall"] == pytest.approx(1.0)
    assert scores["f1-score"] == pytest.approx(1.0)


def test_evaluate_aligned_scores_for_an_imperfect_prediction():
    # Mapping x -> a, y -> b matches 2 + 1 = 3 labels (y -> a, x -> b only 1),
    # so aligned_pred = [a, a, b, b] and accuracy = 3 / 4.
    # Class a: precision 2/2, recall 2/3; class b: precision 1/2, recall 1/1.
    # Weighted by support (3, 1): precision (3 * 1 + 1 * 0.5) / 4 = 0.875,
    # recall (3 * 2/3 + 1 * 1) / 4 = 0.75.
    labels_true = np.array(["a", "a", "a", "b"])
    labels_pred = np.array(["x", "x", "y", "y"])
    scores = evaluate(labels_true, labels_pred)

    assert scores["label_mapping"] == {"x": "a", "y": "b"}
    assert scores["aligned_pred"].tolist() == ["a", "a", "b", "b"]
    assert scores["accuracy"] == pytest.approx(0.75)
    assert scores["precision"] == pytest.approx(0.875)
    assert scores["recall"] == pytest.approx(0.75)
    assert scores["support"] == 4


def test_evaluate_marks_extra_predicted_labels_unmatched():
    # Three predicted labels for two true ones: y (2 of b) is matched to b,
    # z (1 of b) is left over. Accuracy = 4 / 5.
    labels_true = np.array(["a", "a", "b", "b", "b"])
    labels_pred = np.array(["x", "x", "y", "y", "z"])
    scores = evaluate(labels_true, labels_pred)

    assert scores["label_mapping"] == {"x": "a", "y": "b"}
    assert scores["aligned_pred"].tolist() == ["a", "a", "b", "b", "unmatched"]
    assert scores["accuracy"] == pytest.approx(0.8)


def test_evaluate_cluster_scores_use_raw_predicted_labels():
    # Each true label is split over two predicted labels. Raw labels:
    # every predicted label is pure, so homogeneity = 1; H(pred) = 2 bits and
    # H(pred | true) = 1 bit, so completeness = 1 - 1/2 = 0.5.
    # On aligned_pred = [a, a, u, u, b, b, u, u] (u = unmatched) the merged
    # unmatched label would mix a and b (homogeneity < 1) and completeness
    # would be 1 - 1/1.5 = 1/3.
    labels_true = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    labels_pred = np.array(["x", "x", "y", "y", "z", "z", "w", "w"])
    scores = evaluate(labels_true, labels_pred)

    assert scores["homogeneity"] == pytest.approx(1.0)
    assert scores["completeness"] == pytest.approx(0.5)
