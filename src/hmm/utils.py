"""Reshaping sequences for hmmlearn, evaluating predicted labels, and storing fits."""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    completeness_score,
    confusion_matrix,
    homogeneity_score,
    v_measure_score,
)


def concat_sequences(
    sequences: Sequence[np.ndarray],
) -> Tuple[np.ndarray, List[int]]:
    """Stack a list of per-chunk sequences into hmmlearn's (X, lengths) format."""
    arrs = []
    for seq in sequences:
        arr = np.asarray(seq, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arrs.append(arr)
    X = np.concatenate(arrs)
    lengths = [len(arr) for arr in arrs]
    return X, lengths


def split_sequences(X: np.ndarray, lengths: Sequence[int]) -> List[np.ndarray]:
    """Inverse of `concat_sequences`: split a stacked array into one array per chunk."""
    split_idx = np.cumsum(lengths)[:-1]
    return np.split(X, split_idx)


def evaluate(
    labels_true: np.ndarray | Sequence[str],
    labels_pred: np.ndarray | Sequence[str],
) -> Dict[str, Any]:
    """Compare predicted labels with annotated labels.

    Predicted labels are arbitrary (state "a" from the HMM has no inherent
    relation to annotated "a"), so this first finds the best one-to-one
    mapping from predicted to true labels via the Hungarian algorithm
    (maximizing the confusion matrix diagonal), then reports plain accuracy
    and weighted-average precision/recall/F1 under that alignment.

    Also reports homogeneity, completeness, V-measure and adjusted Rand
    index on the unaligned labels. These do not require a one-to-one
    mapping, which matters when the number of states differs from the
    number of annotated labels.

    Returns a dict with:
        accuracy: overall accuracy after alignment.
        precision/recall/f1-score/support: sklearn `classification_report`'s
            weighted-average row (weighted by true class support), after
            alignment.
        homogeneity: each predicted state contains only one true label.
        completeness: each true label stays within one predicted state.
        v_measure: harmonic mean of homogeneity and completeness.
        adjusted_rand_score: chance-corrected pairwise agreement.
        confusion_matrix: raw (unaligned) confusion matrix.
        confusion_matrix_labels: row/column label order for `confusion_matrix`.
        label_mapping: predicted label -> true label, from the alignment.
        aligned_pred: `labels_pred` remapped through `label_mapping`
            ("unmatched" for predicted labels with no true-label match,
            e.g. when there are more predicted than true classes).
    """
    true_classes = sorted(set(labels_true))
    pred_classes = sorted(set(labels_pred))
    all_classes = true_classes + [c for c in pred_classes if c not in true_classes]

    cm = confusion_matrix(labels_true, labels_pred, labels=all_classes)
    row_ind, col_ind = linear_sum_assignment(-cm[: len(true_classes), :])

    label_mapping = {all_classes[c]: true_classes[r] for r, c in zip(row_ind, col_ind)}
    aligned_pred = np.array([label_mapping.get(p, "unmatched") for p in labels_pred])

    report = classification_report(
        labels_true,
        aligned_pred,
        labels=true_classes,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels_true, aligned_pred),
        **report["weighted avg"],
        "homogeneity": homogeneity_score(labels_true, labels_pred),
        "completeness": completeness_score(labels_true, labels_pred),
        "v_measure": v_measure_score(labels_true, labels_pred),
        "adjusted_rand_score": adjusted_rand_score(labels_true, labels_pred),
        "confusion_matrix": cm,
        "confusion_matrix_labels": all_classes,
        "label_mapping": label_mapping,
        "aligned_pred": aligned_pred,
    }


def save_fit(result: Dict[str, Any], path: Path) -> None:
    """Save a dict returned by `fit` (including the GaussianHMM) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)


def load_fit(path: Path) -> Dict[str, Any]:
    """Load a dict saved by `save_fit`."""
    return joblib.load(Path(path))
