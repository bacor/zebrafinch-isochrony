"""Fit a Gaussian HMM to one bird's duration sequences and label its states.

Used for label prediction: states are the predicted labels, emissions the
unit durations (see STAR Methods).
"""

from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from hmmlearn.hmm import GaussianHMM

from .utils import concat_sequences, split_sequences

## Relabeling


def canonical_order(model: GaussianHMM) -> List[int]:
    """Order HMM states by greedily following the most likely transition path.

    Starting from the state most likely to begin a sequence, repeatedly
    jump to the most probable next state (skipping states already visited)
    until every state has been placed, so that labels a, b, c, ... follow
    the dominant order of states within a song.
    """
    n = model.n_components

    # Start at the state the model considers most likely to begin a sequence.
    order = [model.startprob_.argmax()]
    seen = set(order)

    while len(order) < n:
        current = order[-1]

        # Candidate next states, ranked from most to least probable.
        next_states = np.argsort(model.transmat_[current])[::-1]

        # Take the highest-probability state we haven't placed yet (possibly
        # one with zero probability, so every state ends up in the order).
        nxt = next(s for s in next_states if s not in seen)
        order.append(nxt)
        seen.add(nxt)

    return order


def decode_alphabetical(
    model: GaussianHMM, states: np.ndarray | Sequence[int], lengths: Sequence[int]
) -> List[np.ndarray]:
    """Relabel HMM states as letters (a, b, c, ...), split into one array per chunk.

    Args:
        model: Fitted model; its `canonical_order` sets the letters.
        states: Flat integer state codes, e.g. from `model.predict(X, lengths)`.
        lengths: Number of units per chunk, as used to fit or predict.

    Returns:
        One array of letter labels per chunk, in the order of `lengths`.
    """
    order = canonical_order(model)
    mapping = {state: chr(ord("a") + i) for i, state in enumerate(order)}

    labels = np.array([mapping[s] for s in states])
    return split_sequences(labels, lengths)


## Fitting primitives


def _fit_one(
    X: np.ndarray,
    lengths: Sequence[int],
    k: int,
    *,
    covariance_type: str,
    min_covar: float,
    n_iter: int,
    random_state: int,
    transmat_prior: float = 1.0,
) -> Tuple[GaussianHMM | None, float]:
    """Fit one GaussianHMM; (model, log-likelihood), or (None, -inf) if degenerate.

    `transmat_prior` < 1 biases the fit toward sparse, near-deterministic
    transition rows (a Dirichlet pseudocount below 1 favors peaked
    distributions). Too small a value can leave a state without transitions,
    which hmmlearn reports as a `ValueError`.
    """
    model = GaussianHMM(
        n_components=k,
        covariance_type=covariance_type,
        n_iter=n_iter,
        min_covar=min_covar,
        transmat_prior=transmat_prior,
        random_state=random_state,
    )
    try:
        model.fit(X, lengths)
        ll = model.score(X, lengths)
    except ValueError:
        return None, -np.inf
    return model, ll


def _fit_restarts(
    X: np.ndarray,
    lengths: Sequence[int],
    k: int,
    *,
    covariance_type: str,
    min_covar: float,
    transmat_prior: float,
    n_restarts: int,
    n_refine: int,
    n_screen_iter: int,
    n_iter: int,
    rng: np.random.Generator,
) -> Tuple[GaussianHMM | None, float]:
    """Fit k states with restarts, to avoid bad local optima.

    Runs `n_restarts` short fits (`n_screen_iter` iterations) and fits the
    `n_refine` most likely ones to convergence (`n_iter`). Returns
    (None, -inf) if every restart degenerates. `rng` is advanced in place
    (one draw per restart).
    """
    screened = []
    for _ in range(n_restarts):
        seed = int(rng.integers(0, 2**31 - 1))
        model, ll = _fit_one(
            X,
            lengths,
            k,
            covariance_type=covariance_type,
            min_covar=min_covar,
            transmat_prior=transmat_prior,
            n_iter=n_screen_iter,
            random_state=seed,
        )
        if model is not None:
            screened.append((ll, seed))
    if not screened:
        return None, -np.inf  # every restart degenerated at this k

    # Only the most promising seeds get run to full convergence.
    screened.sort(key=lambda x: x[0], reverse=True)
    best_model, best_ll = None, -np.inf
    for _, seed in screened[:n_refine]:
        model, ll = _fit_one(
            X,
            lengths,
            k,
            covariance_type=covariance_type,
            min_covar=min_covar,
            transmat_prior=transmat_prior,
            n_iter=n_iter,
            random_state=seed,
        )
        if model is not None and ll > best_ll:
            best_model, best_ll = model, ll
    return best_model, best_ll


def _sweep_k(
    X: np.ndarray,
    lengths: Sequence[int],
    ks: Iterable[int],
    fit_fn: Callable[[int], "GaussianHMM | None"],
) -> Tuple[GaussianHMM | None, Dict[int, float]]:
    """Fit `fit_fn(k)` for each k and return the best model by BIC and the BIC per k.

    A k for which `fit_fn` returns None (e.g. every restart degenerated) is
    skipped.
    """
    bics: Dict[int, float] = {}
    best_model, best_bic = None, np.inf
    for k in ks:
        model = fit_fn(k)
        if model is None:
            continue
        k_bic = model.bic(X, lengths)
        bics[k] = k_bic
        if k_bic < best_bic:
            best_model, best_bic = model, k_bic
    return best_model, bics


## K selection


def select_k_bic(
    X: np.ndarray,
    lengths: Sequence[int],
    k_min: int = 1,
    k_max: int = 12,
    *,
    covariance_type: str = "diag",
    n_iter: int = 100,
    min_covar: float = 1e-3,
    random_state: int = 0,
) -> Tuple[int, Dict[int, float]]:
    """Number of states K with the lowest BIC, fitting one GaussianHMM per K.

    Returns that K and the BIC of every K with a valid fit. K values that
    give a degenerate fit (a state without observed transitions, common
    when K is too large for the data) are skipped.
    """

    def fit_fn(k: int) -> GaussianHMM | None:
        model, _ = _fit_one(
            X,
            lengths,
            k,
            covariance_type=covariance_type,
            min_covar=min_covar,
            n_iter=n_iter,
            random_state=random_state,
        )
        return model

    _, bics = _sweep_k(X, lengths, range(k_min, k_max + 1), fit_fn)
    if not bics:
        raise ValueError(
            f"No k in [{k_min}, {k_max}] produced a valid fit "
            f"(all had unused states); try lowering k_max."
        )
    return min(bics, key=lambda k: bics[k]), bics


def fit_near_k(
    X: np.ndarray,
    lengths: Sequence[int],
    k_hat: int,
    window: int = 2,
    *,
    covariance_type: str = "diag",
    min_covar: float = 1e-3,
    transmat_prior: float = 1.0,
    n_restarts: int = 20,
    n_refine: int = 3,
    n_screen_iter: int = 15,
    n_iter: int = 100,
    random_state: int = 0,
) -> Tuple[GaussianHMM, Dict[int, float]]:
    """Best GaussianHMM by BIC for K in [k_hat - window, k_hat + window], with restarts.

    A single-fit BIC sweep (`select_k_bic`) can be off by one or two states
    because of a bad local optimum, so neighbouring K are refitted with
    restarts (see `_fit_restarts`; K is at least 1). See `_fit_one` for
    `transmat_prior`.

    Returns the best model and the BIC of every K with a valid fit.
    """
    rng = np.random.default_rng(random_state)
    k_min = max(1, k_hat - window)
    k_max = k_hat + window

    def fit_fn(k: int) -> GaussianHMM | None:
        model, _ = _fit_restarts(
            X,
            lengths,
            k,
            covariance_type=covariance_type,
            min_covar=min_covar,
            transmat_prior=transmat_prior,
            n_restarts=n_restarts,
            n_refine=n_refine,
            n_screen_iter=n_screen_iter,
            n_iter=n_iter,
            rng=rng,
        )
        return model

    best_model, bics = _sweep_k(X, lengths, range(k_min, k_max + 1), fit_fn)
    if best_model is None:
        raise ValueError(
            f"No k in [{k_min}, {k_max}] produced a valid fit "
            f"(all had unused states); try a smaller window or k_hat."
        )
    return best_model, bics


## Pipeline


def fit(
    sequences: Sequence[np.ndarray],
    *,
    covariance_type: str = "diag",
    min_covar: float = 1e-3,
    transmat_prior: float = 1.0,
    # K estimation (select_k_bic)
    k_min: int = 1,
    k_max: int = 12,
    select_n_iter: int = 100,
    # Final fit around k_hat (fit_near_k)
    window: int = 2,
    n_restarts: int = 50,
    n_refine: int = 5,
    n_screen_iter: int = 15,
    n_iter: int = 100,
    seed: int = 0,
) -> Dict[str, Any]:
    """Fit an HMM to one bird's duration sequences (one per chunk) and predict labels.

    1. `select_k_bic` picks a cheap initial K, `k_initial` (one fit per K).
    2. `fit_near_k` refits with restarts for a window of K around `k_initial`.
    3. States are relabeled a, b, c, ... in `canonical_order`.

    Hyperparameters are passed on to `select_k_bic` and `fit_near_k`.

    Returns a dict with:
        model: the fitted GaussianHMM.
        n_states: number of states in `model` (may differ from `k_initial`).
        k_initial: the K chosen by `select_k_bic`.
        bics: BIC per K considered by `fit_near_k`.
        states: flat integer state codes (`model.predict` output).
        labels: letter labels, one array per chunk in the order of `sequences`.
        params: the hyperparameters of this call.
    """
    params = dict(
        covariance_type=covariance_type,
        min_covar=min_covar,
        transmat_prior=transmat_prior,
        k_min=k_min,
        k_max=k_max,
        select_n_iter=select_n_iter,
        window=window,
        n_restarts=n_restarts,
        n_refine=n_refine,
        n_screen_iter=n_screen_iter,
        n_iter=n_iter,
        seed=seed,
    )
    X, lengths = concat_sequences(sequences)
    rng = np.random.default_rng(seed)

    k_initial, _ = select_k_bic(
        X,
        lengths,
        k_min=k_min,
        k_max=k_max,
        covariance_type=covariance_type,
        n_iter=select_n_iter,
        min_covar=min_covar,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )

    model, bics = fit_near_k(
        X,
        lengths,
        k_initial,
        window,
        covariance_type=covariance_type,
        min_covar=min_covar,
        transmat_prior=transmat_prior,
        n_restarts=n_restarts,
        n_refine=n_refine,
        n_screen_iter=n_screen_iter,
        n_iter=n_iter,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )

    states = model.predict(X, lengths)
    labels = decode_alphabetical(model, states, lengths)

    return {
        "model": model,
        "n_states": model.n_components,
        "k_initial": k_initial,
        "bics": bics,
        "states": states,
        "labels": labels,
        "params": params,
    }
