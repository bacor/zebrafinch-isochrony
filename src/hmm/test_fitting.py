import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from hmm.fitting import (
    _fit_restarts,
    _sweep_k,
    canonical_order,
    decode_alphabetical,
    fit,
    fit_near_k,
    select_k_bic,
)
from hmm.utils import concat_sequences


def _hand_built_model(startprob, transmat, means, covars, n_features=1):
    model = GaussianHMM(n_components=len(startprob), covariance_type="diag")
    model.n_features = n_features
    model.startprob_ = np.asarray(startprob)
    model.transmat_ = np.asarray(transmat)
    model.means_ = np.asarray(means)
    model.covars_ = np.asarray(covars)
    return model


def _alternating_sequences(
    n_chunks=8, len_per_state=4, mean_a=10.0, mean_b=50.0, std=0.5, seed=0
):
    """n_chunks sequences alternating a short/long state, well separated."""
    rng = np.random.default_rng(seed)
    sequences = []
    for _ in range(n_chunks):
        a = rng.normal(mean_a, std, len_per_state)
        b = rng.normal(mean_b, std, len_per_state)
        seq = np.empty(2 * len_per_state)
        seq[0::2] = a
        seq[1::2] = b
        sequences.append(seq)
    return sequences


# --- Relabeling ---


def test_canonical_order_follows_dominant_transition_path():
    # 2 -> 1 -> 0 is the dominant path; state 3 is never reached.
    model = _hand_built_model(
        startprob=[0.1, 0.1, 0.7, 0.1],
        transmat=[
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        means=[[0], [0], [0], [0]],
        covars=[[1], [1], [1], [1]],
    )
    order = canonical_order(model)
    # Starts at state 2 (highest startprob), follows 2 -> 1 -> 0; from 0 the
    # only unplaced state is 3, which the walk takes despite its zero probability.
    assert order == [2, 1, 0, 3]


def test_decode_alphabetical_maps_and_splits_by_canonical_order():
    model = _hand_built_model(
        startprob=[0.0, 1.0],
        transmat=[[0.0, 1.0], [1.0, 0.0]],
        means=[[0], [0]],
        covars=[[1], [1]],
    )
    # Canonical order is [1, 0] (state 1 starts), so 1 -> 'a', 0 -> 'b'.
    states = np.array([1, 0, 1, 0, 1, 0])
    lengths = [2, 4]
    labels = decode_alphabetical(model, states, lengths)
    assert [seq.tolist() for seq in labels] == [["a", "b"], ["a", "b", "a", "b"]]


# --- Fitting primitives ---


def test_fit_restarts_recovers_well_separated_states():
    sequences = _alternating_sequences()
    X, lengths = concat_sequences(sequences)
    rng = np.random.default_rng(0)
    model, ll = _fit_restarts(
        X,
        lengths,
        k=2,
        covariance_type="diag",
        min_covar=1e-3,
        transmat_prior=1.0,
        n_restarts=5,
        n_refine=2,
        n_screen_iter=10,
        n_iter=30,
        rng=rng,
    )
    assert model is not None
    assert np.isfinite(ll)
    means = sorted(model.means_.ravel())
    assert means[0] == pytest.approx(10.0, abs=1.0)
    assert means[1] == pytest.approx(50.0, abs=1.0)


def test_sweep_k_skips_none_and_picks_lowest_bic():
    sequences = _alternating_sequences(n_chunks=1)
    X, lengths = concat_sequences(sequences)

    good_model = _hand_built_model(
        startprob=[0.5, 0.5],
        transmat=[[0.5, 0.5], [0.5, 0.5]],
        means=[[10], [50]],
        covars=[[1], [1]],
    )
    bad_model = _hand_built_model(
        startprob=[1 / 3] * 3,
        transmat=[[1 / 3] * 3] * 3,
        means=[[10], [50], [30]],
        covars=[[1], [1], [1]],
    )
    models = {2: good_model, 3: bad_model, 4: None}

    best_model, bics = _sweep_k(X, lengths, [2, 3, 4], lambda k: models[k])
    assert best_model is good_model
    assert set(bics) == {2, 3}  # k=4 (None) skipped entirely
    assert bics[2] < bics[3]


def test_sweep_k_returns_none_and_empty_dict_when_every_k_fails():
    sequences = _alternating_sequences(n_chunks=1)
    X, lengths = concat_sequences(sequences)
    best_model, bics = _sweep_k(X, lengths, [1, 2, 3], lambda k: None)
    assert best_model is None
    assert bics == {}


# --- K selection ---


def test_select_k_bic_recovers_true_k():
    sequences = _alternating_sequences(n_chunks=10)
    X, lengths = concat_sequences(sequences)
    k_hat, bics = select_k_bic(X, lengths, k_min=1, k_max=4, n_iter=30, random_state=0)
    assert k_hat == 2


# --- Final fit ---


def test_fit_near_k_raises_when_no_k_gets_a_fit():
    # Without restarts no k gets a model, as when every restart degenerates.
    sequences = _alternating_sequences(n_chunks=3)
    X, lengths = concat_sequences(sequences)
    with pytest.raises(ValueError, match="No k in"):
        fit_near_k(X, lengths, k_hat=2, window=0, n_restarts=0, n_refine=1)


def test_fit_near_k_recovers_well_separated_states():
    sequences = _alternating_sequences(n_chunks=10)
    X, lengths = concat_sequences(sequences)
    model, bics = fit_near_k(
        X,
        lengths,
        k_hat=2,
        window=1,
        n_restarts=5,
        n_refine=2,
        n_screen_iter=10,
        n_iter=30,
    )
    assert model.n_components == 2
    assert set(bics) <= {1, 2, 3}  # k_hat +- window
    means = sorted(model.means_.ravel())
    assert means[0] == pytest.approx(10.0, abs=1.0)
    assert means[1] == pytest.approx(50.0, abs=1.0)


# --- Pipeline ---


def test_fit_recovers_two_states_labelled_in_song_order():
    sequences = _alternating_sequences(n_chunks=10)
    kwargs = dict(
        k_min=1,
        k_max=3,
        select_n_iter=30,
        n_restarts=3,
        n_refine=1,
        n_screen_iter=5,
        n_iter=15,
    )
    result = fit(sequences, **kwargs)

    assert result["k_initial"] == 2
    assert result["n_states"] == result["model"].n_components == 2
    assert len(result["states"]) == sum(len(seq) for seq in sequences)
    # Every chunk starts with the short state, so canonical order calls it 'a'.
    expected = ["a", "b"] * 4
    assert [seq.tolist() for seq in result["labels"]] == [expected] * len(sequences)


def test_fit_is_reproducible_for_a_given_seed():
    sequences = _alternating_sequences(n_chunks=4)
    kwargs = dict(k_max=3, n_restarts=3, n_refine=1, n_iter=15, seed=1)
    first = fit(sequences, **kwargs)
    second = fit(sequences, **kwargs)
    np.testing.assert_array_equal(first["states"], second["states"])
    np.testing.assert_array_equal(first["model"].means_, second["model"].means_)
