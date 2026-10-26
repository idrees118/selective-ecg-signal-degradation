import numpy as np
import pytest

from ptbxl_reliability.bootstrap import (
    patient_record_weights,
    weighted_grouped_aurc,
)
from ptbxl_reliability.selective_metrics import (
    ErrorStacker,
    aurc_augrc,
    bootstrap_two_sided_p,
    cluster_weight_matrix,
    fit_classwise_temperature,
    holm_adjust,
    percentile_selector,
    risk_at_coverages,
    temperature_scaled_confidence,
    threshold_margin,
    weighted_aurc_augrc,
)
from ptbxl_reliability.selective_prediction import (
    grouped_risk_coverage,
    midrank_ecdf,
)


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    losses = rng.integers(0, 6, size=n) / 5.0
    # Rounded scores create exact ties.
    reliability = np.round(-losses + rng.normal(0, 0.4, size=n), 1)
    return reliability, losses


def test_aurc_matches_frozen_implementation():
    reliability, losses = make_data()
    aurc, _ = aurc_augrc(reliability, losses)
    expected = grouped_risk_coverage(reliability, losses)["aurc"]
    assert aurc == pytest.approx(expected, abs=1e-14)


def test_weighted_aurc_matches_frozen_bootstrap():
    reliability, losses = make_data()
    patients = np.arange(len(losses)) // 2
    for replicate in range(5):
        weights = patient_record_weights(patients, seed=1, replicate=replicate)
        aurc, _ = weighted_aurc_augrc(reliability, losses, weights)
        expected = weighted_grouped_aurc(reliability, losses, weights)
        assert float(aurc) == pytest.approx(expected, abs=1e-13)


def test_matrix_weights_match_row_by_row():
    reliability, losses = make_data()
    weights = cluster_weight_matrix(np.arange(len(losses)), n_replicates=20, seed=3)
    aurc, augrc = weighted_aurc_augrc(reliability, losses, weights)
    for b in range(len(weights)):
        a, g = weighted_aurc_augrc(reliability, losses, weights[b])
        assert aurc[b] == pytest.approx(float(a), abs=1e-14)
        assert augrc[b] == pytest.approx(float(g), abs=1e-14)


def test_augrc_matches_explicit_definition_without_ties():
    rng = np.random.default_rng(4)
    losses = rng.random(50)
    reliability = rng.random(50)
    order = np.argsort(-reliability)
    n = len(losses)
    expected = np.sum(np.cumsum(losses[order]) / n) / n
    _, augrc = aurc_augrc(reliability, losses)
    assert augrc == pytest.approx(expected, abs=1e-14)


def test_oracle_beats_random_on_both_metrics():
    rng = np.random.default_rng(5)
    losses = rng.integers(0, 2, size=500).astype(float)
    oracle = aurc_augrc(-losses + 1e-6 * rng.random(500), losses)
    random = aurc_augrc(rng.random(500), losses)
    assert oracle[0] < random[0]
    assert oracle[1] < random[1]


def test_risk_at_coverage_is_reached():
    reliability, losses = make_data()
    result = risk_at_coverages(reliability, losses, (0.5, 0.8))
    for target, (coverage, risk) in result.items():
        assert coverage >= target
        assert 0.0 <= risk <= 1.0


def test_threshold_margin_zero_at_threshold():
    tau = np.array([0.2, 0.5, 0.7, 0.4, 0.9])
    p = np.vstack([tau, np.clip(tau + 0.05, 0, 1)])
    margin = threshold_margin(p, tau)
    assert margin[0] == pytest.approx(0.0, abs=1e-12)
    assert margin[1] > 0.0


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(6)
    true_logits = rng.normal(0, 1.5, size=(4000, 5))
    y = (rng.random((4000, 5)) < 1 / (1 + np.exp(-true_logits))).astype(int)
    overconfident = 1 / (1 + np.exp(-3.0 * true_logits))
    temperatures = fit_classwise_temperature(overconfident, y)
    np.testing.assert_allclose(temperatures, 3.0, rtol=0.15)
    scaled = temperature_scaled_confidence(overconfident, temperatures)
    assert scaled.shape == (4000,)


def test_percentile_selector_matches_frozen_rule():
    rng = np.random.default_rng(7)
    ref_q, ref_u = rng.random(100), rng.random(100)
    q_bad, u_bad = rng.random(30), rng.random(30)
    expected = -(midrank_ecdf(ref_q, q_bad) + midrank_ecdf(ref_u, u_bad)) / 2.0
    np.testing.assert_allclose(
        percentile_selector([q_bad, u_bad], [ref_q, ref_u]), expected,
    )


def test_stacker_ranks_low_error_features_first():
    rng = np.random.default_rng(8)
    confidence = rng.random(600)
    losses = (rng.random(600) > confidence).astype(float) * 0.2
    stacker = ErrorStacker(["confidence"]).fit({"confidence": confidence}, losses)
    selector = stacker.selector({"confidence": np.array([0.05, 0.95])})
    assert selector[1] > selector[0]
    assert set(stacker.coefficients()) == {"intercept", "confidence"}


def test_cluster_weights_share_multiplicity():
    clusters = np.array([1, 1, 2, 3, 3, 3])
    weights = cluster_weight_matrix(clusters, n_replicates=50, seed=9)
    assert np.all(weights[:, 0] == weights[:, 1])
    assert np.all(weights[:, 3] == weights[:, 5])
    # Each draw resamples 3 clusters.
    np.testing.assert_array_equal(
        weights[:, [0, 2, 3]].sum(axis=1), np.full(50, 3.0),
    )


def test_holm_adjust_known_values():
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03, 0.5]))
    np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.5])


def test_bootstrap_p_value_extremes():
    assert bootstrap_two_sided_p(np.full(1000, 0.1)) == pytest.approx(1 / 1000)
    deltas = np.concatenate([np.full(20, -0.1), np.full(980, 0.1)])
    assert bootstrap_two_sided_p(deltas) == pytest.approx(0.04)
    assert bootstrap_two_sided_p(np.linspace(-1, 1, 1001)) == pytest.approx(1.0)
