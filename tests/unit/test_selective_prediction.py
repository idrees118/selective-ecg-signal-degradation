import numpy as np
import pytest

from ptbxl_reliability.selective_prediction import (
    grouped_risk_coverage,
    mean_bernoulli_confidence,
    midrank_ecdf,
    sample_hamming_error,
)


def test_sample_hamming_error_manual():
    y = np.array(
        [
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )

    pred = np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    observed = sample_hamming_error(
        y,
        pred,
    )

    np.testing.assert_allclose(
        observed,
        [0.4, 0.0],
    )


def test_mean_bernoulli_confidence():
    p = np.array(
        [
            [
                0.9,
                0.1,
                0.8,
                0.2,
                0.5,
            ]
        ]
    )

    observed = (
        mean_bernoulli_confidence(
            p
        )[0]
    )

    expected = np.mean(
        [
            0.9,
            0.9,
            0.8,
            0.8,
            0.5,
        ]
    )

    assert observed == pytest.approx(
        expected
    )


def test_midrank_ecdf_known_values():
    reference = np.array(
        [0.0, 0.0, 1.0, 2.0]
    )

    observed = midrank_ecdf(
        reference,
        np.array(
            [0.0, 1.0, 2.0]
        ),
    )

    expected = np.array(
        [
            0.25,
            0.625,
            0.875,
        ]
    )

    np.testing.assert_allclose(
        observed,
        expected,
    )


def test_good_ranking_has_lower_aurc():
    losses = np.array(
        [0.0, 1.0]
    )

    good = grouped_risk_coverage(
        np.array(
            [1.0, 0.0]
        ),
        losses,
    )

    bad = grouped_risk_coverage(
        np.array(
            [0.0, 1.0]
        ),
        losses,
    )

    assert good["aurc"] == pytest.approx(
        0.25
    )

    assert bad["aurc"] == pytest.approx(
        0.75
    )


def test_all_tied_scores_equal_full_risk():
    losses = np.array(
        [
            0.0,
            0.2,
            0.4,
            0.8,
        ]
    )

    result = grouped_risk_coverage(
        np.ones(4),
        losses,
    )

    expected = float(
        np.mean(losses)
    )

    assert result[
        "aurc"
    ] == pytest.approx(
        expected
    )

    assert result[
        "full_coverage_risk"
    ] == pytest.approx(
        expected
    )


def test_tie_order_cannot_change_result():
    scores_a = np.array(
        [1.0, 1.0, 0.0, 0.0]
    )

    losses_a = np.array(
        [0.0, 1.0, 0.2, 0.8]
    )

    scores_b = np.array(
        [1.0, 1.0, 0.0, 0.0]
    )

    losses_b = np.array(
        [1.0, 0.0, 0.8, 0.2]
    )

    first = grouped_risk_coverage(
        scores_a,
        losses_a,
    )

    second = grouped_risk_coverage(
        scores_b,
        losses_b,
    )

    assert first[
        "aurc"
    ] == pytest.approx(
        second["aurc"]
    )

    np.testing.assert_allclose(
        first["risk"],
        second["risk"],
    )
