import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ptbxl_reliability.metrics import (
    classwise_auroc,
    macro_auroc,
)


def test_perfect_ranking_has_auc_one():
    targets = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=float,
    )

    probabilities = np.array(
        [
            [0.1] * 5,
            [0.2] * 5,
            [0.8] * 5,
            [0.9] * 5,
        ],
        dtype=float,
    )

    assert macro_auroc(
        targets,
        probabilities,
    ) == pytest.approx(1.0)


def test_reversed_ranking_has_auc_zero():
    targets = np.array(
        [
            [0] * 5,
            [0] * 5,
            [1] * 5,
            [1] * 5,
        ],
        dtype=float,
    )

    probabilities = np.array(
        [
            [0.9] * 5,
            [0.8] * 5,
            [0.2] * 5,
            [0.1] * 5,
        ],
        dtype=float,
    )

    assert macro_auroc(
        targets,
        probabilities,
    ) == pytest.approx(0.0)


def test_manual_half_auc_case():
    # One positive outranks one negative,
    # while the other positive does not.
    #
    # Pairwise positive-negative comparisons:
    # 0.9 > 0.8 : correct
    # 0.9 > 0.2 : correct
    # 0.1 > 0.8 : wrong
    # 0.1 > 0.2 : wrong
    #
    # AUC = 2 / 4 = 0.5.

    labels = np.array(
        [1, 1, 0, 0],
        dtype=float,
    )

    scores = np.array(
        [0.9, 0.1, 0.8, 0.2],
        dtype=float,
    )

    targets = np.column_stack(
        [labels] * 5
    )

    probabilities = np.column_stack(
        [scores] * 5
    )

    assert macro_auroc(
        targets,
        probabilities,
    ) == pytest.approx(0.5)


def test_classwise_matches_independent_sklearn_calls():
    targets = np.array(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
        ],
        dtype=float,
    )

    probabilities = np.array(
        [
            [0.8, 0.2, 0.9, 0.1, 0.7],
            [0.3, 0.8, 0.2, 0.9, 0.1],
            [0.7, 0.7, 0.3, 0.2, 0.8],
            [0.2, 0.1, 0.8, 0.8, 0.2],
        ],
        dtype=float,
    )

    observed = classwise_auroc(
        targets,
        probabilities,
    )

    expected = np.array(
        [
            roc_auc_score(
                targets[:, index],
                probabilities[:, index],
            )
            for index in range(5)
        ]
    )

    np.testing.assert_allclose(
        observed,
        expected,
        rtol=0.0,
        atol=0.0,
    )
