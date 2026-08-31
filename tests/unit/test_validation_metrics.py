import numpy as np
import pytest

from ptbxl_reliability.validation_metrics import (
    classwise_average_precision,
    classwise_brier,
    macro_average_precision,
    macro_brier,
    precision_recall_f1,
    select_f1_threshold,
)


def test_perfect_average_precision():
    y = np.array(
        [
            [0] * 5,
            [0] * 5,
            [1] * 5,
            [1] * 5,
        ],
        dtype=float,
    )

    p = np.array(
        [
            [0.1] * 5,
            [0.2] * 5,
            [0.8] * 5,
            [0.9] * 5,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        classwise_average_precision(y, p),
        np.ones(5),
    )

    assert macro_average_precision(
        y,
        p,
    ) == pytest.approx(1.0)


def test_brier_manual_formula():
    y = np.array(
        [
            [0] * 5,
            [1] * 5,
        ],
        dtype=float,
    )

    p = np.array(
        [
            [0.25] * 5,
            [0.75] * 5,
        ],
        dtype=float,
    )

    # Both errors are 0.25, squared = 0.0625.
    expected = 0.0625

    np.testing.assert_allclose(
        classwise_brier(y, p),
        np.full(5, expected),
    )

    assert macro_brier(
        y,
        p,
    ) == pytest.approx(expected)


def test_precision_recall_f1_manual():
    y = np.array(
        [1, 1, 0, 0]
    )

    pred = np.array(
        [1, 0, 1, 0]
    )

    precision, recall, f1 = (
        precision_recall_f1(
            y,
            pred,
        )
    )

    assert precision == pytest.approx(
        0.5
    )

    assert recall == pytest.approx(
        0.5
    )

    assert f1 == pytest.approx(
        0.5
    )


def test_known_optimal_threshold():
    y = np.array(
        [1, 0, 1, 0]
    )

    p = np.array(
        [0.9, 0.8, 0.7, 0.1]
    )

    result = select_f1_threshold(
        y,
        p,
    )

    assert result[
        "threshold"
    ] == pytest.approx(0.7)

    assert result[
        "f1"
    ] == pytest.approx(0.8)

    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["tn"] == 1
