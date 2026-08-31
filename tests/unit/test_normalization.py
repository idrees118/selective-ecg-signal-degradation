import math

import numpy as np
import pandas as pd
import pytest

from ptbxl_reliability.normalization import (
    GlobalZScore,
    RunningPopulationStats,
    standardize,
    training_rows,
)


def test_manual_population_statistics():
    # Hand calculation:
    #
    # values = 1, 2, 3, 4
    # mean = 2.5
    #
    # squared deviations:
    # 2.25, 0.25, 0.25, 2.25
    #
    # sum = 5
    # population variance = 5 / 4 = 1.25

    stats = RunningPopulationStats()

    stats.update(
        np.array(
            [1.0, 2.0, 3.0, 4.0]
        )
    )

    assert stats.count == 4
    assert stats.mean == pytest.approx(2.5)
    assert stats.variance == pytest.approx(1.25)
    assert stats.std == pytest.approx(
        math.sqrt(1.25)
    )


def test_multiple_batches_equal_numpy_population_statistics():
    data = np.array(
        [
            -3.0,
            -1.0,
            0.0,
            2.0,
            4.0,
            8.0,
            10.0,
        ],
        dtype=np.float64,
    )

    stats = RunningPopulationStats()

    stats.update(data[:2])
    stats.update(data[2:5])
    stats.update(data[5:])

    assert stats.count == data.size

    assert stats.mean == pytest.approx(
        np.mean(data),
        rel=1e-14,
        abs=1e-14,
    )

    assert stats.variance == pytest.approx(
        np.var(data, ddof=0),
        rel=1e-14,
        abs=1e-14,
    )


def test_batching_does_not_change_definition():
    data = np.linspace(
        -7.5,
        12.5,
        1001,
        dtype=np.float64,
    )

    one_batch = RunningPopulationStats()
    one_batch.update(data)

    many_batches = RunningPopulationStats()

    for chunk in np.array_split(
        data,
        17,
    ):
        many_batches.update(chunk)

    assert many_batches.mean == pytest.approx(
        one_batch.mean,
        rel=1e-13,
        abs=1e-13,
    )

    assert many_batches.variance == pytest.approx(
        one_batch.variance,
        rel=1e-13,
        abs=1e-13,
    )


def test_standardization_known_example():
    parameters = GlobalZScore(
        mean=2.0,
        std=2.0,
    )

    values = np.array(
        [0.0, 2.0, 4.0]
    )

    result = standardize(
        values,
        parameters,
    )

    np.testing.assert_allclose(
        result,
        [-1.0, 0.0, 1.0],
        rtol=0.0,
        atol=0.0,
    )


def test_nonfinite_values_are_rejected():
    stats = RunningPopulationStats()

    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        stats.update(
            np.array(
                [1.0, np.nan]
            )
        )


def test_zero_standard_deviation_is_rejected():
    stats = RunningPopulationStats()

    stats.update(
        np.ones(10)
    )

    with pytest.raises(
        ValueError,
        match="finite and > 0",
    ):
        _ = stats.std


def test_training_rows_never_returns_validation_or_test():
    manifest = pd.DataFrame(
        {
            "ecg_id": [
                1, 2, 3, 4
            ],
            "split": [
                "train",
                "validation",
                "test",
                "train",
            ],
        }
    )

    selected = training_rows(
        manifest
    )

    assert selected["ecg_id"].tolist() == [
        1, 4
    ]

    assert selected["split"].eq(
        "train"
    ).all()
