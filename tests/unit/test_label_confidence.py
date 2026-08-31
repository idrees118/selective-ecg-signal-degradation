import numpy as np
import pytest

from ptbxl_reliability.label_confidence import (
    positive_superclass_confidence,
    record_annotation_confidence,
)


def test_single_known_likelihood():
    assert (
        positive_superclass_confidence(
            [80.0]
        )
        == pytest.approx(0.8)
    )


def test_zero_means_unknown_not_zero_confidence():
    result = (
        positive_superclass_confidence(
            [0.0, 50.0]
        )
    )

    assert result == pytest.approx(
        0.5
    )


def test_all_unknown_returns_nan():
    result = (
        positive_superclass_confidence(
            [0.0, 0.0]
        )
    )

    assert np.isnan(result)


def test_superclass_uses_strongest_support():
    # The superclass target is positive through
    # logical OR over constituent diagnostic codes.
    result = (
        positive_superclass_confidence(
            [35.0, 100.0, 50.0]
        )
    )

    assert result == pytest.approx(
        1.0
    )


def test_record_mean_across_positive_superclasses():
    result = record_annotation_confidence(
        [
            [100.0],
            [50.0],
        ]
    )

    assert result[
        "label_confidence"
    ] == pytest.approx(
        0.75
    )

    assert result[
        "label_ambiguity"
    ] == pytest.approx(
        0.25
    )

    assert result[
        "coverage"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "complete_coverage"
    ] is True


def test_missing_superclass_is_not_treated_as_zero():
    result = record_annotation_confidence(
        [
            [100.0],
            [0.0],
        ]
    )

    assert result[
        "label_confidence"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "coverage"
    ] == pytest.approx(
        0.5
    )

    assert result[
        "complete_coverage"
    ] is False


def test_all_unknown_record_remains_missing():
    result = record_annotation_confidence(
        [
            [0.0],
        ]
    )

    assert np.isnan(
        result[
            "label_confidence"
        ]
    )

    assert result[
        "coverage"
    ] == pytest.approx(
        0.0
    )
