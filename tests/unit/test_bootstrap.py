import numpy as np
import pytest

from ptbxl_reliability.bootstrap import (
    patient_record_weights,
    weighted_grouped_aurc,
)
from ptbxl_reliability.selective_prediction import (
    grouped_risk_coverage,
)


def test_weighted_aurc_matches_explicit_replication():
    reliability = np.asarray(
        [0.9, 0.9, 0.4, 0.1],
        dtype=np.float64,
    )
    losses = np.asarray(
        [0.0, 1.0, 0.5, 1.0],
        dtype=np.float64,
    )
    weights = np.asarray(
        [2, 0, 3, 1],
        dtype=np.int64,
    )

    observed = weighted_grouped_aurc(
        reliability,
        losses,
        weights,
    )

    expanded_reliability = np.repeat(
        reliability,
        weights,
    )
    expanded_losses = np.repeat(
        losses,
        weights,
    )

    expected = grouped_risk_coverage(
        expanded_reliability,
        expanded_losses,
    )["aurc"]

    assert observed == pytest.approx(
        expected,
        abs=1e-15,
    )


def test_all_one_weights_match_standard_grouped_aurc():
    reliability = np.asarray(
        [0.8, 0.8, 0.6, 0.2, 0.1],
        dtype=np.float64,
    )
    losses = np.asarray(
        [0.0, 0.5, 0.0, 1.0, 0.5],
        dtype=np.float64,
    )

    observed = weighted_grouped_aurc(
        reliability,
        losses,
        np.ones(
            len(losses),
            dtype=np.float64,
        ),
    )

    expected = grouped_risk_coverage(
        reliability,
        losses,
    )["aurc"]

    assert observed == pytest.approx(
        expected,
        abs=1e-15,
    )


def test_patient_bootstrap_weights_are_deterministic_and_clustered():
    patient_ids = np.asarray(
        [10, 10, 20, 30, 30, 30],
        dtype=np.int64,
    )

    first = patient_record_weights(
        patient_ids,
        seed=20260826,
        replicate=17,
    )

    second = patient_record_weights(
        patient_ids,
        seed=20260826,
        replicate=17,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )

    assert first[0] == first[1]
    assert first[3] == first[4] == first[5]


def test_negative_weight_rejected():
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        weighted_grouped_aurc(
            np.asarray([0.9, 0.1]),
            np.asarray([0.0, 1.0]),
            np.asarray([1.0, -1.0]),
        )
