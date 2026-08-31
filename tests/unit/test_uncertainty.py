import numpy as np
import pytest
import torch
from torch import nn

from ptbxl_reliability.model import (
    ResNet1DWang,
)
from ptbxl_reliability.uncertainty import (
    bernoulli_entropy,
    enable_mc_dropout,
    mc_uncertainty,
)


def test_bernoulli_entropy_half_is_ln2():
    observed = bernoulli_entropy(
        np.array([0.5])
    )[0]

    assert observed == pytest.approx(
        np.log(2.0),
        rel=1e-12,
    )


def test_entropy_near_zero_for_certain_probability():
    observed = bernoulli_entropy(
        np.array(
            [0.0, 1.0]
        )
    )

    assert np.all(
        observed < 1e-9
    )


def test_identical_mc_predictions_have_zero_mi():
    probabilities = np.full(
        (30, 4, 5),
        0.7,
        dtype=float,
    )

    result = mc_uncertainty(
        probabilities
    )

    np.testing.assert_allclose(
        result[
            "mutual_information"
        ],
        0.0,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        result[
            "mean_mutual_information"
        ],
        0.0,
        atol=1e-14,
    )


def test_disagreement_produces_positive_mi():
    probabilities = np.empty(
        (2, 1, 5),
        dtype=float,
    )

    probabilities[0, 0, :] = 0.1
    probabilities[1, 0, :] = 0.9

    result = mc_uncertainty(
        probabilities
    )

    assert np.all(
        result[
            "mutual_information"
        ] > 0.0
    )

    assert (
        result[
            "mean_mutual_information"
        ][0]
        > 0.0
    )


def test_manual_mutual_information_formula():
    probabilities = np.array(
        [
            [[0.2] * 5],
            [[0.8] * 5],
        ],
        dtype=float,
    )

    result = mc_uncertainty(
        probabilities
    )

    expected = (
        np.log(2.0)
        - bernoulli_entropy(
            np.array([0.2])
        )[0]
    )

    np.testing.assert_allclose(
        result[
            "mutual_information"
        ][0],
        np.full(
            5,
            expected,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_mc_dropout_keeps_batchnorm_frozen():
    model = ResNet1DWang()

    count = enable_mc_dropout(
        model
    )

    # ResNet1DWang head has two Dropout modules.
    assert count == 2

    dropout_modules = [
        module
        for module in model.modules()
        if isinstance(
            module,
            nn.Dropout,
        )
    ]

    assert all(
        module.training
        for module in dropout_modules
    )

    batchnorm_modules = [
        module
        for module in model.modules()
        if isinstance(
            module,
            nn.BatchNorm1d,
        )
    ]

    assert all(
        not module.training
        for module in batchnorm_modules
    )


def test_mc_dropout_produces_stochastic_logits():
    torch.manual_seed(
        123
    )

    model = ResNet1DWang()

    enable_mc_dropout(
        model
    )

    x = torch.randn(
        4,
        12,
        1000,
    )

    with torch.no_grad():
        first = model(x)
        second = model(x)

    assert not torch.equal(
        first,
        second,
    )
