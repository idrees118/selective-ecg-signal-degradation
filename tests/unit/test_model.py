import torch
from torch import nn

from ptbxl_reliability.model import (
    ResNet1DWang,
    count_trainable_parameters,
)


def test_model_output_shape():
    model = ResNet1DWang()

    model.eval()

    x = torch.zeros(
        4,
        12,
        1000,
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (
        4,
        5,
    )

    assert logits.dtype == (
        torch.float32
    )

    assert torch.isfinite(
        logits
    ).all()


def test_model_contains_no_sigmoid():
    model = ResNet1DWang()

    assert not any(
        isinstance(
            module,
            nn.Sigmoid,
        )
        for module
        in model.modules()
    )


def test_reference_kernel_sizes():
    model = ResNet1DWang()

    assert (
        model.stem[0].kernel_size
        == (7,)
    )

    assert (
        model.block1.conv1.kernel_size
        == (5,)
    )

    assert (
        model.block1.conv2.kernel_size
        == (3,)
    )

    assert (
        model.block2.conv1.kernel_size
        == (5,)
    )

    assert (
        model.block3.conv1.kernel_size
        == (5,)
    )


def test_temporal_downsampling():
    model = ResNet1DWang()

    model.eval()

    x = torch.zeros(
        2,
        12,
        1000,
    )

    with torch.no_grad():

        x = model.stem(x)

        assert x.shape == (
            2,
            128,
            1000,
        )

        x = model.block1(x)

        assert x.shape == (
            2,
            128,
            1000,
        )

        x = model.block2(x)

        assert x.shape == (
            2,
            128,
            500,
        )

        x = model.block3(x)

        assert x.shape == (
            2,
            128,
            250,
        )


def test_concat_pool_shape():
    model = ResNet1DWang()

    model.eval()

    x = torch.zeros(
        2,
        12,
        1000,
    )

    with torch.no_grad():
        features = (
            model.forward_features(x)
        )

        pooled = model.pool(
            features
        )

    assert pooled.shape == (
        2,
        256,
        1,
    )


def test_trainable_parameter_count_is_frozen():
    model = ResNet1DWang()

    assert (
        count_trainable_parameters(
            model
        )
        == 473349
    )


def test_loss_definition_matches_multilabel_task():
    model = ResNet1DWang()

    model.eval()

    x = torch.randn(
        2,
        12,
        1000,
    )

    targets = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )

    loss_fn = (
        nn.BCEWithLogitsLoss()
    )

    with torch.no_grad():
        logits = model(x)

        loss = loss_fn(
            logits,
            targets,
        )

    assert loss.ndim == 0

    assert torch.isfinite(
        loss
    )
