"""Reference-based 1-D ResNet classifier for PTB-XL.

Architecture basis
------------------
Adapted from the ResNet1D-Wang implementation used in the PTB-XL benchmark:

    Strodthoff et al.,
    "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL"

Important:
- Input is the full 10-second ECG: (B, 12, 1000).
- Output is five raw logits.
- No sigmoid is applied inside the network.
- Training loss must use BCEWithLogitsLoss.
"""

from __future__ import annotations

import torch
from torch import nn


INPUT_CHANNELS = 12
NUM_CLASSES = 5
BASE_CHANNELS = 128


def conv_same(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
) -> nn.Conv1d:
    """Odd-kernel 1-D convolution preserving length when stride=1."""

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be a positive odd integer"
        )

    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=False,
    )


class ResidualBlock1D(nn.Module):
    """Basic two-convolution residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        kernel_sizes: tuple[int, int] = (5, 3),
    ) -> None:
        super().__init__()

        k1, k2 = kernel_sizes

        self.conv1 = conv_same(
            in_channels,
            out_channels,
            k1,
            stride=stride,
        )

        self.bn1 = nn.BatchNorm1d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = conv_same(
            out_channels,
            out_channels,
            k2,
            stride=1,
        )

        self.bn2 = nn.BatchNorm1d(
            out_channels
        )

        if (
            stride != 1
            or in_channels != out_channels
        ):
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(
                    out_channels
                ),
            )
        else:
            self.downsample = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(
                identity
            )

        if out.shape != identity.shape:
            raise RuntimeError(
                "Residual and main-path shapes "
                f"do not match: {out.shape} vs "
                f"{identity.shape}"
            )

        out = out + identity
        out = self.relu(out)

        return out


class AdaptiveConcatPool1D(nn.Module):
    """Concatenate adaptive max and mean pooling."""

    def __init__(self) -> None:
        super().__init__()

        self.max_pool = (
            nn.AdaptiveMaxPool1d(1)
        )

        self.avg_pool = (
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        max_value = self.max_pool(x)
        mean_value = self.avg_pool(x)

        return torch.cat(
            [max_value, mean_value],
            dim=1,
        )


class ResNet1DWang(nn.Module):
    """PTB-XL ResNet1D-Wang style classifier.

    Full architecture:

        input: (B, 12, 1000)

        stem:
            Conv1d(12 -> 128, kernel=7, stride=1)
            BatchNorm
            ReLU

        residual block 1:
            128 -> 128
            kernels 5, 3
            stride 1

        residual block 2:
            128 -> 128
            kernels 5, 3
            stride 2

        residual block 3:
            128 -> 128
            kernels 5, 3
            stride 2

        adaptive max + average pooling
            128 + 128 = 256 features

        head:
            BatchNorm(256)
            Dropout(0.25)
            Linear(256 -> 128)
            ReLU
            BatchNorm(128)
            Dropout(0.50)
            Linear(128 -> 5)

        output:
            five raw logits
    """

    def __init__(
        self,
        *,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()

        if input_channels != 12:
            raise ValueError(
                "This study is frozen to "
                "12-lead PTB-XL ECGs"
            )

        if num_classes != 5:
            raise ValueError(
                "This study is frozen to the "
                "five diagnostic superclasses"
            )

        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                BASE_CHANNELS,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(
                BASE_CHANNELS
            ),
            nn.ReLU(
                inplace=True
            ),
        )

        self.block1 = ResidualBlock1D(
            BASE_CHANNELS,
            BASE_CHANNELS,
            stride=1,
            kernel_sizes=(5, 3),
        )

        self.block2 = ResidualBlock1D(
            BASE_CHANNELS,
            BASE_CHANNELS,
            stride=2,
            kernel_sizes=(5, 3),
        )

        self.block3 = ResidualBlock1D(
            BASE_CHANNELS,
            BASE_CHANNELS,
            stride=2,
            kernel_sizes=(5, 3),
        )

        self.pool = (
            AdaptiveConcatPool1D()
        )

        self.head = nn.Sequential(
            nn.Flatten(),

            nn.BatchNorm1d(
                2 * BASE_CHANNELS
            ),

            nn.Dropout(
                p=0.25
            ),

            nn.Linear(
                2 * BASE_CHANNELS,
                128,
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.Dropout(
                p=0.50
            ),

            nn.Linear(
                128,
                num_classes,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(
        self,
    ) -> None:
        """Match the reference Kaiming/BN initialization."""

        for module in self.modules():

            if isinstance(
                module,
                (nn.Conv1d, nn.Linear),
            ):
                nn.init.kaiming_normal_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.BatchNorm1d,
            ):
                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Return final convolutional feature map."""

        if x.ndim != 3:
            raise ValueError(
                "Expected input with shape "
                "(batch, 12, 1000)"
            )

        if x.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                f"Expected 12 leads, "
                f"found {x.shape[1]}"
            )

        if x.shape[2] != 1000:
            raise ValueError(
                f"Expected 1000 samples, "
                f"found {x.shape[2]}"
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "Model input contains NaN or Inf"
            )

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.forward_features(
            x
        )

        x = self.pool(x)

        logits = self.head(x)

        expected_shape = (
            x.shape[0],
            NUM_CLASSES,
        )

        if logits.shape != expected_shape:
            raise RuntimeError(
                f"Expected logits shape "
                f"{expected_shape}, "
                f"found {logits.shape}"
            )

        if not torch.isfinite(
            logits
        ).all():
            raise RuntimeError(
                "Model produced NaN or Inf logits"
            )

        return logits


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """Return number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )
