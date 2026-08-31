"""Waveform-derived PTB-XL signal-quality features."""

from __future__ import annotations

import numpy as np
from scipy.signal import periodogram


EXPECTED_LEADS = 12
EXPECTED_SAMPLES = 1000
EXPECTED_FS = 100.0


def _validate_signal(
    signal: np.ndarray,
    fs: float,
) -> np.ndarray:
    x = np.asarray(
        signal,
        dtype=np.float64,
    )

    if x.shape != (
        EXPECTED_LEADS,
        EXPECTED_SAMPLES,
    ):
        raise ValueError(
            "Expected signal shape (12,1000), "
            f"found {x.shape}"
        )

    if float(fs) != EXPECTED_FS:
        raise ValueError(
            f"Expected fs=100 Hz, found {fs}"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "Signal contains NaN/Inf"
        )

    return x


def spectral_burdens(
    signal: np.ndarray,
    *,
    fs: float = EXPECTED_FS,
) -> tuple[float, float]:
    """Worst-lead low- and high-frequency power burdens."""

    x = _validate_signal(
        signal,
        fs,
    )

    frequencies, psd = periodogram(
        x,
        fs=fs,
        window="hann",
        detrend="constant",
        scaling="density",
        axis=1,
    )

    total_mask = (
        (frequencies >= 0.1)
        & (frequencies <= 50.0)
    )

    baseline_mask = (
        (frequencies >= 0.1)
        & (frequencies <= 1.0)
    )

    high_mask = (
        (frequencies >= 40.0)
        & (frequencies <= 50.0)
    )

    total_power = np.sum(
        psd[:, total_mask],
        axis=1,
        dtype=np.float64,
    )

    if not np.isfinite(
        total_power
    ).all():
        raise ValueError(
            "Non-finite spectral power"
        )

    # A flat/constant lead has zero AC spectral power.
    # Its spectral ratio is undefined, so it is excluded
    # from spectral aggregation rather than assigned an
    # artificial epsilon-based ratio. Such leads are
    # explicitly captured by the dropout/clipping features.
    valid_power = (
        total_power > 0.0
    )

    if not np.any(
        valid_power
    ):
        raise ValueError(
            "All leads have non-positive spectral power"
        )

    baseline_power = np.sum(
        psd[:, baseline_mask],
        axis=1,
        dtype=np.float64,
    )

    high_power = np.sum(
        psd[:, high_mask],
        axis=1,
        dtype=np.float64,
    )

    baseline_ratio = (
        baseline_power[
            valid_power
        ]
        / total_power[
            valid_power
        ]
    )

    high_ratio = (
        high_power[
            valid_power
        ]
        / total_power[
            valid_power
        ]
    )

    if (
        not np.isfinite(
            baseline_ratio
        ).all()
        or not np.isfinite(
            high_ratio
        ).all()
    ):
        raise ValueError(
            "Non-finite spectral ratio"
        )

    return (
        float(
            np.max(
                baseline_ratio
            )
        ),
        float(
            np.max(
                high_ratio
            )
        ),
    )


def lead_dropout_burden(
    signal: np.ndarray,
    *,
    fs: float = EXPECTED_FS,
) -> float:
    """Continuous flat/weak-lead burden.

    AC RMS is used so DC baseline offsets do not determine this score.
    """

    x = _validate_signal(
        signal,
        fs,
    )

    centered = (
        x
        - np.mean(
            x,
            axis=1,
            keepdims=True,
            dtype=np.float64,
        )
    )

    ac_rms = np.sqrt(
        np.mean(
            centered**2,
            axis=1,
            dtype=np.float64,
        )
    )

    median_rms = float(
        np.median(
            ac_rms
        )
    )

    if median_rms <= 0.0:
        raise ValueError(
            "Median lead AC RMS is non-positive"
        )

    minimum_rms = float(
        np.min(
            ac_rms
        )
    )

    ratio = (
        minimum_rms
        / median_rms
    )

    ratio = float(
        np.clip(
            ratio,
            0.0,
            1.0,
        )
    )

    return float(
        1.0 - ratio
    )


def clipping_burden(
    signal: np.ndarray,
    *,
    fs: float = EXPECTED_FS,
) -> float:
    """Worst-lead occupancy at exact minimum/maximum signal values.

    Sustained clipping produces repeated extrema. A completely constant
    lead is assigned maximal burden.
    """

    x = _validate_signal(
        signal,
        fs,
    )

    occupancies = []

    for lead in x:
        minimum = float(
            np.min(
                lead
            )
        )

        maximum = float(
            np.max(
                lead
            )
        )

        if minimum == maximum:
            occupancy = 1.0
        else:
            extreme = (
                (lead == minimum)
                | (lead == maximum)
            )

            occupancy = float(
                np.mean(
                    extreme,
                    dtype=np.float64,
                )
            )

        occupancies.append(
            occupancy
        )

    return float(
        np.max(
            np.asarray(
                occupancies,
                dtype=np.float64,
            )
        )
    )


def waveform_badness_features(
    signal: np.ndarray,
    *,
    fs: float = EXPECTED_FS,
) -> dict[str, float]:
    baseline, high = spectral_burdens(
        signal,
        fs=fs,
    )

    dropout = lead_dropout_burden(
        signal,
        fs=fs,
    )

    clipping = clipping_burden(
        signal,
        fs=fs,
    )

    values = {
        "baseline_burden":
            baseline,

        "high_frequency_burden":
            high,

        "lead_dropout_burden":
            dropout,

        "clipping_burden":
            clipping,
    }

    if not np.isfinite(
        np.asarray(
            list(
                values.values()
            ),
            dtype=np.float64,
        )
    ).all():
        raise ValueError(
            "Waveform quality feature contains NaN/Inf"
        )

    return values
