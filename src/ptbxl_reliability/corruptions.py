"""Deterministic waveform corruptions for PTB-XL stress testing."""

from __future__ import annotations

import numpy as np


EXPECTED_SHAPE = (
    12,
    1000,
)

EXPECTED_FS = 100.0

BASE_SEED = 20260826

CORRUPTION_CODES = {
    "baseline_wander": 101,
    "additive_white_noise": 202,
    "lead_masking": 303,
}


def _validate(
    signal: np.ndarray,
    fs: float,
) -> np.ndarray:
    x = np.asarray(
        signal,
        dtype=np.float64,
    )

    if x.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected {EXPECTED_SHAPE}, "
            f"found {x.shape}"
        )

    if float(fs) != EXPECTED_FS:
        raise ValueError(
            f"Expected fs=100, found {fs}"
        )

    if not np.isfinite(
        x
    ).all():
        raise ValueError(
            "Signal contains NaN/Inf"
        )

    return x


def _rng(
    ecg_id: int,
    corruption: str,
    severity_index: int,
) -> np.random.Generator:
    if corruption not in CORRUPTION_CODES:
        raise ValueError(
            f"Unknown corruption {corruption!r}"
        )

    if int(ecg_id) <= 0:
        raise ValueError(
            "ecg_id must be positive"
        )

    if severity_index not in (
        0,
        1,
        2,
    ):
        raise ValueError(
            "severity_index must be 0, 1, or 2"
        )

    seed_sequence = np.random.SeedSequence(
        [
            BASE_SEED,
            int(ecg_id),
            CORRUPTION_CODES[
                corruption
            ],
            int(severity_index),
        ]
    )

    return np.random.default_rng(
        seed_sequence
    )


def _ac_rms_per_lead(
    signal: np.ndarray,
) -> np.ndarray:
    centered = (
        signal
        - np.mean(
            signal,
            axis=1,
            keepdims=True,
            dtype=np.float64,
        )
    )

    return np.sqrt(
        np.mean(
            centered**2,
            axis=1,
            dtype=np.float64,
        )
    )


def _target_noise_rms(
    signal_rms: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    if not np.isfinite(
        snr_db
    ):
        raise ValueError(
            "SNR must be finite"
        )

    return (
        signal_rms
        / (
            10.0
            ** (
                float(
                    snr_db
                )
                / 20.0
            )
        )
    )


def add_baseline_wander(
    signal: np.ndarray,
    *,
    ecg_id: int,
    snr_db: float,
    severity_index: int,
    frequency_hz: float = 0.2,
    fs: float = EXPECTED_FS,
) -> np.ndarray:
    """Add deterministic low-frequency sinusoidal baseline wander."""

    x = _validate(
        signal,
        fs,
    )

    if (
        frequency_hz <= 0.0
        or frequency_hz
        >= fs / 2.0
    ):
        raise ValueError(
            "Invalid baseline-wander frequency"
        )

    rng = _rng(
        ecg_id,
        "baseline_wander",
        severity_index,
    )

    signal_rms = (
        _ac_rms_per_lead(
            x
        )
    )

    target_rms = (
        _target_noise_rms(
            signal_rms,
            snr_db,
        )
    )

    time = (
        np.arange(
            x.shape[1],
            dtype=np.float64,
        )
        / fs
    )

    noise = np.empty_like(
        x,
        dtype=np.float64,
    )

    phases = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=x.shape[0],
    )

    for lead_index in range(
        x.shape[0]
    ):
        raw_noise = np.sin(
            2.0
            * np.pi
            * frequency_hz
            * time
            + phases[
                lead_index
            ]
        )

        raw_noise = (
            raw_noise
            - np.mean(
                raw_noise,
                dtype=np.float64,
            )
        )

        raw_rms = float(
            np.sqrt(
                np.mean(
                    raw_noise**2,
                    dtype=np.float64,
                )
            )
        )

        if (
            signal_rms[
                lead_index
            ]
            <= 0.0
        ):
            noise[
                lead_index
            ] = 0.0
            continue

        if raw_rms <= 0.0:
            raise ValueError(
                "Generated baseline noise "
                "has zero RMS"
            )

        noise[
            lead_index
        ] = (
            raw_noise
            * (
                target_rms[
                    lead_index
                ]
                / raw_rms
            )
        )

    corrupted = (
        x
        + noise
    )

    if not np.isfinite(
        corrupted
    ).all():
        raise ValueError(
            "Baseline corruption produced NaN/Inf"
        )

    return corrupted


def add_white_noise(
    signal: np.ndarray,
    *,
    ecg_id: int,
    snr_db: float,
    severity_index: int,
    fs: float = EXPECTED_FS,
) -> np.ndarray:
    """Add deterministic zero-mean Gaussian noise at target leadwise SNR."""

    x = _validate(
        signal,
        fs,
    )

    rng = _rng(
        ecg_id,
        "additive_white_noise",
        severity_index,
    )

    signal_rms = (
        _ac_rms_per_lead(
            x
        )
    )

    target_rms = (
        _target_noise_rms(
            signal_rms,
            snr_db,
        )
    )

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=x.shape,
    )

    noise = (
        noise
        - np.mean(
            noise,
            axis=1,
            keepdims=True,
            dtype=np.float64,
        )
    )

    noise_rms = np.sqrt(
        np.mean(
            noise**2,
            axis=1,
            dtype=np.float64,
        )
    )

    if (
        noise_rms <= 0.0
    ).any():
        raise ValueError(
            "Generated white noise has zero RMS"
        )

    scale = np.zeros(
        x.shape[0],
        dtype=np.float64,
    )

    nonflat = (
        signal_rms > 0.0
    )

    scale[
        nonflat
    ] = (
        target_rms[
            nonflat
        ]
        / noise_rms[
            nonflat
        ]
    )

    noise = (
        noise
        * scale[:, None]
    )

    corrupted = (
        x
        + noise
    )

    if not np.isfinite(
        corrupted
    ).all():
        raise ValueError(
            "White-noise corruption produced NaN/Inf"
        )

    return corrupted


def amplitude_clip(
    signal: np.ndarray,
    *,
    quantile: float,
    fs: float = EXPECTED_FS,
) -> np.ndarray:
    """Symmetric leadwise clipping around each lead mean."""

    x = _validate(
        signal,
        fs,
    )

    if (
        quantile <= 0.5
        or quantile >= 1.0
    ):
        raise ValueError(
            "Clipping quantile must lie in (0.5,1)"
        )

    mean = np.mean(
        x,
        axis=1,
        keepdims=True,
        dtype=np.float64,
    )

    centered = (
        x
        - mean
    )

    thresholds = np.quantile(
        np.abs(
            centered
        ),
        quantile,
        axis=1,
    )

    corrupted = centered.copy()

    for lead_index in range(
        x.shape[0]
    ):
        threshold = float(
            thresholds[
                lead_index
            ]
        )

        if threshold <= 0.0:
            # Already-flat lead.
            continue

        corrupted[
            lead_index
        ] = np.clip(
            centered[
                lead_index
            ],
            -threshold,
            threshold,
        )

    corrupted = (
        corrupted
        + mean
    )

    if not np.isfinite(
        corrupted
    ).all():
        raise ValueError(
            "Clipping produced NaN/Inf"
        )

    return corrupted


def mask_leads(
    signal: np.ndarray,
    *,
    ecg_id: int,
    number_of_leads: int,
    severity_index: int,
    fs: float = EXPECTED_FS,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Deterministically replace selected leads by flat zero mV."""

    x = _validate(
        signal,
        fs,
    )

    if (
        number_of_leads < 1
        or number_of_leads
        > x.shape[0]
    ):
        raise ValueError(
            "Invalid number_of_leads"
        )

    rng = _rng(
        ecg_id,
        "lead_masking",
        severity_index,
    )

    selected = np.sort(
        rng.choice(
            x.shape[0],
            size=number_of_leads,
            replace=False,
        )
    )

    corrupted = x.copy()

    corrupted[
        selected,
        :,
    ] = 0.0

    return (
        corrupted,
        selected.astype(
            np.int64
        ),
    )
