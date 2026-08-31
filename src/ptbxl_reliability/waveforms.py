"""Validated loading of PTB-XL 100-Hz ECG waveforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb


EXPECTED_FS = 100.0
EXPECTED_SAMPLES = 1000
EXPECTED_CHANNELS = 12

EXPECTED_LEADS = (
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


@dataclass(frozen=True)
class ECGRecord:
    """One validated PTB-XL waveform in native WFDB orientation."""

    signal: np.ndarray
    sampling_rate_hz: float
    lead_names: tuple[str, ...]


def load_ecg_100hz(record_base: Path) -> ECGRecord:
    """Load and validate one PTB-XL 100-Hz waveform.

    The returned waveform uses WFDB's native time-major layout:
    (1000 samples, 12 leads).
    """

    record_base = Path(record_base)

    dat_path = record_base.with_suffix(".dat")
    hea_path = record_base.with_suffix(".hea")

    if not dat_path.is_file():
        raise FileNotFoundError(f"Missing waveform file: {dat_path}")

    if not hea_path.is_file():
        raise FileNotFoundError(f"Missing header file: {hea_path}")

    signal, fields = wfdb.rdsamp(str(record_base))

    signal = np.asarray(signal)

    expected_shape = (EXPECTED_SAMPLES, EXPECTED_CHANNELS)

    if signal.shape != expected_shape:
        raise ValueError(
            f"{record_base}: expected waveform shape "
            f"{expected_shape}, found {signal.shape}"
        )

    sampling_rate = float(fields["fs"])

    if not np.isclose(
        sampling_rate,
        EXPECTED_FS,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{record_base}: expected {EXPECTED_FS} Hz, "
            f"found {sampling_rate} Hz"
        )

    lead_names = tuple(str(name) for name in fields["sig_name"])

    if lead_names != EXPECTED_LEADS:
        raise ValueError(
            f"{record_base}: unexpected lead order. "
            f"Expected {EXPECTED_LEADS}, found {lead_names}"
        )

    if not np.isfinite(signal).all():
        bad_count = int((~np.isfinite(signal)).sum())
        raise ValueError(
            f"{record_base}: waveform contains "
            f"{bad_count} NaN/Inf values"
        )

    return ECGRecord(
        signal=signal,
        sampling_rate_hz=sampling_rate,
        lead_names=lead_names,
    )


def to_model_layout(signal: np.ndarray) -> np.ndarray:
    """Convert ECG from (1000, 12) to (12, 1000).

    No filtering, normalization, scaling, resampling, or dtype conversion
    is performed.
    """

    signal = np.asarray(signal)

    expected_shape = (EXPECTED_SAMPLES, EXPECTED_CHANNELS)

    if signal.shape != expected_shape:
        raise ValueError(
            f"Expected input shape {expected_shape}, "
            f"found {signal.shape}"
        )

    result = np.ascontiguousarray(signal.T)

    if result.shape != (EXPECTED_CHANNELS, EXPECTED_SAMPLES):
        raise AssertionError(
            "Internal error while converting ECG orientation"
        )

    return result
