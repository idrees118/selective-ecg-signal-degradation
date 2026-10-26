"""ECG signal-quality indices: bSQI (Li et al. 2008) plus pSQI, basSQI, kSQI.

bSQI uses the Pan-Tompkins and Elgendi detectors from NeuroKit2.
Flat leads score 0.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis


MATCH_TOLERANCE_S = 0.15

DETECTORS = (
    "pantompkins1985",
    "elgendi2010",
)


def _detect(
    lead: np.ndarray,
    fs: float,
    method: str,
) -> np.ndarray:
    import neurokit2 as nk

    if np.ptp(lead) == 0.0:
        return np.empty(0, dtype=np.int64)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned = nk.ecg_clean(
                lead,
                sampling_rate=int(fs),
                method=method,
            )
            peaks = nk.ecg_findpeaks(
                cleaned,
                sampling_rate=int(fs),
                method=method,
            )["ECG_R_Peaks"]
    except (ValueError, IndexError, ZeroDivisionError):
        # e.g. heavily clipped leads: count as no beats.
        return np.empty(0, dtype=np.int64)

    return np.asarray(peaks, dtype=np.int64)


def beat_agreement(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    fs: float,
    tolerance_s: float = MATCH_TOLERANCE_S,
) -> float:
    """bSQI = matched / (N_a + N_b - matched); 0 when neither finds beats.

    Beats are matched one-to-one, in time order, within the tolerance.
    """
    a = np.sort(np.asarray(peaks_a, dtype=np.int64))
    b = np.sort(np.asarray(peaks_b, dtype=np.int64))

    if a.size == 0 and b.size == 0:
        return 0.0

    tolerance = tolerance_s * fs
    matched = 0
    i = j = 0

    while i < a.size and j < b.size:
        difference = a[i] - b[j]
        if abs(difference) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif difference < 0:
            i += 1
        else:
            j += 1

    return matched / float(a.size + b.size - matched)


def _band_power(
    frequencies: np.ndarray,
    power: np.ndarray,
    low: float,
    high: float,
) -> float:
    band = (frequencies >= low) & (frequencies <= high)
    return float(np.trapezoid(power[band], frequencies[band]))


def lead_sqi(
    lead: np.ndarray,
    fs: float,
) -> dict[str, float]:
    x = np.asarray(lead, dtype=np.float64)

    peaks = [_detect(x, fs, method) for method in DETECTORS]

    if np.ptp(x) == 0.0:
        return {
            "bsqi": 0.0,
            "psqi": 0.0,
            "bassqi": 0.0,
            "ksqi": 0.0,
        }

    frequencies, power = welch(
        x - x.mean(),
        fs=fs,
        nperseg=min(x.size, int(4 * fs)),
    )

    total_5_40 = _band_power(frequencies, power, 5.0, 40.0)
    total_0_40 = _band_power(frequencies, power, 0.0, 40.0)

    return {
        "bsqi": beat_agreement(peaks[0], peaks[1], fs),
        "psqi": (
            _band_power(frequencies, power, 5.0, 15.0) / total_5_40
            if total_5_40 > 0.0 else 0.0
        ),
        "bassqi": (
            1.0 - _band_power(frequencies, power, 0.0, 1.0) / total_0_40
            if total_0_40 > 0.0 else 0.0
        ),
        "ksqi": float(kurtosis(x, fisher=False)),
    }


def record_sqi(
    signal: np.ndarray,
    fs: float = 100.0,
) -> dict[str, float]:
    """Record-level indices from a (leads, samples) waveform in mV."""
    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("signal must be (leads, samples)")

    if not np.isfinite(x).all():
        raise ValueError("signal contains NaN/Inf")

    leads = [lead_sqi(lead, fs) for lead in x]

    bsqi = np.asarray([lead["bsqi"] for lead in leads])

    return {
        "bsqi_mean": float(bsqi.mean()),
        "bsqi_median": float(np.median(bsqi)),
        "psqi_mean": float(np.mean([lead["psqi"] for lead in leads])),
        "bassqi_mean": float(np.mean([lead["bassqi"] for lead in leads])),
        "ksqi_median": float(np.median([lead["ksqi"] for lead in leads])),
    }


SQI_FIELDS = (
    "bsqi_mean",
    "bsqi_median",
    "psqi_mean",
    "bassqi_mean",
    "ksqi_median",
)
