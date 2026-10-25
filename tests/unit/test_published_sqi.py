import numpy as np
import pytest

from ptbxl_reliability.published_sqi import (
    SQI_FIELDS,
    beat_agreement,
    record_sqi,
)


def test_beat_agreement_perfect_and_partial():
    a = np.array([100, 200, 300])
    assert beat_agreement(a, a, fs=100.0) == 1.0
    # One beat each unmatched: 2 / (3 + 3 - 2).
    assert beat_agreement(a, np.array([105, 205, 500]), fs=100.0) == 0.5


def test_beat_agreement_tolerance_and_empty():
    assert beat_agreement(np.array([100]), np.array([116]), fs=100.0) == 0.0
    assert beat_agreement(np.array([100]), np.array([115]), fs=100.0) == 1.0
    assert beat_agreement(np.array([], int), np.array([], int), fs=100.0) == 0.0


def test_masked_leads_lower_record_bsqi():
    nk = pytest.importorskip("neurokit2")
    ecg = nk.ecg_simulate(duration=10, sampling_rate=100, heart_rate=70,
                          random_state=1)
    x = np.tile(ecg, (12, 1))
    masked = x.copy()
    masked[:6] = 0.0
    clean = record_sqi(x)
    degraded = record_sqi(masked)
    assert set(clean) == set(SQI_FIELDS)
    assert clean["bsqi_mean"] > 0.5
    assert degraded["bsqi_mean"] == pytest.approx(clean["bsqi_mean"] / 2)
