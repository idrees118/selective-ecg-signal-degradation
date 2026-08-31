import numpy as np
import pytest

from ptbxl_reliability.waveform_signal_quality import (
    clipping_burden,
    lead_dropout_burden,
    spectral_burdens,
    waveform_badness_features,
)


FS = 100.0
N = 1000


def base_ecg_like_signal():
    t = np.arange(
        N,
        dtype=np.float64,
    ) / FS

    leads = []

    for index in range(12):
        phase = (
            index
            * 0.07
        )

        lead = (
            np.sin(
                2.0
                * np.pi
                * 10.0
                * t
                + phase
            )
            + 0.25
            * np.sin(
                2.0
                * np.pi
                * 5.0
                * t
                + phase
            )
        )

        leads.append(
            lead
        )

    return np.asarray(
        leads,
        dtype=np.float64,
    )


def test_clean_signal_features_are_finite():
    x = base_ecg_like_signal()

    result = waveform_badness_features(
        x
    )

    assert set(result) == {
        "baseline_burden",
        "high_frequency_burden",
        "lead_dropout_burden",
        "clipping_burden",
    }

    assert np.isfinite(
        list(
            result.values()
        )
    ).all()


def test_baseline_wander_increases_baseline_burden():
    clean = base_ecg_like_signal()

    t = np.arange(
        N,
        dtype=np.float64,
    ) / FS

    corrupted = (
        clean
        + 2.0
        * np.sin(
            2.0
            * np.pi
            * 0.2
            * t
        )[None, :]
    )

    clean_baseline, _ = (
        spectral_burdens(
            clean
        )
    )

    corrupted_baseline, _ = (
        spectral_burdens(
            corrupted
        )
    )

    assert (
        corrupted_baseline
        > clean_baseline
    )


def test_high_frequency_noise_increases_high_frequency_burden():
    clean = base_ecg_like_signal()

    t = np.arange(
        N,
        dtype=np.float64,
    ) / FS

    corrupted = (
        clean
        + 1.5
        * np.sin(
            2.0
            * np.pi
            * 45.0
            * t
        )[None, :]
    )

    _, clean_high = (
        spectral_burdens(
            clean
        )
    )

    _, corrupted_high = (
        spectral_burdens(
            corrupted
        )
    )

    assert (
        corrupted_high
        > clean_high
    )


def test_masked_lead_has_maximal_dropout_burden():
    x = base_ecg_like_signal()

    masked = x.copy()
    masked[3] = 0.0

    assert (
        lead_dropout_burden(
            masked
        )
        == pytest.approx(
            1.0
        )
    )


def test_masked_lead_has_maximal_clipping_burden():
    x = base_ecg_like_signal()

    masked = x.copy()
    masked[5] = 0.0

    assert (
        clipping_burden(
            masked
        )
        == pytest.approx(
            1.0
        )
    )


def test_clipping_increases_extreme_occupancy():
    x = base_ecg_like_signal()

    clipped = np.clip(
        x,
        -0.4,
        0.4,
    )

    assert (
        clipping_burden(
            clipped
        )
        > clipping_burden(
            x
        )
    )


def test_bad_shape_is_rejected():
    with pytest.raises(
        ValueError
    ):
        waveform_badness_features(
            np.zeros(
                (
                    12,
                    999,
                )
            )
        )


def test_single_flat_lead_does_not_break_spectral_features():
    x = base_ecg_like_signal()

    x[4] = 0.0

    result = waveform_badness_features(
        x
    )

    assert np.isfinite(
        list(
            result.values()
        )
    ).all()

    assert result[
        "lead_dropout_burden"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "clipping_burden"
    ] == pytest.approx(
        1.0
    )


def test_all_flat_leads_rejected_for_spectral_features():
    x = np.zeros(
        (
            12,
            1000,
        ),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="All leads have non-positive spectral power",
    ):
        spectral_burdens(
            x
        )
