import numpy as np

from ptbxl_reliability.corruptions import (
    add_baseline_wander,
    add_white_noise,
    amplitude_clip,
    mask_leads,
)


FS = 100.0
N = 1000


def make_signal():
    t = np.arange(
        N,
        dtype=np.float64,
    ) / FS

    return np.asarray(
        [
            (
                0.8
                + 0.03 * index
            )
            * np.sin(
                2.0
                * np.pi
                * (
                    4.0
                    + 0.2 * index
                )
                * t
                + 0.1 * index
            )
            for index
            in range(12)
        ],
        dtype=np.float64,
    )


def ac_rms(x):
    centered = (
        x
        - np.mean(
            x,
            axis=1,
            keepdims=True,
        )
    )

    return np.sqrt(
        np.mean(
            centered**2,
            axis=1,
        )
    )


def test_baseline_wander_is_deterministic():
    x = make_signal()

    first = add_baseline_wander(
        x,
        ecg_id=123,
        snr_db=12.0,
        severity_index=1,
    )

    second = add_baseline_wander(
        x,
        ecg_id=123,
        snr_db=12.0,
        severity_index=1,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_baseline_wander_hits_target_snr():
    x = make_signal()

    corrupted = add_baseline_wander(
        x,
        ecg_id=123,
        snr_db=12.0,
        severity_index=1,
    )

    noise = corrupted - x

    observed = (
        20.0
        * np.log10(
            ac_rms(x)
            / ac_rms(noise)
        )
    )

    np.testing.assert_allclose(
        observed,
        12.0,
        rtol=0.0,
        atol=1e-10,
    )


def test_white_noise_is_deterministic():
    x = make_signal()

    first = add_white_noise(
        x,
        ecg_id=456,
        snr_db=6.0,
        severity_index=2,
    )

    second = add_white_noise(
        x,
        ecg_id=456,
        snr_db=6.0,
        severity_index=2,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_white_noise_hits_target_snr():
    x = make_signal()

    corrupted = add_white_noise(
        x,
        ecg_id=456,
        snr_db=18.0,
        severity_index=0,
    )

    noise = corrupted - x

    observed = (
        20.0
        * np.log10(
            ac_rms(x)
            / ac_rms(noise)
        )
    )

    np.testing.assert_allclose(
        observed,
        18.0,
        rtol=0.0,
        atol=1e-10,
    )


def test_clipping_severity_is_monotonic():
    x = make_signal()

    mild = amplitude_clip(
        x,
        quantile=0.995,
    )

    moderate = amplitude_clip(
        x,
        quantile=0.975,
    )

    severe = amplitude_clip(
        x,
        quantile=0.950,
    )

    mild_change = np.mean(
        np.abs(
            mild - x
        )
    )

    moderate_change = np.mean(
        np.abs(
            moderate - x
        )
    )

    severe_change = np.mean(
        np.abs(
            severe - x
        )
    )

    assert (
        mild_change
        <= moderate_change
        <= severe_change
    )


def test_lead_masking_exact_count():
    x = make_signal()

    corrupted, selected = mask_leads(
        x,
        ecg_id=789,
        number_of_leads=3,
        severity_index=1,
    )

    assert selected.shape == (
        3,
    )

    assert np.unique(
        selected
    ).size == 3

    for index in selected:
        assert np.all(
            corrupted[
                index
            ] == 0.0
        )


def test_lead_selection_is_deterministic():
    x = make_signal()

    first_x, first_ids = mask_leads(
        x,
        ecg_id=789,
        number_of_leads=6,
        severity_index=2,
    )

    second_x, second_ids = mask_leads(
        x,
        ecg_id=789,
        number_of_leads=6,
        severity_index=2,
    )

    np.testing.assert_array_equal(
        first_ids,
        second_ids,
    )

    np.testing.assert_array_equal(
        first_x,
        second_x,
    )
