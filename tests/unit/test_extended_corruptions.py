import numpy as np
import pytest

from ptbxl_reliability.corruptions import (
    add_recorded_noise,
    mask_leads,
    mask_leads_nested,
)


FS = 100.0
N = 1000


def make_signal():
    t = np.arange(N, dtype=np.float64) / FS
    return np.asarray(
        [
            (0.8 + 0.03 * index)
            * np.sin(2.0 * np.pi * (4.0 + 0.2 * index) * t + 0.1 * index)
            for index in range(12)
        ],
        dtype=np.float64,
    )


def make_noise(m=20000):
    rng = np.random.default_rng(7)
    return rng.normal(size=(2, m)).cumsum(axis=1) * 0.01


def ac_rms(x):
    centered = x - np.mean(x, axis=1, keepdims=True)
    return np.sqrt(np.mean(centered**2, axis=1))


def test_nested_masks_are_nested_and_deterministic():
    x = make_signal()
    masks = [
        set(mask_leads_nested(x, ecg_id=42, number_of_leads=k)[1].tolist())
        for k in (1, 3, 6)
    ]
    assert len(masks[0]) == 1 and len(masks[1]) == 3 and len(masks[2]) == 6
    assert masks[0] <= masks[1] <= masks[2]

    first, _ = mask_leads_nested(x, ecg_id=42, number_of_leads=3)
    second, _ = mask_leads_nested(x, ecg_id=42, number_of_leads=3)
    np.testing.assert_array_equal(first, second)


def test_nested_masking_zeroes_selected_leads_only():
    x = make_signal()
    corrupted, selected = mask_leads_nested(x, ecg_id=5, number_of_leads=6)
    assert np.all(corrupted[selected] == 0.0)
    keep = np.setdiff1d(np.arange(12), selected)
    np.testing.assert_array_equal(corrupted[keep], x[keep])


def test_frozen_lead_masking_unchanged_by_new_codes():
    x = make_signal()
    _, selected = mask_leads(x, ecg_id=42, number_of_leads=3, severity_index=1)
    _, again = mask_leads(x, ecg_id=42, number_of_leads=3, severity_index=1)
    np.testing.assert_array_equal(selected, again)


@pytest.mark.parametrize("snr_db", [18.0, 12.0, 6.0])
def test_recorded_noise_hits_target_snr(snr_db):
    x = make_signal()
    y = add_recorded_noise(
        x,
        ecg_id=11,
        noise=make_noise(),
        noise_name="em",
        snr_db=snr_db,
        severity_index=0,
    )
    noise = y - x
    achieved = 20.0 * np.log10(ac_rms(x) / ac_rms(noise))
    np.testing.assert_allclose(achieved, snr_db, atol=1e-9)


def test_recorded_noise_shares_channel_within_lead_groups():
    x = make_signal()
    y = add_recorded_noise(
        x, ecg_id=3, noise=make_noise(), noise_name="ma",
        snr_db=12.0, severity_index=1,
    )
    noise = y - x
    normalized = noise / ac_rms(noise)[:, None]
    np.testing.assert_allclose(normalized[0], normalized[5], atol=1e-12)
    np.testing.assert_allclose(normalized[6], normalized[11], atol=1e-12)
    assert not np.allclose(normalized[0], normalized[6])


def test_recorded_noise_keeps_flat_leads_flat():
    x = make_signal()
    x[2] = 0.0
    y = add_recorded_noise(
        x, ecg_id=3, noise=make_noise(), noise_name="bw",
        snr_db=6.0, severity_index=2,
    )
    assert np.all(y[2] == 0.0)


def test_recorded_noise_rejects_short_noise():
    with pytest.raises(ValueError):
        add_recorded_noise(
            make_signal(), ecg_id=3, noise=np.ones((2, 500)),
            noise_name="bw", snr_db=6.0, severity_index=0,
        )
