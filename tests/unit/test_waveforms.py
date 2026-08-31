import numpy as np
import pytest

from ptbxl_reliability.waveforms import to_model_layout


def test_to_model_layout_transposes_only():
    signal = np.arange(
        1000 * 12,
        dtype=np.float64,
    ).reshape(1000, 12)

    result = to_model_layout(signal)

    assert result.shape == (12, 1000)
    np.testing.assert_array_equal(
        result,
        signal.T,
    )


def test_to_model_layout_rejects_wrong_shape():
    signal = np.zeros((12, 1000))

    with pytest.raises(ValueError):
        to_model_layout(signal)
