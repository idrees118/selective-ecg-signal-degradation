import pandas as pd
import pytest

from ptbxl_reliability.metadata import (
    parse_scp_codes,
    validate_patient_fold_integrity,
)


def test_parse_scp_codes_preserves_values():
    raw = "{'NORM': 100.0, 'SR': 0.0, 'LVH': 50}"

    result = parse_scp_codes(raw)

    assert result == {
        "NORM": 100.0,
        "SR": 0.0,
        "LVH": 50.0,
    }


def test_parse_scp_codes_zero_is_not_discarded():
    result = parse_scp_codes("{'SR': 0}")

    assert "SR" in result
    assert result["SR"] == 0.0


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "'NORM'",
        "{'NORM': -1}",
        "{'NORM': 101}",
        "{'NORM': float('nan')}",
    ],
)
def test_parse_scp_codes_rejects_invalid_values(raw):
    with pytest.raises((TypeError, ValueError)):
        parse_scp_codes(raw)


def test_patient_fold_integrity_accepts_patient_safe_split():
    frame = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 3, 3],
            "strat_fold": [1, 1, 9, 10, 10],
        }
    )

    validate_patient_fold_integrity(frame)


def test_patient_fold_integrity_detects_leakage():
    frame = pd.DataFrame(
        {
            "patient_id": [1, 1, 2],
            "strat_fold": [1, 10, 2],
        }
    )

    with pytest.raises(ValueError, match="Patient leakage"):
        validate_patient_fold_integrity(frame)
