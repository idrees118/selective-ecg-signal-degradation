import pytest

from ptbxl_reliability.metadata import (
    EXPECTED_PATIENT_COUNT,
    EXPECTED_RECORD_COUNT,
    load_ptbxl_database,
    load_scp_statements,
    validate_patient_fold_integrity,
    validate_scp_code_coverage,
)
from ptbxl_reliability.paths import (
    PTBXL_DATABASE_CSV,
    PTBXL_ROOT,
    SCP_STATEMENTS_CSV,
)
from ptbxl_reliability.waveforms import (
    EXPECTED_LEADS,
    load_ecg_100hz,
    to_model_layout,
)


pytestmark = pytest.mark.integration


def test_ptbxl_metadata_contract():
    database = load_ptbxl_database(
        PTBXL_DATABASE_CSV
    )
    statements = load_scp_statements(
        SCP_STATEMENTS_CSV
    )

    assert len(database) == EXPECTED_RECORD_COUNT

    assert (
        database["patient_id"].nunique()
        == EXPECTED_PATIENT_COUNT
    )

    validate_patient_fold_integrity(database)

    observed_codes = validate_scp_code_coverage(
        database,
        statements,
    )

    assert observed_codes


def test_first_ptbxl_waveform_contract():
    database = load_ptbxl_database(
        PTBXL_DATABASE_CSV
    )

    row = database.sort_values("ecg_id").iloc[0]

    record = load_ecg_100hz(
        PTBXL_ROOT / str(row["filename_lr"])
    )

    assert record.signal.shape == (1000, 12)
    assert record.sampling_rate_hz == 100.0
    assert record.lead_names == EXPECTED_LEADS

    model_signal = to_model_layout(record.signal)

    assert model_signal.shape == (12, 1000)
