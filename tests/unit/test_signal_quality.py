import numpy as np
import pytest

from ptbxl_reliability.signal_quality import (
    QUALITY_FIELDS,
    metadata_annotation_present,
    signal_quality_from_presence,
)


def test_missing_metadata_is_not_present():
    assert metadata_annotation_present(None) is False
    assert metadata_annotation_present(np.nan) is False
    assert metadata_annotation_present("") is False
    assert metadata_annotation_present("nan") is False


def test_free_text_metadata_is_present():
    assert metadata_annotation_present(", I-AVR,") is True
    assert metadata_annotation_present("alles") is True
    assert metadata_annotation_present("V6") is True


def test_no_artifacts_gives_quality_one():
    presence = {
        field: False
        for field in QUALITY_FIELDS
    }

    result = signal_quality_from_presence(
        presence
    )

    assert result[
        "n_quality_artifact_categories"
    ] == 0

    assert result[
        "artifact_burden"
    ] == pytest.approx(0.0)

    assert result[
        "signal_quality"
    ] == pytest.approx(1.0)

    assert result[
        "high_quality_no_recorded_artifact"
    ] is True


def test_one_artifact_category_gives_quality_three_quarters():
    presence = {
        field: False
        for field in QUALITY_FIELDS
    }

    presence["static_noise"] = True

    result = signal_quality_from_presence(
        presence
    )

    assert result[
        "artifact_burden"
    ] == pytest.approx(0.25)

    assert result[
        "signal_quality"
    ] == pytest.approx(0.75)

    assert result[
        "any_quality_artifact"
    ] is True


def test_all_four_categories_give_quality_zero():
    presence = {
        field: True
        for field in QUALITY_FIELDS
    }

    result = signal_quality_from_presence(
        presence
    )

    assert result[
        "n_quality_artifact_categories"
    ] == 4

    assert result[
        "artifact_burden"
    ] == pytest.approx(1.0)

    assert result[
        "signal_quality"
    ] == pytest.approx(0.0)


def test_wrong_field_set_rejected():
    with pytest.raises(ValueError):
        signal_quality_from_presence(
            {
                "baseline_drift": False,
            }
        )


def test_nonboolean_presence_rejected():
    presence = {
        field: False
        for field in QUALITY_FIELDS
    }

    presence["burst_noise"] = 1

    with pytest.raises(TypeError):
        signal_quality_from_presence(
            presence
        )
