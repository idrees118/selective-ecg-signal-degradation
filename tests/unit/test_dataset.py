import pytest

from ptbxl_reliability.dataset import (
    resolve_record_base,
    validate_split_name,
)
from ptbxl_reliability.paths import RECORDS100_DIR


@pytest.mark.parametrize(
    "split",
    [
        "train",
        "validation",
        "test",
    ],
)
def test_valid_split_names(split):
    assert validate_split_name(
        split
    ) == split


@pytest.mark.parametrize(
    "split",
    [
        "val",
        "testing",
        "fold10",
        "",
    ],
)
def test_invalid_split_names_are_rejected(split):
    with pytest.raises(ValueError):
        validate_split_name(
            split
        )


def test_record_path_resolves_inside_records100():
    path = resolve_record_base(
        "records100/00000/00001_lr"
    )

    assert path.is_relative_to(
        RECORDS100_DIR.resolve()
    )


def test_path_traversal_is_rejected():
    with pytest.raises(ValueError):
        resolve_record_base(
            "records100/../../secret"
        )
