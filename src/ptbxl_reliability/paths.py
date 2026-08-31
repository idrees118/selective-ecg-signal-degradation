"""Canonical project paths.

Paths are resolved relative to this installed source package, never from the
current working directory. This prevents experiments from silently reading a
different dataset when invoked from another directory.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

PTBXL_VERSION = "1.0.3"
PTBXL_ROOT = RAW_DATA_DIR / "ptb-xl" / PTBXL_VERSION

PTBXL_DATABASE_CSV = PTBXL_ROOT / "ptbxl_database.csv"
SCP_STATEMENTS_CSV = PTBXL_ROOT / "scp_statements.csv"
RECORDS100_DIR = PTBXL_ROOT / "records100"

LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"


def validate_required_paths() -> None:
    """Fail immediately if required PTB-XL inputs are unavailable."""

    required = (
        PTBXL_ROOT,
        PTBXL_DATABASE_CSV,
        SCP_STATEMENTS_CSV,
        RECORDS100_DIR,
    )

    missing = [path for path in required if not path.exists()]

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required PTB-XL files/directories are missing:\n"
            f"{formatted}"
        )
