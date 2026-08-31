"""Reproducibility and file-integrity utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Files are streamed in fixed-size blocks so large files do not need to be
    loaded fully into memory.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()
