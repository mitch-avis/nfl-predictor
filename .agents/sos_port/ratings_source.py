"""The frozen ``nfl-sos-ratings`` data the ratings port is checked against, and how to read it.

Shared by ``ratings_fixtures.py`` and ``ratings_parity.py`` beside it. That project is read-only:
these helpers only read its published Parquet files and ask git which commit its checkout holds.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

# The frozen commit the port starts from (.agents/sos_port_source.md).
SOURCE_COMMIT = "9081bced89b0d51c99e28f80f8f62b9fcd0118e0"


def _git(*args: str) -> str:
    """Return the stripped output of one read-only git command."""
    git = shutil.which("git")
    if git is None:
        message = "git is not on PATH"
        raise SystemExit(message)
    return subprocess.run(  # noqa: S603  # fixed argv: git plus this module's own arguments
        [git, *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def repo_root() -> Path:
    """Return the root of the checkout this script runs from (a worktree's own root)."""
    return Path(__file__).resolve().parents[2]


def default_source() -> Path:
    """Return the ``nfl-sos-ratings`` checkout beside this repository's main checkout."""
    common = _git("-C", str(repo_root()), "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(common).parent.parent / "nfl-sos-ratings"


def source_head(source: Path) -> str:
    """Return the commit the ``nfl-sos-ratings`` checkout at ``source`` holds."""
    return _git("-C", str(source), "rev-parse", "HEAD")


def check_source(source: Path) -> str:
    """Return the source checkout's HEAD after asserting it is the frozen commit.

    Raises:
        SystemExit: If the checkout is at another commit.

    """
    head = source_head(source)
    if head != SOURCE_COMMIT:
        message = f"{source} is at {head}, not the frozen commit {SOURCE_COMMIT}"
        raise SystemExit(message)
    return head


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def published_file(source: Path, season: int, name: str) -> Path:
    """Return the path of one published ``{season}_{name}.parquet`` file."""
    return source / "data" / f"{season}_{name}.parquet"
