"""The frozen ``nfl-sos-ratings`` data the ratings port is checked against, and how to read it.

Shared by ``ratings_fixtures.py`` and ``ratings_parity.py`` beside it. That project is read-only:
these helpers only read its published Parquet files and its git metadata (which commit its
checkout holds), without running git.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# The frozen commit the port starts from (.agents/sos_port_source.md).
SOURCE_COMMIT = "9081bced89b0d51c99e28f80f8f62b9fcd0118e0"
_GITDIR_PREFIX = "gitdir: "
_REF_PREFIX = "ref: "


def repo_root() -> Path:
    """Return the root of the checkout this script runs from (a worktree's own root)."""
    return Path(__file__).resolve().parents[2]


def _git_dir(root: Path) -> Path:
    """Return a checkout's git directory: ``.git`` itself, or the one a worktree's file names."""
    dot_git = root / ".git"
    if dot_git.is_dir():
        return dot_git
    pointer = dot_git.read_text().strip()
    if not pointer.startswith(_GITDIR_PREFIX):
        message = f"{dot_git} is neither a git directory nor a gitdir pointer"
        raise SystemExit(message)
    return (root / pointer.removeprefix(_GITDIR_PREFIX)).resolve()


def _common_dir(git_dir: Path) -> Path:
    """Return the repository's common git directory (a worktree's ``commondir``, else itself)."""
    pointer = git_dir / "commondir"
    if pointer.exists():
        return (git_dir / pointer.read_text().strip()).resolve()
    return git_dir


def _packed_ref(common: Path, ref: str) -> str | None:
    """Return the commit a ref has in ``packed-refs``, or ``None`` if it is not there."""
    packed = common / "packed-refs"
    if not packed.exists():
        return None
    for line in packed.read_text().splitlines():
        if line.startswith(("#", "^")):
            continue
        commit, _, name = line.partition(" ")
        if name == ref:
            return commit
    return None


def source_head(root: Path) -> str:
    """Return the commit the checkout at ``root`` holds, read from its git metadata.

    Raises:
        SystemExit: If HEAD names a ref that neither a ref file nor ``packed-refs`` holds.

    """
    git_dir = _git_dir(root)
    head = (git_dir / "HEAD").read_text().strip()
    if not head.startswith(_REF_PREFIX):
        return head
    ref = head.removeprefix(_REF_PREFIX)
    common = _common_dir(git_dir)
    for candidate in (git_dir / ref, common / ref):
        if candidate.exists():
            return candidate.read_text().strip()
    commit = _packed_ref(common, ref)
    if commit is None:
        message = f"{root}: HEAD names {ref}, which no ref file holds"
        raise SystemExit(message)
    return commit


def default_source() -> Path:
    """Return the ``nfl-sos-ratings`` checkout beside this repository's main checkout."""
    return _common_dir(_git_dir(repo_root())).parent.parent / "nfl-sos-ratings"


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


def package_hashes(root: Path, package: str) -> dict[str, str]:
    """Return the SHA-256 of every Python file of ``package``, keyed by its path under ``root``."""
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted((root / package).glob("*.py"))
    }


def published_file(source: Path, season: int, name: str) -> Path:
    """Return the path of one published ``{season}_{name}.parquet`` file."""
    return source / "data" / f"{season}_{name}.parquet"
