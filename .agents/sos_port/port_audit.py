"""Shared inputs, provenance and failure handling for the `nfl-sos-ratings` port audit.

Both audit scripts in this directory (``inventory.py`` and ``parity.py``) read the same three
things, read-only, and record where each came from:

- `nfl-sos-ratings` at its frozen commit: code through ``git show`` (never the working tree) and
  its generated Parquet outputs in its checkout's ``data/``, which must sit at that commit;
- this repository's data, from the main checkout (the first entry of ``git worktree list``), so a
  run from a linked worktree reads the same files as a run from the main checkout, or from the
  data directory ``--data-dir`` names (``parse_args``), such as a candidate build;
- this repository's code, from the checkout that runs the script.

Nothing here writes anywhere except the temporary copy of the frozen commit that ``SosCode``
extracts under ``~/scratch`` and removes again.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from nfl_predictor.constants import normalize_team_abbr
from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# The commit the port starts from (`.agents/sos_port_source.md`) and the commit its data was built
# from, recorded there as code-identical for every output; `check_sos_checkout` verifies both.
SOS_COMMIT = "9081bced89b0d51c99e28f80f8f62b9fcd0118e0"
SOS_BUILD_COMMIT = "4d23890"
# Paths whose content decides every output: the package, its dependencies and the lock file.
SOS_OUTPUT_CODE_PATHS = ("nfl_sos_ratings", "pyproject.toml", "uv.lock")
# The completed seasons its pipeline builds (its `config.START_YEAR` and `config.END_YEAR`).
FIRST_SEASON = 1999
LAST_SEASON = 2025
SEASONS = tuple(range(FIRST_SEASON, LAST_SEASON + 1))
SCRATCH = Path.home() / "scratch"
# nfl-sos-ratings reads play-by-play through nflreadpy's on-disk cache. This read-only snapshot of
# that cache (written 2026-10-08, the day before its data build) is the play-by-play it read; the
# sha256 manifest was made when the snapshot was taken.
NFLREADPY_SNAPSHOT = SCRATCH / "nflreadpy_cache_snapshot_20261008"
NFLREADPY_SNAPSHOT_MANIFEST = SCRATCH / "nflreadpy_cache_snapshot_20261008.sha256"


def _git(*args: str, cwd: Path | None = None) -> str:
    """Run a read-only git command and return its standard output.

    ``--no-optional-locks`` keeps ``git status`` from refreshing, and so rewriting, the index of
    the read-only repository it inspects.
    """
    return subprocess.run(  # fixed git argv built in this file
        ["git", "--no-optional-locks", *args],  # git from PATH
        cwd=cwd or ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main_checkout() -> Path:
    """Return the repository's main checkout: the first worktree ``git worktree list`` names."""
    for line in _git("worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            return Path(line.removeprefix("worktree ")).resolve()
    msg = "git worktree list named no worktree"
    raise RuntimeError(msg)


MAIN = main_checkout()
DATA = MAIN / "data"
SOS_REPO = (MAIN.parent / "nfl-sos-ratings").resolve()
SOS_DATA = SOS_REPO / "data"


def parse_args(description: str, argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the options both audit scripts take.

    ``--data-dir`` is the data directory of this repository to read (default: the main
    checkout's ``data/``), so a candidate build outside the main checkout can be audited; it must
    exist and is returned resolved.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA,
        help="this repository's data directory to read (default: the main checkout's data/)",
    )
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir.expanduser().resolve()
    if not args.data_dir.is_dir():
        parser.error(f"--data-dir {args.data_dir} is not a directory")
    return args


@dataclass
class Failure:
    """A check that must hold and did not."""

    what: str
    detail: str


@dataclass
class Audit:
    """Failures and provenance collected over one run."""

    failures: list[Failure] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)

    def fail(self, what: str, detail: str) -> None:
        """Record a failure; the run exits non-zero at the end."""
        log.error("FAILURE %s: %s", what, detail)
        self.failures.append(Failure(what, detail))

    def record(self, path: Path) -> Path:
        """Record ``path``'s sha256 as an input of this run and return it."""
        key = portable(str(path))
        if key not in self.files:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            self.files[key] = digest.hexdigest()
        return path


def portable(text: str) -> str:
    """Replace machine-specific paths so reports are identical across checkouts and machines."""
    text = text.replace(str(SOS_REPO), "<nfl-sos-ratings>")
    text = text.replace(str(MAIN), "<main>")
    text = text.replace(str(ROOT), "<repo>")
    return text.replace(str(Path.home()), "~")


def check_sos_checkout(audit: Audit) -> dict[str, object]:
    """Verify the frozen commit, the checkout's HEAD and the build commit's code identity."""
    head = _git("rev-parse", "HEAD", cwd=SOS_REPO).strip()
    if head != SOS_COMMIT:
        audit.fail("sos checkout", f"HEAD is {head}, not the frozen commit {SOS_COMMIT}")
    dirty = _git("status", "--porcelain", "--untracked-files=no", cwd=SOS_REPO).strip()
    if dirty:
        audit.fail("sos checkout", f"tracked files differ from HEAD: {dirty.splitlines()[:5]}")
    changed = _git(
        "diff",
        "--name-only",
        SOS_BUILD_COMMIT,
        SOS_COMMIT,
        "--",
        *SOS_OUTPUT_CODE_PATHS,
        cwd=SOS_REPO,
    ).split()
    if changed:
        audit.fail(
            "sos build commit",
            f"{SOS_BUILD_COMMIT} and {SOS_COMMIT} differ in output code: {changed[:10]}",
        )
    return {
        "frozen_commit": SOS_COMMIT,
        "checkout_head": head,
        "tracked_changes": dirty.splitlines(),
        "data_build_commit": SOS_BUILD_COMMIT,
        "output_code_changed_since_build": changed,
    }


class SosCode:
    """Read `nfl-sos-ratings` source at the frozen commit through git, never its working tree."""

    def __init__(self) -> None:
        """List the frozen commit's tracked files."""
        self.paths = _git("ls-tree", "-r", "--name-only", SOS_COMMIT, cwd=SOS_REPO).split()
        self._extracted: Path | None = None

    def show(self, path: str) -> str:
        """Return one file's text at the frozen commit."""
        return _git("show", f"{SOS_COMMIT}:{path}", cwd=SOS_REPO)

    def blob(self, path: str) -> str:
        """Return one file's git blob id at the frozen commit (its content hash)."""
        return _git("rev-parse", f"{SOS_COMMIT}:{path}", cwd=SOS_REPO).strip()

    def glob(self, pattern: str) -> list[str]:
        """Return the tracked paths matching a regular expression, sorted."""
        return sorted(path for path in self.paths if re.fullmatch(pattern, path))

    def extract(self, prefixes: Iterable[str]) -> Path:
        """Extract the files under ``prefixes`` into a new directory under ``~/scratch``."""
        if self._extracted is None:
            SCRATCH.mkdir(parents=True, exist_ok=True)
            target = Path(tempfile.mkdtemp(prefix="sos_port_", dir=SCRATCH))
            archive = subprocess.run(  # fixed git argv built in this file
                ["git", "--no-optional-locks", "archive", SOS_COMMIT, *prefixes],  # git from PATH
                cwd=SOS_REPO,
                check=True,
                capture_output=True,
            ).stdout
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(target, filter="data")
            self._extracted = target
        return self._extracted

    def cleanup(self) -> None:
        """Remove the extracted copy, if any."""
        if self._extracted is not None:
            shutil.rmtree(self._extracted, ignore_errors=True)
            self._extracted = None


def sos_output(season: int, kind: str) -> Path:
    """Return the path of one `nfl-sos-ratings` output file."""
    return SOS_DATA / f"{season}_{kind}.parquet"


def team_code_map(codes: Iterable[str]) -> dict[str, str]:
    """Map `nfl-sos-ratings` team codes to this repository's canonical codes.

    Both repositories normalize team codes, to different canonical spellings (Washington is
    ``WAS`` there and ``WSH`` here); this repository's own alias table decides.
    """
    return {code: normalize_team_abbr(code) for code in sorted(set(codes))}
