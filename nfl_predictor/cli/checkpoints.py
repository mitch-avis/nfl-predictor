"""List walk-forward checkpoint directories and which runs reference them (read-only).

Walk-forward runs save every finished week under ``models/wf_checkpoints/<fingerprint>/`` so a
stopped run resumes. Nothing prunes that directory. This command lists each checkpoint
directory with its fold count, size and age, and whether anything names it: a file under
``models/`` outside the checkpoint root (a run's ``metrics_report.json``, a review, a launcher or
a log), or the project's documentation (the top-level Markdown files and ``.agents/``). It never
deletes anything: removing checkpoints is a decision for the user.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nfl_predictor import constants
from nfl_predictor.utils.logger import log

# Files that can name a checkpoint directory: reports, reviews, launchers, logs and scripts.
REFERENCE_SUFFIXES = frozenset({".json", ".md", ".txt", ".sh", ".log", ".py", ".csv", ".yaml"})
BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class CheckpointDir:
    """One checkpoint directory and the files that reference it."""

    name: str
    folds: int
    size_bytes: int
    newest: datetime | None
    referenced_by: tuple[Path, ...]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    models = Path(constants.ROOT_DIR) / "models"
    parser = argparse.ArgumentParser(
        description="List walk-forward checkpoint directories and what references them."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=models / "wf_checkpoints",
        help="Checkpoint root to list (default: models/wf_checkpoints).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=models,
        help="Directory searched for references (default: models).",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path(constants.ROOT_DIR),
        help="Repository root whose Markdown files and .agents/ are also searched.",
    )
    parser.add_argument(
        "--unreferenced-only",
        action="store_true",
        help="List only the directories that nothing references.",
    )
    return parser.parse_args(argv)


def _reference_files(models_dir: Path, checkpoint_dir: Path, docs_dir: Path | None) -> list[Path]:
    """Return the files that can name a checkpoint directory.

    That is every text file under ``models_dir`` outside ``checkpoint_dir``, plus, when
    ``docs_dir`` is given, its top-level Markdown files and the Markdown under ``.agents/``
    (except the separately cloned ``skills``).
    """
    root = checkpoint_dir.resolve()
    files = {
        path
        for path in models_dir.rglob("*")
        if path.is_file()
        and path.suffix in REFERENCE_SUFFIXES
        and root not in path.resolve().parents
    }
    if docs_dir is not None:
        files.update(docs_dir.glob("*.md"))
        agents = docs_dir / ".agents"
        if agents.is_dir():
            files.update(
                path
                for path in agents.rglob("*.md")
                if "skills" not in path.relative_to(agents).parts
            )
    return sorted(files)


def scan_checkpoints(
    checkpoint_dir: Path, models_dir: Path, docs_dir: Path | None = None
) -> list[CheckpointDir]:
    """Return every checkpoint directory with its size, fold count and references."""
    directories = sorted(path for path in checkpoint_dir.iterdir() if path.is_dir())
    if not directories:
        return []
    # Longest names first: one directory's name can extend another's (a sliced copy such as
    # "<fingerprint>_2023_2025"), and the alternation must credit the longer mention to it.
    names = sorted((path.name for path in directories), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(name) for name in names))
    references: dict[str, set[Path]] = {path.name: set() for path in directories}
    for file in _reference_files(models_dir, checkpoint_dir, docs_dir):
        text = file.read_text(encoding="utf-8", errors="ignore")
        for match in set(pattern.findall(text)):
            references[match].add(file)

    rows = []
    for directory in directories:
        files = [path for path in directory.rglob("*") if path.is_file()]
        newest = max((path.stat().st_mtime for path in files), default=None)
        rows.append(
            CheckpointDir(
                name=directory.name,
                folds=sum(1 for path in files if path.name.startswith("fold_")),
                size_bytes=sum(path.stat().st_size for path in files),
                newest=None if newest is None else datetime.fromtimestamp(newest, tz=UTC),
                referenced_by=tuple(sorted(references[directory.name])),
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Log the checkpoint listing; return 2 when the checkpoint root does not exist."""
    args = _parse_args(argv)
    if not args.checkpoint_dir.is_dir():
        log.error("No checkpoint directory at %s", args.checkpoint_dir)
        return 2

    rows = scan_checkpoints(args.checkpoint_dir, args.models_dir, args.docs_dir)
    unreferenced = [row for row in rows if not row.referenced_by]
    for row in unreferenced if args.unreferenced_only else rows:
        first = str(row.referenced_by[0]) if row.referenced_by else "-"
        log.info(
            "%s  folds=%3d  %7.1f MB  newest=%s  references=%d  %s",
            row.name,
            row.folds,
            row.size_bytes / BYTES_PER_MB,
            row.newest.strftime("%Y-%m-%d") if row.newest else "-",
            len(row.referenced_by),
            first,
        )
    log.info(
        "%d checkpoint directories (%.1f MB); %d unreferenced (%.1f MB). Nothing was deleted.",
        len(rows),
        sum(row.size_bytes for row in rows) / BYTES_PER_MB,
        len(unreferenced),
        sum(row.size_bytes for row in unreferenced) / BYTES_PER_MB,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
