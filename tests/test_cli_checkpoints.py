"""Tests for the read-only checkpoint listing."""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_predictor.cli import checkpoints


def _layout(tmp_path: Path) -> tuple[Path, Path]:
    """Build a models tree with one referenced and one unreferenced checkpoint directory."""
    models = tmp_path / "models"
    root = models / "wf_checkpoints"
    for name, folds in (("aaaaaaaaaaaaaaaaaaaa", 2), ("bbbbbbbbbbbbbbbbbbbb", 1)):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "run.json").write_text("{}", encoding="utf-8")
        for index in range(folds):
            (directory / f"fold_2024_w{index + 1:02d}.joblib").write_bytes(b"x" * 100)
    run = models / "wf_run"
    run.mkdir()
    (run / "metrics_report.json").write_text(
        f'{{"checkpoint": {{"dir": "{root / "aaaaaaaaaaaaaaaaaaaa"}"}}}}', encoding="utf-8"
    )
    (run / "model.joblib").write_bytes(b"bbbbbbbbbbbbbbbbbbbb")
    return models, root


def test_scan_reports_folds_sizes_and_references(tmp_path: Path) -> None:
    """Only text files outside the checkpoint root count as references."""
    models, root = _layout(tmp_path)

    rows = {row.name: row for row in checkpoints.scan_checkpoints(root, models)}

    referenced = rows["aaaaaaaaaaaaaaaaaaaa"]
    assert referenced.folds == 2
    assert referenced.size_bytes == 202
    assert referenced.referenced_by == (models / "wf_run" / "metrics_report.json",)
    assert referenced.newest is not None
    assert rows["bbbbbbbbbbbbbbbbbbbb"].referenced_by == ()


def test_main_lists_without_deleting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The listing logs every directory and a summary, and leaves every file in place."""
    models, root = _layout(tmp_path)
    before = sorted(path for path in root.rglob("*"))
    messages: list[str] = []
    monkeypatch.setattr(
        checkpoints.log, "info", lambda message, *args: messages.append(message % args)
    )

    code = checkpoints.main(
        [
            "--checkpoint-dir",
            str(root),
            "--models-dir",
            str(models),
            "--docs-dir",
            str(tmp_path),
            "--unreferenced-only",
        ]
    )

    assert code == 0
    assert sorted(path for path in root.rglob("*")) == before
    assert len(messages) == 2
    assert messages[0].startswith("bbbbbbbbbbbbbbbbbbbb")
    assert "2 checkpoint directories" in messages[1]
    assert "1 unreferenced" in messages[1]


def test_main_reports_a_missing_root(tmp_path: Path) -> None:
    """A checkpoint root that does not exist exits with status 2."""
    assert checkpoints.main(["--checkpoint-dir", str(tmp_path / "missing")]) == 2


def test_an_empty_root_lists_nothing(tmp_path: Path) -> None:
    """An empty checkpoint root gives an empty listing."""
    root = tmp_path / "wf_checkpoints"
    root.mkdir()

    assert checkpoints.scan_checkpoints(root, tmp_path) == []


def test_project_docs_count_as_references(tmp_path: Path) -> None:
    """A checkpoint named only in AGENTS.md or .agents/ is referenced; skills are skipped."""
    models, root = _layout(tmp_path)
    (tmp_path / "AGENTS.md").write_text("slice bbbbbbbbbbbbbbbbbbbb", encoding="utf-8")
    skills = tmp_path / ".agents" / "skills"
    skills.mkdir(parents=True)
    (skills / "note.md").write_text("aaaaaaaaaaaaaaaaaaaa", encoding="utf-8")

    rows = {row.name: row for row in checkpoints.scan_checkpoints(root, models, tmp_path)}

    assert rows["bbbbbbbbbbbbbbbbbbbb"].referenced_by == (tmp_path / "AGENTS.md",)
    assert skills / "note.md" not in rows["aaaaaaaaaaaaaaaaaaaa"].referenced_by


def test_a_name_that_extends_another_is_credited_to_the_longer_directory(tmp_path: Path) -> None:
    """A mention of "<fingerprint>_slice" references that directory, not the bare fingerprint."""
    models = tmp_path / "models"
    root = models / "wf_checkpoints"
    for name in ("cccccccccccccccccccc", "cccccccccccccccccccc_2023_2025"):
        (root / name).mkdir(parents=True)
    note = models / "review" / "REVIEW.md"
    note.parent.mkdir()
    note.write_text("slice at wf_checkpoints/cccccccccccccccccccc_2023_2025/", encoding="utf-8")

    rows = {row.name: row for row in checkpoints.scan_checkpoints(root, models)}

    assert rows["cccccccccccccccccccc_2023_2025"].referenced_by == (note,)
    assert rows["cccccccccccccccccccc"].referenced_by == ()
