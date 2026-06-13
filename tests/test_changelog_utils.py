"""Tests for extracting release notes from the Common Changelog file."""

from __future__ import annotations

from pathlib import Path

import pytest

from nfl_predictor.utils.changelog import ReleaseNotes, extract_release_notes


def _write_changelog(tmp_path: Path, text: str) -> Path:
    """Write a temporary changelog file and return its path."""
    changelog_path = tmp_path / "CHANGELOG.md"
    changelog_path.write_text(text, encoding="utf-8")
    return changelog_path


def test_extract_release_notes_matches_plain_semver_tag(tmp_path: Path) -> None:
    """A plain semantic-version tag should resolve to the matching changelog section."""
    changelog_path = _write_changelog(
        tmp_path,
        """# Changelog

## [0.2.5] - 2026-06-13

### Changed

- Align release workflow docs.

### Fixed

- Keep validation green.

## [0.2.4] - 2026-06-12

### Changed

- Earlier release.
""",
    )

    release_notes = extract_release_notes(changelog_path, "0.2.5")

    assert release_notes == ReleaseNotes(
        version="0.2.5",
        date="2026-06-13",
        body=(
            "### Changed\n\n- Align release workflow docs.\n\n### Fixed\n\n- Keep validation green."
        ),
    )


def test_extract_release_notes_accepts_v_prefixed_tag(tmp_path: Path) -> None:
    """A leading `v` in the tag should be ignored for changelog lookup."""
    changelog_path = _write_changelog(
        tmp_path,
        """# Changelog

## [0.2.5] - 2026-06-13

### Added

- Introduce release automation.
""",
    )

    release_notes = extract_release_notes(changelog_path, "v0.2.5")

    assert release_notes.version == "0.2.5"
    assert release_notes.date == "2026-06-13"
    assert release_notes.body == "### Added\n\n- Introduce release automation."


def test_extract_release_notes_accepts_full_tag_ref(tmp_path: Path) -> None:
    """A full Git ref name should normalize to the changelog version number."""
    changelog_path = _write_changelog(
        tmp_path,
        """# Changelog

## [0.2.5] - 2026-06-13

### Added

- Introduce release automation.
""",
    )

    release_notes = extract_release_notes(changelog_path, "refs/tags/v0.2.5")

    assert release_notes.version == "0.2.5"
    assert release_notes.date == "2026-06-13"
    assert release_notes.body == "### Added\n\n- Introduce release automation."


def test_extract_release_notes_raises_for_missing_version(tmp_path: Path) -> None:
    """Missing changelog sections should fail fast with a helpful error."""
    changelog_path = _write_changelog(
        tmp_path,
        """# Changelog

## [0.2.4] - 2026-06-12

### Changed

- Earlier release.
""",
    )

    with pytest.raises(ValueError, match="No changelog entry found for version 0\\.2\\.5"):
        extract_release_notes(changelog_path, "0.2.5")


def test_extract_release_notes_raises_for_empty_release_body(tmp_path: Path) -> None:
    """A matching version without any release notes body should fail fast."""
    changelog_path = _write_changelog(
        tmp_path,
        """# Changelog

## [0.2.5] - 2026-06-13

## [0.2.4] - 2026-06-12

### Changed

- Earlier release.
""",
    )

    with pytest.raises(
        ValueError,
        match="Changelog entry for version 0\\.2\\.5 has no release notes body",
    ):
        extract_release_notes(changelog_path, "0.2.5")
