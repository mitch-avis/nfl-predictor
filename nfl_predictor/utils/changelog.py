"""Helpers for extracting release notes from the repository changelog."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_RELEASE_HEADER_PATTERN = re.compile(
    r"^## \[(?P<version>[^\]]+)\] - (?P<date>\d{4}-\d{2}-\d{2})\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ReleaseNotes:
    """Release metadata and markdown body for a single changelog section."""

    version: str
    date: str
    body: str


def normalize_release_version(tag_name: str) -> str:
    """Normalize a tag or ref name to the semantic version used in `CHANGELOG.md`."""
    normalized = tag_name.strip()
    if normalized.startswith("refs/tags/"):
        normalized = normalized.removeprefix("refs/tags/")
    return normalized.removeprefix("v")


def extract_release_notes(changelog_path: Path, tag_name: str) -> ReleaseNotes:
    """Return the changelog section that matches a release tag.

    Args:
        changelog_path: Path to the repository `CHANGELOG.md` file.
        tag_name: Release tag or ref name such as `0.2.5`, `v0.2.5`, or
            `refs/tags/v0.2.5`.

    Raises:
        ValueError: If the changelog has no matching section or the matched section is empty.

    """
    changelog_text = changelog_path.read_text(encoding="utf-8")
    version = normalize_release_version(tag_name)
    matches = list(_RELEASE_HEADER_PATTERN.finditer(changelog_text))

    for index, match in enumerate(matches):
        if match.group("version") != version:
            continue

        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(changelog_text)
        body = changelog_text[match.end() : next_start].strip()
        if not body:
            raise ValueError(f"Changelog entry for version {version} has no release notes body")

        return ReleaseNotes(version=version, date=match.group("date"), body=body)

    raise ValueError(f"No changelog entry found for version {version}")


__all__ = ["ReleaseNotes", "extract_release_notes", "normalize_release_version"]
