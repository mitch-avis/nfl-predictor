"""Keep inline lint and type-checker suppressions out of the tracked Python and stub files.

The repository fixes what a linter or type checker flags; a genuine false positive goes in
``[tool.ruff.lint.per-file-ignores]`` in ``pyproject.toml`` with its reason. These tests scan the
comments of every tracked Python and stub file for the inline forms that would silence a finding.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tokenize
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Matches a directive that silences a finding anywhere in a comment, including a second ``#``
# after a reason: line and file-level noqa, the type-checker ignores, pyright's basic mode or a
# pyright rule switched off, and the coverage pragmas. Directives that tighten do not match.
SUPPRESSION = re.compile(
    r"#\s*(?:"
    r"(?:ruff:\s*|flake8:\s*)?noqa\b"
    r"|(?:type|pyright|ty):\s*ignore\b"
    r"|pyright:[^#]*\bbasic\b"
    r"|pyright:[^#]*\breport\w+\s*=\s*(?:false|none)\b"
    r"|pragma:\s*no\s*(?:cover|branch)\b"
    r")",
    re.IGNORECASE,
)


def suppression_hits(source: str, path: str) -> list[str]:
    """Return ``path:line`` with the comment for every suppression comment in ``source``."""
    return [
        f"{path}:{token.start[0]}: {token.string.strip()}"
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and SUPPRESSION.search(token.string)
    ]


def tracked_python_files() -> list[Path]:
    """Return the tracked Python and stub files, skipping the test when git cannot list them."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not available to list the tracked Python files")
    listing = subprocess.run(
        [git, "-C", str(REPO_ROOT), "ls-files", "-z", "*.py", "*.pyi"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:
        pytest.skip(f"git ls-files failed: {listing.stderr.strip()}")
    return [REPO_ROOT / name for name in listing.stdout.split("\0") if name]


class NotACheckoutError(Exception):
    """Raised when the directory to scan is not a git checkout."""


def tracked_sources(root: Path) -> list[Path]:
    """Return the tracked Python and stub files under ``root``."""
    listing = subprocess.run(
        ["/usr/bin/git", "-C", str(root), "ls-files", "-z", "*.py", "*.pyi"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:
        raise RuntimeError(listing.stderr)
    return [root / name for name in listing.stdout.split("\0") if name]


def test_tracked_python_files_carry_no_inline_suppressions() -> None:
    # Arrange
    files = tracked_python_files()

    # Act
    hits = [
        hit
        for path in files
        for hit in suppression_hits(
            path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT))
        )
    ]

    # Assert
    assert files
    assert hits == [], "fix each finding or give it a per-file ignore:\n" + "\n".join(hits)


@pytest.mark.parametrize(
    "comment",
    [
        "# noqa",
        "# noqa: S603",
        "# NOQA: E501",
        "# type: ignore",
        "# type: ignore[arg-type]",
        "# pyright: ignore[reportPrivateUsage]",
        "# ty: ignore[unresolved-import]",
        "# pragma: no cover",
        "# fixed argv  # noqa: S603",
        "# ruff: noqa",
        "# ruff: noqa: E501, S603",
        "# flake8: noqa",
        "# pyright: basic",
        "# pyright: reportPrivateUsage=false",
        "# pyright: reportPrivateUsage = False",
        "# pyright: strict, reportUnknownMemberType=false",
        "# pyright: reportPrivateUsage=none",
        "# pragma: no branch",
    ],
)
def test_suppression_hits_names_the_file_and_line_of_each_directive(comment: str) -> None:
    # Arrange
    source = f"import os\nos.getcwd()  {comment}\n"

    # Act
    hits = suppression_hits(source, "pkg/mod.py")

    # Assert
    assert hits == [f"pkg/mod.py:2: {comment}"]


def test_suppression_hits_ignores_a_directive_inside_a_string() -> None:
    # Arrange
    source = 'TEXT = "# noqa: E501 and # type: ignore"\n'

    # Act
    hits = suppression_hits(source, "pkg/mod.py")

    # Assert
    assert hits == []


def test_suppression_hits_ignores_a_comment_that_only_mentions_the_words() -> None:
    # Arrange
    source = "x = 1  # no noqa or type-ignore comments here\n"

    # Act
    hits = suppression_hits(source, "pkg/mod.py")

    # Assert
    assert hits == []


@pytest.mark.parametrize(
    "comment",
    [
        "# pyright: strict",
        "# pyright: reportPrivateUsage=true",
        "# pyright: reportPrivateUsage=error",
        "# pyright: reportPrivateUsage=warning",
        "# ruff: isort: skip_file",
        "# pragma: allowlist secret",
    ],
)
def test_suppression_hits_ignores_a_directive_that_does_not_silence(comment: str) -> None:
    # Arrange
    source = f"import os\nos.getcwd()  {comment}\n"

    # Act
    hits = suppression_hits(source, "pkg/mod.py")

    # Assert
    assert hits == []


def test_tracked_python_files_include_the_type_stubs() -> None:
    # Act
    files = tracked_python_files()

    # Assert
    assert REPO_ROOT / "typings" / "joblib.pyi" in files


def _git_or_skip() -> str:
    """Return the git executable, skipping the test when there is none."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not available")
    return git


def test_tracked_sources_ignores_an_inherited_foreign_git_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    git = _git_or_skip()
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    subprocess.run([git, "init", "-q", str(tmp_path / "other")], check=True, env=env)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "other" / ".git"))

    # Act
    files = tracked_sources(REPO_ROOT)

    # Assert
    assert Path(__file__).resolve() in files


def test_tracked_sources_ignores_an_inherited_relative_index_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    _git_or_skip()
    monkeypatch.setenv("GIT_INDEX_FILE", ".git/index")

    # Act
    files = tracked_sources(REPO_ROOT)

    # Assert
    assert Path(__file__).resolve() in files


def test_tracked_sources_outside_a_git_checkout_raises_not_a_checkout(tmp_path: Path) -> None:
    # Arrange
    _git_or_skip()

    # Act & Assert
    with pytest.raises(NotACheckoutError, match="not a git checkout"):
        tracked_sources(tmp_path)
