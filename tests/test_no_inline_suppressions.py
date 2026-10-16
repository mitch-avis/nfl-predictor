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


class NotACheckoutError(Exception):
    """Raised when there is no git or the directory to scan is not a git checkout."""


def _git(git: str, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in ``root`` with every inherited ``GIT_*`` variable removed."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    return subprocess.run(
        [git, "-C", str(root), *args], capture_output=True, text=True, check=False, env=env
    )


def tracked_sources(root: Path) -> list[Path]:
    """Return the tracked Python and stub files under ``root``.

    Raises:
        NotACheckoutError: git is missing or ``root`` is not a git checkout.
        RuntimeError: git failed to list the files for any other reason.
    """
    git = shutil.which("git")
    if git is None:
        msg = "git is not available to list the tracked files"
        raise NotACheckoutError(msg)
    inside = _git(git, root, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        msg = f"{root} is not a git checkout"
        raise NotACheckoutError(msg)
    listing = _git(git, root, "ls-files", "-z", "*.py", "*.pyi")
    if listing.returncode != 0:
        msg = f"git ls-files failed: {listing.stderr.strip()}"
        raise RuntimeError(msg)
    return [root / name for name in listing.stdout.split("\0") if name]


def tracked_python_files() -> list[Path]:
    """Return the repository's tracked Python and stub files, skipping outside a checkout."""
    try:
        return tracked_sources(REPO_ROOT)
    except NotACheckoutError as exc:
        pytest.skip(str(exc))


def scan_files(root: Path, files: list[Path]) -> list[str]:
    """Return the suppression hits of ``files``, each named relative to ``root``.

    A tracked file deleted from the working tree has nothing to scan and is skipped.
    """
    return [
        hit
        for path in files
        if path.is_file()
        for hit in suppression_hits(path.read_text(encoding="utf-8"), str(path.relative_to(root)))
    ]


def test_tracked_python_files_carry_no_inline_suppressions() -> None:
    # Arrange
    files = tracked_python_files()

    # Act
    hits = scan_files(REPO_ROOT, files)

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


@pytest.mark.parametrize(
    "source",
    ["x = (\n", "def f():\n        a = 1\n    b = 2\n"],
    ids=["unclosed bracket", "inconsistent dedent"],
)
def test_suppression_hits_reports_a_file_that_does_not_tokenize(source: str) -> None:
    # Act
    hits = suppression_hits(source, "pkg/mod.py")

    # Assert
    assert hits == ["pkg/mod.py: does not tokenize"]


def test_scan_files_skips_a_tracked_file_deleted_from_the_working_tree(tmp_path: Path) -> None:
    # Arrange
    (tmp_path / "kept.py").write_text("import os  # noqa: F401\n", encoding="utf-8")
    files = [tmp_path / "kept.py", tmp_path / "deleted.py"]

    # Act
    hits = scan_files(tmp_path, files)

    # Assert
    assert hits == ["kept.py:1: # noqa: F401"]


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
