"""The installed package and the workflows the repository ships.

The root ``__init__`` stays docstring-only (``tests/test_etl_fingerprint_coverage.py``), so the
version is read from the installed distribution rather than a ``__version__`` attribute.
"""

import tomllib
from importlib.metadata import version
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_the_installed_package_is_the_version_pyproject_declares() -> None:
    """A version bump without the sync after it leaves the old version installed."""
    # Arrange
    declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]

    # Act
    installed = version("nfl-predictor")

    # Assert
    assert installed == declared


def test_ci_runs_only_the_validation_workflow() -> None:
    """The project is private and publishes no releases, so no workflow builds one."""
    # Arrange
    workflows = PYPROJECT.parent / ".github" / "workflows"

    # Act
    names = sorted(path.name for path in workflows.iterdir())

    # Assert
    assert names == ["validation.yml"]
