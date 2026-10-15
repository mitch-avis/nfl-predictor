"""The installed package."""

from importlib.metadata import version

import nfl_predictor


def test_the_package_reports_its_installed_version() -> None:
    # Act
    reported = nfl_predictor.__version__

    # Assert
    assert reported == version("nfl-predictor")
