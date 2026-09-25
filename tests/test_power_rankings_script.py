"""Tests for the power rankings command-line parser."""

from __future__ import annotations

from pathlib import Path

from nfl_predictor.reporting.power_rankings import DEFAULT_STRENGTH_SNAPSHOTS
from scripts import power_rankings


def test_cli_exposes_the_method_and_snapshot_path() -> None:
    """The script parses the ranking method and the snapshot file location."""
    args = power_rankings._parse_args(
        [
            "--model-in",
            "model.joblib",
            "--season",
            "2024",
            "--through-week",
            "3",
            "--method",
            "bradley_terry",
            "--strength-snapshots",
            "snapshots.csv",
        ]
    )
    defaults = power_rankings._parse_args(
        ["--model-in", "model.joblib", "--season", "2024", "--through-week", "3"]
    )

    assert args.method == "bradley_terry"
    assert args.strength_snapshots == Path("snapshots.csv")
    assert defaults.method is None
    assert defaults.strength_snapshots == DEFAULT_STRENGTH_SNAPSHOTS
