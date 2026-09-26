"""Tests for the power rankings command-line parser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nfl_predictor.cli import rankings
from nfl_predictor.reporting.power_rankings import DEFAULT_STRENGTH_SNAPSHOTS


def test_cli_exposes_the_method_and_snapshot_path() -> None:
    """The script parses the ranking method and the snapshot file location."""
    args = rankings._parse_args(
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
    defaults = rankings._parse_args(
        ["--model-in", "model.joblib", "--season", "2024", "--through-week", "3"]
    )

    assert args.method == "bradley_terry"
    assert args.strength_snapshots == Path("snapshots.csv")
    assert defaults.method is None
    assert defaults.strength_snapshots == DEFAULT_STRENGTH_SNAPSHOTS


def _base_args(tmp_path: Path) -> list[str]:
    """Return arguments naming a model, both datasets and an output directory under tmp_path."""
    for name in ("model.joblib", "ml.csv", "schedule.csv", "snapshots.csv"):
        (tmp_path / name).write_text("x\n", encoding="utf-8")
    return [
        "--model-in",
        str(tmp_path / "model.joblib"),
        "--season",
        "2024",
        "--through-week",
        "3",
        "--data-ml",
        str(tmp_path / "ml.csv"),
        "--data-schedule",
        str(tmp_path / "schedule.csv"),
        "--strength-snapshots",
        str(tmp_path / "snapshots.csv"),
        "--out-dir",
        str(tmp_path / "out"),
    ]


def test_main_ranks_and_writes_the_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command loads the model, ranks through the week, and writes the three tables."""
    computed: list[tuple[object, dict[str, Any]]] = []
    written: list[tuple[object, dict[str, Any]]] = []
    monkeypatch.setattr(
        rankings.ml_model_core, "load_model_checkpoint", lambda path, kind: ("model", path, kind)
    )

    def fake_compute(model: object, **kwargs: Any) -> str:
        """Record the ranking request."""
        computed.append((model, kwargs))
        return "result"

    def fake_write(result: object, **kwargs: Any) -> None:
        """Record the output request."""
        written.append((result, kwargs))

    monkeypatch.setattr(rankings, "compute_power_rankings", fake_compute)
    monkeypatch.setattr(rankings, "write_ranking_outputs", fake_write)

    assert rankings.main(_base_args(tmp_path)) == 0

    [(model, kwargs)] = computed
    assert model == ("model", tmp_path / "model.joblib", "margin_total")
    assert kwargs["season"] == 2024
    assert kwargs["through_week"] == 3
    assert kwargs["options"].method == "composite"
    assert written == [("result", {"out_dir": tmp_path / "out", "season": 2024, "through_week": 3})]


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("model.joblib", "Missing model checkpoint"),
        ("ml.csv", "Missing ML dataset"),
        ("schedule.csv", "Missing schedule dataset"),
        ("snapshots.csv", "Missing strength snapshot file"),
    ],
)
def test_main_names_the_missing_input(tmp_path: Path, missing: str, message: str) -> None:
    """Each required input is checked before any model is loaded."""
    args = _base_args(tmp_path)
    (tmp_path / missing).unlink()

    with pytest.raises(FileNotFoundError, match=message):
        rankings.main(args)


def test_main_rejects_the_legacy_fit_with_the_composite(tmp_path: Path) -> None:
    """An inconsistent ranking method is a usage error, not a traceback."""
    args = [*_base_args(tmp_path), "--method", "composite", "--legacy-franchise-fit"]

    with pytest.raises(SystemExit, match="legacy-franchise-fit"):
        rankings.main(args)
