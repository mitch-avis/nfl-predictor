"""Characterization test: the weekly run's outputs when it submits the deterministic floor.

The deterministic floor maps the predicted margin to a home win probability through the fixed
normal curve, ``Phi(margin / SCORE_DIFF_STD_DEV)``, with no fitted calibrator and no market
blend or clamp. This test runs the weekly run on the synthetic seasons of
``tests/weekly_fixture.py`` with its walk-forward limited to the floor candidate, and pins the
week's outputs (predictions, confidence picks, betting report, power rankings, projected
standings) and the floor candidate's walk-forward metrics, under
``tests/fixtures/weekly_run_floor_characterization/``.

Rewrite the snapshots with ``NFLP_UPDATE_SNAPSHOTS=1`` only for an intended output change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_predictor.ml import walk_forward
from nfl_predictor.weekly_run import pipeline, stage1
from tests import snapshots
from tests.weekly_fixture import (
    CURRENT_SEASON,
    PREDICT_WEEK,
    build_fixture,
    write_weekly_config,
)

SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "weekly_run_floor_characterization"
TEST_BOOTSTRAP_SAMPLES = 200

# The walk-forward metrics of the configuration the final model is trained with.
WF_METRIC_KEYS = (
    "brier",
    "log_loss",
    "deterministic_brier",
    "deterministic_log_loss",
    "market_brier",
    "market_log_loss",
    "deterministic_brier_vs_market",
    "deterministic_brier_vs_market_ci_low",
    "deterministic_brier_vs_market_ci_high",
    "deterministic_log_loss_vs_market",
    "deterministic_log_loss_vs_market_ci_low",
    "deterministic_log_loss_vs_market_ci_high",
    "reliability_ece",
    "pick_accuracy",
    "deterministic_pick_accuracy",
    "market_pick_accuracy",
    "margin_mae",
    "total_mae",
    "expected_points_avg",
    "actual_points_avg",
    "market_margin_resid_mae",
    "market_total_resid_mae",
    "games",
    "weeks",
    "market_mode",
)


def _run_weekly_on_the_floor(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the weekly entrypoint with its walk-forward limited to the floor candidate."""
    monkeypatch.setattr(walk_forward, "BOOTSTRAP_SAMPLES", TEST_BOOTSTRAP_SAMPLES)
    monkeypatch.setattr(stage1, "_WF_MATRIX", [("auto_base", "auto", 0.0, 0.0)])
    monkeypatch.setattr(sys, "argv", ["weekly_run.py", "--config", str(config_path)])
    assert pipeline.main() == 0


def _week_outputs(output_dir: Path) -> dict[str, Path]:
    """Return the week's outputs this test pins, keyed by snapshot file name."""
    suffix = f"season_{CURRENT_SEASON}_week_{PREDICT_WEEK:02d}"
    through = f"season_{CURRENT_SEASON}_week_{PREDICT_WEEK - 1:02d}"
    return {
        "predictions.csv": output_dir / f"{suffix}_predictions.csv",
        "confidence_picks.csv": output_dir / f"{suffix}_confidence_picks.csv",
        "betting_report.csv": output_dir / f"{suffix}_betting_report.csv",
        "power_rankings.csv": output_dir / f"power_rankings_{through}.csv",
        "projected_standings.csv": output_dir / f"projected_standings_{through}.csv",
        "projected_division_standings.csv": output_dir
        / f"projected_division_standings_{through}.csv",
    }


def _wf_metrics(run_dir: Path) -> dict[str, Any]:
    """Return the walk-forward metrics the run recorded for its trained configuration."""
    payload = json.loads((run_dir / "wf_best.json").read_text(encoding="utf-8"))
    return {key: payload[key] for key in WF_METRIC_KEYS}


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_weekly_run_floor_outputs_match_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The floor's week outputs and walk-forward metrics equal their committed snapshots."""
    paths = build_fixture(tmp_path)
    output_dir = tmp_path / "out"
    _run_weekly_on_the_floor(write_weekly_config(tmp_path, paths, output_dir), monkeypatch)

    for name, path in _week_outputs(output_dir).items():
        snapshots.check_csv(name, pd.read_csv(path), SNAPSHOT_DIR / name)
    snapshots.check_json(
        "wf_metrics.json", _wf_metrics(tmp_path / "run"), SNAPSHOT_DIR / "wf_metrics.json"
    )
    if snapshots.updating():
        pytest.skip(f"snapshots rewritten in {SNAPSHOT_DIR}")
