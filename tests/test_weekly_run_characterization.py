"""Characterization test: the weekly run's outputs on a fixed synthetic dataset.

The weekly run is the production path: it compares probability candidates in a walk-forward
(stage 1), trains the final margin/total model (stage 2), predicts the week (stage 3), and
writes the betting report, power rankings and projected standings (stage 4). This test runs
all four stages end to end on the synthetic seasons of ``tests/weekly_fixture.py``, with the
shipped ``config/weekly_run.yaml`` settings except where the fixture needs otherwise (paths,
one evaluation season, CPU and one thread for determinism, no data refresh), and compares
every output with the snapshots under ``tests/fixtures/weekly_run_characterization/``.

The walk-forward's paired bootstrap runs with 200 resamples instead of 5,000: the code path
is the same and still seeded, and at the shipped count the bootstrap alone takes minutes.
Fields that differ on every run (timestamps, durations, and fingerprints that hash the git
commit or the temporary paths) are dropped before comparing.

Restructuring the code must leave the snapshots untouched. To rewrite them after an
intended output change, run this test with ``NFLP_UPDATE_SNAPSHOTS=1`` and commit the diff
with the change that caused it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_predictor.ml import walk_forward
from nfl_predictor.weekly_run import pipeline
from tests import snapshots
from tests.weekly_fixture import (
    CURRENT_SEASON,
    PREDICT_WEEK,
    build_fixture,
    write_weekly_config,
)

SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "weekly_run_characterization"

TEST_BOOTSTRAP_SAMPLES = 200
VOLATILE_FIELDS = frozenset({"completed_at", "duration_seconds", "wf_run_fingerprint"})


def _run_weekly(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the weekly entrypoint on ``config_path``.

    This is the one line to change when the entrypoint moves; the snapshots must not change.
    """
    monkeypatch.setattr(walk_forward, "BOOTSTRAP_SAMPLES", TEST_BOOTSTRAP_SAMPLES)
    monkeypatch.setattr(sys, "argv", ["weekly_run.py", "--config", str(config_path)])
    assert pipeline.main() == 0


def _snapshot_outputs(root: Path, output_dir: Path) -> dict[str, Path]:
    """Return the weekly outputs this test pins, keyed by snapshot file name."""
    suffix = f"season_{CURRENT_SEASON}_week_{PREDICT_WEEK:02d}"
    through = f"season_{CURRENT_SEASON}_week_{PREDICT_WEEK - 1:02d}"
    return {
        "wf_compare.csv": root / "run" / "wf_compare.csv",
        "wf_best.json": root / "run" / "wf_best.json",
        "predictions.csv": output_dir / f"{suffix}_predictions.csv",
        "confidence_picks.csv": output_dir / f"{suffix}_confidence_picks.csv",
        "betting_report.csv": output_dir / f"{suffix}_betting_report.csv",
        "power_rankings.csv": output_dir / f"power_rankings_{through}.csv",
        "projected_standings.csv": output_dir / f"projected_standings_{through}.csv",
        "projected_division_standings.csv": output_dir
        / f"projected_division_standings_{through}.csv",
    }


def _stable_payload(path: Path) -> dict[str, Any]:
    """Return a JSON output without the fields that change on every run."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: value for key, value in payload.items() if key not in VOLATILE_FIELDS}


def _stable_frame(path: Path) -> pd.DataFrame:
    """Return a CSV output without the columns that change on every run."""
    frame = pd.read_csv(path)
    return frame.drop(columns=[column for column in frame.columns if column in VOLATILE_FIELDS])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_weekly_run_outputs_match_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every weekly output on the fixed fixture equals its committed snapshot."""
    paths = build_fixture(tmp_path)
    output_dir = tmp_path / "out"
    _run_weekly(write_weekly_config(tmp_path, paths, output_dir), monkeypatch)

    outputs = _snapshot_outputs(tmp_path, output_dir)
    missing = [name for name, path in outputs.items() if not path.exists()]
    assert not missing, f"weekly run did not write: {missing}"

    for name, path in outputs.items():
        if name.endswith(".json"):
            snapshots.check_json(name, _stable_payload(path), SNAPSHOT_DIR / name)
        else:
            snapshots.check_csv(name, _stable_frame(path), SNAPSHOT_DIR / name)
    if snapshots.updating():
        pytest.skip(f"snapshots rewritten in {SNAPSHOT_DIR}")
