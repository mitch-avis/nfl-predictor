"""Characterization tests for the backtest, sweep and leakage-audit entrypoints.

Each runs its command-line ``main`` on the synthetic seasons of ``tests/weekly_fixture.py``
and compares the report it writes with a snapshot under
``tests/fixtures/entrypoints_characterization/``, so moving these entrypoints cannot change
what they report. Runs use 20 trees and a 200-resample paired bootstrap to stay fast; fields
that change on every run (timestamps, run ids, temporary paths, the checkpoint fingerprint,
durations) are removed first. Rewrite the snapshots with ``NFLP_UPDATE_SNAPSHOTS=1``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_predictor.ml import walk_forward
from scripts import leakage_audit, walk_forward_backtest, wf_compare
from tests import snapshots
from tests.weekly_fixture import build_fixture

SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "entrypoints_characterization"
TEST_BOOTSTRAP_SAMPLES = 200
SWEEP_VOLATILE_COLUMNS = ("duration_seconds", "completed_at", "wf_run_fingerprint")


def _run_main(module: Any, argv: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    """Run an entrypoint's ``main`` with ``argv`` and a small paired bootstrap."""
    monkeypatch.setattr(walk_forward, "BOOTSTRAP_SAMPLES", TEST_BOOTSTRAP_SAMPLES)
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    return int(module.main() or 0)


def _portable(payload: Any, root: Path) -> Any:
    """Return ``payload`` with the temporary directory written as ``<tmp>``."""
    return json.loads(json.dumps(payload).replace(str(root), "<tmp>"))


def test_leakage_audit_report_matches_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audit passes the fixture and writes the same report as before."""
    paths = build_fixture(tmp_path)
    out_json = tmp_path / "leakage_audit.json"
    code = _run_main(
        leakage_audit,
        ["--data-path", str(paths["completed"]), "--out-json", str(out_json)],
        monkeypatch,
    )
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert code == 0
    assert report["ok"] is True
    snapshots.check_json(
        "leakage_audit.json",
        _portable(report, tmp_path),
        SNAPSHOT_DIR / "leakage_audit.json",
    )


def test_leakage_audit_fails_on_a_planted_target_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A feature that copies the home score is flagged and the audit exits non-zero."""
    paths = build_fixture(tmp_path)
    games = pd.read_csv(paths["completed"])
    position = list(games.columns).index("away_rest") + 1
    games.insert(position, "home_points_leak", games["home_score"])
    leaky = tmp_path / "leaky.csv"
    games.to_csv(leaky, index=False)
    out_json = tmp_path / "leakage_audit.json"

    code = _run_main(
        leakage_audit, ["--data-path", str(leaky), "--out-json", str(out_json)], monkeypatch
    )

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert code == 1
    assert report["ok"] is False
    assert "home_points_leak" in json.dumps(report["flagged_columns"])


def test_leakage_audit_reports_a_missing_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing input file exits with status 2 and writes nothing."""
    out_json = tmp_path / "leakage_audit.json"
    code = _run_main(
        leakage_audit,
        ["--data-path", str(tmp_path / "missing.csv"), "--out-json", str(out_json)],
        monkeypatch,
    )
    assert code == 2
    assert not out_json.exists()


def test_walk_forward_backtest_report_matches_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The benchmark configuration reports the same metrics on the fixture as before."""
    paths = build_fixture(tmp_path)
    out_json = tmp_path / "wf" / "metrics_report.json"
    _run_main(
        walk_forward_backtest,
        [
            "--data-path",
            str(paths["completed"]),
            "--eval-last-n-seasons",
            "1",
            "--wf-start-week",
            "1",
            "--calibration",
            "auto",
            "--market-anchor",
            "--market-transform",
            "--n-estimators",
            "20",
            "--xgb-n-jobs",
            "1",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--out-json",
            str(out_json),
        ],
        monkeypatch,
    )
    report = json.loads(out_json.read_text(encoding="utf-8"))
    for key in ("created_at", "run_id"):
        report.pop(key)
    report["config"].pop("run_id")
    report["config"]["checkpoint"].pop("dir")
    snapshots.check_json(
        "metrics_report.json",
        _portable(report, tmp_path),
        SNAPSHOT_DIR / "walk_forward_metrics_report.json",
    )


def test_wf_compare_summary_matches_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The calibration and market-blend sweep ranks the same candidates the same way."""
    paths = build_fixture(tmp_path)
    out_csv = tmp_path / "wf_compare.csv"
    code = _run_main(
        wf_compare,
        [
            "--data-path",
            str(paths["completed"]),
            "--eval-last-n-seasons",
            "1",
            "--market-mode",
            "anchor",
            "--n-estimators",
            "20",
            "--n-jobs",
            "1",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--out",
            str(out_csv),
        ],
        monkeypatch,
    )
    assert code == 0
    summary = pd.read_csv(out_csv)
    summary = summary.drop(columns=[c for c in SWEEP_VOLATILE_COLUMNS if c in summary.columns])
    snapshots.check_csv("wf_compare.csv", summary, SNAPSHOT_DIR / "wf_compare.csv")
