"""Characterization tests for the SHAP analysis entrypoint.

A small market-anchored margin/total model shaped like the weekly run's is trained once on the
synthetic seasons of ``tests/weekly_fixture.py``. The command-line ``main`` then writes a SHAP
report for each head, and the report is compared with a snapshot under
``tests/fixtures/shap_analysis_characterization/``, so moving the entrypoint cannot change what
it reports. Rewrite the snapshots with ``NFLP_UPDATE_SNAPSHOTS=1``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pytest
from tests import snapshots
from tests.weekly_fixture import build_fixture

from nfl_predictor.cli import explain
from nfl_predictor.ml import ml_model_core, ml_model_training
from nfl_predictor.ml.ml_model_core import OptunaConfig

SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "shap_analysis_characterization"
OPTUNA_OFF = OptunaConfig(
    enabled=False,
    timeout_seconds=0,
    n_trials=None,
    cv_splits=2,
    objective="mae",
    early_stopping_rounds=50,
    tree_method=None,
    device=None,
    storage=None,
    study_name=None,
    best_params_out=None,
    xgb_n_jobs=1,
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Train the model on the fixture and return the model and data paths."""
    root = tmp_path_factory.mktemp("shap_fixture")
    completed = build_fixture(root)["completed"]
    margin_total = ml_model_training.train_margin_total_model(
        ml_model_training.TrainingOptions(
            data_path=completed,
            holdout_seasons=0,
            include_market=True,
            max_cardinality_ratio=0.5,
            optuna_config=OPTUNA_OFF,
            market_transform=True,
            market_anchor=True,
        )
    )
    assert isinstance(margin_total, ml_model_core.MarginTotalModel)
    paths = {"data": completed, "margin_total": root / "margin_total" / "model.joblib"}
    paths["margin_total"].parent.mkdir()
    joblib.dump(margin_total, paths["margin_total"])
    return paths


def _run_shap(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    """Run the entrypoint's ``main`` with ``argv``."""
    monkeypatch.setattr(sys, "argv", ["shap_analysis.py", *argv])
    return explain.main()


@pytest.mark.parametrize(
    ("extra_args", "snapshot_name"),
    [
        (["--target", "margin"], "margin_total_margin.json"),
        (
            ["--target", "total", "--sample-size", "100", "--random-seed", "3"],
            "margin_total_total_sampled.json",
        ),
    ],
)
def test_shap_report_matches_snapshot(
    trained: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    snapshot_name: str,
) -> None:
    """Each head and component reports the same features and mean absolute SHAP as before."""
    out_json = tmp_path / "shap_report.json"
    code = _run_shap(
        [
            "--model-path",
            str(trained["margin_total"]),
            "--data-path",
            str(trained["data"]),
            "--output-path",
            str(out_json),
            *extra_args,
        ],
        monkeypatch,
    )
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert code == 0
    assert report["rows"], "the report lists no features"
    snapshots.check_json(snapshot_name, report, SNAPSHOT_DIR / snapshot_name)


def test_shap_report_defaults_to_the_model_directory(
    trained: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--output-path`` the report is written beside the model."""
    default_path = trained["margin_total"].parent / "shap_report.json"
    code = _run_shap(
        ["--model-path", str(trained["margin_total"]), "--data-path", str(trained["data"])],
        monkeypatch,
    )
    assert code == 0
    assert json.loads(default_path.read_text(encoding="utf-8"))["target"] == "margin"


def test_shap_rejects_a_score_target(
    trained: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the margin and total heads exist, so a home/away target is a usage error."""
    with pytest.raises(SystemExit):
        _run_shap(
            [
                "--model-path",
                str(trained["margin_total"]),
                "--data-path",
                str(trained["data"]),
                "--output-path",
                str(tmp_path / "unused.json"),
                "--target",
                "home",
            ],
            monkeypatch,
        )


def test_shap_reports_a_missing_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing model file raises before anything is written."""
    with pytest.raises(FileNotFoundError, match="Missing model"):
        _run_shap(
            [
                "--model-path",
                str(tmp_path / "missing.joblib"),
                "--data-path",
                str(tmp_path / "missing.csv"),
            ],
            monkeypatch,
        )
