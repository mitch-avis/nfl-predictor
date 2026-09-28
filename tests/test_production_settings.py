"""The walk-forward report's settings-versus-production section.

A backtest lists every model-affecting setting where its resolved configuration differs from
what the production weekly run would use, for stage 1's walk-forward and for the final fit, with
the production side read through the weekly run's own parser and config file.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_predictor.cli import backtest
from nfl_predictor.cli import main as front_door
from nfl_predictor.ml import ml_model_core, ml_model_xgb_utils, walk_forward
from nfl_predictor.reporting import production_settings
from nfl_predictor.weekly_run import config as weekly_config
from tests import test_run_comparison

SECTION = production_settings.SECTION_KEY


def _games() -> pd.DataFrame:
    """Return a tiny dataset with market lines, as the real one has."""
    return pd.DataFrame({"season": [2023], "home_spread": [-3.0], "total_line": [44.5]})


def _run_backtest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_argv: list[str],
    *,
    production_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the backtest entrypoint on fakes and return its written metrics report.

    The production side reads a config file under ``tmp_path`` (the code defaults unless
    ``production_config`` sets keys), and a GPU counts as usable, as on the production machine.
    """
    config_path = tmp_path / "weekly_run.json"
    config_path.write_text(json.dumps(production_config or {}), encoding="utf-8")
    monkeypatch.setattr(weekly_config, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: True)

    def fake_run(
        _df: pd.DataFrame, _config: walk_forward.WalkForwardConfig, **_kwargs: object
    ) -> dict[str, object]:
        return {"checkpoint": {"dir": "somewhere"}, "per_week": []}

    def fake_report(
        _run_id: str, _created_at: str, payload: dict[str, object], _results: object
    ) -> dict[str, object]:
        return {"config": payload, "metrics": {}}

    monkeypatch.setattr(walk_forward, "load_games", lambda _path: _games())
    monkeypatch.setattr(walk_forward, "dataset_fingerprint", lambda _path: "hash")
    monkeypatch.setattr(walk_forward, "run_walk_forward_backtest", fake_run)
    monkeypatch.setattr(walk_forward, "build_metrics_report", fake_report)
    monkeypatch.setattr(walk_forward, "build_metadata", lambda *_args: {})
    out_json = tmp_path / "metrics_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "walk_forward_backtest.py",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--out-json",
            str(out_json),
            *extra_argv,
        ],
    )
    backtest.main()
    return json.loads(out_json.read_text())


def _settings(section: dict[str, Any], stage: str) -> list[str]:
    """Return the names of the model-affecting settings that differ for one stage."""
    return [row["setting"] for row in section[stage]["differences"]]


def test_a_run_with_the_production_settings_shows_no_model_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backtest's defaults are production's: neither stage lists a difference."""
    section = _run_backtest(tmp_path, monkeypatch, [])[SECTION]

    assert section["stage1"]["differences"] == []
    assert section["final_fit"]["differences"] == []
    assert section["stage1"]["run_only"] == {}
    assert section["stage1"]["production_only"] == {}
    assert section["production_config"] == str(tmp_path / "weekly_run.json")


def test_a_recency_half_life_is_the_only_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weighting by recency differs from production's unweighted stage 1 and final fit."""
    section = _run_backtest(tmp_path, monkeypatch, ["--recency-half-life-seasons", "16"])[SECTION]

    expected = [{"setting": "recency_half_life_seasons", "run": 16.0, "production": None}]
    assert section["stage1"]["differences"] == expected
    assert section["final_fit"]["differences"] == expected


def test_training_on_the_cpu_is_the_only_difference_from_the_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a usable GPU, production's ``auto`` trains on it; a CPU run differs only there."""
    section = _run_backtest(tmp_path, monkeypatch, ["--xgb-device", "cpu"])[SECTION]

    expected = [{"setting": "xgb.device", "run": "cpu", "production": "cuda"}]
    assert section["stage1"]["differences"] == expected
    assert section["final_fit"]["differences"] == expected


def test_the_production_side_reads_the_weekly_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value in the weekly config file is production's; the final fit follows stage 1's."""
    section = _run_backtest(
        tmp_path, monkeypatch, [], production_config={"wf_recency_half_life_seasons": 8}
    )[SECTION]

    expected = [{"setting": "recency_half_life_seasons", "run": None, "production": 8}]
    assert section["stage1"]["differences"] == expected
    assert section["final_fit"]["differences"] == expected


def test_xgboost_and_market_overrides_are_named_setting_by_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tree budget, regularization, anchoring and pruning each show under their own name."""
    section = _run_backtest(
        tmp_path,
        monkeypatch,
        ["--n-estimators", "400", "--gamma", "1.5", "--no-market-anchor", "--disable-pruning"],
    )[SECTION]

    assert _settings(section, "stage1") == [
        "disable_pruning",
        "market_anchor",
        "xgb.gamma",
        "xgb.n_estimators",
    ]
    assert section["stage1"]["differences"][3] == {
        "setting": "xgb.n_estimators",
        "run": 400,
        "production": ml_model_core.DEFAULT_XGB_PARAMS["n_estimators"],
    }
    assert _settings(section, "final_fit") == ["market_anchor", "xgb.gamma", "xgb.n_estimators"]
    assert section["final_fit"]["run_only"]["disable_pruning"] is True


def test_postseason_and_dropped_features_are_model_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Postseason games, dropped feature groups and dropped trend features change the model."""
    section = _run_backtest(
        tmp_path,
        monkeypatch,
        ["--include-postseason", "--disable-feature-groups", "pbp", "--disable-trend-features"],
    )[SECTION]

    assert _settings(section, "stage1") == ["disabled_feature_groups", "include_postseason"]
    assert section["stage1"]["run_only"] == {"disable_trend_features": True}
    assert section["final_fit"]["run_only"]["disabled_feature_groups"] == ["pbp"]


def test_settings_on_one_side_only_are_listed_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final fit's own options have no backtest counterpart, and the reverse."""
    section = _run_backtest(tmp_path, monkeypatch, [])[SECTION]

    assert section["final_fit"]["production_only"] == {
        "holdout_seasons": 0,
        "postseason_weight": 1.0,
        "tune": False,
    }
    assert section["final_fit"]["run_only"] == {
        "disable_pruning": False,
        "disabled_feature_groups": [],
    }


def test_scope_settings_are_listed_apart_from_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which seasons and weeks are scored, the seed and the paths are scope, not differences."""
    section = _run_backtest(
        tmp_path,
        monkeypatch,
        [
            "--eval-last-n-seasons",
            "6",
            "--wf-start-week",
            "1",
            "--random-seed",
            "7",
            "--xgb-n-jobs",
            "3",
        ],
    )[SECTION]

    assert section["stage1"]["differences"] == []
    assert section["final_fit"]["differences"] == []
    scope = section["scope"]
    assert scope["run"]["eval_last_n_seasons"] == 6
    assert scope["run"]["wf_start_week"] == 1
    assert scope["run"]["random_seed"] == 7
    assert scope["run"]["checkpoint_dir"] == str(tmp_path / "checkpoints")
    assert scope["stage1"]["eval_last_n_seasons"] == 3
    assert scope["stage1"]["random_seed"] == walk_forward.DEFAULT_RANDOM_SEED
    assert scope["final_fit"]["random_seed"] == ml_model_core.DEFAULT_XGB_PARAMS["random_state"]
    assert section["runtime"]["run"]["xgb.n_jobs"] == 3


def test_the_log_prints_the_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The backtest's log carries the same section, one line per stage."""
    caplog.set_level(logging.INFO)

    _run_backtest(tmp_path, monkeypatch, ["--recency-half-life-seasons", "16"])

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith(production_settings.HEADING) for message in messages)
    assert (
        "- stage 1 walk-forward: recency_half_life_seasons: run 16.0, production null" in messages
    )
    assert "- final fit: recency_half_life_seasons: run 16.0, production null" in messages


def test_an_unreadable_production_config_leaves_the_run_report_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A weekly config that fails to load says why; the finished run still writes its report."""
    caplog.set_level(logging.INFO)

    report = _run_backtest(tmp_path, monkeypatch, [], production_config={"not_a_weekly_option": 1})

    reason = report[SECTION]["unavailable"]
    assert "not_a_weekly_option" in reason
    assert report["config"]["checkpoint"] == {"dir": "somewhere"}
    messages = [record.getMessage() for record in caplog.records]
    assert f"{production_settings.HEADING}: unavailable ({reason})" in messages


def test_a_weekly_config_the_parser_rejects_is_unavailable_too(tmp_path: Path) -> None:
    """Option combinations the weekly parser exits on give a reason instead of exiting."""
    config_path = tmp_path / "weekly_run.json"
    config_path.write_text(
        json.dumps({"legacy_franchise_fit": True, "power_rankings_method": "composite"}),
        encoding="utf-8",
    )

    section = production_settings.settings_versus_production(
        walk_forward.WalkForwardConfig(xgb_params_overrides={"device": "cpu"}),
        _games(),
        production_argv=["--config", str(config_path)],
    )

    assert section == {"unavailable": "the production weekly configuration does not parse"}


def test_format_says_when_nothing_differs() -> None:
    """A stage with no difference says so rather than printing nothing."""
    section = production_settings.settings_versus_production(
        walk_forward.with_resolved_xgb_device(
            walk_forward.WalkForwardConfig(
                include_quantiles=False, xgb_params_overrides={"device": "cpu"}
            )
        ),
        _games(),
        production_argv=["--xgb-device", "cpu"],
    )

    lines = production_settings.format_section(section)

    assert "- stage 1 walk-forward: no model-affecting differences" in lines
    assert "- final fit: no model-affecting differences" in lines


def _write_compare_run(
    root: Path, name: str, config: dict[str, Any], data_path: Path | None
) -> Path:
    """Write a comparable run directory whose metadata records ``config``."""
    run_dir = test_run_comparison._write_run(root, name, test_run_comparison._predictions(), 42)
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config"].update(config)
    if data_path is not None:
        metadata["config"]["data_path"] = str(data_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir


def test_compare_shows_each_runs_settings_versus_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``compare`` adds the section for every run whose metadata and dataset it can read."""
    config_path = tmp_path / "weekly_run.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(weekly_config, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: True)
    data_path = tmp_path / "games.csv"
    _games().to_csv(data_path, index=False)
    recorded = {
        "calibration": "auto",
        "include_quantiles": True,
        "xgb_params_overrides": {"device": "cuda"},
        "disabled_feature_groups": [],
    }
    candidate = _write_compare_run(
        tmp_path, "cand", {**recorded, "recency_half_life_seasons": 16.0}, data_path
    )
    reference = _write_compare_run(tmp_path, "ref", recorded, None)
    out_json = tmp_path / "compare.json"
    out_md = tmp_path / "compare.md"

    code = front_door.main(
        [
            "compare",
            "--candidate",
            str(candidate),
            "--reference",
            str(reference),
            "--resamples",
            "25",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert code == 0
    sections = json.loads(out_json.read_text(encoding="utf-8"))[SECTION]
    assert sections["cand"]["stage1"]["differences"] == [
        {"setting": "recency_half_life_seasons", "run": 16.0, "production": None}
    ]
    assert "dataset" in sections["ref"]["unavailable"]
    markdown = out_md.read_text(encoding="utf-8")
    assert f"## {production_settings.HEADING}" in markdown
    assert "- stage 1 walk-forward: recency_half_life_seasons: run 16.0, production null" in (
        markdown
    )


def test_a_run_without_metadata_has_no_section() -> None:
    """A bare checkpoint directory records no settings to compare."""
    assert "metadata" in production_settings.section_for_metadata(None)["unavailable"]


def test_a_run_whose_recorded_config_no_longer_loads_has_no_section(tmp_path: Path) -> None:
    """A retired setting in an old run's metadata is reported, not raised."""
    data_path = tmp_path / "games.csv"
    _games().to_csv(data_path, index=False)

    section = production_settings.section_for_metadata(
        {"config": {"calibration": "platt", "data_path": str(data_path)}}
    )

    assert "platt" in section["unavailable"]
