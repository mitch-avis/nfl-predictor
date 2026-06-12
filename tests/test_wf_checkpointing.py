"""Tests for walk-forward checkpointing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

import pandas as pd
import pytest

from nfl_predictor.ml import wf_compare_utils
from nfl_predictor.utils import fingerprints
from scripts import weekly_run


class Candidate(TypedDict):
    """Typed candidate payload for checkpoint tests."""

    candidate_key: str
    label: str
    calibration: str
    market_prob_weight: float
    market_prob_clamp: float
    market_prob_source: str
    market_prob_blend_method: str
    win_prob_use_uncertainty: bool
    market_mode: str
    include_market: bool
    market_anchor: bool


def _candidate(candidate_key: str) -> Candidate:
    """Return a minimal candidate dictionary for checkpoint tests."""
    return {
        "candidate_key": candidate_key,
        "label": candidate_key,
        "calibration": "none",
        "market_prob_weight": 0.0,
        "market_prob_clamp": 0.0,
        "market_prob_source": "raw",
        "market_prob_blend_method": "prob",
        "win_prob_use_uncertainty": False,
        "market_mode": "features",
        "include_market": True,
        "market_anchor": False,
    }


def _stub_results(tag: int) -> dict[str, object]:
    """Return stubbed walk-forward results for checkpoint tests."""
    return {
        "overall": {
            "brier": 0.2 + (tag * 0.01),
            "log_loss": 0.7 + (tag * 0.01),
            "pick_accuracy": 0.55,
            "margin_mae": 9.5,
            "total_mae": 10.1,
            "expected_points_avg": 210.0,
            "actual_points_avg": 205.0,
            "games": 2,
            "weeks": 1,
        },
        "per_week": [{"season": 2024, "week": 3, "games": 2}],
        "per_season": [{"season": 2024, "games": 2}],
        "reliability": [{"bin_lower": 0.0, "bin_upper": 0.1, "count": 2}],
        "resolved_settings": {},
    }


def test_candidate_key_stability() -> None:
    """Candidate keys should be stable for identical inputs."""
    key_a = wf_compare_utils.build_candidate_key(
        model_kind="margin_total",
        feature_start="away_rest",
        feature_end="home_moneyline",
        calibration="none",
        market_mode="features",
        market_prob_source="raw",
        market_prob_blend_method="prob",
        win_prob_use_uncertainty=False,
        market_prob_weight=0.0,
        market_prob_clamp=0.0,
        include_quantiles=False,
        xgb_params_overrides={"n_estimators": 50},
    )
    key_b = wf_compare_utils.build_candidate_key(
        model_kind="margin_total",
        feature_start="away_rest",
        feature_end="home_moneyline",
        calibration="none",
        market_mode="features",
        market_prob_source="raw",
        market_prob_blend_method="prob",
        win_prob_use_uncertainty=False,
        market_prob_weight=0.0,
        market_prob_clamp=0.0,
        include_quantiles=False,
        xgb_params_overrides={"n_estimators": 50},
    )
    key_c = wf_compare_utils.build_candidate_key(
        model_kind="margin_total",
        feature_start="away_rest",
        feature_end="home_moneyline",
        calibration="platt",
        market_mode="features",
        market_prob_source="raw",
        market_prob_blend_method="prob",
        win_prob_use_uncertainty=False,
        market_prob_weight=0.0,
        market_prob_clamp=0.0,
        include_quantiles=False,
        xgb_params_overrides={"n_estimators": 50},
    )

    assert key_a == key_b
    assert key_a != key_c


def test_dataset_fingerprint_changes_on_content_change(tmp_path: Path) -> None:
    """Dataset fingerprints should change when file contents change."""
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    fp_a = fingerprints.dataset_fingerprint(path)
    fp_b = fingerprints.dataset_fingerprint(path)
    assert fp_a["sha256"] == fp_b["sha256"]

    path.write_text("a,b\n1,3\n", encoding="utf-8")
    fp_c = fingerprints.dataset_fingerprint(path)
    assert fp_a["sha256"] != fp_c["sha256"]


def test_resume_skips_completed_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume should skip candidates with valid artifacts."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_fp = {"sha256": "deadbeef", "path": "x", "size": 1, "mtime": 0.0}
    wf_fp = "wf123"
    candidates = [_candidate("cand1"), _candidate("cand2")]
    monkeypatch.setattr(weekly_run, "_build_wf_candidates", lambda **_kwargs: candidates)

    call_count = {"count": 0}

    def _fake_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        call_count["count"] += 1
        return _stub_results(call_count["count"])

    monkeypatch.setattr(weekly_run.walk_forward, "run_walk_forward_backtest", _fake_run)

    summary_row = weekly_run._build_summary_row(
        cast(dict[str, object], candidates[0]),
        _stub_results(1),
        dataset_sha256="deadbeef",
        wf_run_fingerprint=wf_fp,
        duration_seconds=1.0,
    )
    payload = {
        "candidate_key": candidates[0]["candidate_key"],
        "candidate": candidates[0],
        "dataset_fingerprint": dataset_fp,
        "wf_run_fingerprint": wf_fp,
        "metrics": _stub_results(1),
        "summary": summary_row,
    }
    artifact_path = weekly_run._candidate_artifact_path(run_dir, candidates[0]["candidate_key"])
    weekly_run._atomic_write_json(artifact_path, payload)

    df = pd.DataFrame()
    result = weekly_run._run_wf_compare(
        df,
        run_dir=run_dir,
        resume=True,
        dataset_fingerprint=dataset_fp,
        wf_run_fingerprint=wf_fp,
        checkpoint_per_fold=False,
        eval_last_n_seasons=1,
        wf_start_week=3,
        calibration_weeks=1,
        include_postseason=False,
        exclude_incomplete_seasons=False,
        recency_half_life_weeks=None,
        recency_half_life_seasons=None,
        market_mode="features",
        market_prob_source="raw",
        market_prob_blend_method="prob",
        win_prob_uncertainty="off",
        xgb_params_overrides={},
        early_stopping_rounds=1,
        include_quantiles=False,
    )

    assert call_count["count"] == 1
    assert set(result["candidate_key"]) == {"cand1", "cand2"}


def test_corrupt_artifact_recomputes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Corrupt candidate artifacts should be moved aside and recomputed."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_fp = {"sha256": "abc123", "path": "x", "size": 1, "mtime": 0.0}
    wf_fp = "wf123"
    candidate = _candidate("cand1")
    monkeypatch.setattr(weekly_run, "_build_wf_candidates", lambda **_kwargs: [candidate])

    artifact_path = weekly_run._candidate_artifact_path(run_dir, candidate["candidate_key"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("{bad json", encoding="utf-8")

    call_count = {"count": 0}

    def _fake_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        call_count["count"] += 1
        return _stub_results(call_count["count"])

    monkeypatch.setattr(weekly_run.walk_forward, "run_walk_forward_backtest", _fake_run)

    df = pd.DataFrame()
    weekly_run._run_wf_compare(
        df,
        run_dir=run_dir,
        resume=True,
        dataset_fingerprint=dataset_fp,
        wf_run_fingerprint=wf_fp,
        checkpoint_per_fold=False,
        eval_last_n_seasons=1,
        wf_start_week=3,
        calibration_weeks=1,
        include_postseason=False,
        exclude_incomplete_seasons=False,
        recency_half_life_weeks=None,
        recency_half_life_seasons=None,
        market_mode="features",
        market_prob_source="raw",
        market_prob_blend_method="prob",
        win_prob_uncertainty="off",
        xgb_params_overrides={},
        early_stopping_rounds=1,
        include_quantiles=False,
    )

    assert call_count["count"] == 1
    corrupt_files = list(artifact_path.parent.glob("wf_candidate_*corrupt*"))
    assert corrupt_files


def test_resume_after_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume should continue after a simulated interruption."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_fp = {"sha256": "abc123", "path": "x", "size": 1, "mtime": 0.0}
    wf_fp = "wf123"
    candidates = [_candidate("cand1"), _candidate("cand2")]
    monkeypatch.setattr(weekly_run, "_build_wf_candidates", lambda **_kwargs: candidates)

    call_count = {"count": 0}

    def _fake_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        call_count["count"] += 1
        if call_count["count"] == 2:
            raise RuntimeError("simulated interrupt")
        return _stub_results(call_count["count"])

    monkeypatch.setattr(weekly_run.walk_forward, "run_walk_forward_backtest", _fake_run)
    df = pd.DataFrame()

    with pytest.raises(RuntimeError):
        weekly_run._run_wf_compare(
            df,
            run_dir=run_dir,
            resume=False,
            dataset_fingerprint=dataset_fp,
            wf_run_fingerprint=wf_fp,
            checkpoint_per_fold=False,
            eval_last_n_seasons=1,
            wf_start_week=3,
            calibration_weeks=1,
            include_postseason=False,
            exclude_incomplete_seasons=False,
            recency_half_life_weeks=None,
            recency_half_life_seasons=None,
            market_mode="features",
            market_prob_source="raw",
            market_prob_blend_method="prob",
            win_prob_uncertainty="off",
            xgb_params_overrides={},
            early_stopping_rounds=1,
            include_quantiles=False,
        )

    assert weekly_run._candidate_artifact_path(run_dir, "cand1").exists()

    call_count["count"] = 0

    def _fake_run_resume(*_args: object, **_kwargs: object) -> dict[str, object]:
        call_count["count"] += 1
        return _stub_results(call_count["count"])

    monkeypatch.setattr(weekly_run.walk_forward, "run_walk_forward_backtest", _fake_run_resume)
    result = weekly_run._run_wf_compare(
        df,
        run_dir=run_dir,
        resume=True,
        dataset_fingerprint=dataset_fp,
        wf_run_fingerprint=wf_fp,
        checkpoint_per_fold=False,
        eval_last_n_seasons=1,
        wf_start_week=3,
        calibration_weeks=1,
        include_postseason=False,
        exclude_incomplete_seasons=False,
        recency_half_life_weeks=None,
        recency_half_life_seasons=None,
        market_mode="features",
        market_prob_source="raw",
        market_prob_blend_method="prob",
        win_prob_uncertainty="off",
        xgb_params_overrides={},
        early_stopping_rounds=1,
        include_quantiles=False,
    )

    assert call_count["count"] == 1
    assert set(result["candidate_key"]) == {"cand1", "cand2"}
