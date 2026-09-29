"""Tests for the weekly run's walk-forward configuration key and dataset fingerprint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nfl_predictor.ml import wf_compare_utils
from nfl_predictor.utils import fingerprints

if TYPE_CHECKING:
    from pathlib import Path


def test_candidate_key_stability() -> None:
    """Candidate keys should be stable for identical inputs."""
    key_a = wf_compare_utils.build_candidate_key(
        wf_compare_utils.CandidateSpec(
            model_kind="margin_total",
            feature_start="away_rest",
            feature_end="home_moneyline",
            market_mode="features",
            include_quantiles=False,
            xgb_params_overrides={"n_estimators": 50},
        )
    )
    key_b = wf_compare_utils.build_candidate_key(
        wf_compare_utils.CandidateSpec(
            model_kind="margin_total",
            feature_start="away_rest",
            feature_end="home_moneyline",
            market_mode="features",
            include_quantiles=False,
            xgb_params_overrides={"n_estimators": 50},
        )
    )
    key_c = wf_compare_utils.build_candidate_key(
        wf_compare_utils.CandidateSpec(
            model_kind="margin_total",
            feature_start="away_rest",
            feature_end="home_moneyline",
            market_mode="anchor",
            include_quantiles=False,
            xgb_params_overrides={"n_estimators": 50},
        )
    )

    assert key_a == key_b
    assert key_a != key_c


def test_candidate_key_uses_default_xgb_tag_for_missing_or_unmapped_overrides() -> None:
    """Missing or unmapped XGBoost overrides should fall back to the default tag."""
    base_kwargs = {
        "model_kind": "margin_total",
        "feature_start": "away_rest",
        "feature_end": "home_moneyline",
        "market_mode": "features",
        "include_quantiles": False,
    }

    key_none = wf_compare_utils.build_candidate_key(
        wf_compare_utils.CandidateSpec(**base_kwargs, xgb_params_overrides=None)
    )
    key_unmapped = wf_compare_utils.build_candidate_key(
        wf_compare_utils.CandidateSpec(**base_kwargs, xgb_params_overrides={"gamma": 1.0})
    )

    assert "_xgb-default_" in key_none
    assert "_xgb-default_" in key_unmapped


def test_candidate_key_formats_float_and_string_xgb_overrides() -> None:
    """Recognized float and string XGBoost overrides should be encoded in the key."""
    key = wf_compare_utils.build_candidate_key(
        wf_compare_utils.CandidateSpec(
            model_kind="margin_total",
            feature_start="away_rest",
            feature_end="home_moneyline",
            market_mode="features",
            include_quantiles=False,
            xgb_params_overrides={"learning_rate": 0.125, "device": "cuda"},
        )
    )

    assert "_xgb-lr0.125-devcuda_" in key


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
