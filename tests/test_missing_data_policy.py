"""Tests for missing-data policy.

These tests focus on:
- invariant ETL schema (missing expected columns are kept as nulls)
- metrics/report visibility for missing-data prevalence

We intentionally keep these tests lightweight (no nflreadpy network calls).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from nfl_predictor import constants, ml_model
from nfl_predictor.utils import polars_utils


def test_select_final_columns_enforces_invariant_schema() -> None:
    """select_final_columns always produces the full expected schema."""
    final_order = polars_utils.build_final_column_order()

    # Minimal input missing most columns.
    df_min = pl.DataFrame(
        {
            "game_id": ["2020_01_BUF_KC"],
            "season": [2020],
            "week": [1],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
            "away_rest": [7],
        }
    )

    # Richer input includes an injury feature.
    df_rich = df_min.with_columns(
        [
            pl.lit(1.5).alias("away_injury_burden_total"),
            pl.lit(None).alias("home_moneyline"),
        ]
    )

    out_min = polars_utils.select_final_columns(df_min)
    out_rich = polars_utils.select_final_columns(df_rich)

    assert out_min.columns == final_order
    assert out_rich.columns == final_order

    # Spot-check that key output groups are always present.
    required_cols = {
        *constants.METADATA_COLUMNS,
        *constants.LINES_COLUMNS,
        *constants.RESULT_COLUMNS,
    }
    assert required_cols.issubset(set(out_min.columns))

    # Injury columns should always exist. When missing from input, they remain null.
    for col in constants.INJURY_FEATURE_COLUMNS:
        assert col in out_min.columns
        assert out_min.select(pl.col(col).is_null().all()).item() is True

    # Provided injury values should be preserved.
    assert out_rich.select(pl.col("away_injury_burden_total").is_null().any()).item() is False


def test_training_report_includes_missing_data_summary(tmp_path: Path, monkeypatch) -> None:
    """Training report includes missing-data prevalence summary."""
    # Minimal training CSV that includes the default feature range endpoints.
    # We leave injury and market fields entirely null to simulate historical missing coverage.
    df = pd.DataFrame(
        {
            "season": [2020, 2020, 2020],
            "week": [1, 2, 3],
            "away_rest": [7, 7, 7],
            "away_injury_burden_total": [None, None, None],
            "home_injury_burden_total": [None, None, None],
            "total_line": [None, None, None],
            "home_moneyline": [None, None, None],
            "away_score": [20, 17, 24],
            "home_score": [17, 21, 20],
        }
    )
    data_path = tmp_path / "minimal_training.csv"
    df.to_csv(data_path, index=False)

    class _DummyModel:
        tuned_cv_summary = None
        xgb_params = None
        tuned_params = None
        feature_spec = ml_model.FeatureSpec(
            feature_columns=["away_rest"],
            categorical_columns=[],
            numeric_columns=["away_rest"],
            dropped_columns=[],
            id_columns=[],
            constant_columns=[],
            high_cardinality_columns=[],
            feature_start="away_rest",
            feature_end="home_moneyline",
            metadata_columns=[],
            post_feature_columns=[],
            market_columns=[],
        )

    monkeypatch.setattr(ml_model, "train_margin_total_model", lambda **_: _DummyModel())

    optuna = ml_model.OptunaConfig(
        enabled=False,
        timeout_seconds=1,
        n_trials=None,
        cv_splits=2,
        objective="mae",
        early_stopping_rounds=5,
        tree_method=None,
        device=None,
        tune_scope="none",
        storage=None,
        study_name=None,
        best_params_out=None,
        xgb_n_jobs=1,
    )

    result = ml_model.train_margin_total_model_with_report(
        data_path=data_path,
        holdout_seasons=0,
        calibration_seasons=0,
        calibration_weeks=0,
        include_market=True,
        max_cardinality_ratio=1.0,
        win_prob_calibration="none",
        optuna_config=optuna,
        market_transform=False,
        market_anchor=False,
        market_prob_config=None,
    )

    missing = result.metrics_report["missing_data"]
    assert missing["total_rows"] == 3
    assert missing["groups"]["injuries"]["columns_all_null"] is not None
    assert missing["groups"]["injuries"]["columns_all_null"] >= 2
    assert missing["groups"]["lines"]["columns_all_null"] is not None
    assert missing["groups"]["lines"]["columns_all_null"] >= 2
