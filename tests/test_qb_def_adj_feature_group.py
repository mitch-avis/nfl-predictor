"""The defense-adjusted quarterback group on the model side: ablation, leakage audit, fallbacks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from nfl_predictor import constants, ml_model
from nfl_predictor.ml import leakage_audit, ml_model_core, walk_forward
from nfl_predictor.utils.polars.finalize import build_final_column_order

_GROUP = "qb_def_adj"
_SEASONS = (2022, 2023)
_WEEKS = (1, 2, 3)
_GAMES_PER_WEEK = 6


def _fixture_dataset() -> pd.DataFrame:
    """Return a dataset in the published column order with random values, new columns included.

    Feature columns are random floats with some nulls; ``*_abbr`` columns are low-cardinality
    strings, so the categorical path runs too; the market lines are always present, because
    the anchor needs them.
    """
    rng = np.random.default_rng(11)
    order = build_final_column_order()
    rows = len(_SEASONS) * len(_WEEKS) * _GAMES_PER_WEEK
    keys = [
        (season, week, game)
        for season in _SEASONS
        for week in _WEEKS
        for game in range(_GAMES_PER_WEEK)
    ]
    start = order.index(ml_model.DEFAULT_FEATURE_START_COLUMN)
    data: dict[str, Any] = {}
    for index, column in enumerate(order):
        if column == "season":
            data[column] = [season for season, _, _ in keys]
        elif column == "week":
            data[column] = [week for _, week, _ in keys]
        elif column == "game_type":
            data[column] = ["REG"] * rows
        elif column == "game_id":
            data[column] = [f"{season}_{week:02d}_{game}" for season, week, game in keys]
        elif column in constants.RESULT_COLUMNS:
            data[column] = rng.integers(0, 40, rows)
        elif index < start or column.endswith("_abbr"):
            data[column] = rng.choice(["AAA", "BBB", "CCC"], rows)
        else:
            values = rng.normal(size=rows)
            if column not in constants.LINES_COLUMNS:
                values[rng.random(rows) < 0.1] = np.nan
            data[column] = values
    return pd.DataFrame(data, columns=order)


def _first_fold_matrix(
    df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, *, disabled: tuple[str, ...]
) -> tuple[list[str], np.ndarray]:
    """Return the feature columns and training matrix the first walk-forward fold fits on."""
    config = walk_forward.WalkForwardConfig(
        eval_seasons=[_SEASONS[-1]],
        wf_start_week=2,
        include_quantiles=False,
        disabled_feature_groups=disabled,
        xgb_params_overrides={"device": "cpu"},
    )
    run = walk_forward._prepare_run(df, config)
    captured: dict[str, Any] = {}
    build_feature_spec = ml_model.build_feature_spec

    class _CapturedError(Exception):
        """Stops the fold once its training matrix is built."""

    def record_spec(frame: pd.DataFrame, selection: ml_model.FeatureSelection) -> Any:
        captured["spec"] = build_feature_spec(frame, selection)
        return captured["spec"]

    def capture(data: ml_model.FitData, _params: dict[str, Any]) -> None:
        captured["x"] = data.x
        raise _CapturedError

    monkeypatch.setattr(ml_model, "build_feature_spec", record_spec)
    monkeypatch.setattr(ml_model, "fit_margin_total_models", capture)
    with pytest.raises(_CapturedError):
        walk_forward._fit_fold(run.folds[0], run, config)
    monkeypatch.undo()
    spec = captured["spec"]
    matrix = captured["x"]
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    return list(spec.feature_columns), dense


def test_disabling_the_group_gives_todays_feature_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the group off, a dataset carrying it trains on exactly the matrix of one without it."""
    with_group = _fixture_dataset()
    today = with_group.drop(columns=list(constants.QB_DEF_ADJ_FEATURE_COLUMNS))
    monkeypatch.setattr(constants, "QB_DEF_ADJ_STATS", [])
    assert list(today.columns) == build_final_column_order()
    monkeypatch.undo()

    columns_off, matrix_off = _first_fold_matrix(with_group, monkeypatch, disabled=(_GROUP,))
    columns_today, matrix_today = _first_fold_matrix(today, monkeypatch, disabled=())
    columns_on, _ = _first_fold_matrix(with_group, monkeypatch, disabled=())

    assert columns_off == columns_today
    np.testing.assert_array_equal(matrix_off, matrix_today)
    assert set(columns_on) - set(columns_today) == set(constants.QB_DEF_ADJ_FEATURE_COLUMNS)


def test_leakage_audit_covers_the_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """A score copied into a defense-adjusted column fails the audit under that column's name."""
    df = _fixture_dataset()
    df["away_qb_def_adj_epa"] = df["away_score"].astype(float)

    report = leakage_audit.run_leakage_audit(df, leakage_audit.LeakageAuditConfig())

    assert not report["ok"]
    assert any("away_qb_def_adj_epa" in failure for failure in report["failures"])


def test_missing_data_summary_reports_the_group() -> None:
    """Training reports the group's null cells and fallback rows like the other groups."""
    columns = list(constants.QB_DEF_ADJ_FEATURE_COLUMNS)
    df = pd.DataFrame({column: [0.1, np.nan, 0.2] for column in columns})
    df.loc[0, columns[0]] = np.nan

    summary = ml_model_core._summarize_missing_data(df)

    assert summary["groups"][_GROUP] == {
        "present_columns": len(columns),
        "missing_columns": 0,
        "null_cells": len(columns) + 1,
        "rows_with_any_null": 2,
        "columns_all_null": 0,
    }
