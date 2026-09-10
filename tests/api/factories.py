"""Builders for synthetic run directories and artifact files.

``data/`` and ``models/`` are gitignored, so every API test fabricates the files it reads. The
headers below mirror the real artifacts written by ``scripts/weekly_run.py`` and the training CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREDICTION_COLUMNS = [
    "game_id", "season", "week", "date", "gametime", "game_datetime",
    "away_abbr", "home_abbr", "away_qb", "home_qb", "away_rest", "home_rest",
    "stadium_name", "is_divisional_matchup",
    "away_elo_pre", "home_elo_pre", "away_qb_elo_pre", "home_qb_elo_pre",
    "away_adj_strength_composite", "home_adj_strength_composite",
    "total_line", "away_spread", "home_spread", "away_moneyline", "home_moneyline",
    "away_score", "home_score",
    "predicted_away_score_raw", "predicted_home_score_raw", "predicted_total_raw",
    "predicted_margin_raw", "predicted_away_score", "predicted_home_score",
    "predicted_total", "predicted_margin", "home_win_prob", "away_win_prob",
    "predicted_winner", "confidence_strength", "confidence_rank",
    "predicted_margin_p10", "predicted_margin_p50", "predicted_margin_p90",
    "predicted_total_p10", "predicted_total_p50", "predicted_total_p90",
    "unregistered_feature",
]  # fmt: skip

PICK_COLUMNS = [
    "season", "week", "date", "game_id", "away_abbr", "home_abbr", "predicted_winner",
    "home_win_prob", "away_win_prob", "confidence_strength", "confidence_rank",
]  # fmt: skip

BETTING_COLUMNS = [
    "game_id", "date", "matchup", "predicted_away_score", "predicted_home_score",
    "predicted_total", "predicted_margin", "model_home_prob", "market_home_prob_novig",
    "edge_home_prob", "away_moneyline", "home_moneyline", "model_fair_away_moneyline",
    "model_fair_home_moneyline", "moneyline_value_side", "moneyline_edge_prob",
    "moneyline_confidence_1_10", "moneyline_action", "away_spread", "home_spread",
    "spread_value_side", "spread_edge_points", "total_line", "total_value_side",
    "total_edge_points",
]  # fmt: skip

POWER_COLUMNS = [
    "team_abbr", "division", "conference", "rating_raw", "power_rating_1_10",
    "power_rating_0_10", "wins", "losses", "ties", "games_played", "home_advantage_logit",
    "home_advantage_prob", "season", "through_week", "rank",
]  # fmt: skip

STANDINGS_COLUMNS = [
    "team_abbr", "wins", "losses", "ties", "games_played", "exp_wins", "games_remaining",
    "projected_wins", "projected_losses", "division", "conference", "projected_win_pct",
    "season", "through_week",
]  # fmt: skip

GAMES: list[tuple[str, str, str, float, float, float, int, int]] = [
    # away, home, date, home_win_prob, home_spread, total_line, away_ml, home_ml
    ("DEN", "KC", "2026-09-14", 0.4023, -3.0, 43.5, 130, -155),
    ("BUF", "HOU", "2026-09-13", 0.3322, 1.5, 44.5, -118, -102),
    ("GB", "MIN", "2026-09-13", 0.3700, -1.5, 46.5, 100, -120),
    ("NYJ", "NE", "2026-09-13", 0.7100, -6.5, 41.0, 240, -290),
]


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> Path:
    """Write ``rows`` under ``columns`` as CSV."""
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join("" if v is None else str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def prediction_rows(season: int, week: int) -> list[list[Any]]:
    """Return prediction rows for the fixture games."""
    rows: list[list[Any]] = []
    for rank, (away, home, date, p_home, home_spread, total, away_ml, home_ml) in enumerate(
        GAMES, 1
    ):
        margin = round((p_home - 0.5) * 30, 1)
        home_score = round((total + margin) / 2, 1)
        away_score = round((total - margin) / 2, 1)
        winner = home if p_home >= 0.5 else away
        rows.append([
            f"{season}_{week:02d}_{away}_{home}", season, week, date, "13:00",
            f"{date} 13:00:00", away, home, "QB A", "QB B", 7, 7, "Stadium", 0,
            1500.0, 1520.0, 100.0, 110.0, 0.5, 0.7,
            total, -home_spread, home_spread, away_ml, home_ml, None, None,
            away_score, home_score, total, margin, away_score, home_score, total, margin,
            p_home, round(1 - p_home, 4), winner, round(abs(p_home - 0.5), 4), rank,
            margin - 17, margin, margin + 17, total - 14, total, total + 16, 1,
        ])  # fmt: skip
    return rows


def write_predictions_csv(path: Path, season: int = 2026, week: int = 1) -> Path:
    """Write a predictions CSV with the real column set (plus one unregistered column)."""
    return _write_csv(path, PREDICTION_COLUMNS, prediction_rows(season, week))


def write_picks_csv(path: Path, season: int = 2026, week: int = 1) -> Path:
    """Write a confidence-picks CSV."""
    rows = []
    for rank, (away, home, date, p_home, *_rest) in enumerate(GAMES, 1):
        winner = home if p_home >= 0.5 else away
        rows.append([
            season, week, date, f"{season}_{week:02d}_{away}_{home}", away, home, winner,
            p_home, round(1 - p_home, 4), round(abs(p_home - 0.5), 4), rank,
        ])  # fmt: skip
    rows.sort(key=lambda r: -int(r[-1]))
    return _write_csv(path, PICK_COLUMNS, rows)


def write_betting_csv(path: Path, season: int = 2026, week: int = 1) -> Path:
    """Write a betting-report CSV."""
    rows = []
    for away, home, date, p_home, home_spread, total, away_ml, home_ml in GAMES:
        rows.append([
            f"{season}_{week:02d}_{away}_{home}", date, f"{away} @ {home}", 21.0, 24.0, total,
            3.0, p_home, 0.55, round(p_home - 0.55, 4), away_ml, home_ml, 150, -170,
            home, 0.05, 5, "SMALL", -home_spread, home_spread, home, 1.5, total, "OVER", 0.4,
        ])  # fmt: skip
    return _write_csv(path, BETTING_COLUMNS, rows)


TEAMS = [("KC", "AFC West", "AFC"), ("DEN", "AFC West", "AFC"), ("BUF", "AFC East", "AFC"),
         ("GB", "NFC North", "NFC"), ("MIN", "NFC North", "NFC")]  # fmt: skip


def write_power_csvs(
    run_dir: Path, season: int, through_week: int, order: list[str] | None = None
) -> None:
    """Write power rankings and both projected-standings CSVs stamped ``through_week``."""
    order = order or [team for team, *_ in TEAMS]
    info = {team: (division, conference) for team, division, conference in TEAMS}
    power_rows = []
    standings_rows = []
    for rank, team in enumerate(order, 1):
        division, conference = info[team]
        rating = 2.0 - rank * 0.4
        power_rows.append([
            team, division, conference, rating, round(1 + 9 / (1 + 2.718 ** -rating), 2),
            round(10 / (1 + 2.718 ** -rating), 2), 10 - rank, rank, 0, 10, 0.25, 0.56,
            season, through_week, rank,
        ])  # fmt: skip
        standings_rows.append([
            team, 10 - rank, rank, 0, 10, 4.5, 7, 14.5 - rank, 2.5 + rank, division,
            conference, round((14.5 - rank) / 17, 4), season, through_week,
        ])  # fmt: skip
    suffix = f"season_{season}_week_{through_week:02d}.csv"
    _write_csv(run_dir / f"power_rankings_{suffix}", POWER_COLUMNS, power_rows)
    _write_csv(run_dir / f"projected_standings_{suffix}", STANDINGS_COLUMNS, standings_rows)
    division_rows = [row + [1 + i % 2] for i, row in enumerate(standings_rows)]
    _write_csv(
        run_dir / f"projected_division_standings_{suffix}",
        [*STANDINGS_COLUMNS, "projected_division_rank"],
        division_rows,
    )


def metadata_payload(
    run_id: str, created_at: str, *, kind: str, season: int, week: int
) -> dict[str, Any]:
    """Return a metadata.json payload shaped like the real artifact."""
    config: dict[str, Any] = {
        "model_kind": "margin_total",
        "data_path": "data/completed_games_ml.csv",
        "feature_start": "away_rest",
        "feature_end": "home_moneyline",
        "win_prob_calibration": "platt",
        "market_prob_blend": 0.2,
        "score_rounding": "nfl",
        "predict_path": f"data/predict/week_{week:02d}_games_to_predict.csv",
    }
    if kind == "weekly":
        config["power_rankings_season"] = season
        config["power_rankings_through_week"] = week - 1
    payload: dict[str, Any] = {
        "created_at": created_at,
        "run_id": run_id,
        "git_commit_hash": "abc123def456",
        "dataset_hash": "5b6af6aa" + "0" * 56,
        "library_versions": {"python": "3.14.7", "xgboost": "3.4.1"},
        "config": config,
        "feature_list": ["away_rest", "home_rest", "away_elo_pre"],
        "splits": {"train_seasons": [2020, 2021], "holdout_seasons": [season - 1]},
        "params": {"max_depth": 4, "n_estimators": 200},
        "tuned_params": None,
        "early_stopping": {"margin_model.best_iteration": 120},
        "optuna_summary": None,
    }
    if kind == "walk_forward":
        payload["splits"] = {
            "eval_window": {},
            "resolved_eval_seasons": [season - 1],
            "wf_start_week": 3,
        }
        del payload["params"]
    return payload


def metrics_payload(run_id: str, created_at: str, *, kind: str) -> dict[str, Any]:
    """Return a metrics_report.json payload for a training or walk-forward run."""
    if kind == "walk_forward":
        return {
            "run_id": run_id,
            "created_at": created_at,
            "metric_strategy": {
                "primary": {"metrics": ["brier", "log_loss"], "direction": "lower"},
            },
            "metrics": {
                "overall": {
                    "brier": 0.2277,
                    "log_loss": 0.7431,
                    "pick_accuracy": 0.6958,
                    "games": 720,
                },
                "per_season": [{"season": 2025, "brier": 0.23, "log_loss": 0.75}],
                "per_week": [{"season": 2025, "week": 3, "brier": 0.21, "log_loss": 0.7}],
                "summary_table": [
                    {
                        "metric": "brier",
                        "priority": "primary",
                        "direction": "lower",
                        "overall": 0.2277,
                    }
                ],
            },
            "calibration": {
                "bin_count": 2,
                "bins": [
                    {
                        "bin_lower": 0.0,
                        "bin_upper": 0.5,
                        "count": 300,
                        "avg_pred": 0.35,
                        "avg_actual": 0.36,
                    },
                    {
                        "bin_lower": 0.5,
                        "bin_upper": 1.0,
                        "count": 420,
                        "avg_pred": 0.66,
                        "avg_actual": 0.64,
                    },
                ],
            },
        }
    return {
        "run_id": run_id,
        "created_at": created_at,
        "metrics": {
            "kind": "training",
            "model_kind": "margin_total",
            "metrics": {
                "holdout": {
                    "winner_accuracy": 0.6471,
                    "brier": 0.2192,
                    "margin_mae": 9.847,
                    "total_mae": 10.997,
                }
            },
            "pool": {
                "weeks": 18,
                "weekly_picks_correct_avg": 9.61,
                "weekly_expected_points_avg": 96.56,
                "weekly_actual_points_avg": 87.22,
            },
            "missing_data": {
                "total_rows": 6952,
                "groups": {
                    "lines": {
                        "present_columns": 5,
                        "missing_columns": 0,
                        "null_cells": 12,
                        "rows_with_any_null": 3,
                    }
                },
            },
        },
    }


def feature_importance_payload(run_id: str) -> dict[str, Any]:
    """Return a feature_importance.json payload."""
    names = ["away_rest", "home_rest", "away_elo_pre"]
    return {
        "run_id": run_id,
        "model_kind": "margin_total",
        "feature_names": [f"num__{n}" for n in names],
        "base_features": {
            "feature_names": names,
            "margin": {"gain": [1.0, 2.0, 3.0], "weight": [1, 2, 3]},
            "total": {"gain": [0.5, 0.5, 0.5], "weight": [1, 1, 1]},
            "combined": {"gain": [1.5, 2.5, 3.5], "weight": [2, 3, 4]},
        },
    }


def make_run_dir(
    models_dir: Path,
    run_id: str,
    *,
    kind: str = "weekly",
    season: int = 2026,
    week: int = 1,
    created_at: str = "2026-09-09T23:29:04+00:00",
    complete: bool = True,
    with_model: bool = True,
    with_xlsx: bool = False,
    power_order: list[str] | None = None,
) -> Path:
    """Create a run directory of ``kind`` (``weekly``, ``training``, or ``walk_forward``)."""
    run_dir = models_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata_payload(run_id, created_at, kind=kind, season=season, week=week)),
        encoding="utf-8",
    )
    (run_dir / "metrics_report.json").write_text(
        json.dumps(metrics_payload(run_id, created_at, kind=kind)), encoding="utf-8"
    )
    if kind == "walk_forward":
        return run_dir
    if with_model:
        (run_dir / "model.joblib").write_bytes(b"not-a-real-model")
    (run_dir / "feature_importance.json").write_text(
        json.dumps(feature_importance_payload(run_id)), encoding="utf-8"
    )
    if kind == "training":
        return run_dir
    prefix = f"season_{season}_week_{week:02d}"
    write_predictions_csv(run_dir / f"{prefix}_predictions.csv", season, week)
    write_picks_csv(run_dir / f"{prefix}_confidence_picks.csv", season, week)
    write_betting_csv(run_dir / f"{prefix}_betting_report.csv", season, week)
    write_power_csvs(run_dir, season, week - 1, power_order)
    (run_dir / "wf_compare.csv").write_text(
        "candidate_key,label,calibration,brier,log_loss,pick_accuracy,rank\n"
        "a,base,platt,0.22,0.74,0.69,1\nb,elo,elo,0.23,0.75,0.68,2\n",
        encoding="utf-8",
    )
    (run_dir / "wf_best.json").write_text(
        json.dumps(
            {"candidate_key": "a", "label": "base", "brier": 0.22, "log_loss": 0.74, "rank": 1}
        ),
        encoding="utf-8",
    )
    if with_xlsx:
        (run_dir / "betting_report.xlsx").write_bytes(b"PK-fake-xlsx")
    stages = ["wf_compare", "train", "predictions"] + (["reports"] if complete else [])
    for stage in stages:
        (run_dir / f"{stage}_state.json").write_text(
            json.dumps({"stage": stage, "created_at": created_at, "outputs": []}), encoding="utf-8"
        )
    return run_dir
