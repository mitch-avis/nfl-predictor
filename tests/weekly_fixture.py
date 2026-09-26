"""A small synthetic NFL dataset shaped like the ETL's outputs, for end-to-end tests.

``build_fixture`` writes ``completed_games_ml.csv``, ``all_data_ml.csv``, ``all_data.csv``,
``strength_snapshots.csv`` and one ``predict/week_05_games_to_predict.csv`` for 32 real team
abbreviations: three history seasons of six weeks, and a current season with four completed
weeks and three scheduled ones. Team strength drifts between seasons; scores, market lines and
the few features are drawn around it from one seeded generator, so every call writes the same
bytes. The column layout mirrors the real dataset: identifiers, then the positional feature
range from ``away_rest`` to ``home_moneyline`` (market columns last), then the scores.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nfl_predictor import constants

SHIPPED_CONFIG = Path(constants.ROOT_DIR) / "config" / "weekly_run.yaml"

TEAMS = sorted(constants.TEAM_TO_DIVISION)
HISTORY_SEASONS = (2021, 2022, 2023)
CURRENT_SEASON = 2024
HISTORY_WEEKS = 6
COMPLETED_CURRENT_WEEKS = 4
PREDICT_WEEK = 5
LAST_SCHEDULED_WEEK = 7
HOME_FIELD_POINTS = 1.5
MARGIN_NOISE = 10.0
SNAPSHOT_COLUMNS = (
    "season",
    "week",
    "team_abbr",
    "adj_off_pass_epa_snap",
    "adj_off_rush_epa_snap",
    "adj_def_pass_epa_snap",
    "adj_def_rush_epa_snap",
    "adj_srs",
    "st_rating",
    "adj_strength_composite",
    "strength_games_played",
    "sos_played_adj",
    "sos_remaining_adj",
    "sos_played_raw",
    "adj_hfa",
)
ID_COLUMNS = (
    "game_id",
    "season",
    "week",
    "game_type",
    "date",
    "gametime",
    "game_datetime",
    "away_abbr",
    "home_abbr",
    "away_qb",
    "home_qb",
    "stadium_name",
    "stadium_city",
    "stadium_state",
    "away_coach",
    "home_coach",
)
# The model reads the columns from ``away_rest`` through ``home_moneyline`` by position, as
# in the real dataset, with the five market columns last and the scores after them.
FEATURE_COLUMNS = (
    "away_rest",
    "home_rest",
    "away_power",
    "home_power",
    "away_points_per_game",
    "home_points_per_game",
    "away_points_allowed_per_game",
    "home_points_allowed_per_game",
    "power_diff",
    "total_line",
    "away_spread",
    "home_spread",
    "away_moneyline",
    "home_moneyline",
)
SCORE_COLUMNS = ("away_score", "home_score")


def american_odds(probability: float) -> int:
    """Return American odds for a probability that already includes the book's margin."""
    if probability >= 0.5:
        return -round(100 * probability / (1 - probability))
    return round(100 * (1 - probability) / probability)


def game_rows(
    rng: np.random.Generator, strength: dict[str, float], season: int, week: int
) -> Iterator[dict[str, Any]]:
    """Yield one synthetic game row per matchup of ``season``/``week``."""
    order = [str(team) for team in rng.permutation(TEAMS)]
    kickoff = date(season, 9, 8) + timedelta(days=7 * (week - 1))
    for away, home in zip(order[0::2], order[1::2], strict=True):
        edge = strength[home] - strength[away]
        expected_margin = edge + HOME_FIELD_POINTS
        expected_total = 44.0 + 0.3 * (abs(strength[home]) + abs(strength[away]))
        spread = round(2 * (expected_margin + rng.normal(0, 1.0))) / 2
        total_line = round(2 * (expected_total + rng.normal(0, 1.5))) / 2
        home_prob = 0.5 * (1 + math.erf(spread / (13.5 * math.sqrt(2))))
        margin = expected_margin + rng.normal(0, MARGIN_NOISE)
        total = max(expected_total + rng.normal(0, 9.0), abs(margin) + 3)
        home_score = max(0, round((total + margin) / 2))
        away_score = max(0, round((total - margin) / 2))
        yield {
            "game_id": f"{season}_{week:02d}_{away}_{home}",
            "season": season,
            "week": week,
            "game_type": "REG",
            "date": kickoff.isoformat(),
            "gametime": "13:00",
            "game_datetime": f"{kickoff.isoformat()}T13:00:00",
            "away_abbr": away,
            "home_abbr": home,
            "away_qb": f"{away} Starter",
            "home_qb": f"{home} Starter",
            "stadium_name": f"{home} Stadium",
            "stadium_city": f"{home} City",
            "stadium_state": "NA",
            "away_coach": f"{away} Coach",
            "home_coach": f"{home} Coach",
            "away_rest": int(rng.choice([6, 7, 7, 7, 10])),
            "home_rest": int(rng.choice([6, 7, 7, 7, 10])),
            "away_power": round(strength[away] + rng.normal(0, 2.0), 3),
            "home_power": round(strength[home] + rng.normal(0, 2.0), 3),
            "away_points_per_game": round(22 + 0.6 * strength[away] + rng.normal(0, 2.5), 3),
            "home_points_per_game": round(22 + 0.6 * strength[home] + rng.normal(0, 2.5), 3),
            "away_points_allowed_per_game": round(
                22 - 0.4 * strength[away] + rng.normal(0, 2.5), 3
            ),
            "home_points_allowed_per_game": round(
                22 - 0.4 * strength[home] + rng.normal(0, 2.5), 3
            ),
            "power_diff": round(edge + rng.normal(0, 2.5), 3),
            "total_line": total_line,
            "away_spread": spread,
            "home_spread": -spread,
            "away_moneyline": american_odds(min(0.97, (1 - home_prob) * 1.0225)),
            "home_moneyline": american_odds(min(0.97, home_prob * 1.0225)),
            "away_score": away_score,
            "home_score": home_score,
        }


def build_fixture(root: Path) -> dict[str, Path]:
    """Write the synthetic datasets the weekly run reads and return their paths."""
    rng = np.random.default_rng(20260924)
    strength = {team: float(rng.normal(0, 4.0)) for team in TEAMS}
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for season in (*HISTORY_SEASONS, CURRENT_SEASON):
        strength = {
            team: 0.7 * value + float(rng.normal(0, 2.5)) for team, value in strength.items()
        }
        last_week = HISTORY_WEEKS if season != CURRENT_SEASON else LAST_SCHEDULED_WEEK
        for week in range(1, last_week + 1):
            rows.extend(game_rows(rng, strength, season, week))
        if season == CURRENT_SEASON:
            for week in range(1, LAST_SCHEDULED_WEEK + 1):
                for team in TEAMS:
                    composite = strength[team] + float(rng.normal(0, 1.0))
                    snapshots.append(
                        {
                            "season": season,
                            "week": week,
                            "team_abbr": team,
                            "adj_off_pass_epa_snap": round(0.01 * composite, 5),
                            "adj_off_rush_epa_snap": round(0.005 * composite, 5),
                            "adj_def_pass_epa_snap": round(-0.01 * composite, 5),
                            "adj_def_rush_epa_snap": round(-0.005 * composite, 5),
                            "adj_srs": round(composite, 4),
                            "st_rating": round(float(rng.normal(0, 0.5)), 4),
                            "adj_strength_composite": round(composite / 4.0, 5),
                            "strength_games_played": week - 1,
                            "sos_played_adj": round(float(rng.normal(0, 1.0)), 4),
                            "sos_remaining_adj": round(float(rng.normal(0, 1.0)), 4),
                            "sos_played_raw": round(float(rng.normal(0, 1.0)), 4),
                            "adj_hfa": HOME_FIELD_POINTS,
                        }
                    )

    games = pd.DataFrame(rows, columns=[*ID_COLUMNS, *FEATURE_COLUMNS, *SCORE_COLUMNS])
    future = (games["season"] == CURRENT_SEASON) & (games["week"] > COMPLETED_CURRENT_WEEKS)
    games.loc[future, list(SCORE_COLUMNS)] = np.nan

    paths = {
        "completed": root / "completed_games_ml.csv",
        "all_ml": root / "all_data_ml.csv",
        "all": root / "all_data.csv",
        "snapshots": root / "strength_snapshots.csv",
        "predict": root / "predict" / f"week_{PREDICT_WEEK:02d}_games_to_predict.csv",
    }
    paths["predict"].parent.mkdir(parents=True, exist_ok=True)
    games.loc[~future].to_csv(paths["completed"], index=False)
    games.to_csv(paths["all_ml"], index=False)
    games.to_csv(paths["all"], index=False)
    predict_rows = (games["season"] == CURRENT_SEASON) & (games["week"] == PREDICT_WEEK)
    games.loc[predict_rows].to_csv(paths["predict"], index=False)
    pd.DataFrame(snapshots, columns=list(SNAPSHOT_COLUMNS)).to_csv(paths["snapshots"], index=False)
    return paths


def write_weekly_config(root: Path, paths: dict[str, Path], output_dir: Path) -> Path:
    """Write the shipped weekly config with only the fixture's required overrides."""
    config = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    config.update(
        {
            "run_id": "characterization",
            "run_dir": str(root / "run"),
            "output_dir": str(output_dir),
            "data_path": str(paths["completed"]),
            "predict_path": str(paths["predict"]),
            "skip_data_refresh": True,
            "resume": False,
            "wf_eval_last_n_seasons": 1,
            "xgb_device": "cpu",
            "xgb_n_jobs": 1,
            "power_rankings_data_ml": str(paths["all_ml"]),
            "power_rankings_data_schedule": str(paths["all"]),
            "power_rankings_strength_snapshots": str(paths["snapshots"]),
        }
    )
    config_path = root / "weekly_run.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return config_path
