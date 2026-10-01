"""Tests that ETL outputs do not depend on the row order Polars happens to produce.

Polars' ``unique()`` and ``group_by()`` return rows in an unspecified order unless asked to
keep it, and a float sum depends on the order of its terms in the last bits. Two identical
ETL runs therefore only write byte-identical files when every reduction either sees its
inputs in a fixed order or sorts them first. Each test here feeds the same rows in many
row orders and requires exactly equal output, values and row order alike.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import nfl_predictor
from nfl_predictor import data_collection
from nfl_predictor.utils.polars import schedule_strength, teamrankings

_SEASON = 2024
_WEEKS = 17
_TEAMS = [f"T{index:02d}" for index in range(32)]
_SHUFFLE_SEEDS = range(12)


def _random_team_games(seed: int = 0) -> pl.DataFrame:
    """Return a 32-team, 17-week season of mirrored team-game rows with random margins.

    Random non-round floats are what make a different summation order visible in the
    last bits; the hand-built fixtures elsewhere use values that sum exactly.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for week in range(1, _WEEKS + 1):
        order = rng.permutation(len(_TEAMS))
        for pair in range(len(_TEAMS) // 2):
            team, opponent = _TEAMS[order[2 * pair]], _TEAMS[order[2 * pair + 1]]
            margin = float(rng.normal())
            plays = float(rng.integers(100, 140))
            for subject, foe, sign in ((team, opponent, 1.0), (opponent, team, -1.0)):
                rows.append(
                    {
                        "season": _SEASON,
                        "week": week,
                        "team_abbr": subject,
                        "opponent_abbr": foe,
                        "epa_margin_per_play": sign * margin,
                        "epa_margin_sum": sign * margin * plays,
                        "total_play_count": plays,
                    }
                )
    return pl.DataFrame(rows)


def _shuffled(frame: pl.DataFrame, seed: int) -> pl.DataFrame:
    return frame.sample(fraction=1.0, shuffle=True, seed=seed)


@pytest.mark.parametrize(
    "margin",
    [
        schedule_strength.MarginSource(numerator="epa_margin_sum", denominator="total_play_count"),
        schedule_strength.PER_GAME_MARGIN,
    ],
    ids=["ratio_of_sums", "mean_of_rates"],
)
def test_raw_schedule_strength_is_independent_of_input_row_order(
    margin: schedule_strength.MarginSource,
) -> None:
    """The one-hop schedule strength is bit-identical however the team-games are ordered."""
    team_games = _random_team_games()
    expected = schedule_strength.compute_schedule_strength_raw(
        team_games, season=_SEASON, week=15, margin=margin
    )
    for seed in _SHUFFLE_SEEDS:
        actual = schedule_strength.compute_schedule_strength_raw(
            _shuffled(team_games, seed), season=_SEASON, week=15, margin=margin
        )
        assert_frame_equal(actual, expected, check_exact=True)


def test_adjusted_schedule_strength_is_independent_of_input_row_order() -> None:
    """Both adjusted columns are bit-identical however the schedule is ordered."""
    team_games = _random_team_games()
    schedule = team_games.filter(pl.col("team_abbr") < pl.col("opponent_abbr")).select(
        "season",
        "week",
        pl.col("team_abbr").alias("home_abbr"),
        pl.col("opponent_abbr").alias("away_abbr"),
    )
    rng = np.random.default_rng(1)
    ratings = pl.DataFrame({"team_abbr": _TEAMS, "rating": rng.normal(size=len(_TEAMS))})
    expected = schedule_strength.compute_schedule_strength_adjusted(
        schedule, ratings, season=_SEASON, week=10
    )
    for seed in _SHUFFLE_SEEDS:
        actual = schedule_strength.compute_schedule_strength_adjusted(
            _shuffled(schedule, seed), _shuffled(ratings, seed), season=_SEASON, week=10
        )
        assert_frame_equal(actual, expected, check_exact=True)


def test_week_aggregation_is_independent_of_input_row_order() -> None:
    """Season-to-date means and their row order do not follow the input row order."""
    rng = np.random.default_rng(2)
    rows: list[dict[str, object]] = []
    for week in range(1, _WEEKS + 1):
        for team in _TEAMS:
            row: dict[str, object] = {
                "season": _SEASON,
                "week": week,
                "team_abbr": team,
                "opponent_abbr": "OPP",
                "points": int(rng.integers(0, 45)),
            }
            row.update({f"stat_{index}": float(rng.normal()) for index in range(8)})
            rows.append(row)
    team_stats = pl.DataFrame(rows)

    expected = teamrankings.aggregate_team_stats_to_week(team_stats, 15, _SEASON)
    for seed in _SHUFFLE_SEEDS:
        actual = teamrankings.aggregate_team_stats_to_week(_shuffled(team_stats, seed), 15, _SEASON)
        assert_frame_equal(actual, expected, check_exact=True)
    assert expected["team_abbr"].to_list() == sorted(_TEAMS)


def _games_on_shared_dates() -> pl.DataFrame:
    """Return two seasons of games where several games share each kickoff date."""
    rows = [
        {"game_id": f"{season}_{week:02d}_{away}_{home}", "date": f"{season}-09-{week + 10}"}
        | {"season": season, "week": week, "away_abbr": away, "home_abbr": home}
        for season in (2023, 2024)
        for week in (1, 2)
        for away, home in (("BUF", "MIA"), ("NE", "NYJ"), ("KC", "DEN"), ("LA", "SEA"))
    ]
    return pl.DataFrame(rows)


def test_combined_seasons_have_a_fixed_row_order_within_a_date() -> None:
    """Games sharing a date come out newest first, then by game id, from any input order."""
    games = _games_on_shared_dates()
    expected_ids = games.sort(["date", "game_id"], descending=[True, False])["game_id"]
    for seed in _SHUFFLE_SEEDS:
        shuffled = _shuffled(games, seed)
        frames = [shuffled.filter(pl.col("season") == season) for season in (2024, 2023)]
        combined = data_collection._combine_seasons(frames)
        assert combined["game_id"].to_list() == expected_ids.to_list()


def test_newest_first_sort_breaks_date_ties_by_game_id() -> None:
    """The final published order is a total order: date descending, then game id."""
    games = _games_on_shared_dates()
    expected = games.sort(["date", "game_id"], descending=[True, False])
    for seed in _SHUFFLE_SEEDS:
        actual = data_collection._sort_newest_first(_shuffled(games, seed))
        assert_frame_equal(actual, expected, check_exact=True)


def test_newest_first_sort_without_game_ids_keeps_same_date_rows_in_input_order() -> None:
    """Without a game id the sort is stable; without a date the frame is unchanged."""
    games = pl.DataFrame(
        {"date": ["2024-09-08", "2024-09-15", "2024-09-08"], "home_abbr": ["NYJ", "MIA", "BUF"]}
    )
    assert data_collection._sort_newest_first(games)["home_abbr"].to_list() == [
        "MIA",
        "NYJ",
        "BUF",
    ]
    undated = games.drop("date")
    assert_frame_equal(data_collection._sort_newest_first(undated), undated)


_PACKAGE = Path(nfl_predictor.__file__).parent
# Modules whose frames reach the files `nfl-predictor data` writes.
_ETL_PATHS = (
    _PACKAGE / "data_collection.py",
    _PACKAGE / "week_builder.py",
    _PACKAGE / "utils" / "game_utils.py",
    _PACKAGE / "utils" / "scraping_utils.py",
    *sorted((_PACKAGE / "utils" / "polars").glob("*.py")),
)
_ORDERED_CALLS = frozenset({"group_by", "unique"})
_DETERMINISTIC_KEEP = frozenset({"first", "last", "none"})


def _keyword(call: ast.Call, name: str) -> object:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _order_violations(path: Path) -> list[str]:
    """Return every group_by/unique call in ``path`` that leaves row order or the kept row open."""
    violations: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in _ORDERED_CALLS:
            continue
        where = f"{path.name}:{node.lineno} {node.func.attr}"
        if _keyword(node, "maintain_order") is not True:
            violations.append(f"{where} without maintain_order=True")
        has_subset = any(keyword.arg == "subset" for keyword in node.keywords)
        if has_subset and _keyword(node, "keep") not in _DETERMINISTIC_KEEP:
            violations.append(f"{where} with a subset but keep='any' or no keep")
    return violations


def test_etl_group_by_and_unique_calls_keep_a_deterministic_order() -> None:
    """Every ETL group_by and unique keeps row order, and every subset dedupe names its row."""
    assert all(path.is_file() for path in _ETL_PATHS)
    violations = [violation for path in _ETL_PATHS for violation in _order_violations(path)]
    assert violations == []


@pytest.mark.xfail(
    strict=True,
    reason=(
        "calculate_league_means reduces an eagerly filtered slice whose chunk boundaries move "
        "with the whole frame's length, so later seasons' rows move the last bits"
    ),
)
def test_league_means_do_not_depend_on_where_the_frame_splits_into_chunks() -> None:
    """Identical values laid out with different chunk splits give identical league means."""
    rng = np.random.default_rng(3)
    frame = pl.DataFrame(
        {
            "season": [2019] * 40 + [2020] * 400 + [2021] * 60,
            "week": rng.integers(1, 18, 500),
            "team_abbr": [f"T{index % 32:02d}" for index in range(500)],
            "passing_epa": rng.normal(0.0, 7.0, 500),
            "rushing_epa": rng.normal(0.0, 4.0, 500),
        }
    )

    def split_at(row: int) -> pl.DataFrame:
        return pl.concat([frame.slice(0, row), frame.slice(row)], rechunk=False)

    expected = teamrankings.calculate_league_means(frame, 2020)
    for row in (97, 211, 333):
        assert teamrankings.calculate_league_means(split_at(row), 2020) == expected
