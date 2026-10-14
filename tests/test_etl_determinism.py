"""Tests that ETL outputs do not depend on the row order Polars happens to produce.

Polars' ``unique()`` and ``group_by()`` return rows in an unspecified order unless asked to
keep it, and a float sum depends on the order of its terms in the last bits. Two identical
ETL runs therefore only write byte-identical files when every reduction either sees its
inputs in a fixed order or sorts them first. Each test here feeds the same rows in many
row orders and requires exactly equal output, values and row order alike.
"""

from __future__ import annotations

import ast
from collections import Counter
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
    # Arrange
    team_games = _random_team_games()
    expected = schedule_strength.compute_schedule_strength_raw(
        team_games, season=_SEASON, week=15, margin=margin
    )

    # Act
    shuffled_runs = [
        schedule_strength.compute_schedule_strength_raw(
            _shuffled(team_games, seed), season=_SEASON, week=15, margin=margin
        )
        for seed in _SHUFFLE_SEEDS
    ]

    # Assert
    for actual in shuffled_runs:
        assert_frame_equal(actual, expected, check_exact=True)


def test_adjusted_schedule_strength_is_independent_of_input_row_order() -> None:
    """Both adjusted columns are bit-identical however the schedule is ordered."""
    # Arrange
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

    # Act
    shuffled_runs = [
        schedule_strength.compute_schedule_strength_adjusted(
            _shuffled(schedule, seed), _shuffled(ratings, seed), season=_SEASON, week=10
        )
        for seed in _SHUFFLE_SEEDS
    ]

    # Assert
    for actual in shuffled_runs:
        assert_frame_equal(actual, expected, check_exact=True)


def test_week_aggregation_is_independent_of_input_row_order() -> None:
    """Season-to-date means and their row order do not follow the input row order."""
    # Arrange
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

    # Act
    shuffled_runs = [
        teamrankings.aggregate_team_stats_to_week(_shuffled(team_stats, seed), 15, _SEASON)
        for seed in _SHUFFLE_SEEDS
    ]

    # Assert
    for actual in shuffled_runs:
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
    # Arrange
    games = _games_on_shared_dates()
    expected_ids = games.sort(["date", "game_id"], descending=[True, False])["game_id"]

    def combined_ids(seed: int) -> list[str]:
        shuffled = _shuffled(games, seed)
        frames = [shuffled.filter(pl.col("season") == season) for season in (2024, 2023)]
        return data_collection._combine_seasons(frames)["game_id"].to_list()

    # Act
    shuffled_runs = [combined_ids(seed) for seed in _SHUFFLE_SEEDS]

    # Assert
    for actual in shuffled_runs:
        assert actual == expected_ids.to_list()


def test_newest_first_sort_breaks_date_ties_by_game_id() -> None:
    """The final published order is a total order: date descending, then game id."""
    # Arrange
    games = _games_on_shared_dates()
    expected = games.sort(["date", "game_id"], descending=[True, False])

    # Act
    shuffled_runs = [
        data_collection._sort_newest_first(_shuffled(games, seed)) for seed in _SHUFFLE_SEEDS
    ]

    # Assert
    for actual in shuffled_runs:
        assert_frame_equal(actual, expected, check_exact=True)


def test_newest_first_sort_without_game_ids_keeps_same_date_rows_in_input_order() -> None:
    """Without a game id the sort is stable."""
    # Arrange
    games = pl.DataFrame(
        {"date": ["2024-09-08", "2024-09-15", "2024-09-08"], "home_abbr": ["NYJ", "MIA", "BUF"]}
    )

    # Act
    ordered = data_collection._sort_newest_first(games)

    # Assert
    assert ordered["home_abbr"].to_list() == ["MIA", "NYJ", "BUF"]


def test_newest_first_sort_without_a_date_leaves_the_frame_unchanged() -> None:
    # Arrange
    undated = pl.DataFrame({"home_abbr": ["NYJ", "MIA", "BUF"]})

    # Act
    ordered = data_collection._sort_newest_first(undated)

    # Assert
    assert_frame_equal(ordered, undated)


_PACKAGE = Path(nfl_predictor.__file__).parent
# Modules whose frames reach the files `nfl-predictor data` writes, the ratings and season
# statistics packages included.
_ETL_PATHS = (
    _PACKAGE / "data_collection.py",
    _PACKAGE / "week_builder.py",
    _PACKAGE / "utils" / "game_utils.py",
    _PACKAGE / "utils" / "scraping_utils.py",
    *sorted((_PACKAGE / "utils" / "polars").glob("*.py")),
    *sorted((_PACKAGE / "ratings").glob("*.py")),
    *sorted((_PACKAGE / "season_stats").rglob("*.py")),
)
_ORDERED_CALLS = frozenset({"group_by", "unique"})
_NUMPY_NAMES = frozenset({"np", "numpy"})
_DETERMINISTIC_KEEP = frozenset({"first", "last", "none"})
_UNORDERED = "without maintain_order=True"
_ANY_KEPT = "with a subset but keep='any' or no keep"
# A call that leaves row order or the kept row open: its file, enclosing function, call and
# problem.
type _Violation = tuple[str, str, str, str]
# The season statistics package's group-bys and uniques that leave row order open, as ported and
# reviewed with the package, counted by file, enclosing function and call. They stay as ported:
# asking for the order could move a published float's last bits, since Polars sums a group's
# floats in another order on another code path (the package's README, "Determinism"), and the
# parity audits match the published files bit for bit. Listing them makes a new one fail below.
_REVIEWED_UNORDERED_CALLS: Counter[_Violation] = Counter(
    {
        ("game_logs.py", "official_weekly_qb_stats", "unique", _UNORDERED): 1,
        ("game_logs.py", "two_team_games_only", "group_by", _UNORDERED): 1,
        ("opponent_stats.py", "compute_all_opponent_profiles", "unique", _UNORDERED): 1,
        ("qb_stats.py", "_canonicalize_qb_rows", "unique", _UNORDERED): 2,
        ("qb_stats.py", "_compute_team_late_game_flags_from_pbp", "group_by", _UNORDERED): 3,
        ("qb_stats.py", "_qb_primary_team_map", "group_by", _UNORDERED): 1,
        ("qb_stats.py", "_qb_spike_stats", "group_by", _UNORDERED): 1,
        ("qb_stats.py", "_with_attempt_qualifier", "group_by", _UNORDERED): 1,
        ("qb_stats.py", "_with_qb_results", "group_by", _UNORDERED): 1,
        ("qb_stats.py", "compute_qb_game_stats_from_pbp", "group_by", _UNORDERED): 2,
        ("qb_stats.py", "compute_qb_game_volumes_from_pbp", "group_by", _UNORDERED): 3,
        ("qb_stats.py", "compute_qb_season_stats", "group_by", _UNORDERED): 1,
        ("team_stats.py", "_aggregate_defense_only_player_stats", "group_by", _UNORDERED): 1,
        ("team_stats.py", "compute_all_teams_per_game", "group_by", _UNORDERED): 1,
        ("team_stats.py", "compute_team_game_stats_from_pbp", "group_by", _UNORDERED): 2,
        ("team_stats.py", "compute_team_snap_counts_from_pbp", "group_by", _UNORDERED): 1,
        ("team_stats.py", "compute_win_totals", "group_by", _UNORDERED): 1,
        ("team_stats_expanded.py", "_aggregate_drive_stats", "group_by", _UNORDERED): 2,
        ("team_stats_expanded.py", "_aggregate_play_stats", "group_by", _UNORDERED): 1,
        ("team_stats_expanded.py", "_aggregate_series_stats", "group_by", _UNORDERED): 2,
        ("team_stats_expanded.py", "_join_committed_penalties", "group_by", _UNORDERED): 1,
        ("team_stats_expanded.py", "_join_touchdown_totals", "group_by", _UNORDERED): 1,
        ("wp_bins.py", "_unit_bins", "group_by", _UNORDERED): 1,
        ("wp_bins.py", "compute_qb_wp_bins", "group_by", _UNORDERED): 1,
    }
)


def _keyword(call: ast.Call, name: str) -> object:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def _enclosing_functions(tree: ast.Module) -> dict[int, str]:
    """Return the name of every node's innermost enclosing function, by node id."""
    owners: dict[int, str] = {}

    def visit(node: ast.AST, owner: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = (
                child.name if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) else owner
            )
            owners[id(child)] = name
            visit(child, name)

    visit(tree, "<module>")
    return owners


def _order_violations(path: Path) -> list[tuple[_Violation, int]]:
    """Return every group_by/unique call in ``path`` that leaves row order or the kept row open.

    Each comes with its line, keyed by the file, the enclosing function, the call and the problem.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owners = _enclosing_functions(tree)
    violations: list[tuple[_Violation, int]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in _ORDERED_CALLS:
            continue
        # numpy.unique always returns its values sorted, and takes no maintain_order.
        if isinstance(node.func.value, ast.Name) and node.func.value.id in _NUMPY_NAMES:
            continue
        where = (path.name, owners[id(node)], node.func.attr)
        if _keyword(node, "maintain_order") is not True:
            violations.append(((*where, _UNORDERED), node.lineno))
        has_subset = any(keyword.arg == "subset" for keyword in node.keywords)
        if has_subset and _keyword(node, "keep") not in _DETERMINISTIC_KEEP:
            violations.append(((*where, _ANY_KEPT), node.lineno))
    return violations


def test_etl_group_by_and_unique_calls_keep_a_deterministic_order() -> None:
    """Every ETL group_by and unique keeps row order, and every subset dedupe names its row.

    The season statistics package's reviewed calls are the only exceptions, each counted, so a new
    call that leaves the order open fails, as does a listed one that is gone.
    """
    # Act
    violations = [violation for path in _ETL_PATHS for violation in _order_violations(path)]

    # Assert
    assert all(path.is_file() for path in _ETL_PATHS)
    found = Counter(key for key, _ in violations)
    unreviewed = [
        f"{file}:{line} {call} in {function} {problem}"
        for (file, function, call, problem), line in violations
        if found[file, function, call, problem]
        > _REVIEWED_UNORDERED_CALLS[file, function, call, problem]
    ]
    assert unreviewed == []
    assert found == _REVIEWED_UNORDERED_CALLS


def test_league_means_do_not_depend_on_where_the_frame_splits_into_chunks() -> None:
    """Identical values laid out with different chunk splits give identical league means."""
    # Arrange
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

    # Act
    split_runs = [
        teamrankings.calculate_league_means(split_at(row), 2020) for row in (97, 211, 333)
    ]

    # Assert
    for actual in split_runs:
        assert actual == expected
