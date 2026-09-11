"""Tests for the early-season blend of season-to-date stats toward the regressed prior."""

from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_predictor import constants, data_collection
from nfl_predictor.data_collection import process_week
from nfl_predictor.utils import polars_utils

# Prior-season (2006) game, one per team. League means: pass_yards 250, pass_epa_sum 7.5,
# dropbacks 30, so the regressed BUF prior (factor 1/3) is pass_yards 283.33,
# pass_epa_sum 10.5 and dropbacks 30, an epa_per_dropback of 0.35.
_PRIOR_ROWS = {
    "season": [2006, 2006],
    "week": [1, 1],
    "team_abbr": ["BUF", "KC"],
    "opponent_abbr": ["KC", "BUF"],
    "pass_yards": [300.0, 200.0],
    "points_scored": [24.0, 17.0],
    "points_allowed": [17.0, 24.0],
    "dropbacks": [30.0, 30.0],
    "pass_epa_sum": [12.0, 3.0],
    "offensive_snaps": [60.0, 60.0],
}

# Current-season (2007) week-1 games. BUF: pass_epa_sum 6 over 60 dropbacks, a rate of
# 0.10. HOU and TEN have no prior season at all.
_CURRENT_ROWS = {
    "season": [2007] * 4,
    "week": [1] * 4,
    "team_abbr": ["BUF", "KC", "HOU", "TEN"],
    "opponent_abbr": ["KC", "BUF", "TEN", "HOU"],
    "pass_yards": [150.0, 250.0, 220.0, 180.0],
    "points_scored": [10.0, 20.0, 21.0, 14.0],
    "points_allowed": [20.0, 10.0, 14.0, 21.0],
    "dropbacks": [60.0, 20.0, 35.0, 32.0],
    "pass_epa_sum": [6.0, 8.0, 4.0, 2.0],
    "offensive_snaps": [70.0, 50.0, 62.0, 58.0],
}

_REGRESSED_BUF_PASS_YARDS = 300.0 * (2.0 / 3.0) + 250.0 / 3.0


def _team_stats() -> pl.DataFrame:
    """Return two seasons of per-game team stats for the blend fixtures."""
    return pl.concat([pl.DataFrame(_PRIOR_ROWS), pl.DataFrame(_CURRENT_ROWS)])


def _schedule(week: int) -> pl.DataFrame:
    """Return a two-game schedule for one week of the 2007 season."""
    return pl.DataFrame(
        {
            "season": [2007, 2007],
            "week": [week, week],
            "away_abbr": ["BUF", "HOU"],
            "home_abbr": ["KC", "TEN"],
        }
    )


def _run_week(
    week: int,
    *,
    min_season: int = 2006,
    blend_stat_prior: bool = True,
    stat_prior_blend_games: float = constants.PRIOR_BLEND_GAMES,
    prior_season_stats: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run `process_week` for one 2007 week on the fixture, sorted for stable comparison."""
    result = process_week(
        2007,
        week,
        _schedule(week),
        _team_stats(),
        min_season=min_season,
        elo_df=None,
        tr_df=None,
        prev_tr_df=None,
        prior_season_stats=prior_season_stats,
        blend_stat_prior=blend_stat_prior,
        stat_prior_blend_games=stat_prior_blend_games,
    )
    return result.sort("away_abbr")


def _row(result: pl.DataFrame, away: str) -> dict[str, Any]:
    """Return one game row keyed by its away team."""
    return result.filter(pl.col("away_abbr") == away).row(0, named=True)


def test_week_one_rows_are_identical_with_and_without_the_blend() -> None:
    """Week 1 has zero in-season games, so the blend weight is zero and nothing moves."""
    blended = _run_week(1)
    ablated = _run_week(1, blend_stat_prior=False)

    assert_frame_equal(blended, ablated, check_exact=True)
    assert _row(blended, "BUF")["away_pass_yards"] == pytest.approx(_REGRESSED_BUF_PASS_YARDS)


def test_one_game_team_is_one_fifth_in_season_and_four_fifths_prior() -> None:
    """With K = 4 games, a one-game team is weighted 1 / (1 + 4) toward its own sample."""
    prior = _row(_run_week(1), "BUF")
    week_two = _row(_run_week(2), "BUF")

    expected_away = 0.2 * 150.0 + 0.8 * float(prior["away_pass_yards"])
    expected_home = 0.2 * 250.0 + 0.8 * float(prior["home_pass_yards"])
    assert week_two["away_pass_yards"] == pytest.approx(expected_away)
    assert week_two["home_pass_yards"] == pytest.approx(expected_home)


def test_blended_rates_are_ratios_of_blended_sums() -> None:
    """A rate is rebuilt from the blended numerator and denominator, not blended itself."""
    week_two = _row(_run_week(2), "BUF")

    blended_epa = 0.2 * 6.0 + 0.8 * 10.5
    blended_dropbacks = 0.2 * 60.0 + 0.8 * 30.0
    ratio_of_blended_sums = blended_epa / blended_dropbacks
    blend_of_rates = 0.2 * (6.0 / 60.0) + 0.8 * 0.35

    assert week_two["away_epa_per_dropback"] == pytest.approx(ratio_of_blended_sums)
    assert week_two["away_epa_per_dropback"] != pytest.approx(blend_of_rates)


def test_blend_games_sets_the_weight() -> None:
    """K = 1 weights a one-game team half in-season and half prior."""
    prior = _row(_run_week(1), "BUF")
    week_two = _row(_run_week(2, stat_prior_blend_games=1.0), "BUF")

    expected = 0.5 * 150.0 + 0.5 * float(prior["away_pass_yards"])
    assert week_two["away_pass_yards"] == pytest.approx(expected)


def test_disabling_the_blend_publishes_raw_in_season_means() -> None:
    """The ablation switch restores the plain mean of prior in-season games."""
    week_two = _row(_run_week(2, blend_stat_prior=False), "BUF")

    assert week_two["away_pass_yards"] == pytest.approx(150.0)
    assert week_two["away_epa_per_dropback"] == pytest.approx(0.1)


def test_games_played_keeps_the_in_season_count() -> None:
    """Blending changes stat values, not the published in-season game count."""
    week_two = _row(_run_week(2), "BUF")

    assert week_two["away_games_played"] == 1
    assert week_two["home_games_played"] == 1


def test_team_without_a_prior_season_keeps_its_raw_in_season_means() -> None:
    """A team with no previous-season rows has nothing to blend toward."""
    week_two = _row(_run_week(2), "HOU")

    assert week_two["away_pass_yards"] == pytest.approx(220.0)
    assert week_two["home_pass_yards"] == pytest.approx(180.0)


def test_first_season_in_the_run_ignores_earlier_seasons() -> None:
    """The first processed season has no prior in this run, even when older rows exist."""
    week_two = _row(_run_week(2, min_season=2007), "BUF")

    assert week_two["away_pass_yards"] == pytest.approx(150.0)


def test_supplied_prior_season_stats_match_the_computed_fallback() -> None:
    """Passing the precomputed prior gives the same rows as computing it inside."""
    prior = data_collection.build_prior_season_stats(_team_stats(), 2007, min_season=2006)

    assert prior is not None
    assert_frame_equal(_run_week(2, prior_season_stats=prior), _run_week(2), check_exact=True)
    assert_frame_equal(_run_week(1, prior_season_stats=prior), _run_week(1), check_exact=True)


def test_prior_season_stats_are_unavailable_for_the_first_season() -> None:
    """No prior is built for the first season in the run or when the data is absent."""
    assert data_collection.build_prior_season_stats(_team_stats(), 2006, min_season=2006) is None
    assert data_collection.build_prior_season_stats(_team_stats(), 2009, min_season=2006) is None
    assert data_collection.build_prior_season_stats(pl.DataFrame(), 2007, min_season=2006) is None


def test_blend_fills_a_missing_in_season_value_from_the_prior() -> None:
    """When only one side has a value, that side is published unchanged."""
    in_season = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB"],
            "pass_yards": [100.0, None],
            "games_played": [2, 2],
        },
        schema_overrides={"games_played": pl.UInt32},
    )
    prior = pl.DataFrame(
        {"team_abbr": ["AAA", "BBB"], "pass_yards": [None, 300.0], "games_played": [17, 17]}
    )

    blended = polars_utils.blend_with_prior_stats(in_season, prior, 4.0).sort("team_abbr")

    assert blended["pass_yards"].to_list() == [pytest.approx(100.0), pytest.approx(300.0)]
    assert blended["games_played"].to_list() == [2, 2]


def test_blend_is_a_no_op_without_a_prior() -> None:
    """An empty prior frame leaves the in-season frame exactly as it was."""
    in_season = pl.DataFrame({"team_abbr": ["AAA"], "pass_yards": [100.0], "games_played": [1]})

    assert_frame_equal(
        polars_utils.blend_with_prior_stats(in_season, pl.DataFrame(), 4.0), in_season
    )


def test_process_season_builds_the_prior_once_per_season(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every week of a season reuses one prior frame rather than rebuilding it."""
    calls: list[int] = []
    seen: list[object] = []
    sentinel = pl.DataFrame({"team_abbr": ["BUF"]})

    def fake_build(_team_stats_df: pl.DataFrame, season: int, *, min_season: int) -> pl.DataFrame:
        """Record the call and hand back a sentinel frame."""
        del min_season
        calls.append(season)
        return sentinel

    def fake_process_week(*_args: object, **kwargs: object) -> pl.DataFrame:
        """Capture the prior frame each week receives."""
        seen.append(kwargs["prior_season_stats"])
        return pl.DataFrame()

    monkeypatch.setattr(data_collection, "build_prior_season_stats", fake_build)
    monkeypatch.setattr(data_collection, "process_week", fake_process_week)

    schedule = pl.DataFrame(
        {
            "season": [2007] * 3,
            "week": [1, 2, 3],
            "away_abbr": ["BUF"] * 3,
            "home_abbr": ["KC"] * 3,
        }
    )
    data_collection.process_season(2007, schedule, pl.DataFrame(), min_season=2006)

    assert calls == [2007]
    assert seen == [sentinel] * 3


def test_prior_blend_games_is_shared_with_the_strength_snapshot() -> None:
    """Both early-season blends read one named constant."""
    assert constants.PRIOR_BLEND_GAMES == 4.0
