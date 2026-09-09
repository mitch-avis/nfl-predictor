"""Tests for the Polars data collection pipeline."""

import polars as pl
import pytest

from nfl_predictor import constants, data_collection
from nfl_predictor.data_collection import _merge_team_rankings, process_week
from nfl_predictor.utils import polars_utils
from nfl_predictor.utils.polars import schedule_strength


def test_process_week_uses_fallback_stats_for_week1() -> None:
    """Week 1 uses prior-season fallbacks when no prior games exist."""
    schedule_df = pl.DataFrame(
        {
            "season": [2007],
            "week": [1],
            "away_abbr": ["BUF"],
            "home_abbr": ["KC"],
        }
    )

    team_stats_df = pl.DataFrame(
        {
            "season": [2006, 2006],
            "week": [1, 1],
            "team_abbr": ["BUF", "KC"],
            "opponent_abbr": ["KC", "BUF"],
            "pass_yards": [300, 200],
            "points_scored": [24, 17],
            "points_allowed": [17, 24],
        }
    )

    result = process_week(
        season=2007,
        week=1,
        schedule_df=schedule_df,
        team_stats_df=team_stats_df,
        min_season=2006,
        elo_df=None,
        tr_df=None,
        prev_tr_df=None,
    )

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["away_pass_yards"] == pytest.approx(283.333, rel=1e-3)
    assert row["home_pass_yards"] == pytest.approx(216.667, rel=1e-3)
    assert row["pass_yards_diff"] == pytest.approx(66.666, rel=1e-3)


def test_merge_team_rankings_week1_uses_prev() -> None:
    """Week 1 TeamRankings merge can fall back to prior-season week 18."""
    merged = pl.DataFrame({"away_abbr": ["BUF"], "home_abbr": ["KC"]})
    prev_tr_df = pl.DataFrame(
        {
            "team_abbr": ["BUF", "KC"],
            "week": [18, 18],
            "predictive_rating": [5.0, 4.0],
        }
    )

    result = _merge_team_rankings(
        merged=merged,
        season=2007,
        week=1,
        tr_df=None,
        prev_tr_df=prev_tr_df,
    )

    row = result.row(0, named=True)
    assert row["away_predictive_rating"] == pytest.approx(5.0)
    assert row["home_predictive_rating"] == pytest.approx(4.0)


def _pbp_play(
    season: int,
    week: int,
    posteam: str,
    defteam: str,
    *,
    epa: float,
    dropback: int = 0,
    rush: int = 0,
    yards: float = 0.0,
    success: int = 0,
    down: int = 1,
) -> dict[str, object]:
    """Build one synthetic scrimmage play row for play-by-play fixtures."""
    return {
        "season": season,
        "week": week,
        "season_type": "REG",
        "posteam": posteam,
        "defteam": defteam,
        "qb_dropback": dropback,
        "rush": rush,
        "qb_kneel": 0,
        "qb_spike": 0,
        "epa": epa,
        "success": success,
        "yards_gained": yards,
        "down": down,
        "play_type": "pass" if dropback else "run",
        "yardline_100": 50,
        "special": 0,
    }


def _three_week_pbp(epa_scale_after_week_1: float) -> pl.DataFrame:
    """Build three weeks of synthetic play-by-play for two teams.

    Weeks 2 and 3 are scaled by `epa_scale_after_week_1` so a perturbation of the
    later weeks can be tested against week-2 features, which must not see them.
    """
    rows: list[dict[str, object]] = []
    for week in (1, 2, 3):
        scale = 1.0 if week == 1 else epa_scale_after_week_1
        for posteam, defteam, base in (("AAA", "BBB", 0.4), ("BBB", "AAA", -0.2)):
            rows.append(
                _pbp_play(
                    2007,
                    week,
                    posteam,
                    defteam,
                    epa=base * scale,
                    dropback=1,
                    yards=25.0,
                    success=1,
                )
            )
            rows.append(
                _pbp_play(
                    2007, week, posteam, defteam, epa=-base * scale, rush=1, yards=2.0, down=2
                )
            )
    return pl.DataFrame(rows)


def _week_two_features(pbp_df: pl.DataFrame) -> dict[str, object]:
    """Run the play-by-play chain end to end and return week-2 feature values."""
    team_games = polars_utils.aggregate_pbp_team_game_stats(pbp_df)
    team_stats_df = pl.DataFrame(
        {
            "season": [2007] * 6,
            "week": [1, 1, 2, 2, 3, 3],
            "team_abbr": ["AAA", "BBB"] * 3,
            "opponent_abbr": ["BBB", "AAA"] * 3,
            "pass_yards": [250.0, 180.0] * 3,
        }
    )
    team_stats_df = data_collection._join_pbp_team_game_stats(team_stats_df, team_games)

    schedule_df = pl.DataFrame(
        {"season": [2007], "week": [2], "away_abbr": ["AAA"], "home_abbr": ["BBB"]}
    )
    result = process_week(
        season=2007,
        week=2,
        schedule_df=schedule_df,
        team_stats_df=team_stats_df,
        min_season=2006,
        elo_df=None,
        tr_df=None,
        prev_tr_df=None,
    )
    assert result.height == 1
    row = result.row(0, named=True)
    return {
        col: row[col]
        for col in row
        if any(marker in col for marker in constants.FEATURE_GROUP_COLUMN_MARKERS["pbp"])
    }


def test_future_week_plays_do_not_change_earlier_week_features() -> None:
    """Perturbing week 2 and week 3 plays leaves week-2 features untouched.

    Week-N features may only use games strictly before week N of the season, so
    rewriting every play from week 2 onward must not move a single week-2 value.
    """
    baseline = _week_two_features(_three_week_pbp(1.0))
    perturbed = _week_two_features(_three_week_pbp(10.0))

    assert baseline, "expected play-by-play features on the week-2 row"
    assert baseline == perturbed


def test_week1_fallback_regresses_pbp_rates_toward_the_league_mean() -> None:
    """Week 1 falls back to the regressed prior season for the play-by-play family.

    Rates are ratios of regressed sums, so each regressed rate is a weighted mediant of
    the team's own rate and the league mean rate and must land between the two.
    """
    prior = pl.DataFrame(
        {
            "season": [2006, 2006],
            "week": [1, 1],
            "team_abbr": ["BUF", "KC"],
            "opponent_abbr": ["KC", "BUF"],
            "pass_yards": [300.0, 200.0],
            "points_scored": [24.0, 17.0],
            "points_allowed": [17.0, 24.0],
            # BUF: 12.0 EPA over 30 dropbacks = 0.40. KC: 3.0 over 30 = 0.10.
            "dropbacks": [30.0, 30.0],
            "pass_epa_sum": [12.0, 3.0],
            "offensive_snaps": [60.0, 60.0],
        }
    )
    schedule_df = pl.DataFrame(
        {"season": [2007], "week": [1], "away_abbr": ["BUF"], "home_abbr": ["KC"]}
    )

    result = process_week(
        season=2007,
        week=1,
        schedule_df=schedule_df,
        team_stats_df=prior,
        min_season=2006,
        elo_df=None,
        tr_df=None,
        prev_tr_df=None,
    )

    row = result.row(0, named=True)
    league_mean_rate = 15.0 / 60.0  # (12.0 + 3.0) EPA over (30 + 30) dropbacks
    assert league_mean_rate < row["away_epa_per_dropback"] < 0.40
    assert 0.10 < row["home_epa_per_dropback"] < league_mean_rate


def _season_pbp(
    season: int,
    weeks: tuple[int, ...],
    *,
    epa_scale: float,
    season_type: str,
) -> pl.DataFrame:
    """Build synthetic play-by-play for two teams over the given weeks of one season."""
    rows: list[dict[str, object]] = []
    for week in weeks:
        for posteam, defteam, base in (("AAA", "BBB", 0.4), ("BBB", "AAA", -0.2)):
            rows.append(
                _pbp_play(
                    season,
                    week,
                    posteam,
                    defteam,
                    epa=base * epa_scale,
                    dropback=1,
                    yards=25.0,
                    success=1,
                )
                | {"season_type": season_type}
            )
            rows.append(
                _pbp_play(
                    season,
                    week,
                    posteam,
                    defteam,
                    epa=-base * epa_scale,
                    rush=1,
                    yards=2.0,
                    down=2,
                )
                | {"season_type": season_type}
            )
    return pl.DataFrame(rows)


def _team_stats_rows(season: int, weeks: tuple[int, ...]) -> pl.DataFrame:
    """Build the per-game team-stat skeleton the play-by-play counts join onto."""
    return pl.DataFrame(
        {
            "season": [season] * (2 * len(weeks)),
            "week": [week for week in weeks for _ in range(2)],
            "team_abbr": ["AAA", "BBB"] * len(weeks),
            "opponent_abbr": ["BBB", "AAA"] * len(weeks),
            "pass_yards": [250.0, 180.0] * len(weeks),
        }
    )


def _pbp_features_for_week(
    pbp_df: pl.DataFrame,
    team_stats_df: pl.DataFrame,
    *,
    season: int,
    week: int,
    min_season: int,
) -> dict[str, object]:
    """Run the play-by-play chain end to end and return one row's feature values."""
    team_games = polars_utils.aggregate_pbp_team_game_stats(pbp_df)
    joined = data_collection._join_pbp_team_game_stats(team_stats_df, team_games)
    schedule_df = pl.DataFrame(
        {"season": [season], "week": [week], "away_abbr": ["AAA"], "home_abbr": ["BBB"]}
    )
    result = process_week(
        season=season,
        week=week,
        schedule_df=schedule_df,
        team_stats_df=joined,
        min_season=min_season,
        elo_df=None,
        tr_df=None,
        prev_tr_df=None,
    )
    assert result.height == 1
    row = result.row(0, named=True)
    return {
        col: row[col]
        for col in row
        if any(marker in col for marker in constants.FEATURE_GROUP_COLUMN_MARKERS["pbp"])
    }


def test_playoff_row_features_ignore_every_postseason_play() -> None:
    """A playoff row is built from the full regular season and never from playoff games.

    The playoff branch of season-to-date aggregation replaces "strictly before
    week N" with "the whole regular season", so it is the one branch where a
    postseason game could leak into its own features. Rewriting every postseason
    play must leave a later playoff row untouched.
    """
    regular_weeks = (1, 2, 3)
    playoff_weeks = (18, 19)
    stats = _team_stats_rows(2007, regular_weeks + playoff_weeks)

    def features(playoff_epa_scale: float) -> dict[str, object]:
        pbp_df = pl.concat(
            [
                _season_pbp(2007, regular_weeks, epa_scale=1.0, season_type="REG"),
                _season_pbp(2007, playoff_weeks, epa_scale=playoff_epa_scale, season_type="POST"),
            ]
        )
        return _pbp_features_for_week(pbp_df, stats, season=2007, week=19, min_season=2006)

    baseline = features(1.0)
    perturbed = features(25.0)

    assert baseline, "expected play-by-play features on the playoff row"
    assert baseline == perturbed


def test_every_playoff_week_sees_the_same_full_regular_season() -> None:
    """Two playoff rows in one season share features, because both use all of it."""
    regular_weeks = (1, 2, 3)
    playoff_weeks = (18, 19)
    stats = _team_stats_rows(2007, regular_weeks + playoff_weeks)
    pbp_df = pl.concat(
        [
            _season_pbp(2007, regular_weeks, epa_scale=1.0, season_type="REG"),
            _season_pbp(2007, playoff_weeks, epa_scale=3.0, season_type="POST"),
        ]
    )

    first_round = _pbp_features_for_week(pbp_df, stats, season=2007, week=18, min_season=2006)
    later_round = _pbp_features_for_week(pbp_df, stats, season=2007, week=19, min_season=2006)

    assert first_round, "expected play-by-play features on the playoff row"
    assert first_round == later_round


def test_week1_row_features_ignore_every_current_season_play() -> None:
    """A Week-1 row falls back to the prior season and never sees the season it opens.

    The Week-1 fallback is the other branch that does not use "strictly before
    week N": it reaches into the previous season instead. Rewriting every play of
    the current season must therefore leave the Week-1 row untouched.
    """
    prior_weeks = (1, 2)
    current_weeks = (1, 2, 3)
    stats = pl.concat([_team_stats_rows(2006, prior_weeks), _team_stats_rows(2007, current_weeks)])

    def features(current_epa_scale: float) -> dict[str, object]:
        pbp_df = pl.concat(
            [
                _season_pbp(2006, prior_weeks, epa_scale=1.0, season_type="REG"),
                _season_pbp(2007, current_weeks, epa_scale=current_epa_scale, season_type="REG"),
            ]
        )
        return _pbp_features_for_week(pbp_df, stats, season=2007, week=1, min_season=2006)

    baseline = features(1.0)
    perturbed = features(40.0)

    assert baseline, "expected play-by-play features on the week-1 row"
    assert baseline == perturbed


def _strength_features_for_week(
    pbp_df: pl.DataFrame,
    team_stats_df: pl.DataFrame,
    *,
    season: int,
    week: int,
    min_season: int,
) -> dict[str, object]:
    """Run the chain end to end and return one row's schedule-adjusted strength values."""
    team_games = polars_utils.aggregate_pbp_team_game_stats(pbp_df)
    joined = data_collection._join_pbp_team_game_stats(team_stats_df, team_games)
    joined = joined.with_columns(
        pl.lit(21.0).alias("points_scored"), pl.lit(17.0).alias("points_allowed")
    )
    schedule_df = pl.DataFrame(
        {
            "season": [season] * 2,
            "week": [week, week + 1],
            "away_abbr": ["AAA", "BBB"],
            "home_abbr": ["BBB", "AAA"],
        }
    )
    result = process_week(
        season=season,
        week=week,
        schedule_df=schedule_df,
        team_stats_df=joined,
        min_season=min_season,
        elo_df=None,
        tr_df=None,
        prev_tr_df=None,
    )
    assert result.height == 1
    row = result.row(0, named=True)
    values = {
        col: row[col]
        for col in row
        if any(marker in col for marker in constants.FEATURE_GROUP_COLUMN_MARKERS["strength"])
    }
    assert values, "expected schedule-adjusted strength columns on the row"
    return values


def test_future_week_plays_do_not_change_earlier_week_strength() -> None:
    """Rewriting week 2 onward leaves every week-2 strength value untouched."""
    stats = _team_stats_rows(2007, (1, 2, 3))

    def features(scale: float) -> dict[str, object]:
        return _strength_features_for_week(
            _three_week_pbp(scale), stats, season=2007, week=2, min_season=2006
        )

    assert features(1.0) == features(10.0)


def test_playoff_row_strength_ignores_every_postseason_play() -> None:
    """A playoff row's strength comes from the regular season, never from playoff games."""
    regular_weeks = (1, 2, 3)
    playoff_weeks = (18, 19)
    stats = _team_stats_rows(2007, regular_weeks + playoff_weeks)

    def features(playoff_scale: float) -> dict[str, object]:
        pbp_df = pl.concat(
            [
                _season_pbp(2007, regular_weeks, epa_scale=1.0, season_type="REG"),
                _season_pbp(2007, playoff_weeks, epa_scale=playoff_scale, season_type="POST"),
            ]
        )
        return _strength_features_for_week(pbp_df, stats, season=2007, week=19, min_season=2006)

    assert features(1.0) == features(25.0)


def test_week1_row_strength_ignores_every_current_season_play() -> None:
    """A Week-1 strength row reaches back to the prior season and never forward."""
    prior_weeks = (1, 2)
    current_weeks = (1, 2, 3)
    stats = pl.concat([_team_stats_rows(2006, prior_weeks), _team_stats_rows(2007, current_weeks)])

    def features(current_scale: float) -> dict[str, object]:
        pbp_df = pl.concat(
            [
                _season_pbp(2006, prior_weeks, epa_scale=1.0, season_type="REG"),
                _season_pbp(2007, current_weeks, epa_scale=current_scale, season_type="REG"),
            ]
        )
        return _strength_features_for_week(pbp_df, stats, season=2007, week=1, min_season=2006)

    assert features(1.0) == features(40.0)


def test_week1_strength_actually_carries_the_prior_season() -> None:
    """The Week-1 leakage test would pass on all-nulls, so prove the prior is really used."""
    prior_weeks = (1, 2)
    stats = pl.concat([_team_stats_rows(2006, prior_weeks), _team_stats_rows(2007, (1,))])
    pbp_df = pl.concat(
        [
            _season_pbp(2006, prior_weeks, epa_scale=1.0, season_type="REG"),
            _season_pbp(2007, (1,), epa_scale=1.0, season_type="REG"),
        ]
    )

    values = _strength_features_for_week(pbp_df, stats, season=2007, week=1, min_season=2006)

    assert any(
        value is not None for name, value in values.items() if "adj_off_pass_epa_snap" in name
    )


def test_strength_feature_join_collapses_duplicate_team_keys() -> None:
    """A duplicate team key annotates the games rather than multiplying them."""
    merged = pl.DataFrame({"away_abbr": ["AAA"], "home_abbr": ["BBB"]})
    features = pl.DataFrame(
        {
            "team_abbr": ["AAA", "AAA", "BBB"],
            **{column: [1.0, 99.0, 2.0] for column in constants.ADJUSTED_STRENGTH_STATS},
        }
    )

    result = data_collection._merge_strength_features(merged, features)

    assert result.height == 1
    assert result.row(0, named=True)["away_adj_srs"] == pytest.approx(1.0)


def test_strength_features_are_added_even_when_no_team_matches() -> None:
    """Every published strength column exists on the row even with nothing to join."""
    merged = pl.DataFrame({"away_abbr": ["AAA"], "home_abbr": ["BBB"]})
    empty = pl.DataFrame(
        schema={
            "team_abbr": pl.String,
            **dict.fromkeys(constants.ADJUSTED_STRENGTH_STATS, pl.Float64),
        }
    )

    result = data_collection._merge_strength_features(merged, empty)

    assert result.height == 1
    for side in ("away", "home"):
        for column in constants.ADJUSTED_STRENGTH_STATS:
            assert f"{side}_{column}" in result.columns
            assert result.row(0, named=True)[f"{side}_{column}"] is None


def test_strength_features_degrade_when_the_schedule_is_incomplete() -> None:
    """A schedule missing the columns the remaining-games lens needs still yields a row.

    The ridge snapshot depends only on played games, so it must survive a schedule that
    cannot answer "who is left to play"; only the adjusted schedule-strength pair goes null.
    """
    team_games = _team_stats_rows(2007, (1, 2)).with_columns(
        pl.lit(60.0).alias("offensive_snaps"),
        pl.lit(60.0).alias("defensive_snaps"),
        pl.lit(3.0).alias("pass_epa_sum"),
        pl.lit(1.0).alias("rush_epa_sum"),
        pl.lit(1.0).alias("pass_epa_allowed_sum"),
        pl.lit(0.5).alias("rush_epa_allowed_sum"),
        pl.lit(21.0).alias("points_scored"),
        pl.lit(17.0).alias("points_allowed"),
        pl.Series("is_home", [True, False, True, False]),
    )
    incomplete_schedule = pl.DataFrame({"season": [2007], "week": [3]})

    features = data_collection.build_strength_features(
        team_games,
        incomplete_schedule,
        season=2007,
        week=3,
        blend_prior=False,
    )

    assert features.columns == ["team_abbr", *constants.ADJUSTED_STRENGTH_STATS]
    assert features.height == 2
    # The solve still produced ratings.
    assert features["adj_off_pass_epa_snap"].null_count() == 0
    # The lens that needs the schedule went null instead of guessing.
    assert features["sos_played_adj"].null_count() == features.height
    assert features["sos_remaining_adj"].null_count() == features.height


def _bracket_schedule(playoff_opponent: str) -> pl.DataFrame:
    """Build a season whose regular slate is fixed and whose playoff opponent varies."""
    return pl.DataFrame(
        {
            "season": [2007] * 7,
            "week": [1, 1, 2, 2, 3, 3, 18],
            "away_abbr": ["AAA", "DDD", "AAA", "DDD", "AAA", "EEE", "AAA"],
            "home_abbr": ["FFF", "EEE", "BBB", "FFF", "CCC", "FFF", playoff_opponent],
        }
    )


def test_remaining_schedule_strength_ignores_the_postseason_bracket() -> None:
    """Who a team meets in the playoffs must not reach a regular-season feature.

    The regular-season schedule is fixed before kickoff and is legitimately known, but
    the bracket is an outcome of the season being predicted. A weak and a strong playoff
    opponent must therefore produce the same week-2 value.
    """
    ratings = pl.DataFrame(
        {
            "team_abbr": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "adj_strength_composite": [0.0, 1.0, 1.0, 10.0, 10.0, 0.0],
        }
    )

    def remaining(playoff_opponent: str) -> float | None:
        adjusted = schedule_strength.compute_schedule_strength_adjusted(
            data_collection._regular_season_schedule(_bracket_schedule(playoff_opponent), 2007),
            ratings,
            season=2007,
            week=2,
            rating_col="adj_strength_composite",
            team_col="team_abbr",
        )
        return adjusted.filter(pl.col("team_abbr") == "AAA")["sos_remaining_adj"].item()

    assert remaining("BBB") == remaining("DDD")


def test_week_one_publishes_strength_before_any_game_is_played() -> None:
    """A new season with a schedule but no played games still carries the prior.

    This is the live pre-kickoff state: the schedule exists, no game has happened, and
    the whole point of the prior blend is to serve exactly this week. Publishing nulls
    here would be a train/serve skew against every historical Week-1 training row.
    """
    played = _team_stats_rows(2006, (1, 2)).with_columns(
        pl.lit(60.0).alias("offensive_snaps"),
        pl.lit(60.0).alias("defensive_snaps"),
        pl.Series("pass_epa_sum", [6.0, -6.0, 6.0, -6.0]),
        pl.Series("rush_epa_sum", [3.0, -3.0, 3.0, -3.0]),
        pl.lit(1.0).alias("pass_epa_allowed_sum"),
        pl.lit(0.5).alias("rush_epa_allowed_sum"),
        pl.Series("points_scored", [28.0, 10.0, 28.0, 10.0]),
        pl.Series("points_allowed", [10.0, 28.0, 10.0, 28.0]),
        pl.Series("is_home", [True, False, True, False]),
    )
    new_season_schedule = pl.DataFrame(
        {"season": [2007], "week": [1], "away_abbr": ["AAA"], "home_abbr": ["BBB"]}
    )
    prior = data_collection.build_prior_strength_snapshot(played, 2007, min_season=2006)
    assert prior is not None, "the prior season should produce a snapshot"

    features = data_collection.build_strength_features(
        played,
        new_season_schedule,
        season=2007,
        week=1,
        prior_snapshot=prior,
    )

    assert sorted(features["team_abbr"].to_list()) == ["AAA", "BBB"]
    assert features["adj_off_pass_epa_snap"].null_count() == 0
    assert features["adj_strength_composite"].null_count() == 0
    # Nothing has been played yet, so the solve rests entirely on the prior.
    assert features["strength_games_played"].to_list() == [0.0, 0.0]
