"""Tests for the per-team weekly strength snapshot the ETL writes next to the game rows."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nfl_predictor import constants, data_collection

# Six teams, so every current-season week after the first leaves two of them on a bye.
_STRENGTH = {"AAA": 0.25, "BBB": 0.15, "CCC": 0.05, "DDD": -0.05, "EEE": -0.15, "FFF": -0.25}

_PRIOR_SLATE = (
    (2006, 1, "AAA", "BBB"),
    (2006, 1, "CCC", "DDD"),
    (2006, 1, "EEE", "FFF"),
    (2006, 2, "BBB", "CCC"),
    (2006, 2, "DDD", "EEE"),
    (2006, 2, "FFF", "AAA"),
)

# Week 2 leaves DDD and FFF on a bye; week 3 leaves AAA and EEE on a bye.
_CURRENT_SLATE = (
    (2007, 1, "AAA", "BBB"),
    (2007, 1, "CCC", "DDD"),
    (2007, 1, "EEE", "FFF"),
    (2007, 2, "AAA", "CCC"),
    (2007, 2, "BBB", "EEE"),
    (2007, 3, "DDD", "BBB"),
    (2007, 3, "FFF", "CCC"),
)


def _team_games(*, scale_from: tuple[int, int] | None = None, scale: float = 1.0) -> pl.DataFrame:
    """Build two rows per game, one per side, from the fixed per-team strengths.

    Args:
        scale_from: Optional ``(season, week)``; every 2007 game from that week on has its
            per-snap edge multiplied by ``scale``. Used to rewrite "later" results.
        scale: Multiplier applied from ``scale_from`` onward.

    """
    rows: list[dict[str, object]] = []
    for season, week, away, home in (*_PRIOR_SLATE, *_CURRENT_SLATE):
        factor = 1.0
        if scale_from is not None and (season, week) >= scale_from:
            factor = scale
        for team, opponent, is_home in ((away, home, False), (home, away, True)):
            edge = (_STRENGTH[team] - _STRENGTH[opponent]) * factor
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "team_abbr": team,
                    "opponent_abbr": opponent,
                    "is_home": is_home,
                    "offensive_snaps": 60.0,
                    "defensive_snaps": 60.0,
                    "pass_epa_sum": 60.0 * edge,
                    "rush_epa_sum": 30.0 * edge,
                    "pass_epa_allowed_sum": -60.0 * edge,
                    "rush_epa_allowed_sum": -30.0 * edge,
                    "points_scored": 20.0 + 20.0 * edge,
                    "points_allowed": 20.0 - 20.0 * edge,
                }
            )
    return pl.DataFrame(rows)


def _schedule() -> pl.DataFrame:
    """Return the 2007 regular-season slate."""
    return pl.DataFrame(
        {
            "season": [game[0] for game in _CURRENT_SLATE],
            "week": [game[1] for game in _CURRENT_SLATE],
            "away_abbr": [game[2] for game in _CURRENT_SLATE],
            "home_abbr": [game[3] for game in _CURRENT_SLATE],
        }
    )


def _assert_same_values(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """Assert two frames from separate runs agree to far below any real change.

    The schedule-strength columns come from parallel Polars group-by sums, whose order can
    move the last one or two bits between identical runs, so separate runs are compared
    within 1e-12 rather than bit for bit.
    """
    assert_frame_equal(left, right, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)


def _run_week(week: int, team_games: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run one 2007 week and return its game rows and the snapshot it recorded."""
    sink: list[pl.DataFrame] = []
    games = data_collection.process_week(
        2007,
        week,
        _schedule(),
        team_games,
        min_season=2006,
        elo_df=None,
        tr_df=None,
        prev_tr_df=None,
        strength_snapshots=sink,
    )
    assert len(sink) == 1, "each processed week records exactly one snapshot"
    return games, sink[0]


def test_every_scheduled_team_gets_a_row_including_teams_on_a_bye() -> None:
    """A bye team has no game row that week, but it still has a snapshot row."""
    games, snapshot = _run_week(2, _team_games())

    assert games.height == 2
    assert snapshot.columns == constants.STRENGTH_SNAPSHOT_FILE_COLUMNS
    assert sorted(snapshot["team_abbr"].to_list()) == sorted(_STRENGTH)
    assert snapshot["season"].unique().to_list() == [2007]
    assert snapshot["week"].unique().to_list() == [2]
    for bye_team in ("DDD", "FFF"):
        row = snapshot.filter(pl.col("team_abbr") == bye_team).row(0, named=True)
        assert row["adj_strength_composite"] is not None
        assert row["strength_games_played"] == pytest.approx(1.0)


@pytest.mark.parametrize("week", [2, 3])
def test_a_week_snapshot_never_reads_that_week_or_later(week: int) -> None:
    """Rewriting the described week and everything after it leaves the snapshot untouched.

    Covers both teams on a bye and teams that play that week: all of them are solved
    from earlier games only.
    """
    _, baseline = _run_week(week, _team_games())
    _, rewritten = _run_week(week, _team_games(scale_from=(2007, week), scale=8.0))

    _assert_same_values(baseline, rewritten)


def test_a_week_snapshot_moves_with_earlier_results() -> None:
    """The leakage check above would pass on a constant, so prove earlier weeks count."""
    _, baseline = _run_week(3, _team_games())
    _, rewritten = _run_week(3, _team_games(scale_from=(2007, 2), scale=8.0))

    baseline_ccc = baseline.filter(pl.col("team_abbr") == "CCC")["adj_off_pass_epa_snap"].item()
    rewritten_ccc = rewritten.filter(pl.col("team_abbr") == "CCC")["adj_off_pass_epa_snap"].item()
    assert baseline_ccc != pytest.approx(rewritten_ccc)


def test_snapshot_rows_equal_the_strength_features_on_the_game_rows() -> None:
    """The file and the model's features come from one solve, so they agree exactly."""
    games, snapshot = _run_week(3, _team_games())

    by_team = {row["team_abbr"]: row for row in snapshot.iter_rows(named=True)}
    assert games.height == 2
    for game in games.iter_rows(named=True):
        for side in ("away", "home"):
            team_row = by_team[game[f"{side}_abbr"]]
            for column in constants.ADJUSTED_STRENGTH_STATS:
                assert game[f"{side}_{column}"] == team_row[column], (side, column)


def test_the_home_field_term_is_one_league_wide_value_per_week() -> None:
    """The snapshot keeps the shared home-field term the game schema leaves out."""
    _, snapshot = _run_week(3, _team_games())

    assert snapshot["adj_hfa"].n_unique() == 1


def test_combining_no_snapshots_keeps_the_published_schema() -> None:
    """A run that solved nothing still writes a file with the documented columns."""
    combined = data_collection.combine_strength_snapshots([])

    assert combined.height == 0
    assert combined.columns == constants.STRENGTH_SNAPSHOT_FILE_COLUMNS
    assert combined.schema["season"] == pl.Int64
    assert combined.schema["week"] == pl.Int64
    assert combined.schema["team_abbr"] == pl.String
    assert combined.schema["adj_strength_composite"] == pl.Float64


def test_combined_snapshots_are_sorted_and_typed() -> None:
    """Combining the weekly frames yields one frame ordered by season, week and team."""
    _, week_three = _run_week(3, _team_games())
    _, week_two = _run_week(2, _team_games())

    combined = data_collection.combine_strength_snapshots([week_three, week_two])

    assert combined.height == 12
    assert combined["week"].to_list() == [2] * 6 + [3] * 6
    assert combined.filter(pl.col("week") == 2)["team_abbr"].to_list() == sorted(_STRENGTH)
    assert combined.columns == constants.STRENGTH_SNAPSHOT_FILE_COLUMNS


def test_process_season_records_every_week_plus_the_week_after_the_regular_season() -> None:
    """A ranking through the final regular-season week needs the full-season snapshot.

    That week has no games until the playoff schedule is published, so the season adds
    it explicitly rather than leaving the last regular-season week unrankable.
    """
    sink: list[pl.DataFrame] = []

    data_collection.process_season(
        2007,
        _schedule(),
        _team_games(),
        min_season=2006,
        strength_snapshots=sink,
    )

    after_regular_season = constants.get_regular_season_weeks(2007) + 1
    weeks = [frame["week"].unique().item() for frame in sink]
    assert weeks == [1, 2, 3, after_regular_season]
    assert all(frame.height == len(_STRENGTH) for frame in sink)
    # Nothing is played after week 3, so the full-season snapshot solves weeks 1-3.
    final = sink[-1].filter(pl.col("team_abbr") == "AAA").row(0, named=True)
    assert final["strength_games_played"] == pytest.approx(2.0)


def test_process_season_does_not_duplicate_a_scheduled_postseason_week() -> None:
    """When the playoff week is already scheduled, its snapshot is recorded once."""
    after_regular_season = constants.get_regular_season_weeks(2007) + 1
    schedule = pl.concat(
        [
            _schedule(),
            pl.DataFrame(
                {
                    "season": [2007],
                    "week": [after_regular_season],
                    "away_abbr": ["BBB"],
                    "home_abbr": ["AAA"],
                }
            ),
        ]
    )
    sink: list[pl.DataFrame] = []

    data_collection.process_season(
        2007, schedule, _team_games(), min_season=2006, strength_snapshots=sink
    )

    weeks = [frame["week"].unique().item() for frame in sink]
    assert weeks.count(after_regular_season) == 1


def test_process_season_without_a_sink_records_nothing() -> None:
    """Callers that do not ask for snapshots get the unchanged game rows only."""
    with_sink = data_collection.process_season(
        2007, _schedule(), _team_games(), min_season=2006, strength_snapshots=[]
    )
    without_sink = data_collection.process_season(2007, _schedule(), _team_games(), min_season=2006)

    _assert_same_values(with_sink, without_sink)


def test_main_writes_the_strength_snapshot_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ETL entry point saves the collected snapshots next to the game datasets."""
    snapshot = data_collection.combine_strength_snapshots([_run_week(2, _team_games())[1]])
    saved: dict[str, pl.DataFrame] = {}

    def fake_collect(
        _seasons: list[int],
        *,
        config: data_collection.DataCollectionConfig,
        strength_snapshots: list[pl.DataFrame] | None = None,
    ) -> pl.DataFrame:
        """Record one weekly snapshot and return a tiny game frame."""
        del config
        assert strength_snapshots is not None
        strength_snapshots.append(snapshot)
        return pl.DataFrame({"season": [2007], "week": [2]})

    def fake_save(df: pl.DataFrame, name: str) -> None:
        """Capture every frame the entry point saves."""
        saved[name] = df

    monkeypatch.setattr(data_collection, "collect_all_data", fake_collect)
    monkeypatch.setattr(data_collection, "save_dataframe", fake_save)
    monkeypatch.setattr(data_collection.polars_utils, "remove_diff_columns", lambda df: df)
    monkeypatch.setattr(data_collection.polars_utils, "filter_completed_games", lambda df: df)
    monkeypatch.setattr(
        data_collection.polars_utils, "filter_upcoming_games", lambda df, *_args: df
    )

    data_collection.main(["--min-season", "2006", "--max-season", "2007"])

    assert constants.STRENGTH_SNAPSHOTS_NAME in saved
    assert_frame_equal(saved[constants.STRENGTH_SNAPSHOTS_NAME], snapshot)
