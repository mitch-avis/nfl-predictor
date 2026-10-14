"""Tests for the team-rating play-set tests (``.agents/sos_port/team_plays_test.py``)."""

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest
from tests.sos_port_scripts import load_sos_port_script

from nfl_predictor.ratings.inputs import team_game_rows
from nfl_predictor.ratings.pbp_inputs import team_games_from_pbp

if TYPE_CHECKING:
    from pathlib import Path

team = load_sos_port_script("team_plays_test")
common = load_sos_port_script("rating_tests_common")

_SEASON = 2010


_FLAG_COLUMNS = {
    "dropback": "qb_dropback",
    "rush": "rush",
    "passed": "pass",
    "kneel": "qb_kneel",
    "spike": "qb_spike",
    "special": "special",
}


def _play(
    game_id: str, posteam: str, play_type: str, epa: float, **flags: int
) -> dict[str, object]:
    """Return one regular-season play of ``game_id`` with the named flags set.

    The game id names both teams, so the defense is the team that is not ``posteam``.
    """
    _, week, away, home = game_id.split("_")
    return {
        "game_id": game_id,
        "season": _SEASON,
        "season_type": "REG",
        "week": int(week),
        "posteam": posteam,
        "defteam": home if posteam == away else away,
        "play_type": play_type,
        "qb_scramble": 0.0,
        **{column: float(flags.get(name, 0)) for name, column in _FLAG_COLUMNS.items()},
        "epa": epa,
    }


def _plays() -> pl.DataFrame:
    """Return two games: every kind of play the variants move, on both sides."""
    first, second = "2010_01_AAA_BBB", "2010_02_BBB_AAA"
    rows = [
        _play(first, "AAA", "pass", 0.4, dropback=1, passed=1),
        _play(first, "AAA", "run", -0.2, rush=1),
        _play(first, "AAA", "qb_kneel", -0.6, kneel=1),
        _play(first, "AAA", "no_play", -0.7, rush=1),
        _play(first, "AAA", "punt", 0.1, special=1),
        _play(first, "BBB", "pass", -0.3, dropback=1, passed=1),
        _play(first, "BBB", "qb_spike", -0.1, spike=1),
        _play(first, "BBB", "no_play", 0.9, passed=1),
        _play(first, "BBB", "no_play", -0.4),
        _play(second, "AAA", "pass", 1.1, dropback=1, passed=1),
        _play(second, "BBB", "qb_kneel", -0.5, kneel=1),
    ]
    return pl.DataFrame(rows)


def _schedule() -> pl.DataFrame:
    """Return the two games: AAA at BBB in week 1 (BBB at home), and BBB at AAA in week 2."""
    return pl.DataFrame(
        {
            "game_id": ["2010_01_AAA_BBB", "2010_02_BBB_AAA"],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "away_abbr": ["AAA", "BBB"],
            "home_abbr": ["BBB", "AAA"],
            "away_score": [20, 10],
            "home_score": [17, 13],
        }
    )


def _snaps(rows: pl.DataFrame) -> dict[tuple[str, str], tuple[int, float]]:
    """Return each team-game's scrimmage snaps and EPA, keyed by game and team."""
    return {
        (game_id, team_code): (snaps, epa)
        for game_id, team_code, snaps, epa in rows.select(
            "game_id", "team", "offensive_snaps", "offensive_epa"
        ).iter_rows()
    }


def _variant_rows(name: str) -> pl.DataFrame:
    """Return the synthetic team-game rows of the named variant."""
    plays = _plays()
    base = team_games_from_pbp(plays, _schedule())
    return team.team_rows_with_snaps(base, team.play_flags(plays), team.VARIANTS[name].snap)


def test_team_rows_with_snaps_today_reproduces_the_adapter_rows_exactly() -> None:
    # Arrange
    plays = _plays()
    base = team_games_from_pbp(plays, _schedule())

    # Act
    rows = team.team_rows_with_snaps(base, team.play_flags(plays), team.VARIANTS["today"].snap)

    # Assert
    assert rows.equals(base)


def test_without_kneels_and_spikes_removes_both_from_their_team_games() -> None:
    # Act
    rows = _variant_rows("without_kneel_spike")

    # Assert
    snaps = _snaps(rows)
    assert snaps[("2010_01_AAA_BBB", "AAA")] == (3, pytest.approx(0.4 - 0.2 - 0.7))
    assert snaps[("2010_01_AAA_BBB", "BBB")] == (1, pytest.approx(-0.3))
    assert snaps[("2010_02_BBB_AAA", "BBB")] == (0, 0.0)


def test_variant_rows_keep_every_team_game_and_the_special_teams_columns() -> None:
    # Arrange
    base = team_games_from_pbp(_plays(), _schedule())

    # Act
    rows = _variant_rows("without_kneel_spike")

    # Assert
    assert rows.drop("offensive_snaps", "offensive_epa").equals(
        base.drop("offensive_snaps", "offensive_epa")
    )


@pytest.mark.parametrize(
    ("name", "aaa_week_one", "bbb_week_one"),
    [
        ("every_wiped_snap", (4, 0.4 - 0.2 - 0.6 - 0.7), (3, -0.3 - 0.1 + 0.9)),
        ("no_wiped_snap", (3, 0.4 - 0.2 - 0.6), (2, -0.3 - 0.1)),
        ("wiped_passes_only", (3, 0.4 - 0.2 - 0.6), (3, -0.3 - 0.1 + 0.9)),
    ],
)
def test_penalty_variants_move_wiped_runs_and_wiped_passes(
    name: str, aaa_week_one: tuple[int, float], bbb_week_one: tuple[int, float]
) -> None:
    # Act
    rows = _variant_rows(name)

    # Assert
    snaps = _snaps(rows)
    assert snaps[("2010_01_AAA_BBB", "AAA")] == (aaa_week_one[0], pytest.approx(aaa_week_one[1]))
    assert snaps[("2010_01_AAA_BBB", "BBB")] == (bbb_week_one[0], pytest.approx(bbb_week_one[1]))


def test_play_flags_leave_a_penalty_without_run_or_pass_out_of_every_set() -> None:
    # Act
    flags = team.play_flags(_plays())

    # Assert
    flagless = flags.filter((pl.col("play_type") == "no_play") & (pl.col("epa") == -0.4))
    assert flagless.select("today_snap", "wiped_run", "wiped_pass").row(0) == (
        False,
        False,
        False,
    )


def _team_logs(weeks: int = 3, season: int = _SEASON) -> pl.DataFrame:
    """Return team-game rows of a four-team round robin, enough for a snapshot fit."""
    pairs = [[("AAA", "BBB"), ("CCC", "DDD")], [("AAA", "CCC"), ("BBB", "DDD")]]
    pairs.append([("AAA", "DDD"), ("BBB", "CCC")])
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(season)
    for week in range(1, weeks + 1):
        for home, away in pairs[(week - 1) % len(pairs)]:
            game_id = f"{season}_{week:02d}_{away}_{home}"
            margin = int(rng.integers(-14, 15))
            for side, opponent, is_home, sign in ((home, away, True, 1), (away, home, False, -1)):
                rows.append(
                    {
                        "game_id": game_id,
                        "week": week,
                        "team": side,
                        "opponent_team": opponent,
                        "is_home": is_home,
                        "offensive_snaps": 60,
                        "offensive_epa": float(rng.normal(0.0, 5.0)),
                        "st_plays": 12,
                        "st_epa": float(rng.normal(0.0, 1.0)),
                        "point_margin": sign * margin,
                    }
                )
    return pl.DataFrame(rows)


def _count_snapshot(prior_games: pl.DataFrame, season_logs: pl.DataFrame) -> pl.DataFrame:
    """Rate every team that has played at its game count, a snapshot that is easy to predict."""
    del season_logs
    return prior_games.group_by("team").agg(pl.len().cast(pl.Float64).alias("team_rating"))


def test_build_snapshot_feature_rows_week_one_has_zero_gap() -> None:
    # Arrange
    logs = _team_logs()

    # Act
    rows = team.build_snapshot_feature_rows(logs, _SEASON, "probe", _count_snapshot)

    # Assert
    week_one = rows.filter(pl.col("week") == 1)
    assert week_one.height == 2
    assert week_one.get_column("rating_diff").to_list() == [0.0, 0.0]


def test_build_snapshot_feature_rows_snapshot_sees_only_earlier_weeks() -> None:
    # Arrange
    logs = _team_logs()
    seen: list[int] = []

    def snapshot(prior_games: pl.DataFrame, season_logs: pl.DataFrame) -> pl.DataFrame:
        seen.extend(prior_games.select(pl.col("week").max()).get_column("week").to_list())
        return _count_snapshot(prior_games, season_logs)

    # Act
    team.build_snapshot_feature_rows(logs, _SEASON, "probe", snapshot)

    # Assert
    assert seen == [1, 2]


def test_build_snapshot_feature_rows_unrated_team_counts_as_zero() -> None:
    # Arrange
    logs = _team_logs()

    def snapshot(prior_games: pl.DataFrame, season_logs: pl.DataFrame) -> pl.DataFrame:
        return _count_snapshot(prior_games, season_logs).filter(pl.col("team") != "AAA")

    # Act
    rows = team.build_snapshot_feature_rows(logs, _SEASON, "probe", snapshot)

    # Assert
    week_two = rows.filter((pl.col("week") == 2) & (pl.col("home_team") == "AAA"))
    assert week_two.get_column("rating_diff").to_list() == [-1.0]


def test_fit_margin_projection_without_rows_is_zero() -> None:
    # Act
    fitted = team.fit_margin_projection(pl.DataFrame(schema=team.FEATURE_SCHEMA))

    # Assert
    assert fitted == (0.0, 0.0)


def test_fit_margin_projection_constant_gap_predicts_the_mean_margin() -> None:
    # Arrange
    rows = pl.DataFrame({"rating_diff": [0.0, 0.0, 0.0], "home_margin": [3.0, -1.0, 7.0]})

    # Act
    fitted = team.fit_margin_projection(rows)

    # Assert
    assert fitted == (0.0, pytest.approx(3.0))


def test_fit_margin_projection_recovers_an_exact_line() -> None:
    # Arrange
    gaps = np.array([-2.0, 0.5, 1.0, 4.0])
    rows = pl.DataFrame({"rating_diff": gaps, "home_margin": 1.5 * gaps + 2.0})

    # Act
    slope, edge = team.fit_margin_projection(rows)

    # Assert
    assert slope == pytest.approx(1.5)
    assert edge == pytest.approx(2.0)


def _feature_rows() -> pl.DataFrame:
    """Return feature rows of two seasons in which a later row would move the fit."""
    rows = [
        (2009, 1, "g1", 1.0, 2.0),
        (2009, 2, "g2", 2.0, 4.0),
        (2010, 1, "g3", 1.0, 2.0),
        (2010, 2, "g4", 3.0, 6.0),
        (2010, 3, "g5", 1.0, 100.0),
    ]
    return pl.DataFrame(
        [
            {
                "season": season,
                "week": week,
                "baseline": "probe",
                "game_id": game_id,
                "home_team": "AAA",
                "away_team": "BBB",
                "rating_diff": gap,
                "home_margin": margin,
            }
            for season, week, game_id, gap, margin in rows
        ],
        schema=team.FEATURE_SCHEMA,
    )


def test_evaluate_feature_rows_fits_each_week_on_earlier_rows_only() -> None:
    # Act
    predictions = team.evaluate_feature_rows(_feature_rows(), start_week=2)

    # Assert
    week_two = predictions.filter((pl.col("season") == 2010) & (pl.col("week") == 2)).row(
        0, named=True
    )
    assert week_two["training_row_count"] == 3
    assert week_two["predicted_margin"] == pytest.approx(6.0)
    assert week_two["error"] == pytest.approx(0.0)


def test_evaluate_feature_rows_scores_only_from_the_start_week() -> None:
    # Act
    predictions = team.evaluate_feature_rows(_feature_rows(), start_week=2)

    # Assert
    assert sorted(predictions.get_column("game_id").to_list()) == ["g2", "g4", "g5"]


def _interval(lower: float, upper: float) -> object:
    """Return an interval with its midpoint as the estimate."""
    return common.Interval(estimate=(lower + upper) / 2.0, lower=lower, upper=upper)


def test_qualifies_needs_overall_below_zero_and_no_band_above_zero() -> None:
    # Arrange
    overall = _interval(-0.2, -0.05)
    bands = {"weeks 5-7": _interval(-0.3, 0.1), "weeks 8+": _interval(-0.2, -0.01)}

    # Act
    result = team.qualifies(overall, bands)

    # Assert
    assert result


def test_qualifies_band_above_zero_disqualifies() -> None:
    # Arrange
    overall = _interval(-0.2, -0.05)
    bands = {"weeks 5-7": _interval(0.01, 0.3), "weeks 8+": _interval(-0.4, -0.1)}

    # Act
    result = team.qualifies(overall, bands)

    # Assert
    assert not result


def _comparison(overall: tuple[float, float], band: tuple[float, float] = (-1.0, 1.0)) -> object:
    """Return a variant's comparison against today's: an overall interval and two bands."""
    return team.Comparison(
        overall=_interval(*overall),
        bands={"weeks 5-7": _interval(*band), "weeks 8+": _interval(*band)},
    )


def test_decide_recommends_the_qualifying_variant_with_the_lowest_mae() -> None:
    # Arrange
    comparisons = {
        "every_wiped_snap": _comparison((-0.3, -0.1)),
        "no_wiped_snap": _comparison((-0.2, -0.01)),
        "wiped_passes_only": _comparison((-0.1, 0.2)),
    }
    maes = {"today": 10.6, "every_wiped_snap": 10.55, "no_wiped_snap": 10.5}

    # Act
    decision = team.decide(comparisons, maes, team.PENALTY_ORDER)

    # Assert
    assert decision.recommended == "no_wiped_snap"
    assert decision.qualifying == ("every_wiped_snap", "no_wiped_snap")


def test_decide_without_a_qualifier_takes_the_simplest_not_significantly_worse() -> None:
    # Arrange
    comparisons = {
        "every_wiped_snap": _comparison((0.02, 0.2)),
        "no_wiped_snap": _comparison((-0.1, 0.1)),
        "wiped_passes_only": _comparison((-0.1, 0.1)),
    }

    # Act
    decision = team.decide(comparisons, {}, team.PENALTY_ORDER)

    # Assert
    assert decision.recommended == "no_wiped_snap"
    assert decision.qualifying == ()


def test_decide_kneel_spike_tie_keeps_today() -> None:
    # Arrange
    comparisons = {"without_kneel_spike": _comparison((-0.1, 0.05))}

    # Act
    decision = team.decide(comparisons, {}, team.KNEEL_SPIKE_ORDER)

    # Assert
    assert decision.recommended == "today"


def test_decide_lists_every_interval_excluding_zero_in_either_direction() -> None:
    # Arrange
    comparisons = {"without_kneel_spike": _comparison((0.01, 0.2), band=(-0.5, -0.1))}

    # Act
    decision = team.decide(comparisons, {}, team.KNEEL_SPIKE_ORDER)

    # Assert
    assert decision.excluding_zero == (
        ("without_kneel_spike", "overall", "worse"),
        ("without_kneel_spike", "weeks 5-7", "better"),
        ("without_kneel_spike", "weeks 8+", "better"),
    )


def _predictions(errors: dict[str, list[float]]) -> pl.DataFrame:
    """Return prediction rows of the named baselines with the given errors, one per game."""
    rows = [
        {"season": 2000 + index // 2, "week": 5 + index % 2, "game_id": f"g{index}"}
        | {"baseline": name, "error": value}
        for name, values in errors.items()
        for index, value in enumerate(values)
    ]
    return pl.DataFrame(rows)


def test_paired_differences_are_absolute_errors_of_variant_minus_today() -> None:
    # Arrange
    predictions = _predictions({"today": [1.0, -2.0], "variant": [-0.5, 3.0]})

    # Act
    differences = team.paired_differences(predictions, "variant")

    # Assert
    assert differences.get_column("difference").to_list() == [-0.5, 1.0]


def test_paired_differences_different_games_raise() -> None:
    # Arrange
    predictions = _predictions({"today": [1.0, -2.0], "variant": [-0.5]})

    # Act & Assert
    with pytest.raises(ValueError, match="same games"):
        team.paired_differences(predictions, "variant")


def test_check_information_set_catches_a_snapshot_that_reads_later_weeks() -> None:
    # Arrange
    logs = _team_logs()

    def leaky(prior_games: pl.DataFrame, season_logs: pl.DataFrame) -> pl.DataFrame:
        later = season_logs.filter(pl.col("week") > prior_games.get_column("week").max())
        return later.group_by("team").agg(pl.col("offensive_epa").sum().alias("team_rating"))

    # Act & Assert
    with pytest.raises(ValueError, match="information set"):
        team.check_information_set(logs, leaky)


def test_check_information_set_catches_a_snapshot_reading_later_special_teams_plays() -> None:
    # Arrange
    def leaky(prior_games: pl.DataFrame, season_logs: pl.DataFrame) -> pl.DataFrame:
        week = int(prior_games.get_column("week").max()) + 1
        signal = float(season_logs.filter(pl.col("week") >= week).get_column("st_plays").sum())
        return pl.DataFrame({"team": ["AAA", "BBB"], "team_rating": [signal, -signal]})

    # Act & Assert
    with pytest.raises(ValueError, match="information set"):
        team.check_information_set(_team_logs(), leaky)


def test_check_information_set_passes_a_snapshot_of_earlier_weeks() -> None:
    # Arrange
    logs = _team_logs()

    # Act
    checked = team.check_information_set(logs, _count_snapshot)

    # Assert
    assert checked == 2


def _league() -> dict[int, pl.DataFrame]:
    """Return two seasons of team-game rows, from the first play-by-play season."""
    return {season: team_game_rows(_team_logs(season=season)) for season in (1999, 2000)}


def _run(name: str, rows: dict[int, pl.DataFrame]) -> Any:
    """Return a variant run of the synthetic league, its information set checked."""
    return team.run_variant(name, rows, sorted(rows), check_information=True)


def test_run_variant_rates_every_home_game_and_checks_every_snapshot() -> None:
    # Arrange
    rows = _league()

    # Act
    run = _run("today", rows)

    # Assert
    assert run.features.height == 12
    assert run.information_checked == 4
    assert sorted(run.full_season) == [1999, 2000]
    assert sorted(run.penalties) == [2000]


def test_check_penalties_passes_a_variants_own_penalties() -> None:
    # Arrange
    run = _run("today", _league())

    # Act
    checked = team.check_penalties({"today": run})

    # Assert
    assert checked == 1


def test_check_penalties_catches_a_penalty_from_another_season() -> None:
    # Arrange
    run = _run("today", _league())
    run.penalties[2000] = (1.0, 1.0)

    # Act & Assert
    with pytest.raises(team.IntegrityError, match="check 5"):
        team.check_penalties({"today": run})


def _predictions_of(run: Any) -> pl.DataFrame:
    """Return the predictions of a run's feature rows from week 2 on."""
    return team.evaluate_feature_rows(run.features, start_week=2)


def test_check_margin_rows_passes_earlier_rows_only() -> None:
    # Arrange
    run = _run("today", _league())

    # Act
    checked = team.check_margin_rows(_predictions_of(run), run.features)

    # Assert
    assert checked == 8


def test_check_margin_rows_catches_a_model_trained_on_a_later_row() -> None:
    # Arrange
    run = _run("today", _league())
    predictions = _predictions_of(run).with_columns(pl.col("training_row_count") + 1)

    # Act & Assert
    with pytest.raises(team.IntegrityError, match="check 6"):
        team.check_margin_rows(predictions, run.features)


def test_check_zero_change_catches_a_moved_prediction() -> None:
    # Arrange
    rows = _league()
    today = _run("today", rows)
    zero = _run("zero_change", rows)
    predictions = team.evaluate_feature_rows(
        pl.concat([today.features, zero.features]), start_week=2
    ).with_columns(
        pl.when(pl.col("baseline") == "zero_change")
        .then(pl.col("predicted_margin") + 1e-6)
        .otherwise(pl.col("predicted_margin"))
        .alias("predicted_margin")
    )

    # Act & Assert
    with pytest.raises(team.IntegrityError, match="check 4"):
        team.check_zero_change(zero, today, predictions)


def test_check_source_validation_catches_another_mae() -> None:
    # Arrange
    run = _run("today", _league())

    # Act & Assert
    with pytest.raises(team.IntegrityError, match="check 2"):
        team.check_source_validation(_predictions_of(run))


def _inputs() -> Any:
    """Return the synthetic play-by-play's team inputs, keyed by its season."""
    plays = _plays()
    return team.TeamInputs(
        team_games={_SEASON: team_games_from_pbp(plays, _schedule())},
        flags={_SEASON: team.play_flags(plays)},
    )


def _rows_run(name: str, rows: pl.DataFrame) -> Any:
    """Return a variant run that holds only its rows, for the row checks."""
    return team.VariantRun(
        name=name,
        rows={_SEASON: rows},
        features=pl.DataFrame(schema=team.FEATURE_SCHEMA),
        snapshots={},
        full_season={},
        penalties={},
    )


def _without_kneel_spike_rows(inputs: Any) -> pl.DataFrame:
    """Return the synthetic team-game rows without kneel-downs and spikes, as the fits take them."""
    return team_game_rows(
        team.team_rows_with_snaps(
            inputs.team_games[_SEASON],
            inputs.flags[_SEASON],
            team.VARIANTS["without_kneel_spike"].snap,
        )
    )


def test_check_variant_rows_counts_the_moved_plays() -> None:
    # Arrange
    inputs = _inputs()
    runs = {"without_kneel_spike": _rows_run("x", _without_kneel_spike_rows(inputs))}

    # Act
    result = team.check_variant_rows(inputs, runs)

    # Assert
    assert result["moved"]["without_kneel_spike"]["removed"] == 3
    assert result["moved"]["without_kneel_spike"]["removed_epa"] == pytest.approx(-1.2)


def test_check_variant_rows_catches_a_moved_play_with_the_wrong_epa() -> None:
    # Arrange
    inputs = _inputs()
    rows = _without_kneel_spike_rows(inputs).with_columns(pl.col("offensive_epa") + 0.01)

    # Act & Assert
    with pytest.raises(team.IntegrityError, match="check 3"):
        team.check_variant_rows(inputs, {"without_kneel_spike": _rows_run("x", rows)})


def test_moved_by_penalty_side_splits_by_the_penalized_team() -> None:
    # Arrange
    plays = _plays().with_columns(
        pl.when(pl.col("play_type") == "no_play")
        .then(pl.col("defteam"))
        .otherwise(None)
        .alias("penalty_team")
    )
    inputs = team.TeamInputs(
        team_games={_SEASON: team_games_from_pbp(plays, _schedule())},
        flags={_SEASON: team.play_flags(plays)},
    )

    # Act
    sides = team.moved_by_penalty_side(inputs)

    # Assert
    assert sides == [
        {"kind": "wiped_run", "side": "defense", "plays": 1, "epa": pytest.approx(-0.7)},
        {"kind": "wiped_pass", "side": "defense", "plays": 1, "epa": pytest.approx(0.9)},
    ]


def test_feature_impact_skips_week_one_zero_snapshots() -> None:
    # Arrange
    rows = _league()
    today = _run("today", rows)
    variant = _run(
        "without_kneel_spike",
        {
            season: frame.with_columns(pl.col("offensive_epa") * 0.5)
            for season, frame in rows.items()
        },
    )

    # Act
    impact = team.feature_impact(variant, today)

    # Assert
    assert impact["team_rating"]["rows"] == 16.0
    assert impact["team_rating"]["rows_moved"] > 0


def test_wiped_play_summary_counts_each_set_by_flag() -> None:
    # Act
    summary = team.wiped_play_summary(_plays())

    # Assert
    counts = {row["set"]: (row["plays"], row["epa"]) for row in summary.iter_rows(named=True)}
    assert counts["every penalty-wiped play"] == (3, pytest.approx(-0.2))
    assert counts["with the rush flag"] == (1, pytest.approx(-0.7))
    assert counts["with the pass flag"] == (1, pytest.approx(0.9))
    assert counts["wiped runs (today's snaps), rush flag"] == (1, pytest.approx(-0.7))
    assert counts["wiped runs (today's snaps), qb_kneel flag"] == (0, 0.0)
    assert counts["wiped passes"] == (1, pytest.approx(0.9))


def test_provenance_records_the_git_commit_command_and_runtime(tmp_path: Path) -> None:
    # Arrange
    ledger = common.InputLedger()

    # Act
    provenance = team._provenance(tmp_path, tmp_path, ledger, "python team.py", 61.0)

    # Assert
    assert re.fullmatch(r"[0-9a-f]{40}", provenance["git_commit"])
    assert (provenance["command"], provenance["seconds"]) == ("python team.py", 61.0)
    assert provenance["input_fingerprint"] == ledger.fingerprint()
