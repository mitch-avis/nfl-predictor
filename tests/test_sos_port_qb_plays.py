"""Tests for the quarterback all-plays test (``.agents/sos_port/qb_plays_test.py``)."""

from typing import Any

import numpy as np
import polars as pl
import pytest
from tests.sos_port_scripts import load_sos_port_script
from threadpoolctl import threadpool_limits

from nfl_predictor.ratings.ridge import UnitFit
from nfl_predictor.ratings.team_rating import fit_team_ratings

qb = load_sos_port_script("qb_plays_test")
common = load_sos_port_script("rating_tests_common")

_GAME = "2010_01_AAA_BBB"
_QB = "00-QB"
_RB = "00-RB"
_C = "00-C"
_FLAG_COLUMNS = ("rush", "qb_scramble", "aborted_play", "two_point_attempt")


def _play(play_type: str, epa: float, **values: object) -> dict[str, object]:
    """Return one play of AAA's offense in the test game; ``values`` sets flags and players."""
    row: dict[str, object] = {
        "game_id": _GAME,
        "posteam": "AAA",
        "play_type": play_type,
        "epa": epa,
        "qb_epa": values.pop("qb_epa", epa),
        "rusher_player_id": None,
        "passer_player_id": None,
        "fumbled_1_player_id": None,
        **dict.fromkeys(_FLAG_COLUMNS, 0.0),
    }
    row.update(values)
    return row


def _attribution_plays() -> pl.DataFrame:
    """Return one play of every kind the candidates count, or must not count."""
    return pl.DataFrame(
        [
            _play("pass", 0.5, passer_player_id=_QB),
            _play("run", 1.2, rusher_player_id=_QB, qb_scramble=1.0),
            _play("run", 0.3, rusher_player_id=_QB, rush=1.0),
            _play(
                "run",
                -2.0,
                rusher_player_id=_RB,
                rush=1.0,
                aborted_play=1.0,
                fumbled_1_player_id=_QB,
            ),
            _play(
                "run",
                -1.0,
                rusher_player_id=_QB,
                rush=1.0,
                aborted_play=1.0,
                fumbled_1_player_id=_C,
            ),
            _play("qb_kneel", -0.6, rusher_player_id=_QB),
            _play("qb_spike", -0.1, passer_player_id=_QB),
            _play("run", 0.8, rusher_player_id=_QB, qb_scramble=1.0, two_point_attempt=1.0),
            _play("no_play", -0.7, rusher_player_id=_QB, rush=1.0),
            _play("run", 0.4, rusher_player_id=_QB),
            _play("run", 0.2, rusher_player_id=_RB, rush=1.0),
        ],
        infer_schema_length=None,
    )


def _parts_of(player: str) -> dict[str, Any]:
    """Return one player's play parts in the test game."""
    parts = qb.qb_play_parts(_attribution_plays())
    return parts.filter(pl.col("qb_id") == player).row(0, named=True)


def test_qb_play_parts_scramble_goes_to_the_rusher_without_a_passer() -> None:
    # Act
    parts = _parts_of(_QB)

    # Assert
    assert (parts["scrambles"], parts["scramble_epa"]) == (1, pytest.approx(1.2))


def test_qb_play_parts_designed_run_and_botched_snaps_by_rusher_and_by_charge() -> None:
    # Act
    parts = _parts_of(_QB)

    # Assert
    assert (parts["designed_runs"], parts["designed_run_epa"]) == (1, pytest.approx(0.3))
    assert (parts["aborted"], parts["aborted_epa"]) == (1, pytest.approx(-1.0))
    assert (parts["charged_aborted"], parts["charged_aborted_epa"]) == (1, pytest.approx(-2.0))


def test_qb_play_parts_kneel_spike_two_point_and_other_runs_land_in_their_own_parts() -> None:
    # Act
    parts = _parts_of(_QB)

    # Assert
    assert (parts["kneels"], parts["spikes"], parts["two_point_runs"]) == (1, 1, 1)
    assert (parts["other_runs"], parts["other_run_epa"]) == (1, pytest.approx(0.4))
    assert parts["pass_spike_qb_epa"] == pytest.approx(0.5 - 0.1)


def test_qb_play_parts_running_back_carry_is_not_charged_to_the_quarterback() -> None:
    # Act
    parts = _parts_of(_RB)

    # Assert
    assert (parts["designed_runs"], parts["aborted"], parts["charged_aborted"]) == (1, 1, 0)


def test_qb_play_parts_scramble_that_is_a_charged_botched_snap_counts_once() -> None:
    # Arrange
    plays = pl.DataFrame(
        [
            _play(
                "run",
                -1.5,
                rusher_player_id=_QB,
                fumbled_1_player_id=_QB,
                qb_scramble=1.0,
                rush=1.0,
                aborted_play=1.0,
            ),
            _play("pass", 0.1, passer_player_id=_QB),
        ],
        infer_schema_length=None,
    )

    # Act
    parts = qb.qb_play_parts(plays).filter(pl.col("qb_id") == _QB).row(0, named=True)

    # Assert
    assert (parts["scrambles"], parts["scramble_epa"]) == (1, pytest.approx(-1.5))
    assert (parts["designed_runs"], parts["charged_aborted"]) == (0, 0)


def _qb_rows(dropbacks: tuple[int, ...] = (10, 0)) -> pl.DataFrame:
    """Return two quarterback rows of AAA in the test game: the starter and a backup."""
    return pl.DataFrame(
        {
            "game_id": [_GAME, _GAME],
            "week": [1, 1],
            "team": ["AAA", "AAA"],
            "opponent_team": ["BBB", "BBB"],
            "is_home": [False, False],
            "qb_id": [_QB, "00-QB2"],
            "qb_name": ["Starter", "Backup"],
            "qb_dropbacks": list(dropbacks),
            "qb_dropback_epa": [2.0, 0.0],
            "qb_attempts": list(dropbacks),
        }
    )


def test_candidate_rows_add_each_candidates_plays_and_epa() -> None:
    # Arrange
    parts = qb.qb_play_parts(_attribution_plays())

    # Act
    rows = {name: qb.candidate_rows(_qb_rows(), parts, name) for name in ("A", "B", "C")}

    # Assert
    assert rows["A"].select("qb_dropbacks", "qb_dropback_epa").row(0) == (10, 2.0)
    assert rows["B"].select("qb_dropbacks", "qb_dropback_epa").row(0) == (11, pytest.approx(3.2))
    assert rows["C"].select("qb_dropbacks", "qb_dropback_epa").row(0) == (
        13,
        pytest.approx(3.2 + 0.3 - 2.0),
    )


def test_candidate_rows_keep_only_rows_with_a_dropback() -> None:
    # Arrange
    parts = qb.qb_play_parts(_attribution_plays())

    # Act
    rows = qb.candidate_rows(_qb_rows(), parts, "C")

    # Assert
    assert rows.get_column("qb_id").to_list() == [_QB]


def test_candidate_rows_with_no_added_plays_equal_passes_and_sacks() -> None:
    # Arrange
    parts = qb.qb_play_parts(_attribution_plays()).with_columns(
        pl.col(column).cast(pl.Float64) * 0 for column in qb.PART_COLUMNS
    )

    # Act
    rows = qb.candidate_rows(_qb_rows(), parts, "C")

    # Assert
    assert rows.equals(qb.candidate_rows(_qb_rows(), parts, "A"))


def _snap_plays() -> pl.DataFrame:
    """Return first-quarter plays: a carry, a passer without a fit row, then the starter."""
    rows = [
        (1, 10, None, _RB),
        (1, 20, "00-WR", None),
        (1, 30, "00-QB2", None),
        (1, 40, None, _QB),
        (1, 50, _QB, None),
    ]
    return pl.DataFrame(
        [
            {
                "game_id": _GAME,
                "posteam": "AAA",
                "qtr": quarter,
                "play_id": play_id,
                "passer_player_id": passer,
                "rusher_player_id": rusher,
            }
            for quarter, play_id, passer, rusher in rows
        ],
        schema={
            "game_id": pl.String,
            "posteam": pl.String,
            "qtr": pl.Int64,
            "play_id": pl.Int64,
            "passer_player_id": pl.String,
            "rusher_player_id": pl.String,
        },
    )


def test_first_qb_snaps_skips_carries_passers_without_a_fit_row_and_gadget_passers() -> None:
    # Act
    starters = qb.first_qb_snaps(_snap_plays(), _qb_rows())

    # Assert
    assert starters.select("game_id", "team", "qb_id").rows() == [(_GAME, "AAA", _QB)]


def test_first_qb_snaps_without_a_first_quarter_snap_leaves_the_starter_unknown() -> None:
    # Arrange
    plays = _snap_plays().with_columns(pl.lit(2, dtype=pl.Int64).alias("qtr"))

    # Act
    starters = qb.first_qb_snaps(plays, _qb_rows())

    # Assert
    assert starters.is_empty()


def _fit() -> UnitFit:
    """Return a snapshot fit with one passer and one defense."""
    return UnitFit(
        intercept=0.05,
        home_field=0.02,
        ridge_lambda=10.0,
        offense={_QB: 0.1},
        defense={"BBB": 0.03},
    )


@pytest.mark.parametrize(
    ("starter", "opponent", "sign", "expected"),
    [
        (_QB, "BBB", 1.0, 0.05 + 0.1 - 0.03 + 0.02),
        (None, "BBB", 1.0, 0.05 - 0.03 + 0.02),
        ("00-NEW", "BBB", -1.0, 0.05 - 0.03 - 0.02),
        (_QB, "ZZZ", 1.0, 0.05 + 0.1 + 0.02),
    ],
)
def test_passer_prediction_uses_zero_for_an_unknown_starter_or_defense(
    starter: str | None, opponent: str, sign: float, expected: float
) -> None:
    # Act
    prediction = qb.passer_prediction(_fit(), starter, opponent, sign)

    # Assert
    assert prediction == pytest.approx(expected)


def test_passer_prediction_without_a_snapshot_is_zero() -> None:
    # Act
    prediction = qb.passer_prediction(None, _QB, "BBB", 1.0)

    # Assert
    assert prediction == 0.0


def test_fit_calibration_without_rows_is_zero() -> None:
    # Act
    coefficients = qb.fit_calibration(np.empty((0, 2)), np.empty(0))

    # Assert
    assert coefficients.tolist() == [0.0, 0.0, 0.0]


def test_fit_calibration_recovers_an_exact_plane() -> None:
    # Arrange
    rng = np.random.default_rng(0)
    features = rng.normal(size=(12, 2))
    target = 0.7 * features[:, 0] - 0.4 * features[:, 1] + 0.05

    # Act
    coefficients = qb.fit_calibration(features, target)

    # Assert
    assert coefficients.tolist() == pytest.approx([0.7, -0.4, 0.05])


def _interval(lower: float, upper: float) -> Any:
    """Return an interval with its midpoint as the estimate."""
    return common.Interval(estimate=(lower + upper) / 2.0, lower=lower, upper=upper)


def test_interval_text_keeps_a_difference_of_order_1e_5_visible() -> None:
    # Arrange
    interval = common.Interval(estimate=1e-5, lower=-2e-5, upper=4e-5)

    # Act
    text = qb._interval_text(interval)

    # Assert
    assert text == "+1.00e-05 (-2.00e-05 to +4.00e-05)"


_TIE = (-0.01, 0.01)
_LOWER = (-0.03, -0.01)
_HIGHER = (0.01, 0.03)


def _comparisons(**overall: tuple[float, float]) -> dict[tuple[str, str], Any]:
    """Return the three comparisons, each overall interval given (bands tie)."""
    return {
        pair: qb.Comparison(
            overall=_interval(*overall[f"{pair[0]}_{pair[1]}"]),
            bands={"weeks 5-8": _interval(*_TIE), "weeks 9+": _interval(*_TIE)},
        )
        for pair in (("B", "A"), ("C", "A"), ("C", "B"))
    }


@pytest.mark.parametrize(
    ("order", "overall", "expected"),
    [
        (("C", "B", "A"), {"B_A": _TIE, "C_A": _TIE, "C_B": _TIE}, "C"),
        (("B", "A", "C"), {"B_A": _HIGHER, "C_A": _TIE, "C_B": _TIE}, "A"),
        (("C", "B", "A"), {"B_A": _LOWER, "C_A": _TIE, "C_B": _TIE}, "C"),
        (("A", "B", "C"), {"B_A": _LOWER, "C_A": _TIE, "C_B": _TIE}, "B"),
        (("A", "B", "C"), {"B_A": _TIE, "C_A": _LOWER, "C_B": _TIE}, "B"),
        (("A", "B", "C"), {"B_A": _TIE, "C_A": _LOWER, "C_B": _LOWER}, "C"),
        (("C", "B", "A"), {"B_A": _TIE, "C_A": _HIGHER, "C_B": _HIGHER}, "B"),
        (("C", "B", "A"), {"B_A": _HIGHER, "C_A": _HIGHER, "C_B": _TIE}, "A"),
    ],
)
def test_decide_recommends_the_first_candidate_in_the_order_no_other_beats(
    order: tuple[str, ...], overall: dict[str, tuple[float, float]], expected: str
) -> None:
    # Act
    decision = qb.decide(_comparisons(**overall), order)

    # Assert
    assert decision.recommended == expected


def test_decide_band_above_zero_is_not_a_win() -> None:
    # Arrange
    comparisons = _comparisons(B_A=_TIE, C_A=_LOWER, C_B=_TIE)
    comparisons[("C", "A")] = qb.Comparison(
        overall=_interval(*_LOWER),
        bands={"weeks 5-8": _interval(*_HIGHER), "weeks 9+": _interval(*_LOWER)},
    )

    # Act
    decision = qb.decide(comparisons, ("A", "B", "C"))

    # Assert
    assert decision.recommended == "A"
    assert decision.wins == ()


def test_decide_reports_every_interval_excluding_zero_in_either_direction() -> None:
    # Act
    decision = qb.decide(_comparisons(B_A=_HIGHER, C_A=_TIE, C_B=_LOWER), ("C", "B", "A"))

    # Assert
    assert decision.excluding_zero == (
        ("B", "A", "overall", "worse"),
        ("C", "B", "overall", "better"),
    )
    assert decision.wins == (("A", "B"), ("C", "B"))


def test_qb_decide_cycle_of_wins_is_not_silently_resolved() -> None:
    # Arrange: plus scrambles beats passes and sacks, every play beats plus scrambles, and
    # passes and sacks beats every play, so every candidate is beaten.
    comparisons = _comparisons(B_A=_LOWER, C_A=_HIGHER, C_B=_LOWER)

    # Act
    decision = qb.decide(comparisons, qb.ORDER)

    # Assert
    assert {loser for _, loser in decision.wins} == set(qb.CANDIDATES)
    assert decision.recommended is None


def test_qb_decide_without_wins_recommends_the_first_in_the_users_order() -> None:
    # Act
    decision = qb.decide(_comparisons(B_A=_TIE, C_A=_TIE, C_B=_TIE), qb.ORDER)

    # Assert
    assert decision.wins == ()
    assert decision.recommended == qb.ORDER[0]


def _results_of(decision: Any) -> Any:
    """Return results whose headline reads ``decision``, the scores and intervals all ties."""
    comparisons = _comparisons(B_A=_TIE, C_A=_TIE, C_B=_TIE)
    return qb.Results(
        integrity={},
        scores={name: {qb.OVERALL: (100, 0.15, 0.2)} for name in qb.CANDIDATES},
        comparisons=comparisons,
        decision=decision,
        extras={},
    )


def test_headline_cycle_of_wins_goes_to_the_user() -> None:
    # Arrange
    decision = qb.decide(_comparisons(B_A=_LOWER, C_A=_HIGHER, C_B=_LOWER), qb.ORDER)

    # Act
    text = " ".join(qb._headline(_results_of(decision)))

    # Assert
    assert "between the rule's branches" in text
    assert "goes to the user" in text
    assert "The recommendation is" not in text


_TEAMS = ("AAA", "BBB", "CCC", "DDD")
_PAIRINGS = (
    (("AAA", "BBB"), ("CCC", "DDD")),
    (("AAA", "CCC"), ("BBB", "DDD")),
    (("AAA", "DDD"), ("BBB", "CCC")),
)
_WINDOW_SEASONS = (2005, 2006, 2007)


def _game_rows(season: int, weeks: int, rng: np.random.Generator) -> list[dict[str, object]]:
    """Return one season's team-game and quarterback values, one dict per team-game."""
    rows: list[dict[str, object]] = []
    for week in range(1, weeks + 1):
        for home, away in _PAIRINGS[(week - 1) % len(_PAIRINGS)]:
            for side, opponent, is_home in ((home, away, True), (away, home, False)):
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{away}_{home}",
                        "week": week,
                        "team": side,
                        "opponent_team": opponent,
                        "is_home": is_home,
                        "qb_id": f"{side}-2" if side == "AAA" and week >= 3 else f"{side}-1",
                        "offensive_snaps": 60,
                        "offensive_epa": float(rng.normal(0.0, 4.0)),
                        "st_plays": 10,
                        "st_epa": float(rng.normal(0.0, 1.0)),
                        "point_margin": int(rng.integers(-10, 11)),
                        "qb_dropbacks": 10,
                        "qb_dropback_epa": float(rng.normal(0.5, 2.0)),
                        "qb_attempts": 9,
                        "scrambles": 1,
                        "scramble_epa": float(rng.normal(0.4, 1.0)),
                    }
                )
    return rows


def _harness_inputs(weeks: int = 4) -> Any:
    """Return a small league's harness inputs: AAA's backup starts from week 3 each season."""
    rng = np.random.default_rng(7)
    team_rows: dict[int, pl.DataFrame] = {}
    qb_rows: dict[int, pl.DataFrame] = {}
    parts: dict[int, pl.DataFrame] = {}
    starters: dict[int, pl.DataFrame] = {}
    for season in _WINDOW_SEASONS:
        games = pl.DataFrame(_game_rows(season, weeks, rng))
        team_rows[season] = games.select(
            "game_id",
            "week",
            "team",
            "opponent_team",
            "is_home",
            "offensive_snaps",
            "offensive_epa",
            "st_plays",
            "st_epa",
            "point_margin",
        )
        qb_rows[season] = games.select(
            "game_id",
            "week",
            "team",
            "opponent_team",
            "is_home",
            "qb_id",
            pl.col("qb_id").alias("qb_name"),
            "qb_dropbacks",
            "qb_dropback_epa",
            "qb_attempts",
        )
        parts[season] = games.select(
            "game_id",
            "team",
            "qb_id",
            *(
                (pl.col(column) if column in {"scrambles", "scramble_epa"} else pl.lit(0)).alias(
                    column
                )
                for column in qb.PART_COLUMNS
            ),
        )
        starters[season] = games.select("game_id", "team", "qb_id")
    candidate_rows = {
        name: {
            season: qb.candidate_rows(qb_rows[season], parts[season], name) for season in qb_rows
        }
        for name in qb.CANDIDATES
    }
    history_rows = dict(team_rows)
    return qb.HarnessInputs(
        team_rows=team_rows,
        candidate_rows=candidate_rows,
        starters=starters,
        team_penalties=lambda season: fit_team_ratings(history_rows[season - 1]),
        season_prior=lambda season: None,
    )


_SMALL_WINDOW = qb.Window(
    penalty_season=2005, warmup_season=2006, first_scored=2007, last_season=2007, start_week=3
)


def test_run_harness_predicts_every_team_game_from_week_two_for_every_candidate() -> None:
    # Arrange
    inputs = _harness_inputs()

    # Act
    result = qb.run_harness(inputs, _SMALL_WINDOW)

    # Assert
    predictions = result.predictions
    assert predictions.height == 3 * 2 * 3 * 4
    assert sorted(predictions.get_column("week").unique().to_list()) == [2, 3, 4]


def test_run_harness_team_term_comes_from_the_snapshot_behind_its_ratings() -> None:
    # Arrange
    inputs = _harness_inputs()

    # Act
    result = qb.run_harness(inputs, _SMALL_WINDOW)

    # Assert
    assert result.audit.team_term_gap <= 1e-12
    assert result.audit.team_snapshots == 2 * 3


def test_check_week_information_passes_the_harness_prediction() -> None:
    # Arrange
    inputs = _harness_inputs()
    lambdas = dict.fromkeys(qb.CANDIDATES, 10.0)

    # Act
    with threadpool_limits(limits=1, user_api="blas"):
        gap = qb.check_week_information(qb.predict_week, inputs, 2007, 3, lambdas)

    # Assert
    assert gap == 0.0


def test_check_week_information_catches_a_prediction_that_reads_its_own_week() -> None:
    # Arrange
    inputs = _harness_inputs()
    lambdas = dict.fromkeys(qb.CANDIDATES, 10.0)

    def leaky(inputs: Any, season: int, week: int, lambdas: Any) -> Any:
        rows = inputs.team_rows[season]
        own = rows.filter(pl.col("week") == week).get_column("offensive_epa").sum()
        result = qb.predict_week(inputs, season, week, lambdas)
        return result.with_columns(pl.col("t") + own)

    # Act & Assert
    with pytest.raises(qb.IntegrityError, match="check 8"):
        qb.check_week_information(leaky, inputs, 2007, 3, lambdas)


def test_check_week_information_catches_a_prediction_reading_later_point_margins() -> None:
    # Arrange
    inputs = _harness_inputs()
    lambdas = dict.fromkeys(qb.CANDIDATES, 10.0)

    def leaky(inputs: Any, season: int, week: int, lambdas: Any) -> Any:
        rows = inputs.team_rows[season].filter(pl.col("week") >= week)
        signal = float(rows.get_column("point_margin").abs().sum())
        return qb.predict_week(inputs, season, week, lambdas).with_columns(pl.col("t") + signal)

    # Act & Assert
    with threadpool_limits(limits=1, user_api="blas"), pytest.raises(qb.IntegrityError):
        qb.check_week_information(leaky, inputs, 2007, 3, lambdas)


def _later_team_values(column: str) -> Any:
    """Return a reader of ``column`` over a season's team rows from the predicted week on."""

    def read(inputs: Any, season: int, week: int) -> float:
        rows = inputs.team_rows[season].filter(pl.col("week") >= week)
        return float(rows.get_column(column).cast(pl.Float64).sum())

    return read


def _next_week_sites(inputs: Any, season: int, week: int) -> float:
    """Return how many of AAA's games after ``week`` are at home."""
    rows = inputs.team_rows[season].filter((pl.col("week") > week) & (pl.col("team") == "AAA"))
    return float(rows.get_column("is_home").cast(pl.Float64).sum())


def _later_attempts(inputs: Any, season: int, week: int) -> float:
    """Return passes and sacks' pass attempts from ``week`` on."""
    rows = inputs.candidate_rows["A"][season].filter(pl.col("week") >= week)
    return float(rows.get_column("qb_attempts").sum())


@pytest.mark.parametrize(
    "reader",
    [
        _later_team_values("st_plays"),
        _later_team_values("st_epa"),
        _next_week_sites,
        _later_attempts,
    ],
    ids=["st_plays", "st_epa", "is_home_after_the_week", "qb_attempts"],
)
def test_check_week_information_catches_a_prediction_reading_any_later_value(reader: Any) -> None:
    # Arrange
    inputs = _harness_inputs()
    lambdas = dict.fromkeys(qb.CANDIDATES, 10.0)

    def leaky(inputs: Any, season: int, week: int, lambdas: Any) -> Any:
        signal = reader(inputs, season, week)
        return qb.predict_week(inputs, season, week, lambdas).with_columns(pl.col("t") + signal)

    # Act & Assert
    with threadpool_limits(limits=1, user_api="blas"), pytest.raises(qb.IntegrityError):
        qb.check_week_information(leaky, inputs, 2007, 3, lambdas)


def _calibration_rows() -> pl.DataFrame:
    """Return prediction rows over three weeks; the last week's target would move any fit."""
    rows = [
        (2006, 2, 1.0, 0.0, 2.0),
        (2006, 2, 2.0, 1.0, 4.0),
        (2006, 3, 3.0, 0.0, 4.0),
        (2006, 3, 0.0, 2.0, 3.0),
        (2006, 4, 1.0, 1.0, 100.0),
    ]
    return pl.DataFrame(
        [
            {"season": s, "week": w, "game_id": f"g{i}", "team": "AAA", "p": p, "t": t, "target": y}
            for i, (s, w, p, t, y) in enumerate(rows)
        ]
    )


def test_calibrate_first_week_has_no_earlier_rows() -> None:
    # Act
    calibrated = qb.calibrate(_calibration_rows(), ("p", "t"), "target")

    # Assert
    assert calibrated.filter(pl.col("week") == 2).get_column("prediction").to_list() == [0.0, 0.0]


def test_calibrate_fits_each_week_on_earlier_rows_only() -> None:
    # Act
    calibrated = qb.calibrate(_calibration_rows(), ("p", "t"), "target")

    # Assert
    week_four = calibrated.filter(pl.col("week") == 4).row(0, named=True)
    assert week_four["prediction"] == pytest.approx(1.0 + 1.0 + 1.0)
    assert week_four["training_rows"] == 4


def _harness_calibrations() -> dict[str, pl.DataFrame]:
    """Return each candidate's calibration of the small league's predictions to a target."""
    with threadpool_limits(limits=1, user_api="blas"):
        result = qb.run_harness(_harness_inputs(), _SMALL_WINDOW)
    rows = result.predictions.with_columns((pl.col("p") * 0.5 + pl.col("t")).alias("target"))
    return qb.calibrated(rows, qb.CANDIDATES, ("p", "t"), "target")


def test_check_calibration_rows_passes_calibrate_output_of_the_harness() -> None:
    # Arrange
    calibrations = _harness_calibrations()

    # Act
    weeks = qb.check_calibration_rows(calibrations)

    # Assert
    assert weeks == len(qb.CANDIDATES) * 2 * 3


def test_check_calibration_rows_catches_a_fit_that_includes_its_own_week() -> None:
    # Arrange
    calibrations = _harness_calibrations()
    calibrations["B"] = calibrations["B"].with_columns(
        pl.col("training_rows") + pl.len().over("season", "week")
    )

    # Act & Assert
    with pytest.raises(qb.IntegrityError, match="check 8"):
        qb.check_calibration_rows(calibrations)


def _with_zero(inputs: Any) -> Any:
    """Return harness inputs with the zero-addition candidate, a copy of passes and sacks."""
    frames = dict(inputs.candidate_rows)
    frames[qb.ZERO] = dict(frames["A"])
    return qb.HarnessInputs(
        team_rows=inputs.team_rows,
        candidate_rows=frames,
        starters=inputs.starters,
        team_penalties=inputs.team_penalties,
        season_prior=inputs.season_prior,
    )


def test_check_zero_addition_passes_identical_rows_and_predictions() -> None:
    # Arrange
    inputs = _with_zero(_harness_inputs())
    result = qb.run_harness(inputs, _SMALL_WINDOW)

    # Act
    gap = qb.check_zero_addition(inputs.candidate_rows, result.predictions)

    # Assert
    assert gap == 0.0


def test_check_zero_addition_catches_rows_that_differ() -> None:
    # Arrange
    inputs = _with_zero(_harness_inputs())
    result = qb.run_harness(inputs, _SMALL_WINDOW)
    frames = dict(inputs.candidate_rows)
    frames[qb.ZERO] = {
        season: rows.with_columns(pl.col("qb_dropbacks") + 1)
        for season, rows in frames[qb.ZERO].items()
    }

    # Act & Assert
    with pytest.raises(qb.IntegrityError, match="check 6"):
        qb.check_zero_addition(frames, result.predictions)


def test_check_penalties_catches_a_penalty_not_from_the_previous_season() -> None:
    # Arrange
    inputs = _harness_inputs()
    result = qb.run_harness(inputs, _SMALL_WINDOW)
    result.audit.lambdas["B"][2007] = 123.0

    # Act & Assert
    with pytest.raises(qb.IntegrityError, match="check 7"):
        qb.check_penalties(inputs.candidate_rows, result.audit)


def test_check_penalties_passes_the_harness_penalties() -> None:
    # Arrange
    inputs = _harness_inputs()
    result = qb.run_harness(inputs, _SMALL_WINDOW)

    # Act
    checked = qb.check_penalties(inputs.candidate_rows, result.audit)

    # Assert
    assert checked == 3 * 2


def _season(**facts: float) -> Any:
    """Return a quarterback season holding only the per-season facts the checks read."""
    empty = pl.DataFrame()
    parts_facts = {"qb_games": 10.0, "scramble_mismatches": 0.0, "kneel_mismatches": 0.0}
    snap_facts = {"scrambles_not_snaps": 0.0, "target_gap": 0.0}
    for name, value in facts.items():
        (parts_facts if name in parts_facts else snap_facts)[name] = value
    return qb.QbSeason(
        qb_rows=empty,
        parts=empty,
        starters=empty,
        targets=empty,
        snap_facts=snap_facts,
        parts_facts=parts_facts,
        dispersion=empty,
    )


def test_check_parts_and_snaps_catches_a_play_part_mismatch() -> None:
    # Arrange
    inputs = qb.QbInputs(team_rows={}, seasons={2006: _season(kneel_mismatches=1.0)})

    # Act & Assert
    with pytest.raises(qb.IntegrityError, match="check 3"):
        qb.check_parts_and_snaps(inputs)


def test_check_parts_and_snaps_catches_a_target_that_does_not_recount() -> None:
    # Arrange
    inputs = qb.QbInputs(team_rows={}, seasons={2006: _season(target_gap=1e-6)})

    # Act & Assert
    with pytest.raises(qb.IntegrityError, match="check 4"):
        qb.check_parts_and_snaps(inputs)


def test_check_parts_and_snaps_sums_the_quarterback_games() -> None:
    # Arrange
    inputs = qb.QbInputs(team_rows={}, seasons={2006: _season(), 2007: _season()})

    # Act
    totals = qb.check_parts_and_snaps(inputs)

    # Assert
    assert totals["qb_games"] == 20.0
    assert totals["kneel_mismatches"] == 0.0


_SIM_TEAMS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")
_SIM_WINDOW = qb.Window(
    penalty_season=2005, warmup_season=2006, first_scored=2007, last_season=2010
)


def _sim_structure(weeks: int = 12) -> dict[int, Any]:
    """Return a synthetic league's structure: two quarterbacks a team, swapping every 3 weeks."""
    structure: dict[int, Any] = {}
    half = len(_SIM_TEAMS) // 2
    for season in range(2005, 2011):
        rotation = list(_SIM_TEAMS)
        teams: list[dict[str, object]] = []
        fits: list[dict[str, object]] = []
        for week in range(1, weeks + 1):
            pairs = zip(rotation[:half], reversed(rotation[half:]), strict=True)
            for home, away in pairs:
                game_id = f"{season}_{week:02d}_{away}_{home}"
                for side, opponent, is_home in ((home, away, True), (away, home, False)):
                    starter = f"{side}-{1 + ((week - 1) // 3) % 2}"
                    teams.append(
                        {
                            "game_id": game_id,
                            "week": week,
                            "team": side,
                            "opponent_team": opponent,
                            "is_home": is_home,
                            "offensive_snaps": 60,
                            "offensive_epa": 0.0,
                            "st_plays": 10,
                            "st_epa": 0.0,
                            "point_margin": 0,
                        }
                    )
                    fits.append(
                        {
                            "game_id": game_id,
                            "week": week,
                            "team": side,
                            "opponent_team": opponent,
                            "is_home": is_home,
                            "qb_id": starter,
                            "qb_name": starter,
                            "qb_attempts": 28,
                            "n_dropback": 30,
                            "n_scramble": 0,
                            "n_designed": 10,
                            "n_botched": 0,
                        }
                    )
            rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
        fit_rows = pl.DataFrame(fits)
        structure[season] = qb.SimSeason(
            team_rows=pl.DataFrame(teams),
            fit_rows=fit_rows,
            starters=fit_rows.select("game_id", "team", "qb_id"),
        )
    return structure


def _spreads(unit: str, designed: float = 1.0) -> Any:
    """Return spreads where quarterbacks differ, if at all, only in designed-run value."""
    return qb.Spreads(
        offense=0.02,
        defense=0.02,
        home_field=0.0,
        passer=0.0,
        scramble=0.0,
        designed=designed,
        play=0.3,
        means={"dropback": 0.05, "scramble": 0.4, "designed": 0.1, "botched": -2.0, "other": 0.0},
        designed_unit=unit,
    )


def _decision(unit: str, seed: int = 3) -> Any:
    """Return the rule's decision on one synthetic window."""
    with threadpool_limits(limits=1, user_api="blas"):
        return qb.simulate_window(
            _sim_structure(),
            _spreads(unit),
            _SIM_WINDOW,
            np.random.default_rng(seed),
            common.season_draws(len(_SIM_WINDOW.scored_seasons)),
        )


def test_synthetic_league_quarterbacks_differing_in_designed_runs_c_beats_a() -> None:
    # Act
    decision = _decision("passer")

    # Assert
    assert ("C", "A") in decision.wins


def test_synthetic_league_designed_runs_as_a_team_effect_c_does_not_beat_a() -> None:
    # Act
    decision = _decision("team")

    # Assert
    assert ("C", "A") not in decision.wins


def test_synthetic_league_designed_runs_as_pure_noise_nothing_beats_a() -> None:
    # Act
    decision = _decision("none")

    # Assert
    assert all(loser != "A" for _, loser in decision.wins)


def test_power_is_reproducible_with_its_seed() -> None:
    # Arrange
    structure, spreads = _sim_structure(), _spreads("passer")
    with threadpool_limits(limits=1, user_api="blas"):
        first = qb.power(structure, spreads, _SIM_WINDOW, windows=2, seed=1)

        # Act
        second = qb.power(structure, spreads, _SIM_WINDOW, windows=2, seed=1)

    # Assert
    assert first == second
    assert set(first) == {"B beats A", "C beats A", "C beats B"}


def _schedule_of(team_rows: pl.DataFrame) -> pl.DataFrame:
    """Return a schedule for synthetic team rows: home and away codes and scores."""
    home = team_rows.filter(pl.col("is_home"))
    return home.select(
        "game_id",
        pl.lit("REG").alias("game_type"),
        pl.col("team").alias("home_abbr"),
        pl.col("opponent_team").alias("away_abbr"),
        pl.lit(24.0).alias("home_score"),
        pl.lit(17.0).alias("away_score"),
    )


def _synthetic_inputs() -> Any:
    """Return quarterback inputs drawn from the synthetic league, one season per year."""
    structure, spreads = _sim_structure(), _spreads("passer")
    rng = np.random.default_rng(11)
    inputs = qb.QbInputs(team_rows={}, seasons={})
    for season, data in structure.items():
        rows, qb_rows, parts = qb.simulate_season(data, spreads, rng)
        inputs.team_rows[season] = rows
        inputs.seasons[season] = qb.QbSeason(
            qb_rows=qb_rows,
            parts=parts,
            starters=data.starters,
            targets=qb.season_targets(rows, _schedule_of(rows)),
            snap_facts={},
            parts_facts={},
            dispersion=pl.DataFrame(),
        )
    return inputs


def _fake_integrity() -> dict[str, Any]:
    """Return integrity results in the shape the report reads."""
    return {
        "reproduction": {"season_gap": 0.0, "week_gap": 0.0, "weeks_compared": 1},
        "parts_and_snaps": {"qb_games": 1.0, "target_gap": 0.0},
        "team_term": {"term_gap": 0.0, "history_gap": 0.0, "snapshots": 1.0},
        "zero_addition_gap": 0.0,
        "penalties": 1,
        "information": {
            "snapshots": 1,
            "gap": 0.0,
            "qb_fit_gap": 0.0,
            "calibration_weeks": 1234,
        },
        "rows": [
            {
                "season": 2007,
                "team_games": 1,
                "unknown_starters": 0,
                "listed_without_qb_row": 0,
                "first_snap_differs": 0,
                "qb_rows_without_dropback": 0,
                "runs_on_those_rows": 0,
            }
        ],
        "stability": (0.455, 601),
        "parts_fingerprint": "0" * 64,
    }


def test_integrity_lines_report_the_calibration_rows_check() -> None:
    # Act
    lines = qb._integrity_lines(_fake_integrity())

    # Assert
    check = next(line for line in lines if line.startswith("8. "))
    assert "1,234 candidate-weeks" in check
    assert "earlier weeks only" not in check


def test_evaluate_and_build_report_run_end_to_end_on_a_synthetic_league() -> None:
    # Arrange
    inputs = _synthetic_inputs()
    frames = qb.candidate_frames(inputs, (*qb.SCORED, qb.ZERO))
    history = qb.PriorHistory(lambda season: inputs.team_rows[season], first_season=2005)
    harness = qb.HarnessInputs(
        team_rows=inputs.team_rows,
        candidate_rows=frames,
        starters={season: data.starters for season, data in inputs.seasons.items()},
        team_penalties=lambda season: history.cross_validated_fit(season - 1),
        season_prior=lambda season: None,
    )
    provenance = {
        "version": "test",
        "code": {"qb_plays_test.py": "0" * 64},
        "inputs": 0,
        "fingerprint": "0" * 64,
        "command": "test",
        "seconds": 0.0,
        "git_commit": "0" * 40,
    }
    with threadpool_limits(limits=1, user_api="blas"):
        result = qb.run_harness(harness, _SIM_WINDOW)

        # Act
        results = qb.evaluate(inputs, frames, result, _fake_integrity(), _SIM_WINDOW)
    lines = qb.build_report(results, provenance, ["power"])

    # Assert
    assert results.decision.recommended in qb.CANDIDATES
    assert any(line.startswith("## Headline") for line in lines)
    assert all(len(line) <= 100 for line in lines if not line.startswith("|"))


def _error_rows(count: int = 400) -> pl.DataFrame:
    """Return scored rows over several seasons and weeks, in no particular order."""
    return pl.DataFrame(
        {
            "season": [2007 + (index % 7) for index in range(count)][::-1],
            "week": [5 + (index % 11) for index in range(count)],
            "game_id": [f"g{index}" for index in range(count)],
            "team": ["AAA"] * count,
            "error": [float(index % 13) - 6.0 for index in range(count)],
        }
    )


def test_paired_errors_do_not_depend_on_the_row_order_of_either_side() -> None:
    # Arrange
    rows = _error_rows()
    first = qb.paired_errors(rows, rows.sample(fraction=1.0, shuffle=True, seed=1))

    # Act
    second = qb.paired_errors(rows.sample(fraction=1.0, shuffle=True, seed=3), rows.reverse())

    # Assert
    assert first.equals(second)
    assert first.select("season", "week").rows() == sorted(first.select("season", "week").rows())
