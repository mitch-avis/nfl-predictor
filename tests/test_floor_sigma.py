"""Tests for the spread of the probability floor, estimated from earlier out-of-fold errors."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from nfl_predictor import constants
from nfl_predictor.ml import floor_sigma


def _errors(rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    """Return an error pool with one game per ``(season, week, error)`` row."""
    return pd.DataFrame(
        {
            "game_id": [
                f"{season}_{week:02d}_{index}" for index, (season, week, _) in enumerate(rows)
            ],
            "season": [season for season, _, _ in rows],
            "week": [week for _, week, _ in rows],
            "squared_error": [error**2 for _, _, error in rows],
        }
    )


def _four_seasons() -> pd.DataFrame:
    """Two games a week, weeks 1-3, in 2019-2022, with errors that differ by week."""
    rows = [
        (season, week, float(season - 2010 + week + game))
        for season in (2019, 2020, 2021, 2022)
        for week in (1, 2, 3)
        for game in (0, 1)
    ]
    return _errors(rows)


def test_sigma_is_the_root_mean_square_error_strictly_before_the_week() -> None:
    """Earlier seasons and the season's earlier weeks count; the week itself and later do not."""
    pool = _four_seasons()
    before = (pool["season"] < 2022) | ((pool["season"] == 2022) & (pool["week"] < 3))

    estimate = floor_sigma.estimate(pool, 2022, 3)

    assert estimate.sigma == np.sqrt(pool.loc[before, "squared_error"].mean())
    assert estimate.fallback is False
    assert estimate.pool_games == int(before.sum())
    assert estimate.pool_seasons == (2019, 2020, 2021, 2022)
    assert (estimate.season, estimate.week) == (2022, 3)


def test_later_errors_never_reach_an_earlier_week() -> None:
    """Changing the errors of the week or anything after it leaves the week's sigma alone."""
    pool = _four_seasons()
    changed = pool.copy()
    later = (changed["season"] == 2022) & (changed["week"] >= 2)
    changed.loc[later, "squared_error"] = 1e6

    assert floor_sigma.estimate(changed, 2022, 2) == floor_sigma.estimate(pool, 2022, 2)


def test_fewer_than_the_minimum_earlier_seasons_falls_back_to_the_constant() -> None:
    """Two earlier seasons plus the current one's weeks are not enough: the constant is used."""
    pool = _four_seasons()
    pool = pool[pool["season"] >= 2020]

    estimate = floor_sigma.estimate(pool, 2022, 3)

    assert constants.FLOOR_SIGMA_MIN_POOL_SEASONS == 3
    assert estimate.sigma == constants.SCORE_DIFF_STD_DEV
    assert estimate.fallback is True
    assert estimate.pool_games == 16
    assert estimate.pool_seasons == (2020, 2021, 2022)


def test_the_minimum_counts_complete_earlier_seasons_only() -> None:
    """At a new season's week 1 the three seasons before it are enough."""
    pool = _four_seasons()

    assert floor_sigma.estimate(pool, 2022, 1).fallback is False
    assert floor_sigma.estimate(pool, 2021, 3).fallback is True


def test_an_empty_pool_falls_back_to_the_constant() -> None:
    """With no earlier errors at all the sigma is the constant, marked as the fallback."""
    estimate = floor_sigma.estimate(floor_sigma.empty_errors(), 2024, 1)

    assert estimate.sigma == constants.SCORE_DIFF_STD_DEV
    assert estimate.fallback is True
    assert estimate.pool_games == 0
    assert estimate.pool_seasons == ()


def test_margin_errors_square_actual_minus_predicted_and_skip_missing() -> None:
    """Each game's squared error is (actual - predicted margin) squared; unplayed games drop."""
    predictions = pd.DataFrame(
        {
            "game_id": ["a", "b", "c"],
            "season": [2020, 2020, 2020],
            "week": [1, 1, 2],
            "actual_margin": [7.0, -3.0, np.nan],
            "predicted_margin": [3.0, 2.0, 1.0],
        }
    )

    errors = floor_sigma.margin_errors(predictions)

    assert list(errors.columns) == list(floor_sigma.ERROR_COLUMNS)
    assert errors["game_id"].tolist() == ["a", "b"]
    assert errors["squared_error"].tolist() == [16.0, 25.0]


def test_runs_are_averaged_per_game_so_each_game_counts_once() -> None:
    """Two seeds of one game average their squared errors; a game in one run keeps its own."""
    seed_a = _errors([(2020, 1, 2.0), (2020, 1, 4.0)])
    seed_b = _errors([(2020, 1, 6.0)])

    combined = floor_sigma.average_over_runs([seed_a, seed_b])

    assert combined.set_index("game_id")["squared_error"].to_dict() == {
        "2020_01_0": (4.0 + 36.0) / 2,
        "2020_01_1": 16.0,
    }


def test_averaging_equals_pooling_when_the_runs_cover_the_same_games() -> None:
    """Over identical games, per-game averaging and pooling every row give one sigma."""
    seed_a = _four_seasons()
    seed_b = seed_a.assign(squared_error=seed_a["squared_error"] * 1.5)

    averaged = floor_sigma.estimate(floor_sigma.average_over_runs([seed_a, seed_b]), 2022, 3)
    pooled = floor_sigma.estimate(pd.concat([seed_a, seed_b], ignore_index=True), 2022, 3)

    assert averaged.sigma == pytest.approx(pooled.sigma, rel=1e-12)


def test_a_runs_own_weeks_replace_the_history_for_those_weeks() -> None:
    """History rows at a week the run predicts itself are dropped; other weeks are kept."""
    history = _errors([(2020, 1, 1.0), (2020, 2, 2.0)])
    own = _errors([(2020, 2, 5.0)]).assign(game_id="own")

    combined = floor_sigma.combine(history, own)

    assert sorted(combined["squared_error"].tolist()) == [1.0, 25.0]


def test_the_floor_is_the_normal_curve_at_the_given_spread() -> None:
    """Margins map through Phi(margin / sigma); a larger sigma pulls toward a coin flip."""
    margin = np.array([-7.0, 0.0, 7.0])

    narrow = floor_sigma.home_win_prob(margin, 10.0)
    wide = floor_sigma.home_win_prob(margin, 20.0)

    assert narrow[1] == 0.5
    assert narrow[2] > wide[2] > 0.5
    with pytest.raises(ValueError, match="positive"):
        floor_sigma.home_win_prob(margin, 0.0)


def test_the_record_round_trips_through_json() -> None:
    """A sigma record written to metadata reads back as the same record."""
    estimate = floor_sigma.estimate(_four_seasons(), 2022, 3, sources=("models/ref",))

    payload = json.loads(json.dumps(estimate.to_dict()))

    assert payload["sigma"] == estimate.sigma
    assert payload["min_pool_seasons"] == constants.FLOOR_SIGMA_MIN_POOL_SEASONS
    assert floor_sigma.FloorSigma.from_dict(payload) == estimate


def _write_reference_run(root: Path, name: str, predictions: pd.DataFrame) -> Path:
    """Write a walk-forward run directory whose metadata points at its fold checkpoints."""
    checkpoint_dir = root / "checkpoints" / name
    checkpoint_dir.mkdir(parents=True)
    for (season, week), fold in predictions.groupby(["season", "week"]):
        joblib.dump(
            {"predictions": fold.reset_index(drop=True)},
            checkpoint_dir / f"fold_{season}_w{week:02d}.joblib",
        )
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"config": {"checkpoint": {"dir": str(checkpoint_dir)}}}), encoding="utf-8"
    )
    return run_dir


def _reference_predictions(offset: float) -> pd.DataFrame:
    """Return fold predictions for two games a week in 2020, weeks 1-2."""
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "season": [2020, 2020, 2020, 2020],
            "week": [1, 1, 2, 2],
            "actual_margin": [3.0, -7.0, 10.0, 0.0],
            "predicted_margin": [1.0 + offset, -1.0, 4.0, 2.0],
        }
    )


def test_reference_runs_load_from_their_fold_checkpoints(tmp_path: Path) -> None:
    """Every reference run's fold predictions are read and the runs averaged per game."""
    seed_a = _write_reference_run(tmp_path, "seed_a", _reference_predictions(0.0))
    seed_b = _write_reference_run(tmp_path, "seed_b", _reference_predictions(2.0))

    pool = floor_sigma.load_reference_pool([seed_a, seed_b])

    assert pool.sources == (str(seed_a), str(seed_b))
    errors = pool.errors.set_index("game_id")["squared_error"]
    assert errors["g1"] == (4.0 + 0.0) / 2
    assert errors["g2"] == 36.0
    assert len(pool.errors) == 4


def test_a_reference_checkpoint_directory_can_be_named_directly(tmp_path: Path) -> None:
    """A directory holding fold checkpoints is read without a metadata file."""
    run_dir = _write_reference_run(tmp_path, "seed_a", _reference_predictions(0.0))
    checkpoint_dir = tmp_path / "checkpoints" / "seed_a"

    assert floor_sigma.load_reference_pool([checkpoint_dir]).errors.equals(
        floor_sigma.load_reference_pool([run_dir]).errors
    )


def test_relative_reference_paths_resolve_against_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative path in the config means the same run from any working directory."""
    _write_reference_run(tmp_path, "seed_a", _reference_predictions(0.0))
    monkeypatch.setattr(floor_sigma.constants, "ROOT_DIR", tmp_path)

    pool = floor_sigma.load_reference_pool([Path("seed_a")])

    assert pool.sources == (str(tmp_path / "seed_a"),)


def test_a_missing_reference_run_is_an_error_naming_it(tmp_path: Path) -> None:
    """A configured run that is not on disk stops the run instead of changing the sigma."""
    missing = tmp_path / "no_such_run"

    with pytest.raises(FileNotFoundError, match="no_such_run"):
        floor_sigma.load_reference_pool([missing])


def test_no_reference_runs_is_an_empty_pool() -> None:
    """An empty list is the explicit way to run without reference errors."""
    pool = floor_sigma.load_reference_pool([])

    assert pool.errors.empty
    assert pool.sources == ()


def test_the_pool_digest_follows_its_contents(tmp_path: Path) -> None:
    """Equal pools share a digest; a changed error changes it."""
    pool = floor_sigma.ErrorPool(_four_seasons(), ("a",))
    same = floor_sigma.ErrorPool(_four_seasons(), ("a",))
    changed_errors = _four_seasons()
    changed_errors.loc[0, "squared_error"] = changed_errors["squared_error"].iloc[0] + 1.0

    assert pool.digest() == same.digest()
    assert pool.digest() != floor_sigma.ErrorPool(changed_errors, ("a",)).digest()
