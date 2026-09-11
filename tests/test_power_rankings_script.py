"""Tests for the power rankings script helpers."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.ml import ml_model_core
from scripts import power_rankings


def test_load_current_records_filters_to_reg_only(tmp_path) -> None:
    """Records should exclude postseason games when computing current standings."""
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 5,
                "game_type": "REG",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 10,
                "home_score": 20,
            },
            {
                "season": 2024,
                "week": 5,
                "game_type": "WC",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 30,
                "home_score": 10,
            },
        ]
    ).to_csv(schedule_path, index=False)

    records = power_rankings._load_current_records(
        schedule_path, season=2024, through_week=5, include_postseason=False
    )
    records = records.set_index("team_abbr")

    assert records.loc["AAA", "wins"] == 0
    assert records.loc["AAA", "losses"] == 1
    assert records.loc["BBB", "wins"] == 1
    assert records.loc["BBB", "losses"] == 0


def test_load_current_records_can_include_postseason(tmp_path) -> None:
    """Records should include postseason games when requested."""
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 5,
                "game_type": "REG",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 10,
                "home_score": 20,
            },
            {
                "season": 2024,
                "week": 19,
                "game_type": "DIV",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 30,
                "home_score": 10,
            },
        ]
    ).to_csv(schedule_path, index=False)

    records = power_rankings._load_current_records(
        schedule_path, season=2024, through_week=19, include_postseason=True
    )
    records = records.set_index("team_abbr")

    assert records.loc["AAA", "wins"] == 1
    assert records.loc["AAA", "losses"] == 1
    assert records.loc["BBB", "wins"] == 1
    assert records.loc["BBB", "losses"] == 1


def test_predict_future_games_requires_feature_columns(tmp_path) -> None:
    """Missing feature columns should raise a clear validation error."""
    data_path = tmp_path / "ml.csv"
    pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 2,
                "game_type": "REG",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
            }
        ]
    ).to_csv(data_path, index=False)

    model = SimpleNamespace(feature_spec=SimpleNamespace(feature_columns=["feat_required"]))

    with pytest.raises(ValueError, match="Missing required feature columns \\(1\\): feat_required"):
        power_rankings._predict_future_games(
            model,
            model_kind="margin_total",
            data_ml=data_path,
            season=2025,
            through_week=1,
        )


def test_score_model_calibration_applied_for_win_prob(tmp_path, monkeypatch) -> None:
    """ScoreModel paths should apply a calibrator when present."""
    data_path = tmp_path / "ml.csv"
    pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 2,
                "game_type": "REG",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "feat1": 1.0,
            }
        ]
    ).to_csv(data_path, index=False)

    class DummyPreprocessor:
        """Minimal preprocessor stub for score model predictions."""

        def transform(self, _df: pd.DataFrame) -> np.ndarray:
            """Return a deterministic feature matrix."""
            return np.zeros((len(_df), 1), dtype=float)

    class DummyModel:
        """Labelled dummy model for predict_xgb."""

        def __init__(self, name: str) -> None:
            self.name = name

    class DummyCalibrator:
        """Predictor stub returning fixed probabilities."""

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            """Return a fixed 0.9 home win probability."""
            return np.tile(np.array([0.1, 0.9], dtype=float), (len(x), 1))

    def fake_predict_xgb(model: DummyModel, x: np.ndarray) -> np.ndarray:
        """Return deterministic away/home scores."""
        if model.name == "away":
            return np.full(len(x), 10.0)
        return np.full(len(x), 20.0)

    monkeypatch.setattr(ml_model_core, "predict_xgb", fake_predict_xgb)

    spec = SimpleNamespace(feature_columns=["feat1"])
    calibrator = ml_model_core.WinProbCalibrator(method="platt", model=DummyCalibrator())
    model = SimpleNamespace(
        feature_spec=spec,
        preprocessor=DummyPreprocessor(),
        away_model=DummyModel("away"),
        home_model=DummyModel("home"),
        market_prob_config=None,
        calibrator=calibrator,
    )

    out = power_rankings._predict_future_games(
        model,
        model_kind="score",
        data_ml=data_path,
        season=2025,
        through_week=1,
    )
    assert np.allclose(out["home_win_prob"].to_numpy(dtype=float), 0.9)


def test_build_games_for_ratings_logs_diagnostics(tmp_path, caplog) -> None:
    """Ratings diagnostics should log once per invocation."""
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "game_type": "REG",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 10,
                "home_score": 20,
            }
        ]
    ).to_csv(schedule_path, index=False)

    future_games = pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 2,
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "home_win_prob": 0.6,
            }
        ]
    )

    caplog.set_level(logging.INFO)
    power_rankings._build_games_for_ratings(
        schedule_path=schedule_path,
        season=2024,
        through_week=1,
        ratings_min_season=None,
        future_games_with_probs=future_games,
        include_postseason=False,
    )

    messages = [record.message for record in caplog.records]
    diag = [msg for msg in messages if msg.startswith("Ratings fit diagnostics:")]
    assert len(diag) == 1


def test_build_games_for_ratings_includes_postseason(tmp_path) -> None:
    """Ratings fit should include postseason games when requested."""
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "game_type": "REG",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 10,
                "home_score": 20,
            },
            {
                "season": 2024,
                "week": 19,
                "game_type": "CON",
                "away_abbr": "AAA",
                "home_abbr": "BBB",
                "away_score": 30,
                "home_score": 10,
            },
        ]
    ).to_csv(schedule_path, index=False)

    games = power_rankings._build_games_for_ratings(
        schedule_path=schedule_path,
        season=2024,
        through_week=19,
        ratings_min_season=None,
        future_games_with_probs=pd.DataFrame(),
        include_postseason=True,
    )

    assert len(games) == 2
    assert set(games["week"]) == {1, 19}


def test_legacy_franchise_fit_restores_the_all_seasons_equal_weight_fit() -> None:
    """The legacy flag reproduces the historical fit: all seasons, equal weight, binary."""
    pr = power_rankings

    schedule = pl.DataFrame(
        {
            "season": [2022, 2022, 2023, 2023],
            "week": [1, 2, 1, 2],
            "game_type": ["REG"] * 4,
            "away_abbr": ["AAA", "BBB", "AAA", "BBB"],
            "home_abbr": ["BBB", "AAA", "BBB", "AAA"],
            "away_score": [30.0, 10.0, 10.0, 30.0],
            "home_score": [10.0, 30.0, 30.0, 10.0],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "schedule.csv"
        schedule.write_csv(path)

        legacy = pr._build_games_for_ratings(
            schedule_path=path,
            season=2023,
            through_week=18,
            ratings_min_season=None,
            future_games_with_probs=pd.DataFrame(),
            include_postseason=False,
            window_seasons=0,
            prior_season_weight=1.0,
            target="binary",
            include_future=True,
        )

    # Every season is in the fit and every game weighs the same.
    assert sorted(legacy["season"].unique().tolist()) == [2022, 2023]
    assert legacy["fit_weight"].nunique() == 1
    assert float(legacy["fit_weight"].iloc[0]) == pytest.approx(1.0)
    # Binary targets take only two values regardless of margin.
    assert legacy["p_home"].nunique() <= 2


def test_default_ratings_fit_windows_seasons_and_downweights_the_past() -> None:
    """The default fit sees a short window and weighs earlier seasons down."""
    pr = power_rankings

    schedule = pl.DataFrame(
        {
            "season": [2020, 2022, 2023],
            "week": [1, 1, 1],
            "game_type": ["REG"] * 3,
            "away_abbr": ["AAA", "AAA", "AAA"],
            "home_abbr": ["BBB", "BBB", "BBB"],
            "away_score": [10.0, 10.0, 10.0],
            "home_score": [30.0, 24.0, 13.0],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "schedule.csv"
        schedule.write_csv(path)

        games = pr._build_games_for_ratings(
            schedule_path=path,
            season=2023,
            through_week=18,
            ratings_min_season=None,
            future_games_with_probs=pd.DataFrame(),
            include_postseason=False,
        )

    # 2020 falls outside the default two-season window.
    assert sorted(games["season"].unique().tolist()) == [2022, 2023]
    weights = dict(zip(games["season"], games["fit_weight"], strict=True))
    assert weights[2023] == pytest.approx(1.0)
    assert weights[2022] == pytest.approx(pr.DEFAULT_PRIOR_SEASON_WEIGHT)
    # Margin targets distinguish the 20-point win from the 3-point win.
    assert games["p_home"].nunique() == 2


def test_future_games_stay_out_of_the_strength_fit_by_default() -> None:
    """The strength fit describes results; the model's own forecasts are excluded."""
    pr = power_rankings

    schedule = pl.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "game_type": ["REG"],
            "away_abbr": ["AAA"],
            "home_abbr": ["BBB"],
            "away_score": [10.0],
            "home_score": [30.0],
        }
    )
    future = pd.DataFrame(
        {
            "season": [2023],
            "week": [2],
            "away_abbr": ["BBB"],
            "home_abbr": ["AAA"],
            "home_win_prob": [0.9],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "schedule.csv"
        schedule.write_csv(path)
        default = pr._build_games_for_ratings(
            schedule_path=path,
            season=2023,
            through_week=1,
            ratings_min_season=None,
            future_games_with_probs=future,
            include_postseason=False,
        )
        with_future = pr._build_games_for_ratings(
            schedule_path=path,
            season=2023,
            through_week=1,
            ratings_min_season=None,
            future_games_with_probs=future,
            include_postseason=False,
            include_future=True,
        )

    assert default["week"].tolist() == [1]
    assert sorted(with_future["week"].tolist()) == [1, 2]


def test_load_current_records_compares_scores_as_numbers(tmp_path: Path) -> None:
    """A file that opens with unplayed games must not turn scores into text.

    The ETL writes newest games first, so the rows Polars samples to infer types can all
    have blank scores; compared as text, a 9-31 loss would read as a win.
    """
    unplayed = [
        {
            "season": 2025,
            "week": index % 18 + 1,
            "game_type": "REG",
            "away_abbr": "CCC",
            "home_abbr": "DDD",
            "away_score": None,
            "home_score": None,
        }
        for index in range(150)
    ]
    played = [
        {
            "season": 2024,
            "week": 5,
            "game_type": "REG",
            "away_abbr": "AAA",
            "home_abbr": "BBB",
            "away_score": 9,
            "home_score": 31,
        }
    ]
    schedule_path = tmp_path / "schedule.csv"
    pd.DataFrame([*unplayed, *played]).to_csv(schedule_path, index=False)

    records = power_rankings._load_current_records(
        schedule_path, season=2024, through_week=5
    ).set_index("team_abbr")

    assert records.loc["BBB", "wins"] == 1
    assert records.loc["AAA", "losses"] == 1
    assert records.loc["AAA", "wins"] == 0


# Two seasons of a four-team league; week 3 of 2024 is still to be played.
_FIXTURE_GAMES = (
    (2023, 1, "AAA", "BBB", 10, 27),
    (2023, 1, "CCC", "DDD", 24, 17),
    (2023, 2, "BBB", "CCC", 20, 23),
    (2023, 2, "DDD", "AAA", 13, 30),
    (2024, 1, "AAA", "CCC", 21, 20),
    (2024, 1, "BBB", "DDD", 35, 14),
    (2024, 2, "CCC", "BBB", 17, 31),
    (2024, 2, "DDD", "AAA", 28, 24),
    (2024, 3, "AAA", "BBB", None, None),
    (2024, 3, "CCC", "DDD", None, None),
)


def _write_fixture_schedule(path: Path) -> Path:
    """Write the fixture league as a schedule/results CSV."""
    frame = pd.DataFrame(
        _FIXTURE_GAMES,
        columns=["season", "week", "away_abbr", "home_abbr", "away_score", "home_score"],
    )
    frame.insert(2, "game_type", "REG")
    frame.to_csv(path, index=False)
    return path


def _write_snapshots(path: Path, weeks: dict[int, dict[str, float]]) -> Path:
    """Write a strength snapshot file with one row per team per 2024 week."""
    rows: list[dict[str, object]] = []
    for week, composites in weeks.items():
        for team, value in composites.items():
            row: dict[str, object] = {"season": 2024, "week": week, "team_abbr": team}
            for column in constants.STRENGTH_SNAPSHOT_FILE_COLUMNS[3:]:
                row[column] = value / 10.0
            row["adj_strength_composite"] = value
            row["adj_srs"] = 7.0 * value
            rows.append(row)
    pl.DataFrame(rows).write_csv(path)
    return path


def _no_future_games(*_args: object, **_kwargs: object) -> pd.DataFrame:
    """Stand in for model predictions: nothing left to project."""
    return pd.DataFrame(columns=["season", "week", "away_abbr", "home_abbr", "home_win_prob"])


def _rank_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: power_rankings.RankingOptions,
    *,
    through_week: int = 2,
) -> pd.DataFrame:
    """Rank the fixture league through a week with the given options."""
    monkeypatch.setattr(power_rankings, "_predict_future_games", _no_future_games)
    schedule = _write_fixture_schedule(tmp_path / "schedule.csv")
    result = power_rankings.compute_power_rankings(
        None,
        model_kind="margin_total",
        data_ml=schedule,
        data_schedule=schedule,
        season=2024,
        through_week=through_week,
        include_postseason=False,
        options=options,
    )
    return result.power_rankings


def test_bradley_terry_method_reproduces_the_current_season_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinned from the ranking the script produced before the composite became the default."""
    options = power_rankings.resolve_ranking_options(
        method="bradley_terry", legacy_franchise_fit=False
    )

    rankings = _rank_fixture(tmp_path, monkeypatch, options)

    assert rankings.columns.tolist() == [
        "team_abbr",
        "division",
        "conference",
        "rating_raw",
        "power_rating_1_10",
        "power_rating_0_10",
        "wins",
        "losses",
        "ties",
        "games_played",
        "home_advantage_logit",
        "home_advantage_prob",
        "season",
        "through_week",
        "rank",
    ]
    assert rankings["team_abbr"].tolist() == ["BBB", "CCC", "AAA", "DDD"]
    assert rankings["rating_raw"].tolist() == pytest.approx(
        [1.0734312909839037, -0.16356451172377265, -0.28861933211784196, -0.6212474471422891],
        abs=1e-12,
    )
    assert rankings["power_rating_1_10"].tolist() == [7.71, 5.13, 4.86, 4.15]
    assert rankings["power_rating_0_10"].tolist() == [7.45, 4.59, 4.28, 3.49]
    assert float(rankings["home_advantage_logit"].iloc[0]) == pytest.approx(
        -0.10168353446110899, abs=1e-12
    )


def test_composite_method_ranks_on_the_snapshot_for_the_next_week(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through week 2 means going into week 3, so week 3's snapshot decides the order."""
    snapshots = _write_snapshots(
        tmp_path / "snapshots.csv",
        {
            2: {"AAA": 2.0, "BBB": 1.0, "CCC": 0.0, "DDD": -1.0},
            3: {"AAA": -0.5, "BBB": 0.5, "CCC": 1.5, "DDD": -1.5},
            4: {"AAA": 3.0, "BBB": -3.0, "CCC": -2.0, "DDD": 2.0},
        },
    )
    options = power_rankings.resolve_ranking_options(
        method=None, legacy_franchise_fit=False, strength_snapshots=snapshots
    )

    rankings = _rank_fixture(tmp_path, monkeypatch, options)

    assert rankings["team_abbr"].tolist() == ["CCC", "BBB", "AAA", "DDD"]
    assert rankings["snapshot_week"].unique().tolist() == [3]
    # Records still come from the results through week 2.
    assert int(rankings.loc[rankings["team_abbr"] == "BBB", "wins"].iloc[0]) == 2


def test_later_snapshot_weeks_never_reach_the_ranking(tmp_path: Path) -> None:
    """Rewriting every week after the one being ranked changes nothing."""
    base = {3: {"AAA": -0.5, "BBB": 0.5}}
    later = {**base, 4: {"AAA": 50.0, "BBB": -50.0}, 5: {"AAA": 9.0, "BBB": 9.0}}

    without_later = power_rankings.load_strength_snapshot(
        _write_snapshots(tmp_path / "base.csv", base), season=2024, through_week=2
    )
    with_later = power_rankings.load_strength_snapshot(
        _write_snapshots(tmp_path / "later.csv", later), season=2024, through_week=2
    )

    pd.testing.assert_frame_equal(without_later, with_later)
    assert with_later["week"].unique().tolist() == [3]


def test_a_missing_snapshot_week_is_a_clear_error(tmp_path: Path) -> None:
    """Ranking a week the ETL never solved fails loudly and names the alternatives."""
    path = _write_snapshots(tmp_path / "snapshots.csv", {3: {"AAA": 0.5, "BBB": -0.5}})

    with pytest.raises(ValueError, match="bradley_terry"):
        power_rankings.load_strength_snapshot(path, season=2024, through_week=9)


def test_a_duplicated_team_in_a_snapshot_week_is_rejected(tmp_path: Path) -> None:
    """Two rows for one team in one week would make the rank ambiguous."""
    path = tmp_path / "snapshots.csv"
    rows = pl.read_csv(_write_snapshots(path, {3: {"AAA": 0.5, "BBB": -0.5}}))
    pl.concat([rows, rows.head(1)]).write_csv(path)

    with pytest.raises(ValueError, match="duplicate"):
        power_rankings.load_strength_snapshot(path, season=2024, through_week=2)


def test_ranking_options_default_to_the_composite() -> None:
    """With no method named, rankings read the adjusted composite."""
    options = power_rankings.resolve_ranking_options(method=None, legacy_franchise_fit=False)

    assert options.method == "composite"
    assert options.strength_snapshots == power_rankings.DEFAULT_STRENGTH_SNAPSHOTS


def test_legacy_franchise_fit_selects_the_old_bradley_terry_settings() -> None:
    """The legacy flag implies Bradley-Terry with every historical setting restored."""
    options = power_rankings.resolve_ranking_options(
        method=None,
        legacy_franchise_fit=True,
        window_seasons=5,
        prior_season_weight=0.1,
        target="margin",
        include_future=False,
        ratings_min_season=2010,
    )

    assert options.method == "bradley_terry"
    assert options.window_seasons == 0
    assert options.prior_season_weight == pytest.approx(1.0)
    assert options.target == "binary"
    assert options.include_future is True
    assert options.ratings_min_season == 2010


def test_legacy_franchise_fit_cannot_be_combined_with_the_composite() -> None:
    """Asking for both is contradictory, so it is refused rather than guessed."""
    with pytest.raises(ValueError, match="legacy"):
        power_rankings.resolve_ranking_options(method="composite", legacy_franchise_fit=True)


def test_an_unknown_ranking_method_is_rejected() -> None:
    """Only the documented methods are accepted."""
    with pytest.raises(ValueError, match="method"):
        power_rankings.resolve_ranking_options(method="elo", legacy_franchise_fit=False)


def test_cli_exposes_the_method_and_snapshot_path() -> None:
    """The script parses the ranking method and the snapshot file location."""
    args = power_rankings._parse_args(
        [
            "--model-in",
            "model.joblib",
            "--season",
            "2024",
            "--through-week",
            "3",
            "--method",
            "bradley_terry",
            "--strength-snapshots",
            "snapshots.csv",
        ]
    )
    defaults = power_rankings._parse_args(
        ["--model-in", "model.joblib", "--season", "2024", "--through-week", "3"]
    )

    assert args.method == "bradley_terry"
    assert args.strength_snapshots == Path("snapshots.csv")
    assert defaults.method is None
    assert defaults.strength_snapshots == power_rankings.DEFAULT_STRENGTH_SNAPSHOTS
