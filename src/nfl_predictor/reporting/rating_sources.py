"""Team ratings in points that a power ranking can rank on, each known before the week's games.

A rating source gives every team one number per week, in points against an average team on a
neutral field, from information available before that week's first game:

- **ported**: the ported team rating (:mod:`nfl_predictor.ratings`) as the ETL publishes it. A
  team that has played rates as the published week-by-week history of the week before, which
  equals the pre-week entry point (:meth:`nfl_predictor.ratings.pre_week.RatingsHistory.pre_week`)
  bit for bit; the game rows' ``ratings`` feature group carries the same pre-week values, the
  preseason prior of week 1 included.
- **model**: a ridge fit of team strengths plus a home term to the trained model's pre-game
  predicted margins for the season's games before the week (:func:`fit_model_ratings`).
- **blend**: ``w * ported + (1 - w) * model`` (:func:`blend_ratings`).

The adjusted-strength composite, the fourth source, is expressed in points by
``power_rankings.composite_points_vs_average``.

Rating frames are long: one row per ``season``, ``week`` and ``team`` with the ``rating``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pickle import UnpicklingError
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd
import polars as pl

from nfl_predictor import constants
from nfl_predictor.reporting import run_comparison

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

# One row per season, week and team, the rating in points against an average team.
RATING_SCHEMA = pl.Schema(
    {"season": pl.Int64(), "week": pl.Int64(), "team": pl.String(), "rating": pl.Float64()}
)

# The ported team rating's column in the published ratings tables, and the name of its pre-week
# value in the game rows' ``ratings`` feature group (``away_`` / ``home_`` in front).
PORTED_RATING_COLUMN = "team_rating"
PORTED_FEATURE = "ratings_team"

# The model rating's penalty on the team strengths: small enough not to shrink them, because the
# targets are already the model's regressed predictions, and large enough to pin a solution when
# the games so far do not identify every strength (one game a team in week 2). The home term is
# not penalized.
MODEL_RATING_RIDGE_LAMBDA = 1e-6

# The first week with an earlier game of its season to fit the model rating on.
FIRST_FITTED_WEEK = 2

_GAME_COLUMNS = ("away_abbr", "home_abbr", "predicted_margin")

# What reading walk-forward fold checkpoints can raise for a missing, foreign, half-written or
# damaged source: the set the walk-forward's own checkpoint loader treats as unreadable, plus the
# ValueError and TypeError this module's reader raises for a source it refuses.
UNREADABLE_PREDICTION_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    EOFError,
    ValueError,
    TypeError,
    AttributeError,
    ImportError,
    KeyError,
    IndexError,
    UnpicklingError,
)

# What the model rating reads from a walk-forward's fold predictions.
PREDICTION_COLUMNS = (
    "game_id",
    "season",
    "week",
    "away_abbr",
    "home_abbr",
    "neutral",
    "predicted_margin",
)
_PREDICTION_SCHEMA = pl.Schema(
    {
        "game_id": pl.String(),
        "season": pl.Int64(),
        "week": pl.Int64(),
        "away_abbr": pl.String(),
        "home_abbr": pl.String(),
        "neutral": pl.Float64(),
        "predicted_margin": pl.Float64(),
    }
)


@dataclass(frozen=True, slots=True)
class ModelRatingFit:
    """Team strengths and the home term fit to predicted margins.

    Attributes:
        ratings: Each team's strength in points against an average team on a neutral field; they
            sum to zero over the teams fit, and a team without a game rates zero.
        home_term: The home side's edge in points at a non-neutral site.
        games: How many games the fit read.

    """

    ratings: dict[str, float]
    home_term: float
    games: int


def fit_model_ratings(
    games: pl.DataFrame,
    teams: Sequence[str],
    *,
    ridge_lambda: float = MODEL_RATING_RIDGE_LAMBDA,
) -> ModelRatingFit:
    """Fit ``predicted_margin = rating[home] - rating[away] + home_term * (1 - neutral)``.

    A ridge regression with the penalty on the team strengths only, so among the strengths that
    fit equally well it returns the smallest, which sum to zero.

    Args:
        games: One row per game with ``away_abbr``, ``home_abbr``, ``predicted_margin`` (home
            minus away, in points) and ``neutral`` (1 at a neutral site; a null or missing flag is
            not neutral).
        teams: The teams to rate; a team without a game rates zero, and a team in ``games`` but
            not here is rated too.
        ridge_lambda: The penalty on the team strengths.

    Returns:
        The fit.

    Raises:
        ValueError: If ``games`` is empty or a team or margin is null.

    """
    if games.is_empty():
        msg = "fit_model_ratings received no games"
        raise ValueError(msg)
    if any(games.get_column(column).null_count() for column in _GAME_COLUMNS):
        msg = "fit_model_ratings needs a team on both sides and a margin for every game"
        raise ValueError(msg)
    home = games.get_column("home_abbr").cast(pl.String).to_list()
    away = games.get_column("away_abbr").cast(pl.String).to_list()
    universe = sorted(set(teams) | set(home) | set(away))
    position = {team: index for index, team in enumerate(universe)}
    neutral = (
        games.get_column("neutral").cast(pl.Float64).fill_null(0.0).to_numpy()
        if "neutral" in games.columns
        else np.zeros(games.height)
    )

    design = np.zeros((games.height, len(universe) + 1), dtype=np.float64)
    rows = np.arange(games.height)
    design[rows, [position[team] for team in home]] = 1.0
    design[rows, [position[team] for team in away]] = -1.0
    design[:, -1] = 1.0 - neutral
    response = games.get_column("predicted_margin").cast(pl.Float64).to_numpy()

    penalty = np.full(len(universe) + 1, float(ridge_lambda))
    penalty[-1] = 0.0
    system = design.T @ design + np.diag(penalty)
    target = design.T @ response
    try:
        solution = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(system, target, rcond=None)
    return ModelRatingFit(
        ratings={team: float(solution[index]) for team, index in position.items()},
        home_term=float(solution[-1]),
        games=games.height,
    )


def model_ratings_before_week(
    predictions: pl.DataFrame, *, season: int, week: int, teams: Sequence[str]
) -> ModelRatingFit:
    """Return the model rating fit on ``season``'s predicted games with a week before ``week``.

    Raises:
        ValueError: If the season has no predicted game before the week (week 1, whose rating
            comes from the previous season: :func:`week_one_model_ratings`).

    """
    games = predictions.filter((pl.col("season") == season) & (pl.col("week") < week))
    if games.is_empty():
        msg = f"no predicted games of {season} before week {week} to fit the model rating on"
        raise ValueError(msg)
    return fit_model_ratings(games, teams)


def model_ratings_by_week(
    predictions: pl.DataFrame, *, teams_by_season: Mapping[int, Sequence[str]] | None = None
) -> pl.DataFrame:
    """Return every team's model rating before every predicted week from the second on.

    For each season and each week from 2 on that has a predicted game, the rating is fit on that
    season's predicted games of the earlier weeks only (:func:`model_ratings_before_week`), so it
    is known before the week's games. Week 1 has no earlier game of its season and no row.

    Args:
        predictions: Pre-game predictions with ``season``, ``week``, ``away_abbr``, ``home_abbr``,
            ``neutral`` and ``predicted_margin``, one row per game.
        teams_by_season: The teams to rate in each season; by default the season's predicted teams.

    Returns:
        A rating frame (``RATING_SCHEMA``), sorted by season, week and team.

    """
    parts: list[pl.DataFrame] = []
    for season in sorted(predictions.get_column("season").unique().to_list()):
        season_rows = predictions.filter(pl.col("season") == season)
        teams = (
            list(teams_by_season[season])
            if teams_by_season is not None and season in teams_by_season
            else _teams_of(season_rows)
        )
        for week in sorted(season_rows.get_column("week").unique().to_list()):
            if week < FIRST_FITTED_WEEK:
                continue
            fit = model_ratings_before_week(season_rows, season=season, week=week, teams=teams)
            parts.append(_rating_frame(season, week, fit.ratings))
    if not parts:
        return pl.DataFrame(schema=RATING_SCHEMA)
    return pl.concat(parts).sort("season", "week", "team")


def carryover_factor(week_one: Mapping[str, float], previous: Mapping[str, float]) -> float:
    """Return how much of last season's team rating the ported prior carries into week 1.

    The ported prior (:mod:`nfl_predictor.ratings.team_prior`) carries last season's offense and
    defense effects into the new season with a slope for each and special teams with none. Its
    factor on the team rating as a whole is the least-squares slope, through the origin over the
    teams of both seasons, of each team's week-1 rating (the prior at full weight) on its rating
    for the whole previous season.

    Args:
        week_one: Each team's ported rating before week 1 of the season.
        previous: Each team's ported rating for the whole previous season.

    Returns:
        The factor.

    Raises:
        ValueError: If no team is in both, or the previous ratings of those teams are all zero.

    """
    shared = sorted(set(week_one) & set(previous))
    before = np.array([previous[team] for team in shared], dtype=np.float64)
    after = np.array([week_one[team] for team in shared], dtype=np.float64)
    spread = float(before @ before)
    if spread == 0.0:
        msg = "the previous season's ratings have no spread over the teams of both seasons"
        raise ValueError(msg)
    return float(before @ after) / spread


def week_one_model_ratings(
    previous_final: Mapping[str, float], factor: float, teams: Sequence[str]
) -> dict[str, float]:
    """Return the week-1 model rating: last season's last fit times ``factor``, centered.

    Args:
        previous_final: The model rating fit on every predicted game of the previous season.
        factor: How much of it carries over (:func:`carryover_factor`).
        teams: The season's teams; a team without a previous rating starts from zero.

    Returns:
        Each team's rating, averaging zero over ``teams``.

    """
    raw = {team: factor * previous_final.get(team, 0.0) for team in teams}
    shift = float(np.mean(list(raw.values()))) if raw else 0.0
    return {team: value - shift for team, value in raw.items()}


def ported_ratings_from_by_week(by_week: pl.DataFrame, *, season: int) -> pl.DataFrame:
    """Return the ported ratings before each week from a season's published weekly history.

    The published ``ratings_by_week`` table rates the teams that have played on the games through
    each week; for the week after, those are the pre-week ratings, bit for bit (the ratings parity
    audit, ``.agents/sos_port/RATINGS_PARITY.md``). A team without a game yet has no row.

    Args:
        by_week: The season's ``ratings_by_week`` table (``week``, ``team``, ``team_rating``).
        season: Its season.

    Returns:
        A rating frame whose week is each row's week plus one.

    """
    return by_week.select(
        pl.lit(season, dtype=pl.Int64).alias("season"),
        (pl.col("week").cast(pl.Int64) + 1).alias("week"),
        pl.col("team").cast(pl.String),
        pl.col(PORTED_RATING_COLUMN).cast(pl.Float64).alias("rating"),
    ).sort("season", "week", "team")


def ported_ratings_from_games(games: pl.DataFrame) -> pl.DataFrame:
    """Return both teams' pre-week ported ratings from game rows' ``ratings`` feature group.

    Args:
        games: Game rows with ``season``, ``week``, ``away_abbr``, ``home_abbr``,
            ``away_ratings_team`` and ``home_ratings_team``.

    Returns:
        A rating frame with one row per team and week that has a rating; a null rating (1999,
        or week 1 before the first prior in 2003) has no row.

    Raises:
        ValueError: If a team carries two different ratings in one week.

    """
    sides = [
        games.select(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col(f"{side}_abbr").cast(pl.String).alias("team"),
            pl.col(f"{side}_{PORTED_FEATURE}").cast(pl.Float64).alias("rating"),
        )
        for side in ("away", "home")
    ]
    long = pl.concat(sides).drop_nulls().unique(maintain_order=True)
    if long.select("season", "week", "team").is_duplicated().any():
        msg = "a team carries two different pre-week ratings in one week of the game rows"
        raise ValueError(msg)
    return long.sort("season", "week", "team")


@dataclass(frozen=True, slots=True)
class PortedRatings:
    """The ported ratings of the game rows and the weekly history, combined.

    Attributes:
        ratings: The rating frame: a team's game-row rating where it plays that week, else its
            weekly-history rating (a team on a bye).
        compared: Team-weeks rated by both sources.
        disagreements: Those whose two ratings differ by more than the tolerance.

    """

    ratings: pl.DataFrame
    compared: int
    disagreements: int


def combine_ported_ratings(
    from_games: pl.DataFrame, from_by_week: pl.DataFrame, *, tolerance: float = 1e-9
) -> PortedRatings:
    """Combine the game rows' ported ratings with the weekly history's, the game rows first.

    The game rows hold the pre-week entry point's own values, the preseason prior of week 1 and
    a team that has yet to play included; the weekly history adds the teams on a bye.

    Args:
        from_games: :func:`ported_ratings_from_games`.
        from_by_week: :func:`ported_ratings_from_by_week`, for one or more seasons.
        tolerance: The largest absolute difference counted as agreement.

    Returns:
        The combined ratings and how well the two sources agree where both rate a team.

    """
    keys = ["season", "week", "team"]
    both = from_games.join(from_by_week, on=keys, how="inner", suffix="_by_week")
    disagreements = both.filter(
        (pl.col("rating") - pl.col("rating_by_week")).abs() > tolerance
    ).height
    byes = from_by_week.join(from_games, on=keys, how="anti")
    return PortedRatings(
        ratings=pl.concat([from_games, byes]).sort(keys),
        compared=both.height,
        disagreements=disagreements,
    )


def blend_ratings(ported: pl.DataFrame, model: pl.DataFrame, weight: float) -> pl.DataFrame:
    """Return ``weight * ported + (1 - weight) * model`` for every team-week both rate.

    Args:
        ported: The ported ratings (a rating frame).
        model: The model ratings (a rating frame).
        weight: The ported rating's weight, from 0 to 1.

    Returns:
        A rating frame, sorted by season, week and team.

    Raises:
        ValueError: If ``weight`` is not between 0 and 1.

    """
    if not 0.0 <= weight <= 1.0:
        msg = f"the blend weight must be between 0 and 1, got {weight}"
        raise ValueError(msg)
    keys = ["season", "week", "team"]
    return (
        ported.join(model, on=keys, how="inner", suffix="_model")
        .select(
            *keys,
            (weight * pl.col("rating") + (1.0 - weight) * pl.col("rating_model")).alias("rating"),
        )
        .sort(keys)
    )


@dataclass(frozen=True, slots=True)
class FoldPredictions:
    """A walk-forward run's pre-game predictions, read from its fold checkpoints.

    Attributes:
        predictions: One row per predicted game with ``PREDICTION_COLUMNS``, sorted by season,
            week and game.
        checkpoint_dir: The directory the folds were read from.
        folds: How many fold files it holds.
        metadata: The run's ``metadata.json`` when read from a run directory, else empty.

    """

    predictions: pl.DataFrame
    checkpoint_dir: Path
    folds: int
    metadata: dict[str, Any]


def read_fold_predictions(path: Path) -> FoldPredictions:
    """Read the pre-game predictions a walk-forward run saved in its fold checkpoints.

    Each finished week of ``nfl-predictor backtest`` is a ``fold_<season>_w<week>.joblib`` file
    whose ``predictions`` frame holds that week's games as predicted by a model trained on the
    games before the week.

    Args:
        path: A run directory whose ``metadata.json`` names its checkpoint directory, or a
            checkpoint directory (``run_comparison.resolve_run``).

    Returns:
        The predictions and where they came from.

    Raises:
        ValueError: If ``path`` is neither, holds no fold, a fold lacks a needed column, or a game
            is predicted more than once.
        TypeError: If a fold file holds no prediction frame.

    """
    run = run_comparison.resolve_run(path)
    files = sorted(run.checkpoint_dir.glob("fold_*.joblib"))
    if not files:
        msg = f"no fold checkpoints in {run.checkpoint_dir}"
        raise ValueError(msg)
    frames = [_fold_predictions(file) for file in files]
    predictions = pl.concat(frames).sort("season", "week", "game_id")
    repeated = predictions.filter(pl.col("game_id").is_duplicated())
    if not repeated.is_empty():
        shown = ", ".join(sorted(set(repeated.get_column("game_id").to_list()))[:5])
        msg = f"{run.checkpoint_dir} predicts a game more than once: {shown}"
        raise ValueError(msg)
    return FoldPredictions(
        predictions=predictions,
        checkpoint_dir=run.checkpoint_dir,
        folds=len(files),
        metadata=dict(run.metadata or {}),
    )


def _fold_predictions(path: Path) -> pl.DataFrame:
    """Return one fold file's predictions with ``PREDICTION_COLUMNS``.

    Raises:
        TypeError: If the file holds no prediction frame.
        RatingSourceUnavailableError: If the frame lacks one of ``PREDICTION_COLUMNS``, the
            ``neutral`` site flag included, or holds games whose ``neutral`` flag is null in
            every row: such a fold would fit neutral-site games as home games. A partly null
            flag is kept, a null read as not neutral (`fit_model_ratings`).

    """
    payload = joblib.load(path)
    frame = payload.get("predictions") if isinstance(payload, dict) else None
    if not isinstance(frame, pd.DataFrame):
        msg = f"{path} holds no prediction frame"
        raise TypeError(msg)
    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        msg = f"{path} lacks {missing}"
        raise RatingSourceUnavailableError(msg)
    if len(frame) and pd.to_numeric(frame["neutral"], errors="coerce").isna().all():
        msg = f"{path} has a null neutral flag in every row"
        raise RatingSourceUnavailableError(msg)
    return pl.DataFrame(
        {
            "game_id": frame["game_id"].astype(str).tolist(),
            "season": pl.Series(pd.to_numeric(frame["season"]).to_numpy(), dtype=pl.Int64),
            "week": pl.Series(pd.to_numeric(frame["week"]).to_numpy(), dtype=pl.Int64),
            "away_abbr": frame["away_abbr"].astype(str).tolist(),
            "home_abbr": frame["home_abbr"].astype(str).tolist(),
            "neutral": pl.Series(
                pd.to_numeric(frame["neutral"], errors="coerce").to_numpy(dtype=float),
                dtype=pl.Float64,
                nan_to_null=True,
            ),
            "predicted_margin": pl.Series(
                pd.to_numeric(frame["predicted_margin"], errors="coerce").to_numpy(dtype=float),
                dtype=pl.Float64,
                nan_to_null=True,
            ),
        },
        schema=_PREDICTION_SCHEMA,
    )


class RatingSourceUnavailableError(ValueError):
    """The files a rating source reads are missing, or hold nothing for the week asked for."""


def ported_ratings_for_week(
    ratings_dir: Path, game_rows: Path, *, season: int, week: int
) -> dict[str, float]:
    """Return every team's ported rating before ``week``, as the ETL published it.

    A team playing that week takes its game row's pre-week ``ratings`` feature (the preseason
    prior in week 1); any other team takes the published weekly history's row for the last
    week before (``{season}_ratings_by_week.parquet`` under ``ratings_dir``), which is the same
    pre-week rating. A postseason week past the history takes its last week, the whole regular
    season.

    Args:
        ratings_dir: The ETL's published ratings tables (``data/ratings``).
        game_rows: A game-row CSV with the ``ratings`` feature group (``data/all_data_ml.csv``).
        season: The season.
        week: The week whose games the ratings come before.

    Returns:
        Each rated team's rating in points.

    Raises:
        RatingSourceUnavailableError: If neither file rates any team before the week.

    """
    from_games = pl.DataFrame(schema=RATING_SCHEMA)
    features = [f"{side}_{PORTED_FEATURE}" for side in ("away", "home")]
    if game_rows.exists() and set(features) <= set(pl.read_csv(game_rows, n_rows=0).columns):
        rows = pl.read_csv(
            game_rows,
            columns=["season", "week", "away_abbr", "home_abbr", *features],
            schema_overrides={
                "season": pl.Int64,
                "week": pl.Int64,
                **dict.fromkeys(features, pl.Float64),
            },
            infer_schema_length=None,
        ).filter((pl.col("season") == season) & (pl.col("week") == week))
        from_games = ported_ratings_from_games(rows)
    from_by_week = pl.DataFrame(schema=RATING_SCHEMA)
    by_week_path = ratings_dir / f"{season}_ratings_by_week.parquet"
    if by_week_path.exists():
        earlier = pl.read_parquet(by_week_path).filter(pl.col("week") < week)
        if not earlier.is_empty():
            last = earlier.filter(pl.col("week") == earlier.get_column("week").max())
            from_by_week = ported_ratings_from_by_week(last, season=season).with_columns(
                pl.lit(week, dtype=pl.Int64).alias("week")
            )
    ratings = combine_ported_ratings(from_games, from_by_week).ratings
    if ratings.is_empty():
        msg = (
            f"no ported team rating for {season} week {week} in {by_week_path} or {game_rows}; "
            "rebuild the data with `nfl-predictor data`"
        )
        raise RatingSourceUnavailableError(msg)
    return dict(zip(ratings.get_column("team"), ratings.get_column("rating"), strict=True))


def season_ported_ratings(ratings_dir: Path, season: int) -> dict[str, float]:
    """Return each team's published ported rating for the whole of ``season``.

    Raises:
        RatingSourceUnavailableError: If the season's ``ratings`` table is missing.

    """
    path = ratings_dir / f"{season}_ratings.parquet"
    if not path.exists():
        msg = f"missing {path}; rebuild the data with `nfl-predictor data`"
        raise RatingSourceUnavailableError(msg)
    table = pl.read_parquet(path, columns=["team", PORTED_RATING_COLUMN])
    return dict(
        zip(
            table.get_column("team").cast(pl.String),
            table.get_column(PORTED_RATING_COLUMN).cast(pl.Float64),
            strict=True,
        )
    )


def read_predictions(paths: Sequence[Path]) -> pl.DataFrame:
    """Return the fold predictions of several walk-forward runs or checkpoint directories.

    Raises:
        ValueError: If a game is predicted in more than one of them, or as
            :func:`read_fold_predictions` does.

    """
    frames = [read_fold_predictions(path).predictions for path in paths]
    if not frames:
        return pl.DataFrame(schema=_PREDICTION_SCHEMA)
    predictions = pl.concat(frames).sort("season", "week", "game_id")
    repeated = sorted(
        set(predictions.filter(pl.col("game_id").is_duplicated()).get_column("game_id"))
    )
    if repeated:
        msg = f"games predicted by more than one source: {', '.join(repeated[:5])}"
        raise ValueError(msg)
    return predictions


def average_predictions(runs: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Return each game's predicted margin averaged over several runs, in the first run's order.

    The margins are summed in the order the runs are given (the same seed order every time
    gives the same bits) and divided by their number.

    Args:
        runs: Each run's predictions (``PREDICTION_COLUMNS``), all of the same games.

    Returns:
        The first run's rows with ``predicted_margin`` the average over the runs.

    Raises:
        ValueError: If no run is given, the runs predict different games, or two runs disagree on
            a game's season, week, teams or neutral site.

    """
    if not runs:
        msg = "no runs to average"
        raise ValueError(msg)
    first = runs[0]
    keys = list(PREDICTION_COLUMNS[1:-1])
    games = set(first.get_column("game_id").to_list())
    joined = first.select("game_id", *keys, pl.col("predicted_margin").alias("_margin_0"))
    total = pl.col("_margin_0")
    for index, run in enumerate(runs[1:], start=1):
        if run.height != first.height or set(run.get_column("game_id").to_list()) != games:
            msg = f"run {index + 1} does not predict the same games as run 1"
            raise ValueError(msg)
        other = run.select(
            "game_id",
            *(pl.col(key).alias(f"_{key}_{index}") for key in keys),
            pl.col("predicted_margin").alias(f"_margin_{index}"),
        )
        joined = joined.join(other, on="game_id", how="left", maintain_order="left")
        differs = pl.any_horizontal(
            pl.col(key).ne_missing(pl.col(f"_{key}_{index}")) for key in keys
        )
        disagreeing = joined.filter(differs).get_column("game_id").to_list()
        if disagreeing:
            msg = f"runs 1 and {index + 1} disagree on games {', '.join(disagreeing[:5])}"
            raise ValueError(msg)
        joined = joined.drop([f"_{key}_{index}" for key in keys])
        total = total + pl.col(f"_margin_{index}")
    return joined.select("game_id", *keys, (total / len(runs)).alias("predicted_margin")).cast(
        _PREDICTION_SCHEMA
    )


def read_split_predictions(
    current: Sequence[Path], reference: Sequence[Path], *, season: int
) -> pl.DataFrame:
    """Return ``season``'s predictions from ``current`` and earlier seasons' from ``reference``.

    ``reference`` holds seeds of one walk-forward configuration, which predict the same games,
    so they are one source: each game's margin is averaged over them in the order given
    (:func:`average_predictions`). Games of ``season`` and later come from ``current`` alone,
    earlier games from the averaged ``reference`` alone, so a game both predict is no conflict.

    Args:
        current: Runs or fold checkpoint directories for ``season`` (:func:`read_predictions`).
        reference: Runs or fold checkpoint directories of the same games, one per seed.
        season: The first season read from ``current``.

    Raises:
        ValueError: As :func:`read_predictions` and :func:`average_predictions` do.

    """
    own = read_predictions(current).filter(pl.col("season") >= season)
    earlier = average_predictions(
        [read_fold_predictions(path).predictions for path in reference]
    ).filter(pl.col("season") < season)
    return pl.concat([earlier, own]).sort("season", "week", "game_id")


def model_ratings_for_week(
    predictions: pl.DataFrame,
    *,
    season: int,
    week: int,
    teams: Sequence[str],
    carryover: float | None = None,
) -> dict[str, float]:
    """Return every team's model rating before ``week``.

    From the second week, the fit on the season's predicted games before the week
    (:func:`model_ratings_before_week`); in week 1, the fit on every predicted regular-season game
    of the previous season (its playoff weeks left out) carried forward by ``carryover``
    (:func:`week_one_model_ratings`).

    Args:
        predictions: Pre-game predictions (:func:`read_predictions`).
        season: The season.
        week: The week whose games the ratings come before.
        teams: The season's teams.
        carryover: Week 1's factor (:func:`carryover_factor`).

    Raises:
        RatingSourceUnavailableError: If the predictions hold no game to fit, or week 1 has no
            ``carryover``.

    """
    if week >= FIRST_FITTED_WEEK:
        try:
            fit = model_ratings_before_week(predictions, season=season, week=week, teams=teams)
        except ValueError as error:
            raise RatingSourceUnavailableError(str(error)) from error
        return fit.ratings
    if carryover is None:
        msg = f"the week-1 model rating of {season} needs the ported prior's carryover factor"
        raise RatingSourceUnavailableError(msg)
    previous_season = predictions.filter(
        (pl.col("season") == season - 1)
        & (pl.col("week") <= constants.get_regular_season_weeks(season - 1))
    )
    try:
        final = fit_model_ratings(previous_season, teams)
    except ValueError as error:
        raise RatingSourceUnavailableError(str(error)) from error
    return week_one_model_ratings(final.ratings, carryover, teams)


def _teams_of(games: pl.DataFrame) -> list[str]:
    """Return the teams on either side of ``games``, sorted."""
    teams = pl.concat([games.get_column("away_abbr"), games.get_column("home_abbr")])
    return sorted(teams.cast(pl.String).drop_nulls().unique().to_list())


def _rating_frame(season: int, week: int, ratings: Mapping[str, float]) -> pl.DataFrame:
    """Return one week's ratings as a rating frame."""
    teams = sorted(ratings)
    return pl.DataFrame(
        {
            "season": [season] * len(teams),
            "week": [week] * len(teams),
            "team": teams,
            "rating": [ratings[team] for team in teams],
        },
        schema=RATING_SCHEMA,
    )


__all__ = [
    "FIRST_FITTED_WEEK",
    "MODEL_RATING_RIDGE_LAMBDA",
    "PORTED_FEATURE",
    "PORTED_RATING_COLUMN",
    "PREDICTION_COLUMNS",
    "RATING_SCHEMA",
    "UNREADABLE_PREDICTION_ERRORS",
    "FoldPredictions",
    "ModelRatingFit",
    "PortedRatings",
    "RatingSourceUnavailableError",
    "average_predictions",
    "blend_ratings",
    "carryover_factor",
    "combine_ported_ratings",
    "fit_model_ratings",
    "model_ratings_before_week",
    "model_ratings_by_week",
    "model_ratings_for_week",
    "ported_ratings_for_week",
    "ported_ratings_from_by_week",
    "ported_ratings_from_games",
    "read_fold_predictions",
    "read_predictions",
    "read_split_predictions",
    "season_ported_ratings",
    "week_one_model_ratings",
]
