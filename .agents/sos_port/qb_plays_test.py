r"""The quarterback rating from all of a quarterback's plays: the pre-registered test.

The protocol is ``protocols/qb_all_plays.md``, a port of ``nfl-sos-ratings``' roadmap section
"Q1" (at ``9081bce``) committed before any candidate was scored. Three candidates rate a
quarterback on different plays with the ported fit unchanged (``nfl_predictor.ratings.qb_rating``
on the shared ridge): "passes and sacks" (A, the rating as ported), "plus scrambles" (B) and
"every quarterback play" (C, plus designed runs and the botched snaps charged to him). Each
predicts a team's offensive EPA per scrimmage play in a game from the starter's rating before the
game and the team fit's own prediction, calibrated on earlier games; paired errors are compared by
a season bootstrap, and the recommendation is the first candidate in the user's order (every
quarterback play, plus scrambles, passes and sacks) that no other beats.

Run from the repository root, without ``-I`` (it imports ``rating_tests_common`` from its own
directory):

    VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/qb_plays_test.py \
        --cache-dir /home/mitch/scratch/step6/data/cache/nflreadpy \
        --ratings-dir /home/mitch/scratch/step6/data/ratings

It reads the play-by-play and nflverse player sources through this repository's loaders with
downloads blocked, and the candidate build's published ratings for the integrity checks; it writes
only its report and JSON beside this file.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import rating_tests_common as common
from threadpoolctl import threadpool_limits

from nfl_predictor import constants
from nfl_predictor.ratings.inputs import team_game_rows
from nfl_predictor.ratings.pbp_inputs import (
    passer_games_from_pbp,
    quarterback_rows,
    team_games_from_pbp,
)
from nfl_predictor.ratings.qb_rating import (
    QB_FACED_DEFENSE_COLUMN,
    QB_ID_COLUMN,
    QB_RATING_COLUMN,
    QB_UNIT_COLUMNS,
    QbRatingFit,
    compute_qb_faced_pass_defense,
    fit_qb_ratings,
    qb_rating_rows,
    qb_season_summary,
)
from nfl_predictor.ratings.ridge import UnitFit, fit_unit_ridge, predict_unit
from nfl_predictor.ratings.team_prior import PRIOR_HORIZON_GAMES, PriorHistory, snapshot_fit
from nfl_predictor.ratings.team_rating import TEAM_UNIT_COLUMNS, scrimmage_rows
from nfl_predictor.season_stats.game_logs import build_qb_game_stats
from nfl_predictor.season_stats.sources import prepare_pbp, repair_one_team_games
from nfl_predictor.utils.polars import nflverse_sources

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt
    from rating_tests_common import Interval

    from nfl_predictor.ratings.pbp_inputs import SeasonSources
    from nfl_predictor.ratings.ridge import UnitPrior
    from nfl_predictor.ratings.team_prior import SeasonPrior
    from nfl_predictor.ratings.team_rating import TeamRatingFit

# --- the candidates --------------------------------------------------------------------------

CANDIDATES = ("A", "B", "C")
WITHOUT_BOTCHED = "C_without_botched"
LABELS = {
    "A": "passes and sacks",
    "B": "plus scrambles",
    "C": "every quarterback play",
    WITHOUT_BOTCHED: "every quarterback play without the botched snaps",
}
# The user's order of simplicity (2026-10-09): every quarterback play, plus scrambles, passes and
# sacks.
ORDER = ("C", "B", "A")
# The play parts each candidate adds to A's dropbacks.
ADDED: dict[str, tuple[str, ...]] = {
    "A": (),
    "B": ("scrambles",),
    "C": ("scrambles", "designed_runs", "charged_aborted"),
    WITHOUT_BOTCHED: ("scrambles", "designed_runs"),
}
# Each play part the play-by-play pass counts, with the column of its summed EPA.
PLAY_TYPES = {
    "scrambles": "scramble_epa",
    "designed_runs": "designed_run_epa",
    "aborted": "aborted_epa",
    "charged_aborted": "charged_aborted_epa",
    "kneels": "kneel_epa",
    "spikes": "spike_epa",
    "two_point_runs": "two_point_run_epa",
    "other_runs": "other_run_epa",
}
# Plus the passer's ``qb_epa`` over his passes and spikes, which is how nflverse's official weekly
# ``passing_epa`` is summed.
PART_COLUMNS = (*(column for pair in PLAY_TYPES.items() for column in pair), "pass_spike_qb_epa")
GAME_KEYS = ["game_id", "team", "qb_id"]


def _flag(column: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, with a missing value read as 0."""
    return pl.col(column).fill_null(0) > 0


def _count_and_epa(name: str, condition: pl.Expr) -> list[pl.Expr]:
    """Return the number of plays meeting ``condition`` and their summed ``epa``."""
    return [
        condition.cast(pl.Int64).sum().alias(name),
        pl.col("epa").fill_null(0.0).filter(condition).sum().alias(PLAY_TYPES[name]),
    ]


def qb_play_parts(plays: pl.DataFrame) -> pl.DataFrame:
    """Return each player's plays of every type in ``PLAY_TYPES`` and their EPA, per game.

    The source's helper (``.agents/findings_2026_10_08/qb_play_types.py`` at ``9081bce``), on
    this repository's play-by-play. Runs, scrambles, botched snaps by rusher and kneel-downs go to
    ``rusher_player_id``; spikes to ``passer_player_id``; the botched snaps a candidate counts to
    the player the official scorer charged with the fumble (``fumbled_1_player_id``), whoever ran
    after it. A botched snap is a carry (a run flagged ``rush`` and not ``qb_scramble``), as in the
    source's by-rusher part, so a scramble that is also a charged botched snap counts once, as a
    scramble. Rows are keyed ``game_id``, ``team`` (the play's ``posteam``) and ``qb_id``.
    Two-point tries are counted only in ``two_point_runs``; ``other_runs`` are run plays flagged
    neither ``rush`` nor ``qb_scramble`` (passes and fumbles coded as runs).
    """
    two_point = _flag("two_point_attempt")
    run = (pl.col("play_type") == "run") & ~two_point
    carry = run & _flag("rush") & ~_flag("qb_scramble")
    rushes = (
        plays.filter(pl.col("posteam").is_not_null() & pl.col("rusher_player_id").is_not_null())
        .group_by("game_id", "posteam", "rusher_player_id", maintain_order=True)
        .agg(
            *_count_and_epa("scrambles", run & _flag("qb_scramble")),
            *_count_and_epa("designed_runs", carry & ~_flag("aborted_play")),
            *_count_and_epa("aborted", carry & _flag("aborted_play")),
            *_count_and_epa("kneels", (pl.col("play_type") == "qb_kneel") & ~two_point),
            *_count_and_epa(
                "two_point_runs", pl.col("play_type").is_in(["run", "qb_kneel"]) & two_point
            ),
            *_count_and_epa("other_runs", run & ~_flag("rush") & ~_flag("qb_scramble")),
        )
        .rename({"posteam": "team", "rusher_player_id": "qb_id"})
    )
    spikes = (
        plays.filter(pl.col("posteam").is_not_null() & pl.col("passer_player_id").is_not_null())
        .group_by("game_id", "posteam", "passer_player_id", maintain_order=True)
        .agg(
            *_count_and_epa("spikes", pl.col("play_type") == "qb_spike"),
            pl.col("qb_epa")
            .fill_null(0.0)
            .filter(pl.col("play_type").is_in(["pass", "qb_spike"]))
            .sum()
            .alias("pass_spike_qb_epa"),
        )
        .rename({"posteam": "team", "passer_player_id": "qb_id"})
    )
    charged = (
        plays.filter(pl.col("posteam").is_not_null() & pl.col("fumbled_1_player_id").is_not_null())
        .group_by("game_id", "posteam", "fumbled_1_player_id", maintain_order=True)
        .agg(*_count_and_epa("charged_aborted", carry & _flag("aborted_play")))
        .rename({"posteam": "team", "fumbled_1_player_id": "qb_id"})
    )
    return (
        rushes.join(spikes, on=GAME_KEYS, how="full", coalesce=True, maintain_order="left_right")
        .join(charged, on=GAME_KEYS, how="full", coalesce=True, maintain_order="left_right")
        .with_columns(pl.col(column).fill_null(0) for column in PART_COLUMNS)
    )


def with_parts(qb_rows: pl.DataFrame, parts: pl.DataFrame) -> pl.DataFrame:
    """Return the quarterback rows with each row's play parts joined on (zero without any)."""
    return qb_rows.join(parts, on=GAME_KEYS, how="left", maintain_order="left").with_columns(
        pl.col(column).fill_null(0) for column in PART_COLUMNS
    )


def candidate_rows(qb_rows: pl.DataFrame, parts: pl.DataFrame, candidate: str) -> pl.DataFrame:
    """Return a candidate's fit rows: the rows with a dropback, its plays added to A's.

    A's plays are the rows' ``qb_dropbacks`` and its EPA their ``qb_dropback_epa`` (the official
    passing EPA less spikes); a candidate adds its parts' counts and EPA to both, so the fit's
    rate (``qb_rating.qb_rating_rows``) is the candidate's EPA per play. The columns keep the
    rows' names and types, so the ported fit reads them unchanged.
    """
    rows = qb_rows.filter(pl.col("qb_dropbacks") > 0)
    added = ADDED[candidate]
    if not added:
        return rows
    joined = with_parts(rows, parts)
    plays = pl.sum_horizontal("qb_dropbacks", *added)
    epa = pl.sum_horizontal("qb_dropback_epa", *(PLAY_TYPES[name] for name in added))
    return joined.with_columns(
        plays.cast(rows.schema["qb_dropbacks"]).alias("qb_dropbacks"),
        epa.cast(rows.schema["qb_dropback_epa"]).alias("qb_dropback_epa"),
    ).select(rows.columns)


def first_qb_snaps(plays: pl.DataFrame, qb_rows: pl.DataFrame) -> pl.DataFrame:
    """Return each team-game's starter, from its first quarterback snap.

    The starter is the passer or rusher of the first first-quarter play, by ``play_id``, who has a
    fit row: a quarterback row of that team and game with at least one dropback. A team-game
    without such a play has no row.
    """
    fit_rows = qb_rows.filter(pl.col("qb_dropbacks") > 0).select(GAME_KEYS).unique()
    first_quarter = plays.filter(pl.col("posteam").is_not_null() & (pl.col("qtr") == 1)).select(
        "game_id",
        pl.col("posteam").alias("team"),
        "play_id",
        "passer_player_id",
        "rusher_player_id",
    )
    candidates = pl.concat(
        [
            first_quarter.select("game_id", "team", "play_id", pl.col(column).alias("qb_id"))
            for column in ("passer_player_id", "rusher_player_id")
        ]
    ).join(fit_rows, on=GAME_KEYS, how="semi")
    return (
        candidates.sort("game_id", "team", "play_id")
        .group_by("game_id", "team", maintain_order=True)
        .first()
        .select("game_id", "team", "qb_id")
    )


def passer_prediction(
    fit: UnitFit | None, starter: str | None, opponent: str, home_sign: float
) -> float:
    """Return ``intercept + passer[starter] - defense[opponent] + home_field * home_sign``.

    An effect the snapshot lacks (a starter or defense without earlier rows, or an unknown
    starter) counts as zero; without a snapshot (no earlier rows at all) the prediction is zero.
    """
    if fit is None:
        return 0.0
    passer = fit.offense.get(starter, 0.0) if starter is not None else 0.0
    return fit.intercept + passer - fit.defense.get(opponent, 0.0) + fit.home_field * home_sign


def fit_calibration(
    features: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return least-squares coefficients of ``target`` on ``features`` and a constant (last).

    Without rows every coefficient is zero.
    """
    count, width = features.shape
    if count == 0:
        return np.zeros(width + 1, dtype=np.float64)
    design = np.column_stack((features, np.ones(count, dtype=np.float64)))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(coefficients, dtype=np.float64)


# --- the decision ----------------------------------------------------------------------------

OVERALL = "overall"
BANDS: tuple[tuple[str, int, int | None], ...] = (("weeks 5-8", 5, 8), ("weeks 9+", 9, None))
PAIRS = (("B", "A"), ("C", "A"), ("C", "B"))


@dataclass(frozen=True)
class Comparison:
    """A paired difference X - Y of absolute errors: overall and in each week band."""

    overall: Interval
    bands: Mapping[str, Interval]

    def flipped(self) -> Comparison:
        """Return the comparison of Y - X."""
        return Comparison(
            overall=self.overall.flipped(),
            bands={name: interval.flipped() for name, interval in self.bands.items()},
        )


def beats(comparison: Comparison) -> bool:
    """Return whether X beats Y: the overall interval below zero and no band interval above."""
    return comparison.overall.below_zero and not any(
        band.above_zero for band in comparison.bands.values()
    )


@dataclass(frozen=True)
class Decision:
    """The rule's outcome: the recommendation, the wins and every interval excluding zero.

    ``wins`` holds ``(winner, loser)``; ``excluding_zero`` holds ``(X, Y, band, "better" or
    "worse")`` for each X - Y interval that excludes zero, read for X. ``recommended`` is
    ``None`` when every candidate is beaten (a cycle of wins), which the rule has no branch for.
    """

    recommended: str | None
    wins: tuple[tuple[str, str], ...]
    excluding_zero: tuple[tuple[str, str, str, str], ...]


def decide(comparisons: Mapping[tuple[str, str], Comparison], order: Sequence[str]) -> Decision:
    """Apply the rule: the first candidate in ``order`` that no other candidate beats.

    When every candidate is beaten, the result lies between the rule's branches: there is no
    recommendation, and the decision goes to the user.
    """
    wins: list[tuple[str, str]] = []
    excluding: list[tuple[str, str, str, str]] = []
    for (first, second), comparison in comparisons.items():
        if beats(comparison):
            wins.append((first, second))
        elif beats(comparison.flipped()):
            wins.append((second, first))
        for band, interval in ((OVERALL, comparison.overall), *comparison.bands.items()):
            if interval.excludes_zero:
                direction = "better" if interval.below_zero else "worse"
                excluding.append((first, second, band, direction))
    beaten = {loser for _, loser in wins}
    recommended = next((name for name in order if name not in beaten), None)
    return Decision(recommended, tuple(wins), tuple(excluding))


# --- the walk-forward ------------------------------------------------------------------------

INFORMATION_TOLERANCE = 1e-12
MATCH_TOLERANCE = 1e-9


class IntegrityError(ValueError):
    """Raised when an integrity check fails; nothing about the candidates is read then."""


@dataclass(frozen=True)
class Window:
    """The seasons and weeks of the test.

    ``penalty_season`` supplies the first predicted season's penalties; ``warmup_season`` is
    predicted and feeds the calibration but is never scored; scoring covers ``first_scored``
    through ``last_season`` from ``start_week``; every week from ``first_week`` is predicted.
    """

    penalty_season: int
    warmup_season: int
    first_scored: int
    last_season: int
    start_week: int = 5
    first_week: int = 2

    @property
    def predicted_seasons(self) -> list[int]:
        """Return the warm-up and scored seasons."""
        return list(range(self.warmup_season, self.last_season + 1))

    @property
    def scored_seasons(self) -> list[int]:
        """Return the scored seasons."""
        return list(range(self.first_scored, self.last_season + 1))


REAL_WINDOW = Window(penalty_season=2005, warmup_season=2006, first_scored=2007, last_season=2025)


@dataclass(frozen=True)
class HarnessInputs:
    """Everything the walk-forward reads, season by season.

    Attributes:
        team_rows: Each season's team-game rows (``inputs.TEAM_GAME_SCHEMA``).
        candidate_rows: Each candidate's fit rows by season (:func:`candidate_rows`).
        starters: Each season's starters (``game_id``, ``team``, ``qb_id``).
        team_penalties: The team fit whose penalties a season's snapshots use.
        season_prior: A season's preseason prior, or ``None``.

    """

    team_rows: Mapping[int, pl.DataFrame]
    candidate_rows: Mapping[str, Mapping[int, pl.DataFrame]]
    starters: Mapping[int, pl.DataFrame]
    team_penalties: Callable[[int], TeamRatingFit]
    season_prior: Callable[[int], SeasonPrior | None]


@dataclass(frozen=True)
class TeamSnapshot:
    """The team fit of a season's games before a week: its scrimmage unit, means and ratings."""

    fit: UnitFit
    means: UnitPrior | None
    ratings: pl.DataFrame
    plays_per_game: float


def team_snapshot(
    prior_games: pl.DataFrame,
    teams: Sequence[str],
    penalties: TeamRatingFit,
    season_prior: SeasonPrior | None,
) -> TeamSnapshot:
    """Return the published team fit of ``prior_games`` and the scrimmage unit behind it.

    ``team_prior.snapshot_fit`` gives the ratings and the prior means it used; the scrimmage unit
    is refit with the same rows, penalty and means, the fit inside ``fit_team_ratings``.
    """
    ratings, means = snapshot_fit(prior_games, teams, penalties, season_prior, PRIOR_HORIZON_GAMES)
    rows = scrimmage_rows(prior_games)
    fit = fit_unit_ridge(
        rows, TEAM_UNIT_COLUMNS, ridge_lambda=penalties.scrimmage_lambda, prior=means
    )
    plays_per_game = float(rows.get_column("plays").sum()) / rows.height
    return TeamSnapshot(fit, means, ratings, plays_per_game)


def _effect(fitted: Mapping[str, float], prior: Mapping[str, float] | None, team: str) -> float:
    """Return a team's fitted effect, else its prior mean, else zero."""
    if team in fitted:
        return fitted[team]
    if prior is not None and team in prior:
        return prior[team]
    return 0.0


def team_term(snapshot: TeamSnapshot, team: str, opponent: str, home_sign: float) -> float:
    """Return the snapshot's predicted scrimmage EPA per play for ``team`` against ``opponent``."""
    fit, means = snapshot.fit, snapshot.means
    offense = _effect(fit.offense, None if means is None else means.offense, team)
    defense = _effect(fit.defense, None if means is None else means.defense, opponent)
    return fit.intercept + offense - defense + fit.home_field * home_sign


def team_term_gap(snapshot: TeamSnapshot) -> float:
    """Return the largest gap between the effects behind ``t`` and the snapshot's ratings."""
    fit, means = snapshot.fit, snapshot.means
    gap = 0.0
    for team, offense, defense in snapshot.ratings.select(
        "team", "offense_rating", "defense_rating"
    ).iter_rows():
        own_offense = _effect(fit.offense, None if means is None else means.offense, team)
        own_defense = _effect(fit.defense, None if means is None else means.defense, team)
        gap = max(
            gap,
            abs(own_offense * snapshot.plays_per_game - offense),
            abs(own_defense * snapshot.plays_per_game - defense),
        )
    return gap


def qb_snapshot(rows: pl.DataFrame, week: int, ridge_lambda: float) -> UnitFit | None:
    """Return the ported quarterback fit of the rows before ``week``, or ``None`` without any."""
    prior_rows = rows.filter(pl.col("week") < week)
    if prior_rows.is_empty():
        return None
    return fit_unit_ridge(qb_rating_rows(prior_rows), QB_UNIT_COLUMNS, ridge_lambda=ridge_lambda)


def _home_sign(row: Mapping[str, object]) -> float:
    """Return +1 when the row's team is at home, -1 away and 0 at a neutral or unknown site."""
    is_home = row["is_home"]
    if is_home is None:
        return 0.0
    return 1.0 if is_home else -1.0


_WEEK_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64(),
    "week": pl.Int64(),
    "game_id": pl.String(),
    "team": pl.String(),
    "opponent_team": pl.String(),
    "is_home": pl.Boolean(),
    "starter": pl.String(),
    "t": pl.Float64(),
}


def _week(
    inputs: HarnessInputs, season: int, week: int, lambdas: Mapping[str, float]
) -> tuple[pl.DataFrame, TeamSnapshot | None, dict[str, UnitFit | None]]:
    """Return a week's predictions and the snapshots behind them."""
    team_rows = inputs.team_rows[season]
    games = team_rows.filter(pl.col("week") == week)
    prior_games = team_rows.filter(pl.col("week") < week)
    teams = sorted(team_rows.get_column("team").unique().to_list())
    snapshot = (
        None
        if prior_games.is_empty()
        else team_snapshot(
            prior_games, teams, inputs.team_penalties(season), inputs.season_prior(season)
        )
    )
    game_ids = games.get_column("game_id").unique().to_list()
    starters = {
        (game_id, team): qb_id
        for game_id, team, qb_id in inputs.starters[season]
        .filter(pl.col("game_id").is_in(game_ids))
        .select("game_id", "team", "qb_id")
        .iter_rows()
    }
    fits = {
        name: qb_snapshot(rows[season], week, lambdas[name])
        for name, rows in inputs.candidate_rows.items()
    }
    records: list[dict[str, object]] = []
    for row in games.iter_rows(named=True):
        sign = _home_sign(row)
        starter = starters.get((row["game_id"], row["team"]))
        record: dict[str, object] = {
            "season": season,
            "week": week,
            "game_id": row["game_id"],
            "team": row["team"],
            "opponent_team": row["opponent_team"],
            "is_home": row["is_home"],
            "starter": starter,
            "t": 0.0
            if snapshot is None
            else team_term(snapshot, row["team"], row["opponent_team"], sign),
        }
        for name, fit in fits.items():
            record[f"p_{name}"] = passer_prediction(fit, starter, row["opponent_team"], sign)
            record[f"r_{name}"] = (
                fit.intercept + fit.offense[starter]
                if fit is not None and starter in fit.offense
                else None
            )
        records.append(record)
    schema = {
        **_WEEK_SCHEMA,
        **{f"{kind}_{name}": pl.Float64() for name in inputs.candidate_rows for kind in ("p", "r")},
    }
    return pl.DataFrame(records, schema=schema), snapshot, fits


def predict_week(
    inputs: HarnessInputs, season: int, week: int, lambdas: Mapping[str, float]
) -> pl.DataFrame:
    """Return each team-game of a week with ``t`` and every candidate's ``p_X``."""
    return _week(inputs, season, week, lambdas)[0]


def _perturbed_inputs(inputs: HarnessInputs, season: int, week: int) -> HarnessInputs:
    """Return ``inputs`` with every value of ``season``'s games from ``week`` on replaced.

    The team rows' EPA and snaps, every candidate's plays and EPA, and the starters of the games
    after ``week`` change; the week's own starters stay (they are known at the first snap).
    """
    later = pl.col("week") >= week
    team_rows = dict(inputs.team_rows)
    team_rows[season] = team_rows[season].with_columns(
        pl.when(later)
        .then(pl.col("offensive_epa") * -3.0 + 7.0)
        .otherwise(pl.col("offensive_epa"))
        .alias("offensive_epa"),
        pl.when(later)
        .then(pl.col("offensive_snaps") + 5)
        .otherwise(pl.col("offensive_snaps"))
        .alias("offensive_snaps"),
    )
    candidate_rows: dict[str, Mapping[int, pl.DataFrame]] = {}
    for name, by_season in inputs.candidate_rows.items():
        rows = dict(by_season)
        rows[season] = rows[season].with_columns(
            pl.when(later)
            .then(pl.col("qb_dropbacks") + 3)
            .otherwise(pl.col("qb_dropbacks"))
            .alias("qb_dropbacks"),
            pl.when(later)
            .then(pl.col("qb_dropback_epa") * -2.0 + 1.0)
            .otherwise(pl.col("qb_dropback_epa"))
            .alias("qb_dropback_epa"),
        )
        candidate_rows[name] = rows
    later_games = (
        inputs.team_rows[season].filter(pl.col("week") > week).get_column("game_id").unique()
    )
    starters = dict(inputs.starters)
    starters[season] = starters[season].with_columns(
        pl.when(pl.col("game_id").is_in(later_games.to_list()))
        .then(pl.lit("perturbed"))
        .otherwise(pl.col("qb_id"))
        .alias("qb_id")
    )
    return HarnessInputs(
        team_rows=team_rows,
        candidate_rows=candidate_rows,
        starters=starters,
        team_penalties=inputs.team_penalties,
        season_prior=inputs.season_prior,
    )


def check_week_information(
    predict: Callable[[HarnessInputs, int, int, Mapping[str, float]], pl.DataFrame],
    inputs: HarnessInputs,
    season: int,
    week: int,
    lambdas: Mapping[str, float],
) -> float:
    """Check 8: a week's predictions ignore every value of the games from that week on.

    Raises:
        IntegrityError: If a prediction moves when the later games change.

    """
    original = predict(inputs, season, week, lambdas)
    again = predict(_perturbed_inputs(inputs, season, week), season, week, lambdas)
    columns = ["t", *(column for column in original.columns if column.startswith("p_"))]
    if original.select("game_id", "team").rows() != again.select("game_id", "team").rows():
        msg = f"check 8: {season} week {week}'s team-games change with the later games"
        raise IntegrityError(msg)
    gap = max(
        common.as_float((original.get_column(c) - again.get_column(c)).abs().max()) for c in columns
    )
    if gap > INFORMATION_TOLERANCE:
        msg = f"check 8: {season} week {week}'s predictions move by {gap:.3g} with later games"
        raise IntegrityError(msg)
    return gap


@dataclass
class HarnessAudit:
    """What the walk-forward found while it ran, for the integrity checks."""

    team_snapshots: int = 0
    team_term_gap: float = 0.0
    qb_fit_gap: float = 0.0
    information_snapshots: int = 0
    information_gap: float = 0.0
    lambdas: dict[str, dict[int, float]] = field(default_factory=dict)
    team_ratings: dict[tuple[int, int], pl.DataFrame] = field(default_factory=dict)


@dataclass
class HarnessResult:
    """The walk-forward's predictions, one row per candidate and team-game, and its audit.

    ``starter_ratings`` holds, per team-game, each candidate's pre-week rating of the starter
    (``r_X``: the snapshot's intercept plus his effect; null without earlier rows).
    """

    predictions: pl.DataFrame
    audit: HarnessAudit
    starter_ratings: pl.DataFrame


def season_lambdas(inputs: HarnessInputs, season: int) -> dict[str, float]:
    """Return each candidate's penalty for ``season``: cross-validated on its previous season."""
    return {
        name: fit_qb_ratings(rows[season - 1]).ridge_lambda
        for name, rows in inputs.candidate_rows.items()
    }


def run_harness(
    inputs: HarnessInputs,
    window: Window,
    *,
    check_information: bool = False,
    audit_fits: bool = True,
) -> HarnessResult:
    """Predict every team-game from ``window.first_week`` of every predicted season.

    ``check_information`` runs check 8 on every week; ``audit_fits`` compares every quarterback
    snapshot with ``qb_rating.fit_qb_ratings`` on the same rows (the synthetic windows skip it).
    """
    audit = HarnessAudit()
    frames: list[pl.DataFrame] = []
    for season in window.predicted_seasons:
        lambdas = season_lambdas(inputs, season)
        for name, value in lambdas.items():
            audit.lambdas.setdefault(name, {})[season] = value
        weeks = sorted(
            week
            for week in inputs.team_rows[season].get_column("week").unique().to_list()
            if week >= window.first_week
        )
        for week in weeks:
            frame, snapshot, fits = _week(inputs, season, week, lambdas)
            if snapshot is not None:
                audit.team_snapshots += 1
                audit.team_term_gap = max(audit.team_term_gap, team_term_gap(snapshot))
                audit.team_ratings[(season, week)] = snapshot.ratings
            if audit_fits:
                audit.qb_fit_gap = max(
                    audit.qb_fit_gap, _qb_fit_gap(inputs, season, week, lambdas, fits)
                )
            if check_information:
                audit.information_gap = max(
                    audit.information_gap,
                    check_week_information(predict_week, inputs, season, week, lambdas),
                )
                audit.information_snapshots += 1
            frames.append(frame)
    wide = pl.concat(frames)
    names = list(inputs.candidate_rows)
    predictions = (
        wide.select(*_WEEK_SCHEMA, *(f"p_{name}" for name in names))
        .unpivot(
            on=[f"p_{name}" for name in names],
            index=list(_WEEK_SCHEMA),
            variable_name="candidate",
            value_name="p",
        )
        .with_columns(pl.col("candidate").str.strip_prefix("p_"))
    )
    starter_ratings = wide.select(
        "season", "week", "game_id", "team", "starter", *(f"r_{name}" for name in names)
    )
    return HarnessResult(predictions, audit, starter_ratings)


def _qb_fit_gap(
    inputs: HarnessInputs,
    season: int,
    week: int,
    lambdas: Mapping[str, float],
    fits: Mapping[str, UnitFit | None],
) -> float:
    """Return the gap between each snapshot and ``qb_rating.fit_qb_ratings`` on the same rows."""
    gap = 0.0
    for name, fit in fits.items():
        if fit is None:
            continue
        prior_rows = inputs.candidate_rows[name][season].filter(pl.col("week") < week)
        ported = fit_qb_ratings(prior_rows, ridge_lambda=lambdas[name]).ratings
        for passer, rating in ported.select(QB_ID_COLUMN, QB_RATING_COLUMN).iter_rows():
            gap = max(gap, abs(fit.intercept + fit.offense[passer] - rating))
    return gap


def calibrate(rows: pl.DataFrame, features: Sequence[str], target: str) -> pl.DataFrame:
    """Return ``rows`` with each week's calibrated prediction, fit on the earlier weeks only.

    The rows are one candidate's, ordered by season, week, game and team; each week's
    coefficients come from least squares of ``target`` on ``features`` and a constant over every
    earlier row (:func:`fit_calibration`). Adds ``prediction``, ``error`` (prediction minus
    target), ``training_rows`` and one ``coef_{feature}`` per feature.
    """
    ordered = rows.sort("season", "week", "game_id", "team")
    design = ordered.select(features).to_numpy().astype(np.float64)
    response = ordered.get_column(target).cast(pl.Float64).to_numpy()
    keys = ordered.select("season", "week").rows()
    predictions = np.zeros(ordered.height, dtype=np.float64)
    coefficients = np.zeros((ordered.height, len(features) + 1), dtype=np.float64)
    training = np.zeros(ordered.height, dtype=np.int64)
    start = 0
    while start < ordered.height:
        end = start
        while end < ordered.height and keys[end] == keys[start]:
            end += 1
        fitted = fit_calibration(design[:start], response[:start])
        predictions[start:end] = design[start:end] @ fitted[:-1] + fitted[-1]
        coefficients[start:end] = fitted
        training[start:end] = start
        start = end
    return ordered.with_columns(
        pl.Series("prediction", predictions),
        pl.Series("error", predictions - response),
        pl.Series("training_rows", training),
        *(pl.Series(f"coef_{name}", coefficients[:, index]) for index, name in enumerate(features)),
    )


# --- inputs ----------------------------------------------------------------------------------

FIRST_TEAM_SEASON = 1999
RUSHING_TOLERANCE = 1e-6


def _snap() -> pl.Expr:
    """Return the ported team rating's scrimmage snap, written out (``pbp_inputs``' rule)."""
    run_play = (pl.col("play_type") == "run").fill_null(value=False)
    return (
        _flag("qb_dropback")
        | (_flag("qb_scramble") & run_play)
        | _flag("rush")
        | _flag("qb_kneel")
        | _flag("qb_spike")
    )


def team_points(schedule: pl.DataFrame) -> pl.DataFrame:
    """Return each regular-season team-game's points scored, from the schedule."""
    games = (
        schedule.filter(pl.col("game_type") == "REG")
        if "game_type" in schedule.columns
        else schedule
    )
    return pl.concat(
        [
            games.select(
                "game_id",
                pl.col(team).alias("team"),
                pl.col(score).cast(pl.Float64).alias("points_for"),
            )
            for team, score in (("home_abbr", "home_score"), ("away_abbr", "away_score"))
        ]
    )


def season_targets(team_rows: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Return each team-game's offensive EPA per scrimmage play and its points scored."""
    return team_rows.select(
        "game_id",
        "team",
        (pl.col("offensive_epa") / pl.col("offensive_snaps")).alias("target"),
    ).join(team_points(schedule), on=["game_id", "team"], how="left", maintain_order="left")


def snap_check(plays: pl.DataFrame, team_rows: pl.DataFrame) -> dict[str, float]:
    """Check 4's facts for one season: added plays outside the snaps, and the target recount.

    Every scramble, designed run and charged botched snap a candidate can add must be a
    scrimmage snap; and each team-game's EPA per snap, recounted from the play-by-play with the
    rule written out here, must equal the adapter's.
    """
    snap = _snap()
    two_point = _flag("two_point_attempt")
    run = (pl.col("play_type") == "run") & ~two_point
    added = {
        "scrambles": run & _flag("qb_scramble") & pl.col("rusher_player_id").is_not_null(),
        "designed_runs": run
        & _flag("rush")
        & ~_flag("qb_scramble")
        & ~_flag("aborted_play")
        & pl.col("rusher_player_id").is_not_null(),
        "charged_aborted": run
        & _flag("rush")
        & _flag("aborted_play")
        & pl.col("fumbled_1_player_id").is_not_null(),
    }
    known = plays.filter(
        pl.col("posteam").is_not_null()
        & (pl.col("posteam") != "")
        & pl.col("defteam").is_not_null()
        & (pl.col("defteam") != "")
    )
    facts: dict[str, float] = {
        f"{name}_not_snaps": float(known.filter(condition & ~snap).height)
        for name, condition in added.items()
    }
    recount = (
        known.group_by("game_id", pl.col("posteam").alias("team"))
        .agg(
            snap.cast(pl.Int64).sum().alias("_snaps"),
            pl.col("epa").fill_null(0.0).filter(snap).sum().alias("_epa"),
        )
        .filter(pl.col("_snaps") > 0)
    )
    joined = team_rows.join(recount, on=["game_id", "team"], how="full", coalesce=True)
    facts["team_games_unmatched"] = float(
        joined.filter(pl.col("_snaps").is_null() | pl.col("offensive_snaps").is_null()).height
    )
    facts["snap_mismatches"] = float(
        joined.filter(pl.col("_snaps") != pl.col("offensive_snaps")).height
    )
    facts["target_gap"] = common.as_float(
        joined.select(
            (
                pl.col("_epa") / pl.col("_snaps")
                - pl.col("offensive_epa") / pl.col("offensive_snaps")
            )
            .abs()
            .max()
        ).item()
    )
    return facts


def season_stats_parts(sources: SeasonSources, cache_dir: Path) -> pl.DataFrame:
    """Return this repository's ported season statistics' quarterback play counts, per game.

    ``season_stats.game_logs.build_qb_game_stats`` on the same play-by-play, snap counts, weekly
    player stats and quarterback identity, an implementation of the source's game logs that does
    not share the code above.
    """
    season = sources.season

    def roster() -> pl.DataFrame:
        return nflverse_sources.load_rosters(
            [season], cache_dir=cache_dir, current_season=common.CACHE_CURRENT_SEASON
        )

    pbp = repair_one_team_games(prepare_pbp(sources.plays, season), roster)
    identity = nflverse_sources.load_season_qb_identity(
        season, cache_dir=cache_dir, current_season=common.CACHE_CURRENT_SEASON
    )
    games = build_qb_game_stats(pbp, sources.snap_counts, identity, sources.player_stats)
    columns = (
        "qb_scrambles",
        "qb_kneels",
        "qb_designed_carries",
        "qb_designed_rush_epa",
        "qb_rushing_epa",
        "qb_passing_epa",
    )
    return games.select(
        "game_id",
        pl.col("team_abbr").alias("team"),
        "qb_id",
        *(pl.col(column).fill_null(0) for column in columns),
    )


def parts_check(parts: pl.DataFrame, stats: pl.DataFrame) -> dict[str, float]:
    """Check 3's facts for one season: the play parts against the ported season statistics."""
    joined = with_parts(stats, parts)
    rushing = pl.sum_horizontal(
        "scramble_epa",
        "designed_run_epa",
        "aborted_epa",
        "kneel_epa",
        "two_point_run_epa",
        "other_run_epa",
    )
    carries = pl.col("designed_runs") + pl.col("aborted")
    carry_epa = pl.col("designed_run_epa") + pl.col("aborted_epa")
    return {
        "qb_games": float(joined.height),
        "scramble_mismatches": float(
            joined.filter(pl.col("scrambles") != pl.col("qb_scrambles")).height
        ),
        "kneel_mismatches": float(joined.filter(pl.col("kneels") != pl.col("qb_kneels")).height),
        "carry_mismatches": float(
            joined.filter(
                (carries != pl.col("qb_designed_carries"))
                | ((carry_epa - pl.col("qb_designed_rush_epa")).abs() > MATCH_TOLERANCE)
            ).height
        ),
        "rushing_epa_gaps": float(
            joined.filter((pl.col("qb_rushing_epa") - rushing).abs() > RUSHING_TOLERANCE).height
        ),
        "passing_epa_gaps": float(
            joined.filter(
                (pl.col("qb_passing_epa") - pl.col("pass_spike_qb_epa")).abs() > RUSHING_TOLERANCE
            ).height
        ),
    }


def dispersion_parts(plays: pl.DataFrame, qb_rows: pl.DataFrame) -> pl.DataFrame:
    """Return each fit-row passer's scramble and designed-run counts, EPA sums and squares.

    The power estimate's method-of-moments spreads read them, per passer and season.
    """
    two_point = _flag("two_point_attempt")
    run = (pl.col("play_type") == "run") & ~two_point
    kinds = {
        "scramble": run & _flag("qb_scramble"),
        "designed": run & _flag("rush") & ~_flag("qb_scramble") & ~_flag("aborted_play"),
    }
    epa = pl.col("epa").fill_null(0.0)
    sums = (
        plays.filter(pl.col("posteam").is_not_null() & pl.col("rusher_player_id").is_not_null())
        .group_by(
            "game_id", pl.col("posteam").alias("team"), pl.col("rusher_player_id").alias("qb_id")
        )
        .agg(
            *(
                expression
                for name, condition in kinds.items()
                for expression in (
                    condition.cast(pl.Int64).sum().alias(f"{name}_n"),
                    epa.filter(condition).sum().alias(f"{name}_sum"),
                    (epa * epa).filter(condition).sum().alias(f"{name}_sumsq"),
                )
            )
        )
    )
    fit_rows = qb_rows.filter(pl.col("qb_dropbacks") > 0).select(GAME_KEYS)
    return (
        fit_rows.join(sums, on=GAME_KEYS, how="inner")
        .group_by("qb_id")
        .agg(pl.col("^(scramble|designed)_.*$").sum())
        .sort("qb_id")
    )


@dataclass
class QbSeason:
    """One season's quarterback rows, play parts, starters, targets and per-season facts."""

    qb_rows: pl.DataFrame
    parts: pl.DataFrame
    starters: pl.DataFrame
    targets: pl.DataFrame
    snap_facts: dict[str, float]
    parts_facts: dict[str, float]
    dispersion: pl.DataFrame


@dataclass
class QbInputs:
    """Every season's team rows (from 1999) and quarterback seasons (from the penalty season)."""

    team_rows: dict[int, pl.DataFrame]
    seasons: dict[int, QbSeason]


def load_qb_inputs(cache_dir: Path, window: Window, ledger: common.InputLedger) -> QbInputs:
    """Load every season through this repository's loaders, downloads blocked.

    The play-by-play of a season is read once and dropped after its rows, parts, starters and
    checks are built.
    """
    common.block_downloads()
    players = common.load_players(cache_dir, ledger)
    inputs = QbInputs(team_rows={}, seasons={})
    for season in range(FIRST_TEAM_SEASON, window.last_season + 1):
        if season < window.penalty_season:
            plays, schedule = common.load_season_plays(cache_dir, season, ledger)
            inputs.team_rows[season] = team_game_rows(team_games_from_pbp(plays, schedule))
            continue
        sources, schedule = common.load_season_sources(cache_dir, season, players, ledger)
        team = team_games_from_pbp(sources.plays, schedule)
        rows = team_game_rows(team)
        inputs.team_rows[season] = rows
        qb_rows = quarterback_rows(passer_games_from_pbp(sources, team.drop("season")))
        parts = qb_play_parts(sources.plays)
        inputs.seasons[season] = QbSeason(
            qb_rows=qb_rows,
            parts=parts,
            starters=first_qb_snaps(sources.plays, qb_rows),
            targets=season_targets(rows, schedule),
            snap_facts=snap_check(sources.plays, rows),
            parts_facts=parts_check(parts, season_stats_parts(sources, cache_dir)),
            dispersion=dispersion_parts(sources.plays, qb_rows),
        )
    return inputs


# --- the real run's inputs and integrity checks ----------------------------------------------

ZERO = "Z"
LABELS[ZERO] = "every quarterback play with every added play set to zero (an integrity check)"
# The source validation's quarterback year-over-year Pearson (docs/validation-report.md).
SOURCE_STABILITY = (0.455, 601)


def candidate_frames(inputs: QbInputs, names: Sequence[str]) -> dict[str, dict[int, pl.DataFrame]]:
    """Return each candidate's fit rows by season; ``ZERO`` is C with its added plays zeroed."""
    frames: dict[str, dict[int, pl.DataFrame]] = {}
    for name in names:
        frames[name] = {}
        for season, data in inputs.seasons.items():
            if name == ZERO:
                zeroed = data.parts.with_columns(pl.col(column) * 0 for column in PART_COLUMNS)
                frames[name][season] = candidate_rows(data.qb_rows, zeroed, "C")
            else:
                frames[name][season] = candidate_rows(data.qb_rows, data.parts, name)
    return frames


def harness_inputs(
    inputs: QbInputs, frames: Mapping[str, Mapping[int, pl.DataFrame]]
) -> HarnessInputs:
    """Return the real run's harness inputs: the published team fit's penalties and prior."""
    history = PriorHistory(lambda season: inputs.team_rows[season], first_season=FIRST_TEAM_SEASON)
    return HarnessInputs(
        team_rows=inputs.team_rows,
        candidate_rows=frames,
        starters={season: data.starters for season, data in inputs.seasons.items()},
        team_penalties=lambda season: history.cross_validated_fit(season - 1),
        season_prior=history.season_prior,
    )


def _read(path: Path, ledger: common.InputLedger) -> pl.DataFrame:
    """Read a published table, recording its hash."""
    ledger.record(path)
    return pl.read_parquet(path)


def _rating_gap(
    left: pl.DataFrame, right: pl.DataFrame, key: str, column: str
) -> tuple[float, int]:
    """Return the largest gap in ``column`` over the keys in both frames, and their count."""
    joined = left.select(key, column).join(right.select(key, column), on=key, suffix="_other")
    gap = common.as_float(
        joined.select((pl.col(column) - pl.col(f"{column}_other")).abs().max()).item()
    )
    return gap, joined.height


def check_reproduction(
    frames: Mapping[str, Mapping[int, pl.DataFrame]],
    ratings_dir: Path,
    window: Window,
    ledger: common.InputLedger,
) -> dict[str, float]:
    """Checks 1 and 2: A reproduces the published quarterback ratings and their weekly history.

    Raises:
        IntegrityError: If a full-season rating or a week's rating differs beyond 1e-9, or a
            published passer has no rating here.

    """
    season_gap = week_gap = 0.0
    weeks_compared = 0
    for season in range(window.penalty_season, window.last_season + 1):
        rows = frames["A"][season]
        fit = fit_qb_ratings(rows)
        published = _read(ratings_dir / f"{season}_qb_ratings.parquet", ledger)
        gap, matched = _rating_gap(fit.ratings, published, QB_ID_COLUMN, QB_RATING_COLUMN)
        if matched != published.height or gap > MATCH_TOLERANCE:
            msg = f"check 1: {season} differs by {gap:.3g} over {matched} of {published.height}"
            raise IntegrityError(msg)
        season_gap = max(season_gap, gap)
        if season < window.warmup_season:
            continue
        history = _read(ratings_dir / f"{season}_qb_ratings_by_week.parquet", ledger)
        history_weeks = sorted(history.get_column("week").unique().to_list())
        for week in sorted(rows.get_column("week").unique().to_list()):
            earlier = [value for value in history_weeks if value < week]
            if week < window.first_week or not earlier:
                continue
            refit = fit_qb_ratings(
                rows.filter(pl.col("week") < week), ridge_lambda=fit.ridge_lambda
            ).ratings
            gap, _ = _rating_gap(
                refit,
                history.filter(pl.col("week") == earlier[-1]),
                QB_ID_COLUMN,
                QB_RATING_COLUMN,
            )
            if gap > MATCH_TOLERANCE:
                msg = f"check 2: {season} week {week} differs from the history by {gap:.3g}"
                raise IntegrityError(msg)
            week_gap = max(week_gap, gap)
            weeks_compared += 1
    return {"season_gap": season_gap, "week_gap": week_gap, "weeks_compared": weeks_compared}


def check_parts_and_snaps(inputs: QbInputs) -> dict[str, float]:
    """Checks 3 and 4: every per-season fact is zero and the target recount matches.

    Raises:
        IntegrityError: If a play part differs from the season statistics, an added play is not
            a scrimmage snap, or the target recount differs.

    """
    totals: dict[str, float] = {}
    for season, data in inputs.seasons.items():
        for name, value in {**data.parts_facts, **data.snap_facts}.items():
            if name == "target_gap":
                totals[name] = max(totals.get(name, 0.0), value)
                limit = MATCH_TOLERANCE
            elif name == "qb_games":
                totals[name] = totals.get(name, 0.0) + value
                continue
            else:
                totals[name] = totals.get(name, 0.0) + value
                limit = 0.0
            if value > limit:
                msg = f"check {3 if name in data.parts_facts else 4}: {season} {name} = {value:g}"
                raise IntegrityError(msg)
    return totals


def check_team_history(
    audit: HarnessAudit, ratings_dir: Path, ledger: common.InputLedger
) -> dict[str, float]:
    """Check 5, second half: each team snapshot equals the published history of the week before.

    Raises:
        IntegrityError: If the effects behind ``t`` miss the snapshot's ratings by 1e-12, or a
            snapshot misses the history by 1e-9.

    """
    if audit.team_term_gap > INFORMATION_TOLERANCE:
        msg = f"check 5: the team term misses its snapshot's ratings by {audit.team_term_gap:.3g}"
        raise IntegrityError(msg)
    gap = 0.0
    compared = 0
    histories: dict[int, pl.DataFrame] = {}
    for (season, week), ratings in sorted(audit.team_ratings.items()):
        if season not in histories:
            histories[season] = _read(ratings_dir / f"{season}_ratings_by_week.parquet", ledger)
        history = histories[season]
        earlier = sorted(
            value for value in history.get_column("week").unique().to_list() if value < week
        )
        if not earlier:
            continue
        week_rows = history.filter(pl.col("week") == earlier[-1])
        for column in ("offense_rating", "defense_rating", "special_teams_rating", "team_rating"):
            column_gap, _ = _rating_gap(ratings, week_rows, "team", column)
            gap = max(gap, column_gap)
        compared += 1
    if gap > MATCH_TOLERANCE:
        msg = f"check 5: team snapshots miss the published history by {gap:.3g}"
        raise IntegrityError(msg)
    return {"term_gap": audit.team_term_gap, "history_gap": gap, "snapshots": float(compared)}


def check_zero_addition(
    frames: Mapping[str, Mapping[int, pl.DataFrame]], predictions: pl.DataFrame
) -> float:
    """Check 6: with the added plays zeroed, C's rows and predictions are A's.

    Raises:
        IntegrityError: If a season's rows differ or a prediction differs beyond 1e-12.

    """
    for season, rows in frames[ZERO].items():
        if not rows.equals(frames["A"][season]):
            msg = f"check 6: zeroed rows of {season} differ from passes and sacks"
            raise IntegrityError(msg)
    keys = ["season", "week", "game_id", "team"]
    zero = predictions.filter(pl.col("candidate") == ZERO).select(*keys, "p")
    joined = (
        predictions.filter(pl.col("candidate") == "A")
        .select(*keys, "p")
        .join(zero, on=keys, suffix="_zero")
    )
    gap = common.as_float(joined.select((pl.col("p") - pl.col("p_zero")).abs().max()).item())
    if joined.height != zero.height or gap > INFORMATION_TOLERANCE:
        msg = f"check 6: zeroed predictions differ from passes and sacks by {gap:.3g}"
        raise IntegrityError(msg)
    return gap


def check_penalties(frames: Mapping[str, Mapping[int, pl.DataFrame]], audit: HarnessAudit) -> int:
    """Check 7: every season's penalty is a separate cross-validation of the previous season.

    Raises:
        IntegrityError: If a penalty differs.

    """
    checked = 0
    for name, by_season in audit.lambdas.items():
        for season, used in by_season.items():
            separate = fit_unit_ridge(
                qb_rating_rows(frames[name][season - 1]), QB_UNIT_COLUMNS
            ).ridge_lambda
            if separate != used:
                msg = f"check 7: {name}'s {season} penalty {used} is not {separate}"
                raise IntegrityError(msg)
            checked += 1
    return checked


def schedule_starters(
    path: Path, seasons: Sequence[int], ledger: common.InputLedger
) -> pl.DataFrame:
    """Return the schedule's starter ids per regular-season team-game, canonical team codes."""
    raw = _read(path, ledger).filter(
        (pl.col("game_type") == "REG") & pl.col("season").is_in(list(seasons))
    )
    return pl.concat(
        [
            raw.select(
                pl.col("season").cast(pl.Int64),
                "game_id",
                pl.col(f"{side}_team").replace(constants.ALIAS_TO_CANONICAL).alias("team"),
                pl.col(f"{side}_qb_id").alias("listed_id"),
            )
            for side in ("home", "away")
        ]
    )


def rows_report(inputs: QbInputs, listed: pl.DataFrame, window: Window) -> list[dict[str, Any]]:
    """Check 9's table: team-games, unknown starters, schedule disagreements and off-row runs."""
    table: list[dict[str, Any]] = []
    for season in window.predicted_seasons:
        data = inputs.seasons[season]
        games = inputs.team_rows[season].select("game_id", "team")
        starters = games.join(data.starters, on=["game_id", "team"], how="left")
        season_listed = listed.filter(pl.col("season") == season).drop("season")
        compared = starters.join(season_listed, on=["game_id", "team"], how="left")
        known = compared.filter(pl.col("qb_id").is_not_null())
        rows_with_ids = data.qb_rows.select(GAME_KEYS)
        listed_rows = season_listed.rename({"listed_id": "qb_id"})
        no_row = listed_rows.filter(pl.col("qb_id").is_not_null()).join(
            rows_with_ids, on=GAME_KEYS, how="anti"
        )
        off_rows = with_parts(data.qb_rows.filter(pl.col("qb_dropbacks") == 0), data.parts)
        table.append(
            {
                "season": season,
                "team_games": games.height,
                "unknown_starters": starters.filter(pl.col("qb_id").is_null()).height,
                "listed_without_qb_row": no_row.height,
                "first_snap_differs": known.filter(
                    pl.col("qb_id").ne_missing(pl.col("listed_id"))
                ).height,
                "qb_rows_without_dropback": off_rows.height,
                "runs_on_those_rows": int(
                    off_rows.select(
                        pl.sum_horizontal("scrambles", "designed_runs", "charged_aborted").sum()
                    ).item()
                ),
            }
        )
    return table


def check_stability(ratings_dir: Path, ledger: common.InputLedger) -> tuple[float, int]:
    """Check 10: A's year-over-year Pearson from the published tables reproduces the source's.

    Raises:
        IntegrityError: If the Pearson (to three decimals) or the pair count differs.

    """
    pairs: list[tuple[float, float]] = []
    for season in range(FIRST_TEAM_SEASON, REAL_WINDOW.last_season):
        current = _read(ratings_dir / f"{season}_qb_ratings.parquet", ledger)
        following = _read(ratings_dir / f"{season + 1}_qb_ratings.parquet", ledger)
        joined = current.select(QB_ID_COLUMN, QB_RATING_COLUMN).join(
            following.select(QB_ID_COLUMN, QB_RATING_COLUMN), on=QB_ID_COLUMN, suffix="_next"
        )
        pairs.extend(joined.select(QB_RATING_COLUMN, f"{QB_RATING_COLUMN}_next").iter_rows())
    values = np.array(pairs, dtype=np.float64)
    pearson = float(np.corrcoef(values.T)[0, 1])
    if round(pearson, 3) != SOURCE_STABILITY[0] or len(pairs) != SOURCE_STABILITY[1]:
        msg = f"check 10: Pearson {pearson:.4f} over {len(pairs)} pairs, not 0.455 over 601"
        raise IntegrityError(msg)
    return pearson, len(pairs)


def parts_fingerprint(inputs: QbInputs) -> str:
    """Return a SHA-256 over every season's play parts and starters, sorted, as CSV."""
    parts = pl.concat(
        [
            data.parts.with_columns(pl.lit(season).alias("season"))
            for season, data in inputs.seasons.items()
        ]
    ).sort("season", *GAME_KEYS)
    starters = pl.concat(
        [
            data.starters.with_columns(pl.lit(season).alias("season"))
            for season, data in inputs.seasons.items()
        ]
    ).sort("season", "game_id", "team")
    text = parts.write_csv() + starters.write_csv()
    return hashlib.sha256(text.encode()).hexdigest()


# --- scoring ---------------------------------------------------------------------------------

_KEYS = ["season", "week", "game_id", "team"]


def all_targets(inputs: QbInputs) -> pl.DataFrame:
    """Return every team-game's targets, with its season."""
    return pl.concat(
        [
            data.targets.with_columns(pl.lit(season, dtype=pl.Int64).alias("season"))
            for season, data in inputs.seasons.items()
        ]
    )


def own_targets(frames: Mapping[str, Mapping[int, pl.DataFrame]]) -> pl.DataFrame:
    """Return each fit row's own EPA per play under A's and C's play sets, with its plays."""
    parts: list[pl.DataFrame] = []
    for name, suffix in (("A", "a"), ("C", "c")):
        rows = pl.concat(
            [
                frame.select(
                    pl.lit(season, dtype=pl.Int64).alias("season"),
                    "game_id",
                    "team",
                    pl.col("qb_id").alias("starter"),
                    (pl.col("qb_dropback_epa") / pl.col("qb_dropbacks")).alias(f"own_{suffix}"),
                    pl.col("qb_dropbacks").cast(pl.Float64).alias(f"plays_{suffix}"),
                )
                for season, frame in frames[name].items()
            ]
        )
        parts.append(rows)
    return parts[0].join(parts[1], on=["season", "game_id", "team", "starter"], how="left")


def with_targets(
    predictions: pl.DataFrame, targets: pl.DataFrame, own: pl.DataFrame
) -> pl.DataFrame:
    """Return the predictions with each team-game's targets and its starter's own targets."""
    return predictions.join(targets, on=["season", "game_id", "team"], how="left").join(
        own, on=["season", "game_id", "team", "starter"], how="left"
    )


def calibrated(
    rows: pl.DataFrame, names: Sequence[str], features: Sequence[str], target: str
) -> dict[str, pl.DataFrame]:
    """Return each candidate's calibrated rows for one target and model."""
    usable = rows.filter(pl.col(target).is_not_null())
    return {
        name: calibrate(usable.filter(pl.col("candidate") == name), features, target)
        for name in names
    }


def scored(rows: pl.DataFrame, window: Window) -> pl.DataFrame:
    """Return the scored rows: the scored seasons, from the start week."""
    return rows.filter(
        (pl.col("season") >= window.first_scored) & (pl.col("week") >= window.start_week)
    )


def band_rows(rows: pl.DataFrame, band: str) -> pl.DataFrame:
    """Return the rows of one week band, or every row for ``OVERALL``."""
    for name, first, last in BANDS:
        if name == band:
            in_band = pl.col("week") >= first
            return rows.filter(in_band if last is None else in_band & (pl.col("week") <= last))
    return rows


def paired_errors(first: pl.DataFrame, second: pl.DataFrame) -> pl.DataFrame:
    """Return each team-game's ``|error first| - |error second|``, sorted by team-game.

    The sort fixes the order the season sums add in, so two runs write the same bits.

    Raises:
        IntegrityError: If the two do not score the same team-games (check 9).

    """
    joined = first.select(*_KEYS, pl.col("error").abs().alias("_first")).join(
        second.select(*_KEYS, pl.col("error").abs().alias("_second")),
        on=_KEYS,
        how="full",
        coalesce=True,
    )
    if joined.select(pl.any_horizontal(pl.col("_first", "_second").is_null()).any()).item():
        msg = "check 9: two candidates do not score the same team-games"
        raise IntegrityError(msg)
    return joined.select(*_KEYS, (pl.col("_first") - pl.col("_second")).alias("difference")).sort(
        _KEYS
    )


def compare(
    first: pl.DataFrame,
    second: pl.DataFrame,
    seasons: Sequence[int],
    draws: npt.NDArray[np.int64],
    confidence: float,
) -> Comparison:
    """Return the season-bootstrap comparison of two candidates' scored rows."""
    differences = paired_errors(first, second)
    present = sorted(set(differences.get_column("season").to_list()))
    if list(present) != list(seasons):
        msg = f"check 9: the scored seasons {present} are not {list(seasons)}"
        raise IntegrityError(msg)
    return Comparison(
        overall=common.season_bootstrap(differences, seasons, draws, confidence),
        bands={
            name: common.season_bootstrap(band_rows(differences, name), seasons, draws, confidence)
            for name, _, _ in BANDS
        },
    )


def compare_subset(first: pl.DataFrame, second: pl.DataFrame, confidence: float) -> Interval | None:
    """Return a season-bootstrap interval over a subset, on the seasons the subset holds."""
    differences = paired_errors(first, second)
    if differences.is_empty():
        return None
    seasons = sorted(set(differences.get_column("season").to_list()))
    draws = common.season_draws(len(seasons))
    return common.season_bootstrap(differences, seasons, draws, confidence)


def t_interval(
    first: pl.DataFrame, second: pl.DataFrame, seasons: Sequence[int], confidence: float
) -> Interval:
    """Return the Student-t interval on the season means of the paired differences."""
    return common.season_t_interval(paired_errors(first, second), seasons, confidence)


def score(rows: pl.DataFrame, weight: str | None = None) -> dict[str, tuple[int, float, float]]:
    """Return games, MAE and RMSE overall and by band (weighted by ``weight`` when given)."""
    result: dict[str, tuple[int, float, float]] = {}
    for band in (OVERALL, *(name for name, _, _ in BANDS)):
        part = band_rows(rows, band)
        errors = part.get_column("error").to_numpy()
        weights = np.ones(errors.size) if weight is None else part.get_column(weight).to_numpy()
        if errors.size == 0 or weights.sum() == 0:
            result[band] = (0, float("nan"), float("nan"))
            continue
        result[band] = (
            int(errors.size),
            float(np.average(np.abs(errors), weights=weights)),
            float(np.sqrt(np.average(errors**2, weights=weights))),
        )
    return result


def mean_coefficients(rows: pl.DataFrame, features: Sequence[str]) -> dict[str, dict[str, float]]:
    """Return each feature's mean fitted coefficient over the scored rows, overall and by band."""
    return {
        band: {
            name: common.as_float(band_rows(rows, band).get_column(f"coef_{name}").mean())
            for name in features
        }
        for band in (OVERALL, *(name for name, _, _ in BANDS))
    }


def starter_changes(inputs: QbInputs) -> pl.DataFrame:
    """Return the team-games whose starter differs from the team's previous game's that season."""
    frames: list[pl.DataFrame] = []
    for season, data in inputs.seasons.items():
        games = inputs.team_rows[season].select("game_id", "week", "team")
        frames.append(
            games.join(data.starters, on=["game_id", "team"], how="left")
            .sort("team", "week")
            .with_columns(pl.col("qb_id").shift(1).over("team").alias("_previous"))
            .filter(
                pl.col("qb_id").is_not_null()
                & pl.col("_previous").is_not_null()
                & (pl.col("qb_id") != pl.col("_previous"))
            )
            .select(pl.lit(season, dtype=pl.Int64).alias("season"), "game_id", "team")
        )
    return pl.concat(frames)


def full_season_ratings(
    frames: Mapping[str, Mapping[int, pl.DataFrame]], name: str, season: int
) -> tuple[pl.DataFrame, QbRatingFit]:
    """Return a candidate's full-season ratings, its penalty cross-validated as published."""
    fit = fit_qb_ratings(frames[name][season])
    return fit.ratings, fit


def eligible_passers(inputs: QbInputs, season: int) -> list[str]:
    """Return the passers the rankings qualify in a season (A's attempts, as published)."""
    summary = qb_season_summary(inputs.seasons[season].qb_rows, inputs.team_rows[season])
    return summary.filter(pl.col("qb_is_eligible")).get_column(QB_ID_COLUMN).to_list()


def stability(
    inputs: QbInputs, frames: Mapping[str, Mapping[int, pl.DataFrame]], name: str, window: Window
) -> dict[str, float]:
    """Return a candidate's year-over-year Pearson and Spearman over qualified passers."""
    pairs: list[tuple[float, float]] = []
    ratings = {
        season: dict(
            full_season_ratings(frames, name, season)[0]
            .select(QB_ID_COLUMN, QB_RATING_COLUMN)
            .iter_rows()
        )
        for season in window.predicted_seasons
    }
    for season in window.predicted_seasons[:-1]:
        both = set(eligible_passers(inputs, season)) & set(eligible_passers(inputs, season + 1))
        pairs.extend(
            (ratings[season][passer], ratings[season + 1][passer])
            for passer in sorted(both)
            if passer in ratings[season] and passer in ratings[season + 1]
        )
    values = np.array(pairs, dtype=np.float64)
    ranks = pl.DataFrame({"x": values[:, 0], "y": values[:, 1]}).select(
        pl.col("x").rank("average"), pl.col("y").rank("average")
    )
    return {
        "pairs": float(len(pairs)),
        "pearson": float(np.corrcoef(values.T)[0, 1]),
        "spearman": float(np.corrcoef(ranks.to_numpy().T)[0, 1]),
    }


def last_season_spotlight(
    inputs: QbInputs,
    frames: Mapping[str, Mapping[int, pl.DataFrame]],
    names: Sequence[str],
    season: int,
) -> dict[str, Any]:
    """Return each candidate's faced pass defense (and its correlation with A's) and two ranks."""
    eligible = eligible_passers(inputs, season)
    qb_names = dict(inputs.seasons[season].qb_rows.select(QB_ID_COLUMN, "qb_name").iter_rows())
    result: dict[str, Any] = {"faced_correlation": {}, "ranks": {}}
    faced: dict[str, dict[str, float | None]] = {}
    for name in names:
        ratings, fit = full_season_ratings(frames, name, season)
        defense = compute_qb_faced_pass_defense(frames[name][season], fit)
        faced[name] = dict(defense.select(QB_ID_COLUMN, QB_FACED_DEFENSE_COLUMN).iter_rows())
        ranked = ratings.filter(pl.col(QB_ID_COLUMN).is_in(eligible)).with_columns(
            pl.col(QB_RATING_COLUMN).rank("min", descending=True).alias("rank")
        )
        result["ranks"][name] = {
            qb_names.get(passer, passer): (int(rank), float(value))
            for passer, value, rank in ranked.select(
                QB_ID_COLUMN, QB_RATING_COLUMN, "rank"
            ).iter_rows()
            if qb_names.get(passer) in SPOTLIGHT_PASSERS
        }
    for name in names:
        common_ids = [
            passer
            for passer in eligible
            if faced[name].get(passer) is not None and faced["A"].get(passer) is not None
        ]
        values = np.array(
            [[faced[name][passer], faced["A"][passer]] for passer in common_ids], dtype=np.float64
        )
        result["faced_correlation"][name] = float(np.corrcoef(values.T)[0, 1])
    return result


SPOTLIGHT_PASSERS = ("Drake Maye", "Josh Allen")


# --- the power estimate ----------------------------------------------------------------------

POWER_WINDOWS = 100
POWER_SEED = 1
MIN_TYPE_PLAYS = 10
_PLAY_KINDS = ("dropback", "scramble", "designed", "botched", "other")


@dataclass(frozen=True)
class Spreads:
    """The synthetic league's spreads and means, per play, estimated from the scored seasons.

    ``designed_unit`` says who owns the designed-run effect: each passer-season (the estimate),
    each team-season (every quarterback of a team shares it) or nobody (pure noise).
    """

    offense: float
    defense: float
    home_field: float
    passer: float
    scramble: float
    designed: float
    play: float
    means: Mapping[str, float]
    designed_unit: str = "passer"


@dataclass(frozen=True)
class SimSeason:
    """One season's real structure: team rows, fit rows with their play counts, starters."""

    team_rows: pl.DataFrame
    fit_rows: pl.DataFrame
    starters: pl.DataFrame


def sim_structure(inputs: QbInputs) -> dict[int, SimSeason]:
    """Return each quarterback season's real structure for the synthetic league."""
    structure: dict[int, SimSeason] = {}
    for season, data in inputs.seasons.items():
        rows = with_parts(data.qb_rows.filter(pl.col("qb_dropbacks") > 0), data.parts)
        structure[season] = SimSeason(
            team_rows=inputs.team_rows[season],
            fit_rows=rows.select(
                "game_id",
                "week",
                "team",
                "opponent_team",
                "is_home",
                "qb_id",
                "qb_name",
                "qb_attempts",
                pl.col("qb_dropbacks").alias("n_dropback"),
                pl.col("scrambles").alias("n_scramble"),
                pl.col("designed_runs").alias("n_designed"),
                pl.col("charged_aborted").alias("n_botched"),
            ),
            starters=data.starters,
        )
    return structure


def _weighted_sd(values: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]) -> float:
    """Return the weighted standard deviation of ``values``."""
    mean = np.average(values, weights=weights)
    return float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def _moments_sd(dispersion: pl.DataFrame, kind: str) -> float:
    """Return the between-passer spread of EPA per play of one kind, by the method of moments.

    Over passer-seasons with at least ``MIN_TYPE_PLAYS`` such plays: the variance of their rates
    less the mean sampling variance (the pooled within-passer variance over each count), floored
    at zero.
    """
    rows = dispersion.filter(pl.col(f"{kind}_n") >= MIN_TYPE_PLAYS)
    counts = rows.get_column(f"{kind}_n").cast(pl.Float64).to_numpy()
    sums = rows.get_column(f"{kind}_sum").to_numpy()
    squares = rows.get_column(f"{kind}_sumsq").to_numpy()
    rates = sums / counts
    within = float((squares - sums**2 / counts).sum() / (counts - 1.0).sum())
    between = float(rates.var(ddof=1)) - float(np.mean(within / counts))
    return float(np.sqrt(max(between, 0.0)))


def estimate_spreads(
    inputs: QbInputs, frames: Mapping[str, Mapping[int, pl.DataFrame]], seasons: Sequence[int]
) -> Spreads:
    """Return the spreads the scored seasons show, for the synthetic league."""
    history = PriorHistory(lambda season: inputs.team_rows[season], first_season=FIRST_TEAM_SEASON)
    offense: list[float] = []
    defense: list[float] = []
    home: list[float] = []
    residual_sum = 0.0
    residual_rows = 0
    passer_effects: list[float] = []
    passer_weights: list[float] = []
    totals = dict.fromkeys(_PLAY_KINDS, 0.0)
    counts = dict.fromkeys(_PLAY_KINDS, 0.0)
    for season in seasons:
        fit = history.published_scrimmage(season)
        offense.extend(fit.offense.values())
        defense.extend(fit.defense.values())
        home.append(fit.home_field)
        rows = scrimmage_rows(inputs.team_rows[season])
        predicted = predict_unit(fit, rows, TEAM_UNIT_COLUMNS)
        residuals = rows.get_column("epa_per_play").to_numpy() - predicted
        residual_sum += float((rows.get_column("plays").to_numpy() * residuals**2).sum())
        residual_rows += rows.height
        unit = fit_unit_ridge(
            qb_rating_rows(frames["A"][season]),
            QB_UNIT_COLUMNS,
            ridge_lambda=fit_qb_ratings(frames["A"][season]).ridge_lambda,
        )
        volume = dict(
            frames["A"][season].group_by(QB_ID_COLUMN).agg(pl.col("qb_dropbacks").sum()).iter_rows()
        )
        for passer, effect in unit.offense.items():
            passer_effects.append(effect)
            passer_weights.append(float(volume[passer]))
        data = inputs.seasons[season]
        parts = with_parts(data.qb_rows.filter(pl.col("qb_dropbacks") > 0), data.parts)
        sums = parts.select(
            pl.col("qb_dropbacks").sum(),
            pl.col("qb_dropback_epa").sum(),
            pl.col("scrambles").sum(),
            pl.col("scramble_epa").sum(),
            pl.col("designed_runs").sum(),
            pl.col("designed_run_epa").sum(),
            pl.col("charged_aborted").sum(),
            pl.col("charged_aborted_epa").sum(),
        ).row(0)
        for index, kind in enumerate(_PLAY_KINDS[:4]):
            counts[kind] += float(sums[2 * index])
            totals[kind] += float(sums[2 * index + 1])
        team_rows = inputs.team_rows[season]
        counts["other"] += float(team_rows.get_column("offensive_snaps").sum()) - sum(
            float(sums[2 * index]) for index in range(4)
        )
        totals["other"] += float(team_rows.get_column("offensive_epa").sum()) - sum(
            float(sums[2 * index + 1]) for index in range(4)
        )
    dispersion = pl.concat([inputs.seasons[season].dispersion for season in seasons])
    return Spreads(
        offense=float(np.std(offense, ddof=1)),
        defense=float(np.std(defense, ddof=1)),
        home_field=float(np.mean(home)),
        passer=_weighted_sd(np.array(passer_effects), np.array(passer_weights)),
        scramble=_moments_sd(dispersion, "scramble"),
        designed=_moments_sd(dispersion, "designed"),
        play=float(np.sqrt(residual_sum / residual_rows)),
        means={kind: totals[kind] / counts[kind] for kind in _PLAY_KINDS},
    )


def _draw(rng: np.random.Generator, keys: Sequence[str], spread: float) -> dict[str, float]:
    """Return one normal effect per key, mean zero, in the keys' order."""
    return dict(zip(keys, rng.normal(0.0, spread, len(keys)).tolist(), strict=True))


def simulate_season(
    season: SimSeason, spreads: Spreads, rng: np.random.Generator
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return one synthetic season: team rows, quarterback rows and play parts.

    Every play's EPA is its kind's mean plus the offense's effect, less the defense's, plus home
    field times the home sign, plus the passer's effect for his kind (none for botched snaps and
    the team's other plays), plus normal noise; a group of ``n`` plays gets ``n`` times the mean
    and noise with ``sqrt(n)`` times the per-play spread.
    """
    teams = sorted(season.team_rows.get_column("team").unique().to_list())
    offense = _draw(rng, teams, spreads.offense)
    defense = _draw(rng, teams, spreads.defense)
    passers = sorted(season.fit_rows.get_column("qb_id").unique().to_list())
    passer = _draw(rng, passers, spreads.passer)
    scramble = _draw(rng, passers, spreads.scramble)
    owners = teams if spreads.designed_unit == "team" else passers
    designed = _draw(rng, owners, spreads.designed) if spreads.designed_unit != "none" else {}
    means, play = spreads.means, spreads.play

    def group(count: float, mean: float) -> float:
        return count * mean + play * float(np.sqrt(count)) * float(rng.normal())

    qb_values: list[dict[str, object]] = []
    team_qb_epa: dict[tuple[str, str], tuple[float, float]] = {}
    for row in season.fit_rows.iter_rows(named=True):
        base = (
            offense[row["team"]]
            - defense[row["opponent_team"]]
            + spreads.home_field * _home_sign(row)
        )
        owner = row["team"] if spreads.designed_unit == "team" else row["qb_id"]
        values = {
            "dropback": group(row["n_dropback"], means["dropback"] + base + passer[row["qb_id"]]),
            "scramble": group(row["n_scramble"], means["scramble"] + base + scramble[row["qb_id"]]),
            "designed": group(
                row["n_designed"], means["designed"] + base + designed.get(owner, 0.0)
            ),
            "botched": group(row["n_botched"], means["botched"] + base),
        }
        plays = row["n_dropback"] + row["n_scramble"] + row["n_designed"] + row["n_botched"]
        key = (row["game_id"], row["team"])
        total_plays, total_epa = team_qb_epa.get(key, (0.0, 0.0))
        team_qb_epa[key] = (total_plays + plays, total_epa + sum(values.values()))
        qb_values.append({**row, **{f"epa_{kind}": value for kind, value in values.items()}})
    team_values: list[dict[str, object]] = []
    for row in season.team_rows.iter_rows(named=True):
        base = (
            offense[row["team"]]
            - defense[row["opponent_team"]]
            + spreads.home_field * _home_sign(row)
        )
        qb_plays, qb_epa = team_qb_epa.get((row["game_id"], row["team"]), (0.0, 0.0))
        other = max(float(row["offensive_snaps"]) - qb_plays, 0.0)
        team_values.append({**row, "offensive_epa": qb_epa + group(other, means["other"] + base)})
    team_rows = pl.DataFrame(team_values, schema=season.team_rows.schema)
    qbs = pl.DataFrame(qb_values)
    qb_rows = qbs.select(
        "game_id",
        "week",
        "team",
        "opponent_team",
        "is_home",
        "qb_id",
        "qb_name",
        pl.col("n_dropback").alias("qb_dropbacks"),
        pl.col("epa_dropback").alias("qb_dropback_epa"),
        "qb_attempts",
    )
    parts = qbs.select(
        "game_id",
        "team",
        "qb_id",
        *(
            pl.col(source).alias(column)
            for column, source in (
                ("scrambles", "n_scramble"),
                ("scramble_epa", "epa_scramble"),
                ("designed_runs", "n_designed"),
                ("designed_run_epa", "epa_designed"),
                ("charged_aborted", "n_botched"),
                ("charged_aborted_epa", "epa_botched"),
            )
        ),
    ).with_columns(
        pl.lit(0).alias(column)
        for column in PART_COLUMNS
        if column
        not in {
            "scrambles",
            "scramble_epa",
            "designed_runs",
            "designed_run_epa",
            "charged_aborted",
            "charged_aborted_epa",
        }
    )
    return team_rows, qb_rows, parts


def simulate_window(
    structure: Mapping[int, SimSeason],
    spreads: Spreads,
    window: Window,
    rng: np.random.Generator,
    draws: npt.NDArray[np.int64],
) -> Decision:
    """Return the rule's decision on one synthetic window run through the real harness."""
    team_rows: dict[int, pl.DataFrame] = {}
    frames: dict[str, dict[int, pl.DataFrame]] = {name: {} for name in CANDIDATES}
    targets: list[pl.DataFrame] = []
    for season, data in structure.items():
        rows, qb_rows, parts = simulate_season(data, spreads, rng)
        team_rows[season] = rows
        for name in CANDIDATES:
            frames[name][season] = candidate_rows(qb_rows, parts, name)
        targets.append(
            rows.select(
                pl.lit(season, dtype=pl.Int64).alias("season"),
                "game_id",
                "team",
                (pl.col("offensive_epa") / pl.col("offensive_snaps")).alias("target"),
            )
        )
    history = PriorHistory(lambda season: team_rows[season], first_season=min(team_rows))
    inputs = HarnessInputs(
        team_rows=team_rows,
        candidate_rows=frames,
        starters={season: data.starters for season, data in structure.items()},
        team_penalties=lambda season: history.cross_validated_fit(season - 1),
        season_prior=lambda _season: None,
    )
    result = run_harness(inputs, window, audit_fits=False)
    rows = result.predictions.join(pl.concat(targets), on=["season", "game_id", "team"])
    calibrations = calibrated(rows, CANDIDATES, ("p", "t"), "target")
    scored_rows = {name: scored(frame, window) for name, frame in calibrations.items()}
    confidence = common.comparison_confidence(len(PAIRS))
    comparisons = {
        pair: compare(
            scored_rows[pair[0]], scored_rows[pair[1]], window.scored_seasons, draws, confidence
        )
        for pair in PAIRS
    }
    return decide(comparisons, ORDER)


def power(
    structure: Mapping[int, SimSeason],
    spreads: Spreads,
    window: Window,
    *,
    windows: int = POWER_WINDOWS,
    seed: int = POWER_SEED,
) -> dict[str, float]:
    """Return the share of synthetic windows in which B beats A, C beats A and C beats B."""
    rng = np.random.default_rng(seed)
    draws = common.season_draws(len(window.scored_seasons))
    wins = dict.fromkeys(PAIRS, 0)
    for _ in range(windows):
        decision = simulate_window(structure, spreads, window, rng, draws)
        for first, second in PAIRS:
            if (first, second) in decision.wins:
                wins[(first, second)] += 1
    return {f"{first} beats {second}": wins[(first, second)] / windows for first, second in PAIRS}


# --- the report ------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
OUT_MD = "QB_PLAYS_TEST.md"
OUT_JSON = "qb_plays_test.json"
SCORED = (*CANDIDATES, WITHOUT_BOTCHED)
DEFAULT_SCHEDULES = (
    Path.home()
    / "scratch"
    / "nflreadpy_cache_snapshot_20261008"
    / "a7d0d7a2405b3d35047e8de4a40035e6.parquet"
)
# The feature-value impact counts the starter ratings a candidate moves by more than this.
IMPACT_CUT = 0.02


def _interval_text(interval: Interval) -> str:
    """Format a paired difference and its interval in scientific notation.

    The differences in EPA per play are of order 1e-5, which fixed decimals would print as zero.
    """
    return f"{interval.estimate:+.2e} ({interval.lower:+.2e} to {interval.upper:+.2e})"


def _pair_label(first: str, second: str) -> str:
    """Return a pair's plain name, ``X - Y``."""
    return f"{LABELS[first]} - {LABELS[second]}"


def _comparison_table(comparisons: Mapping[tuple[str, str], Comparison]) -> list[str]:
    """Return a table of paired differences, overall and by band."""
    bands = [OVERALL, *(name for name, _, _ in BANDS)]
    lines = ["| pair | " + " | ".join(bands) + " |", "| --- |" + " --- |" * len(bands)]
    for (first, second), comparison in comparisons.items():
        cells = [_interval_text(comparison.overall)] + [
            _interval_text(comparison.bands[band]) for band in bands[1:]
        ]
        lines.append(f"| {_pair_label(first, second)} | " + " | ".join(cells) + " |")
    return lines


def _score_table(scores: Mapping[str, Mapping[str, tuple[int, float, float]]]) -> list[str]:
    """Return a table of team-games, MAE and RMSE by band."""
    lines = ["| candidate | band | team-games | MAE | RMSE |", "| --- | --- | --- | --- | --- |"]
    for name, by_band in scores.items():
        lines.extend(
            f"| {LABELS.get(name, name)} | {band} | {games:,} | {mae:.6f} | {rmse:.6f} |"
            for band, (games, mae, rmse) in by_band.items()
        )
    return lines


def starter_rating_impact(starter_ratings: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Return how far each candidate's pre-week starter rating moves from passes and sacks'."""
    impact: dict[str, dict[str, float]] = {}
    for name in ("B", "C", WITHOUT_BOTCHED):
        moved = (pl.col(f"r_{name}") - pl.col("r_A")).abs()
        rows = starter_ratings.filter(
            pl.col(f"r_{name}").is_not_null() & pl.col("r_A").is_not_null()
        )
        impact[name] = {
            "rows": float(rows.height),
            "mean": common.as_float(rows.select(moved.mean()).item()),
            "largest": common.as_float(rows.select(moved.max()).item()),
            "beyond_cut": float(rows.filter(moved > IMPACT_CUT).height),
        }
    return impact


@dataclass(frozen=True)
class Results:
    """Everything the report reads once the integrity checks have passed."""

    integrity: Mapping[str, Any]
    scores: Mapping[str, Mapping[str, tuple[int, float, float]]]
    comparisons: Mapping[tuple[str, str], Comparison]
    decision: Decision
    extras: Mapping[str, Any]


def _headline(results: Results) -> list[str]:
    """Return the headline bullets: each candidate's MAE and the three paired differences."""
    overall = {name: results.scores[name][OVERALL] for name in CANDIDATES}
    lines = [
        f"- MAE of a team's offensive EPA per play over {overall['A'][0]:,} team-games "
        "(2007-2025, weeks 5 on): "
        + "; ".join(f"{LABELS[name]} {overall[name][1]:.6f}" for name in CANDIDATES)
        + ".",
    ]
    lines.extend(
        f"- {_pair_label(first, second)}: {_interval_text(comparison.overall)}."
        for (first, second), comparison in results.comparisons.items()
    )
    decision = results.decision
    wins = "; ".join(f"{LABELS[a]} beats {LABELS[b]}" for a, b in decision.wins) or "none"
    if decision.recommended is None:
        lines.append(
            f"- Rule as written: wins: {wins}. Every candidate is beaten, so the result lies "
            "between the rule's branches: there is no recommendation, and it goes to the user."
        )
        return lines
    lines.append(
        f"- Rule as written: wins: {wins}. The recommendation is "
        f"**{LABELS[decision.recommended]}**, the first candidate in the user's order (every "
        "quarterback play, plus scrambles, passes and sacks) that no other candidate beats."
    )
    return lines


def _integrity_lines(integrity: Mapping[str, Any]) -> list[str]:
    """Return the integrity checks' results as numbered items."""
    reproduction = integrity["reproduction"]
    parts = integrity["parts_and_snaps"]
    team = integrity["team_term"]
    information = integrity["information"]
    stability = integrity["stability"]
    return [
        (
            "1. Passes and sacks' full-season fits of 2005-2025 equal the candidate build's "
            f"`{{season}}_qb_ratings.parquet` (largest gap {reproduction['season_gap']:.1e})."
        ),
        (
            "2. Its snapshots refit with the season's own penalty equal the published history of "
            f"the week before in {reproduction['weeks_compared']:,} weeks of 2006-2025 (largest "
            f"gap {reproduction['week_gap']:.1e}). The history is read only here, to check the "
            "port: every prediction uses the previous season's penalty."
        ),
        (
            f"3. On {int(parts['qb_games']):,} quarterback-games of 2005-2025 the play parts equal "
            "the ported season statistics: 0 scramble, kneel or carry mismatches, 0 rushing-EPA or "
            "passing-EPA gaps."
        ),
        (
            "4. Every added play is a scrimmage snap (0 outside), and the target recounted from "
            f"the play-by-play equals the adapter's (largest gap {parts['target_gap']:.1e})."
        ),
        (
            "5. The team term's effects equal its snapshot's ratings (largest gap "
            f"{team['term_gap']:.1e}), and {int(team['snapshots']):,} snapshots equal the "
            f"published team history of the week before (largest gap {team['history_gap']:.1e})."
        ),
        (
            "6. With the added plays zeroed, the rows equal passes and sacks' and the predictions "
            f"differ by at most {integrity['zero_addition_gap']:.1e}."
        ),
        (
            f"7. {integrity['penalties']:,} season penalties equal separate cross-validations of "
            "each candidate's previous season."
        ),
        (
            f"8. On {information['snapshots']:,} weekly snapshots, replacing every later game's "
            f"values moves no prediction (largest gap {information['gap']:.1e}); every snapshot "
            "equals `qb_rating.fit_qb_ratings` on the same rows (largest gap "
            f"{information['qb_fit_gap']:.1e}); every calibration used earlier weeks only."
        ),
        "9. Every candidate scores the same team-games; by season below.",
        (
            "10. Passes and sacks' year-over-year Pearson from the published tables is "
            f"{stability[0]:.4f} over {stability[1]} pairs (the source: 0.455 over 601). The QBR "
            "half is not run (no QBR file among the allowed inputs)."
        ),
        (
            "11. Every input's SHA-256 is recorded; the play parts and starters fingerprint is "
            f"`{integrity['parts_fingerprint']}`."
        ),
    ]


def _rows_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return check 9's table."""
    columns = (
        "season",
        "team_games",
        "unknown_starters",
        "listed_without_qb_row",
        "first_snap_differs",
        "qb_rows_without_dropback",
        "runs_on_those_rows",
    )
    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    lines.extend("| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows)
    return lines


def _calibration_lines(extras: Mapping[str, Any]) -> list[str]:
    """Return the calibration slopes, the one-term model and the team-only baseline."""
    lines = [
        "Mean fitted calibration slopes, `k1` on the quarterback term and `k2` on the team term:",
        "",
        "| candidate | band | k1 | k2 |",
        "| --- | --- | --- | --- |",
    ]
    for name, by_band in extras["coefficients"].items():
        lines.extend(
            f"| {LABELS[name]} | {band} | {values['p']:.3f} | {values['t']:.3f} |"
            for band, values in by_band.items()
        )
    team_only = extras["team_only"][OVERALL]
    return [
        *lines,
        "",
        "One-term model (`k * p_X + m`, the source draft's primary):",
        "",
        *_score_table(extras["one_term_scores"]),
        "",
        *_comparison_table(extras["one_term"]),
        "",
        (
            f"Team-only baseline (`k2 * t + m`): MAE {team_only[1]:.6f} over {team_only[0]:,} "
            "team-games."
        ),
    ]


def _band_lines(extras: Mapping[str, Any]) -> list[str]:
    """Return the starter-change band and the t-interval sensitivity."""
    lines = [
        (
            "Starter-change band (the starter differs from the team's previous game's), season "
            "bootstrap on the seasons it holds:"
        ),
        "",
    ]
    for (first, second), interval in extras["starter_change"].items():
        text = "no team-games" if interval is None else _interval_text(interval)
        lines.append(f"- {_pair_label(first, second)}: {text}.")
    lines += ["", "t-interval sensitivity on the 19 season means (same level):", ""]
    lines.extend(
        f"- {_pair_label(first, second)}: {_interval_text(interval)}."
        for (first, second), interval in extras["t_intervals"].items()
    )
    return lines


def _other_target_lines(extras: Mapping[str, Any]) -> list[str]:
    """Return the points-scored comparison and the starter's own EPA targets."""
    lines = [
        "Points scored (`points_for`, calibrated to points):",
        "",
        *_score_table(extras["points_scores"]),
        "",
        *_comparison_table(extras["points"]),
        "",
        (
            "The starter's own EPA per play in the game, play-weighted MAE (each target is one "
            "candidate's play set, so it favors that candidate):"
        ),
        "",
    ]
    for target, label in (
        ("own_a", "per dropback, which favors passes and sacks"),
        ("own_c", "per quarterback play, which favors every quarterback play"),
    ):
        cells = "; ".join(
            f"{LABELS[name]} {by_band[OVERALL][1]:.6f}" for name, by_band in extras[target].items()
        )
        lines.append(f"- {label}: {cells}.")
    return lines


def _rating_lines(extras: Mapping[str, Any]) -> list[str]:
    """Return stability, the 2025 spotlight and the feature-value impact."""
    lines = [
        (
            "Year-over-year stability of each full-season rating, 2006-2025, passers qualifying in "
            "both seasons:"
        ),
        "",
    ]
    lines.extend(
        f"- {LABELS[name]}: Pearson {values['pearson']:.3f}, Spearman {values['spearman']:.3f} "
        f"({int(values['pairs']):,} pairs)."
        for name, values in extras["stability"].items()
    )
    spotlight = extras["spotlight"]
    lines += [
        "",
        (
            "2025: faced pass defense against passes and sacks' (Pearson over qualified "
            "passers), and two passers' ranks and ratings:"
        ),
        "",
    ]
    for name in SCORED:
        ranks = "; ".join(
            f"{passer} {rank} ({value:+.3f})"
            for passer, (rank, value) in sorted(spotlight["ranks"][name].items())
        )
        lines.append(
            f"- {LABELS[name]}: faced-defense correlation "
            f"{spotlight['faced_correlation'][name]:.3f}; {ranks}."
        )
    lines += [
        "",
        (
            "Feature-value impact: each candidate's pre-week rating of the starter against passes "
            "and sacks' (the fit behind the `ratings_qb` feature, with the season's identity):"
        ),
        "",
    ]
    lines.extend(
        f"- {LABELS[name]}: {int(values['rows']):,} team-games, mean absolute change "
        f"{values['mean']:.4f}, largest {values['largest']:.4f} EPA per play, "
        f"{int(values['beyond_cut']):,} beyond {IMPACT_CUT}."
        for name, values in extras["impact"].items()
    )
    return lines


def _power_lines(power_result: Mapping[str, float], spreads: Spreads, windows: int) -> list[str]:
    """Return the power estimate and the spreads it drew from."""
    shares = []
    for key, share in power_result.items():
        first, _, second = key.split(" ")
        shares.append(f"{LABELS[first]} beats {LABELS[second]} {share:.0%}")
    return [
        (
            f"Over {windows} synthetic windows (seed {POWER_SEED}), the share in which the rule "
            f"finds each win: {'; '.join(shares)}."
        ),
        "",
        f"Spreads per play, from 2007-2025: team offense {spreads.offense:.4f}, defense "
        f"{spreads.defense:.4f}, home field {spreads.home_field:+.4f}; passer "
        f"{spreads.passer:.4f}, scramble {spreads.scramble:.4f}, designed run "
        f"{spreads.designed:.4f}; play noise {spreads.play:.4f}. Means: "
        + ", ".join(f"{kind} {value:+.3f}" for kind, value in spreads.means.items())
        + ".",
    ]


def build_report(
    results: Results, provenance: Mapping[str, Any], power_section: Sequence[str]
) -> list[str]:
    """Return the report's lines, wrapped for the Markdown linter."""
    excluding = "; ".join(
        f"{_pair_label(first, second)}, {band}: {direction}"
        for first, second, band, direction in results.decision.excluding_zero
    )
    extras = results.extras
    lines = [
        "# The quarterback rating from all of a quarterback's plays: test result",
        "",
        (
            "Pre-registered in `protocols/qb_all_plays.md` before any candidate was scored, a port "
            "of `nfl-sos-ratings`' roadmap section Q1 at `9081bce`. Candidates: passes and sacks "
            "(the rating as ported), plus scrambles, and every quarterback play."
        ),
        "",
        "## Headline",
        "",
        *_headline(results),
        "",
        "## Decision",
        "",
        f"- Intervals that exclude zero, either direction: {excluding or 'none'}.",
        (
            "- The decision goes to the user; adoption is held (it changes the `ratings` feature "
            "group and the published quarterback tables)."
        ),
        "",
        "## Paired differences (98.33% season-bootstrap intervals, 10,000 resamples, seed 0)",
        "",
        "Mean of `|error X| - |error Y|` per team-game; negative favors X.",
        "",
        *_comparison_table(results.comparisons),
        "",
        "## Integrity checks (all passed before any result was read)",
        "",
        *_integrity_lines(results.integrity),
        "",
        *_rows_table(results.integrity["rows"]),
        "",
        "## Power (descriptive, never changes the rule)",
        "",
        *power_section,
        "",
        "## Descriptive extras (never decision inputs)",
        "",
        "MAE and RMSE by band (every quarterback play without the botched snaps included):",
        "",
        *_score_table(results.scores),
        "",
        *_calibration_lines(extras),
        "",
        *_band_lines(extras),
        "",
        *_other_target_lines(extras),
        "",
        "Every quarterback play without the botched snaps, same draws and level:",
        "",
        *_comparison_table(extras["without_botched"]),
        "",
        *_rating_lines(extras),
        "",
        "## Not run",
        "",
        (
            "- The per-season correlations with ESPN QBR and the QBR half of check 10: no QBR file "
            "is cached among the inputs this test may read, and the script does not download."
        ),
        (
            "- The passer holdout: it needs postseason quarterback rows, which this repository's "
            "adapter does not build."
        ),
        "",
        "## Provenance",
        "",
        f"- nfl-predictor {provenance['version']}; code SHA-256: "
        + ", ".join(f"`{name}` {digest[:12]}" for name, digest in provenance["code"].items())
        + ".",
        (
            f"- Inputs: {provenance['inputs']} files, fingerprint `{provenance['fingerprint']}` "
            f"(each file's SHA-256 is in `{OUT_JSON}`), read only, downloads blocked."
        ),
        f"- Command: `{provenance['command']}`; {provenance['seconds']:.0f} seconds.",
        "",
    ]
    return common.wrap_markdown(lines)


# --- the run ---------------------------------------------------------------------------------


def run_integrity_checks(
    inputs: QbInputs,
    frames: Mapping[str, Mapping[int, pl.DataFrame]],
    result: HarnessResult,
    paths: tuple[Path, Path],
    ledger: common.InputLedger,
) -> dict[str, Any]:
    """Run checks 1-11 in order; the first that fails raises ``IntegrityError``.

    ``paths`` holds the published ratings directory and the raw schedule file.
    """
    ratings_dir, schedules_file = paths
    window = REAL_WINDOW
    if result.audit.qb_fit_gap > INFORMATION_TOLERANCE:
        msg = f"check 8: a snapshot misses qb_rating.fit_qb_ratings by {result.audit.qb_fit_gap}"
        raise IntegrityError(msg)
    keys = result.predictions.group_by("candidate").agg(pl.struct(_KEYS).sort().alias("_keys"))
    if keys.get_column("_keys").n_unique() != 1:
        msg = "check 9: the candidates do not predict the same team-games"
        raise IntegrityError(msg)
    return {
        "reproduction": check_reproduction(frames, ratings_dir, window, ledger),
        "parts_and_snaps": check_parts_and_snaps(inputs),
        "team_term": check_team_history(result.audit, ratings_dir, ledger),
        "zero_addition_gap": check_zero_addition(frames, result.predictions),
        "penalties": check_penalties(frames, result.audit),
        "information": {
            "snapshots": result.audit.information_snapshots,
            "gap": result.audit.information_gap,
            "qb_fit_gap": result.audit.qb_fit_gap,
        },
        "rows": rows_report(
            inputs, schedule_starters(schedules_file, window.predicted_seasons, ledger), window
        ),
        "stability": check_stability(ratings_dir, ledger),
        "parts_fingerprint": parts_fingerprint(inputs),
    }


def _scored_calibrations(
    rows: pl.DataFrame, features: Sequence[str], target: str, window: Window
) -> dict[str, pl.DataFrame]:
    """Return each scored candidate's calibrated, scored rows for one model and target."""
    return {
        name: scored(frame, window)
        for name, frame in calibrated(rows, SCORED, features, target).items()
    }


def evaluate(
    inputs: QbInputs,
    frames: Mapping[str, Mapping[int, pl.DataFrame]],
    result: HarnessResult,
    integrity: Mapping[str, Any],
    window: Window = REAL_WINDOW,
) -> Results:
    """Score the candidates, apply the rule and gather the descriptive extras."""
    seasons = window.scored_seasons
    rows = with_targets(result.predictions, all_targets(inputs), own_targets(frames))
    draws = common.season_draws(len(seasons))
    confidence = common.comparison_confidence(len(PAIRS))
    primary = _scored_calibrations(rows, ("p", "t"), "target", window)

    def pairwise(
        source: Mapping[str, pl.DataFrame], pairs: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], Comparison]:
        return {
            pair: compare(source[pair[0]], source[pair[1]], seasons, draws, confidence)
            for pair in pairs
        }

    comparisons = pairwise(primary, PAIRS)
    one_term = _scored_calibrations(rows, ("p",), "target", window)
    points = _scored_calibrations(rows, ("p", "t"), "points_for", window)
    own_a = _scored_calibrations(rows, ("p", "t"), "own_a", window)
    own_c = _scored_calibrations(rows, ("p", "t"), "own_c", window)
    team_only = scored(
        calibrate(
            rows.filter((pl.col("candidate") == "A") & pl.col("target").is_not_null()),
            ("t",),
            "target",
        ),
        window,
    )
    changes = starter_changes(inputs)
    keys = ["season", "game_id", "team"]
    extras: dict[str, Any] = {
        "coefficients": {name: mean_coefficients(primary[name], ("p", "t")) for name in SCORED},
        "one_term_scores": {name: score(one_term[name]) for name in SCORED},
        "one_term": pairwise(one_term, PAIRS),
        "team_only": score(team_only),
        "starter_change": {
            pair: compare_subset(
                primary[pair[0]].join(changes, on=keys, how="semi"),
                primary[pair[1]].join(changes, on=keys, how="semi"),
                confidence,
            )
            for pair in PAIRS
        },
        "t_intervals": {
            pair: t_interval(primary[pair[0]], primary[pair[1]], seasons, confidence)
            for pair in PAIRS
        },
        "points_scores": {name: score(points[name]) for name in SCORED},
        "points": pairwise(points, PAIRS),
        "own_a": {name: score(own_a[name], "plays_a") for name in SCORED},
        "own_c": {name: score(own_c[name], "plays_c") for name in SCORED},
        "without_botched": pairwise(primary, [(WITHOUT_BOTCHED, "A"), ("C", WITHOUT_BOTCHED)]),
        "stability": {name: stability(inputs, frames, name, window) for name in SCORED},
        "spotlight": last_season_spotlight(inputs, frames, SCORED, window.last_season),
        "impact": starter_rating_impact(result.starter_ratings),
    }
    return Results(
        integrity=integrity,
        scores={name: score(primary[name]) for name in SCORED},
        comparisons=comparisons,
        decision=decide(comparisons, ORDER),
        extras=extras,
    )


def _comparison_json(comparison: Comparison | Interval | None) -> object:
    """Return a comparison's or an interval's numbers for the JSON record."""
    if comparison is None:
        return None
    if isinstance(comparison, Comparison):
        return {
            OVERALL: _comparison_json(comparison.overall),
            **{band: _comparison_json(interval) for band, interval in comparison.bands.items()},
        }
    return [comparison.estimate, comparison.lower, comparison.upper]


def _json_ready(value: object) -> object:
    """Return ``value`` with comparisons, intervals and tuple keys in JSON-friendly form."""
    if isinstance(value, Comparison | common.Interval):
        return _comparison_json(value)
    if isinstance(value, Mapping):
        return {
            " - ".join(key) if isinstance(key, tuple) else str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def run(
    paths: tuple[Path, Path, Path], out_dir: Path, *, power_windows: int = POWER_WINDOWS
) -> int:
    """Run the test end to end and write its report; return the exit code.

    ``paths`` holds the nflreadpy cache, the published ratings directory and the raw schedule
    file.
    """
    cache_dir, ratings_dir, schedules_file = paths
    start = time.perf_counter()
    ledger = common.InputLedger()
    report = out_dir / OUT_MD
    with threadpool_limits(limits=1, user_api="blas"):
        inputs = load_qb_inputs(cache_dir, REAL_WINDOW, ledger)
        frames = candidate_frames(inputs, (*SCORED, ZERO))
        result = run_harness(harness_inputs(inputs, frames), REAL_WINDOW, check_information=True)
        try:
            integrity = run_integrity_checks(
                inputs, frames, result, (ratings_dir, schedules_file), ledger
            )
        except IntegrityError as error:
            report.write_text(
                "# The quarterback all-plays test: integrity check failed\n\n"
                f"No result is read. {error}\n"
            )
            sys.stderr.write(f"{error}\n")
            return 1
        results = evaluate(inputs, frames, result, integrity)
        spreads = estimate_spreads(inputs, frames, REAL_WINDOW.scored_seasons)
        power_result = power(sim_structure(inputs), spreads, REAL_WINDOW, windows=power_windows)
    code = [HERE / "qb_plays_test.py", HERE / "rating_tests_common.py"]
    command = (
        'VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/qb_plays_test.py '
        f"--cache-dir {cache_dir} --ratings-dir {ratings_dir} --schedules-file {schedules_file}"
        + ("" if power_windows == POWER_WINDOWS else f" --power-windows {power_windows}")
    )
    provenance = {
        "version": metadata.version("nfl-predictor"),
        "code": {path.name: common.sha256_file(path) for path in code},
        "inputs": len(ledger.files),
        "fingerprint": ledger.fingerprint(),
        "command": command,
        "seconds": time.perf_counter() - start,
    }
    lines = build_report(results, provenance, _power_lines(power_result, spreads, power_windows))
    report.write_text("\n".join(lines).rstrip("\n") + "\n")
    payload = {
        "integrity": results.integrity,
        "decision": {
            "recommended": results.decision.recommended,
            "wins": results.decision.wins,
            "excluding_zero": results.decision.excluding_zero,
            "order": ORDER,
        },
        "scores": results.scores,
        "comparisons": results.comparisons,
        "extras": results.extras,
        "power": {"windows": power_windows, "seed": POWER_SEED, "shares": power_result},
        "spreads": {
            "offense": spreads.offense,
            "defense": spreads.defense,
            "home_field": spreads.home_field,
            "passer": spreads.passer,
            "scramble": spreads.scramble,
            "designed": spreads.designed,
            "play": spreads.play,
            "means": spreads.means,
        },
        "provenance": {**provenance, "input_files": ledger.files},
    }
    common.write_json_record(out_dir / OUT_JSON, _json_ready(payload))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True, help="nflreadpy cache, read only.")
    parser.add_argument(
        "--ratings-dir", type=Path, required=True, help="Published ratings tables, read only."
    )
    parser.add_argument(
        "--schedules-file",
        type=Path,
        default=DEFAULT_SCHEDULES,
        help="The raw nflverse schedule file with the starter ids, read only.",
    )
    parser.add_argument(
        "--power-windows", type=int, default=POWER_WINDOWS, help="Synthetic windows."
    )
    parser.add_argument("--out-dir", type=Path, default=HERE, help="Where the report goes.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the test."""
    args = _parse_args(argv)
    return run(
        (args.cache_dir, args.ratings_dir, args.schedules_file),
        args.out_dir,
        power_windows=args.power_windows,
    )


if __name__ == "__main__":
    sys.exit(main())
