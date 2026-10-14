r"""Which plays the team rating's scrimmage EPA should count: two pre-registered walk-forward tests.

The protocols are ``protocols/kneel_spike.md`` (kneel-downs and spikes) and
``protocols/penalty_wiped.md`` (plays a penalty wiped out), both committed before any variant was
scored. Each variant is the ported team rating (``nfl_predictor.ratings``) run unchanged on its
own play set: its own penalties, preseason prior and per-game scale. Each is scored by the
source's walk-forward validation, ported here from ``nfl_sos_ratings/validation/walk_forward.py``
at ``9081bce``: every week of 1999-2025 is rated on the season's earlier games, a margin model
``home_margin = k * rating_gap + home_edge`` fit on the variant's earlier predictions turns the
gap into a predicted home margin, and prediction weeks 5 on are scored. Paired differences from
today's rating are resampled by whole season (``rating_tests_common``).

Run from the repository root (``kneel-spike`` or ``penalty-wiped``):

    VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/team_plays_test.py \
        kneel-spike --cache-dir /home/mitch/scratch/step6/data/cache/nflreadpy \
        --ratings-dir /home/mitch/scratch/step6/data/ratings

It reads the play-by-play and schedules through this repository's loaders with downloads blocked,
and the candidate build's published ratings for the integrity checks; it writes only its report
and JSON beside this file. Run it without ``-I``: it imports ``rating_tests_common`` from its own
directory.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import rating_tests_common as common
from threadpoolctl import threadpool_limits

from nfl_predictor.ratings.inputs import team_game_rows
from nfl_predictor.ratings.pbp_inputs import team_games_from_pbp
from nfl_predictor.ratings.team_prior import (
    PRIOR_HORIZON_GAMES,
    PriorHistory,
    snapshot_fit,
    snapshot_prior,
)
from nfl_predictor.ratings.team_rating import (
    TEAM_RATING_COLUMNS,
    fit_team_ratings,
    fit_team_ratings_with_previous_penalties,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from rating_tests_common import Interval

HERE = Path(__file__).resolve().parent
FIRST_SEASON = 1999
LAST_SEASON = 2025
START_WEEK = 5
TODAY = "today"
# The source validation's early and late splits: prediction weeks 5-7 and 8 on.
BANDS: tuple[tuple[str, int, int | None], ...] = (("weeks 5-7", 5, 7), ("weeks 8+", 8, None))
OVERALL = "overall"
FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Int64(),
    "week": pl.Int64(),
    "baseline": pl.String(),
    "game_id": pl.String(),
    "home_team": pl.String(),
    "away_team": pl.String(),
    "rating_diff": pl.Float64(),
    "home_margin": pl.Float64(),
}
_PREDICTION_SCHEMA: dict[str, pl.DataType] = {
    **FEATURE_SCHEMA,
    "predicted_margin": pl.Float64(),
    "error": pl.Float64(),
    "training_row_count": pl.Int64(),
    "fitted_k": pl.Float64(),
    "fitted_hfa_points": pl.Float64(),
}
# A snapshot rates the teams from the season's games before a week: it takes those games and the
# whole season's rows (for the season's teams) and returns ``team`` and ``team_rating``.
type Snapshot = Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]
INFORMATION_TOLERANCE = 1e-12
_GAME_KEYS = ("season", "game_id", "team")
_PLAY_GROUP = ("season", "game_id", "week", "posteam", "defteam")


# --- the source's walk-forward harness ------------------------------------------------------


def build_home_game_frame(game_logs: pl.DataFrame, season: int) -> pl.DataFrame:
    """Return one row per home game with the realized home margin."""
    return (
        game_logs.filter(pl.col("is_home"))
        .select(
            pl.lit(season, dtype=pl.Int64).alias("season"),
            "game_id",
            pl.col("week").cast(pl.Int64),
            pl.col("team").alias("home_team"),
            pl.col("opponent_team").alias("away_team"),
            pl.col("point_margin").cast(pl.Float64).alias("home_margin"),
        )
        .sort(["week", "game_id"])
    )


def week_snapshot(game_logs: pl.DataFrame, week: int, snapshot: Snapshot) -> pl.DataFrame:
    """Return the ratings of the season's games before ``week``, or zeros without any."""
    prior_games = game_logs.filter(pl.col("week") < week)
    if prior_games.is_empty():
        teams = sorted(game_logs.get_column("team").unique().to_list())
        return pl.DataFrame({"team": teams, "team_rating": [0.0] * len(teams)})
    return snapshot(prior_games, game_logs)


def build_snapshot_feature_rows(
    game_logs: pl.DataFrame,
    season: int,
    baseline: str,
    snapshot: Snapshot,
    on_snapshot: Callable[[int, pl.DataFrame], None] | None = None,
) -> pl.DataFrame:
    """Build one row per home game with the pregame rating gap and the realized home margin.

    Each week is rated on the season's games before it (:func:`week_snapshot`); a team the
    snapshot does not rate counts as zero. ``on_snapshot`` receives each week and its ratings.
    """
    home_games = build_home_game_frame(game_logs, season)
    frames: list[pl.DataFrame] = []
    for week in sorted(home_games.get_column("week").unique().to_list()):
        ratings = week_snapshot(game_logs, week, snapshot)
        if on_snapshot is not None:
            on_snapshot(week, ratings)
        lookup = ratings.select("team", pl.col("team_rating").alias("rating"))
        frames.append(
            home_games.filter(pl.col("week") == week)
            .join(
                lookup.rename({"team": "home_team", "rating": "home"}), on="home_team", how="left"
            )
            .join(
                lookup.rename({"team": "away_team", "rating": "away"}), on="away_team", how="left"
            )
            .with_columns(
                pl.lit(baseline).alias("baseline"),
                (pl.col("home").fill_null(0.0) - pl.col("away").fill_null(0.0)).alias(
                    "rating_diff"
                ),
            )
            .select(list(FEATURE_SCHEMA))
        )
    return pl.concat(frames) if frames else pl.DataFrame(schema=FEATURE_SCHEMA)


def fit_margin_projection(training_rows: pl.DataFrame) -> tuple[float, float]:
    """Fit ``home_margin = k * rating_diff + home_edge`` on earlier rows only."""
    if training_rows.is_empty():
        return 0.0, 0.0
    rating_diff = training_rows.get_column("rating_diff").cast(pl.Float64).to_numpy()
    home_margin = training_rows.get_column("home_margin").cast(pl.Float64).to_numpy()
    if training_rows.height == 1 or np.isclose(rating_diff.std(), 0.0):
        return 0.0, float(home_margin.mean())
    design = np.column_stack((rating_diff, np.ones(len(rating_diff), dtype=np.float64)))
    coefficients, *_ = np.linalg.lstsq(design, home_margin, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def evaluate_feature_rows(feature_rows: pl.DataFrame, start_week: int) -> pl.DataFrame:
    """Fit earlier-rows-only margin models and predict every game from ``start_week`` on."""
    prediction_rows: list[dict[str, object]] = []
    eval_rows = feature_rows.filter(pl.col("week") >= start_week).sort(
        ["baseline", "season", "week", "game_id"]
    )
    for (baseline, season, week), week_rows in eval_rows.group_by(
        ["baseline", "season", "week"], maintain_order=True
    ):
        training_rows = feature_rows.filter(
            (pl.col("baseline") == baseline)
            & (
                (pl.col("season") < season)
                | ((pl.col("season") == season) & (pl.col("week") < week))
            )
        )
        fitted_k, fitted_home_edge = fit_margin_projection(training_rows)
        for row in week_rows.iter_rows(named=True):
            predicted = fitted_k * float(row["rating_diff"]) + fitted_home_edge
            prediction_rows.append(
                {
                    **{column: row[column] for column in FEATURE_SCHEMA},
                    "predicted_margin": predicted,
                    "error": predicted - float(row["home_margin"]),
                    "training_row_count": training_rows.height,
                    "fitted_k": fitted_k,
                    "fitted_hfa_points": fitted_home_edge,
                }
            )
    if not prediction_rows:
        return pl.DataFrame(schema=_PREDICTION_SCHEMA)
    return pl.DataFrame(prediction_rows, schema=_PREDICTION_SCHEMA).sort(
        ["baseline", "season", "week", "game_id"]
    )


# --- the play sets ---------------------------------------------------------------------------


def _flag(column: str) -> pl.Expr:
    """Return a play-by-play 0/1 flag as a boolean, false when missing."""
    return pl.col(column).fill_null(0) > 0


def _known_team(column: str) -> pl.Expr:
    """Return whether a team-code column holds a team (old play-by-play leaves some empty)."""
    return pl.col(column).is_not_null() & (pl.col(column) != "")


def play_flags(plays: pl.DataFrame) -> pl.DataFrame:
    """Return the regular-season plays with both teams known and the flags the variants read.

    ``today_snap`` is the ported team rating's scrimmage snap, written out here rather than
    imported so the integrity checks recount it independently: a dropback, a run play flagged
    ``qb_scramble``, a run (``rush``), a kneel-down or a spike. ``kneel`` and ``spike`` are the
    snaps flagged ``qb_kneel`` and ``qb_spike``; ``wiped_run`` is a snap a penalty wiped out
    (``play_type == "no_play"``), and ``wiped_pass`` a penalty-wiped play with the ``pass`` flag
    that is not a snap.
    """
    regular = (
        plays.filter(pl.col("season_type") == "REG") if "season_type" in plays.columns else plays
    )
    run_play = (pl.col("play_type") == "run").fill_null(value=False)
    no_play = (pl.col("play_type") == "no_play").fill_null(value=False)
    snap = (
        _flag("qb_dropback")
        | (_flag("qb_scramble") & run_play)
        | _flag("rush")
        | _flag("qb_kneel")
        | _flag("qb_spike")
    )
    extra = [column for column in ("penalty_team",) if column in regular.columns]
    return (
        regular.filter(_known_team("posteam") & _known_team("defteam"))
        .with_columns(
            snap.alias("today_snap"),
            (snap & _flag("qb_kneel")).alias("kneel"),
            (snap & _flag("qb_spike")).alias("spike"),
            (snap & no_play).alias("wiped_run"),
            (no_play & _flag("pass") & ~snap).alias("wiped_pass"),
        )
        .select(
            pl.col("season").cast(pl.Int64),
            "game_id",
            pl.col("week").cast(pl.Int64),
            "posteam",
            "defteam",
            "play_type",
            "epa",
            "today_snap",
            "kneel",
            "spike",
            "wiped_run",
            "wiped_pass",
            *extra,
        )
    )


_SNAP_FLAGS = ("rush", "qb_kneel", "qb_spike", "qb_dropback")


def wiped_play_summary(plays: pl.DataFrame) -> pl.DataFrame:
    """Return the penalty-wiped plays' counts and EPA, by set, for one season's play-by-play.

    The first sets take every regular-season ``no_play`` play, as the phase-1 parity check
    counted them (both teams known or not); the rest take the plays with both teams known, as the
    rating does: the wiped runs (``no_play`` plays today's snap rule counts) by each flag that
    makes them a snap (a play can carry more than one), and the wiped passes.
    """
    regular = (
        plays.filter(pl.col("season_type") == "REG") if "season_type" in plays.columns else plays
    )
    wiped = regular.filter((pl.col("play_type") == "no_play").fill_null(value=False))
    flags = play_flags(wiped)
    epa = pl.col("epa").fill_null(0.0)
    sets: list[tuple[str, pl.DataFrame]] = [
        ("every penalty-wiped play", wiped),
        ("with the rush flag", wiped.filter(_flag("rush"))),
        ("with the pass flag", wiped.filter(_flag("pass"))),
        ("both teams known", flags),
        ("wiped runs (today's snaps)", flags.filter(pl.col("wiped_run"))),
    ]
    known = wiped.filter(_known_team("posteam") & _known_team("defteam"))
    sets.extend(
        (
            f"wiped runs (today's snaps), {flag} flag",
            known.filter(_flag(flag)),
        )
        for flag in _SNAP_FLAGS
    )
    sets.append(("wiped passes", flags.filter(pl.col("wiped_pass"))))
    return pl.DataFrame(
        [
            {
                "set": name,
                "plays": frame.height,
                "epa": common.as_float(frame.select(epa.sum()).item()),
            }
            for name, frame in sets
        ],
        schema={"set": pl.String, "plays": pl.Int64, "epa": pl.Float64},
    )


@dataclass(frozen=True)
class Variant:
    """One play set: which plays count as scrimmage snaps, over :func:`play_flags`' columns."""

    name: str
    label: str
    snap: pl.Expr


_TODAY = pl.col("today_snap")
VARIANTS: dict[str, Variant] = {
    variant.name: variant
    for variant in (
        Variant(TODAY, "today's rating (every scrimmage snap)", _TODAY),
        Variant(
            "without_kneel_spike",
            "without kneel-downs and spikes",
            _TODAY & ~pl.col("kneel") & ~pl.col("spike"),
        ),
        Variant("without_kneels", "without kneel-downs", _TODAY & ~pl.col("kneel")),
        Variant("without_spikes", "without spikes", _TODAY & ~pl.col("spike")),
        Variant("every_wiped_snap", "every wiped snap", _TODAY | pl.col("wiped_pass")),
        Variant("no_wiped_snap", "no wiped snap", _TODAY & ~pl.col("wiped_run")),
        Variant(
            "wiped_passes_only",
            "wiped passes only",
            (_TODAY & ~pl.col("wiped_run")) | pl.col("wiped_pass"),
        ),
    )
}


def team_rows_with_snaps(
    team_games: pl.DataFrame, flags: pl.DataFrame, snap: pl.Expr
) -> pl.DataFrame:
    """Return ``team_games`` with ``offensive_snaps`` and ``offensive_epa`` over ``snap`` plays.

    The sums mirror the adapter's (``pbp_inputs.team_games_from_pbp``): per team-game, in the
    plays' order, a missing EPA counting as 0. Every team-game of ``team_games`` stays, with 0 and
    0.0 when none of its plays counts, and every other column is kept as it is.
    """
    epa = pl.col("epa").fill_null(0.0)
    sums = (
        flags.group_by(list(_PLAY_GROUP), maintain_order=True)
        .agg(
            snap.cast(pl.Int64).sum().alias("_snaps"),
            epa.filter(snap).sum().fill_null(0.0).alias("_epa"),
        )
        .select("season", "game_id", pl.col("posteam").alias("team"), "_snaps", "_epa")
    )
    columns = team_games.columns
    return (
        team_games.join(sums, on=list(_GAME_KEYS), how="left", maintain_order="left")
        .with_columns(
            pl.col("_snaps")
            .fill_null(0)
            .cast(team_games.schema["offensive_snaps"])
            .alias("offensive_snaps"),
            pl.col("_epa")
            .fill_null(0.0)
            .cast(team_games.schema["offensive_epa"])
            .alias("offensive_epa"),
        )
        .select(columns)
    )


# --- the decision ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """A variant's paired difference from today's rating, overall and in each week band."""

    overall: Interval
    bands: Mapping[str, Interval]


def qualifies(overall: Interval, bands: Mapping[str, Interval]) -> bool:
    """Return whether the overall interval lies below zero and no band interval above zero."""
    return overall.below_zero and not any(band.above_zero for band in bands.values())


@dataclass(frozen=True)
class Decision:
    """The decision rule's outcome, with every interval that excludes zero.

    ``excluding_zero`` holds ``(variant, band, "better" or "worse")``; ``band_guard`` the variants
    whose overall interval lay below zero but which a week band disqualified.
    """

    recommended: str
    qualifying: tuple[str, ...]
    band_guard: tuple[str, ...]
    excluding_zero: tuple[tuple[str, str, str], ...]


KNEEL_SPIKE_ORDER = (TODAY, "without_kneel_spike")
PENALTY_ORDER = ("every_wiped_snap", "no_wiped_snap", TODAY, "wiped_passes_only")


def decide(
    comparisons: Mapping[str, Comparison], maes: Mapping[str, float], order: Sequence[str]
) -> Decision:
    """Apply the pre-registered rule: the best qualifier, or the simplest not significantly worse.

    A variant qualifies when :func:`qualifies` holds for its comparison with today's rating; the
    recommendation is the qualifier with the lowest overall MAE (ties to the earlier in
    ``order``). With none, it is the first of ``order`` (the order of simplicity) that is today's
    rating or whose overall interval does not lie entirely above zero.
    """
    named = [name for name in order if name in comparisons]
    qualifying = tuple(
        name for name in named if qualifies(comparisons[name].overall, comparisons[name].bands)
    )
    band_guard = tuple(
        name for name in named if comparisons[name].overall.below_zero and name not in qualifying
    )
    excluding: list[tuple[str, str, str]] = []
    for name in named:
        comparison = comparisons[name]
        for band, interval in ((OVERALL, comparison.overall), *comparison.bands.items()):
            if interval.excludes_zero:
                excluding.append((name, band, "better" if interval.below_zero else "worse"))
    if qualifying:
        recommended = min(qualifying, key=lambda name: (maes[name], order.index(name)))
    else:
        recommended = next(
            name
            for name in order
            if name == TODAY or (name in comparisons and not comparisons[name].overall.above_zero)
        )
    return Decision(recommended, qualifying, band_guard, tuple(excluding))


def paired_differences(
    predictions: pl.DataFrame, variant: str, reference: str = TODAY
) -> pl.DataFrame:
    """Return each game's absolute error under ``variant`` minus the one under ``reference``.

    Raises:
        ValueError: If the two do not predict the same games, or one predicts a game twice.

    """
    keys = ["season", "week", "game_id"]
    pair = predictions.filter(pl.col("baseline").is_in([reference, variant]))
    if pair.select(pl.struct("baseline", *keys).is_duplicated().any()).item():
        msg = f"{variant} or {reference} predicts a game more than once"
        raise ValueError(msg)
    errors = pair.with_columns(pl.col("error").abs()).pivot(
        on="baseline", index=keys, values="error", aggregate_function="first"
    )
    if (
        not {reference, variant} <= set(errors.columns)
        or errors.select(pl.any_horizontal(pl.col(reference, variant).is_null()).any()).item()
    ):
        msg = f"{variant} and {reference} do not predict the same games"
        raise ValueError(msg)
    return errors.select(*keys, (pl.col(variant) - pl.col(reference)).alias("difference")).sort(
        keys
    )


def band_rows(rows: pl.DataFrame, band: str) -> pl.DataFrame:
    """Return the rows of one week band, or every row for ``OVERALL``."""
    for name, first, last in BANDS:
        if name == band:
            in_band = pl.col("week") >= first
            return rows.filter(in_band if last is None else in_band & (pl.col("week") <= last))
    return rows


# --- integrity helpers -----------------------------------------------------------------------


def _perturbed(game_logs: pl.DataFrame, week: int) -> pl.DataFrame:
    """Return ``game_logs`` with every numeric value of the games from ``week`` on replaced."""
    return common.perturbed_numbers(game_logs, pl.col("week") >= week)


def largest_gap(left: pl.DataFrame, right: pl.DataFrame, key: str = "team") -> float:
    """Return the largest absolute difference over the float columns two frames share.

    Returns infinity when the frames do not hold the same keys.
    """
    if sorted(left.get_column(key).to_list()) != sorted(right.get_column(key).to_list()):
        return float("inf")
    columns = [
        name
        for name, dtype in left.schema.items()
        if name != key and name in right.columns and dtype.is_float()
    ]
    joined = left.join(right, on=key, suffix="_other")
    gaps = [
        common.as_float(joined.select((pl.col(name) - pl.col(f"{name}_other")).abs().max()).item())
        for name in columns
    ]
    return max(gaps, default=0.0)


def check_information_set(game_logs: pl.DataFrame, snapshot: Snapshot) -> int:
    """Check that each week's snapshot ignores the games from that week on; return the count.

    Every week with earlier games is rated twice: on the season's rows, and on the same rows with
    every value of the games from that week on replaced.

    Raises:
        ValueError: If a week's ratings move when the later games change.

    """
    checked = 0
    for week in sorted(game_logs.get_column("week").unique().to_list()):
        if game_logs.filter(pl.col("week") < week).is_empty():
            continue
        original = week_snapshot(game_logs, week, snapshot)
        again = week_snapshot(_perturbed(game_logs, week), week, snapshot)
        gap = largest_gap(original, again)
        if gap > INFORMATION_TOLERANCE:
            msg = f"information set: week {week}'s snapshot moves by {gap:.3g} with later games"
            raise ValueError(msg)
        checked += 1
    return checked


# --- inputs and runs -------------------------------------------------------------------------


@dataclass
class TeamInputs:
    """Every season's adapter team-game rows, its plays' flags and its penalty-wiped plays."""

    team_games: dict[int, pl.DataFrame]
    flags: dict[int, pl.DataFrame]
    wiped: dict[int, pl.DataFrame] = field(default_factory=dict)


def load_team_inputs(
    cache_dir: Path, seasons: Iterable[int], ledger: common.InputLedger
) -> TeamInputs:
    """Load each season's play-by-play and schedule and build its rows, downloads blocked."""
    common.block_downloads()
    inputs = TeamInputs(team_games={}, flags={})
    for season in seasons:
        plays, schedule = common.load_season_plays(cache_dir, season, ledger)
        inputs.team_games[season] = team_games_from_pbp(plays, schedule)
        inputs.flags[season] = play_flags(plays)
        inputs.wiped[season] = wiped_play_summary(plays)
    return inputs


def wiped_totals(inputs: TeamInputs) -> list[dict[str, Any]]:
    """Return the penalty-wiped plays' counts and EPA by set, summed over every season."""
    if not inputs.wiped:
        return []
    return (
        pl.concat(list(inputs.wiped.values()))
        .group_by("set", maintain_order=True)
        .agg(pl.col("plays").sum(), pl.col("epa").sum())
        .to_dicts()
    )


def variant_rows(inputs: TeamInputs, variant: Variant) -> dict[int, pl.DataFrame]:
    """Return each season's team-game rows of ``variant``, as the ETL hands them to the fits.

    Today's rating takes the adapter's rows themselves; every other variant replaces their
    scrimmage snaps and EPA by its own play set (:func:`team_rows_with_snaps`).
    """
    return {
        season: team_game_rows(
            rows
            if variant.name == TODAY
            else team_rows_with_snaps(rows, inputs.flags[season], variant.snap)
        )
        for season, rows in inputs.team_games.items()
    }


def season_snapshot(history: PriorHistory, season: int) -> Snapshot:
    """Return the published fit of a season's games before a week, as the source validates it.

    The previous season's cross-validated penalties (none in the first season, whose snapshots
    cross-validate their own, as the source's ``previous_season_fit`` gives) and the season's
    preseason prior at the published horizon (``team_prior.snapshot_fit``).
    """
    previous = history.cross_validated_fit(season - 1) if season > history.first_season else None
    season_prior = history.season_prior(season)

    def snapshot(prior_games: pl.DataFrame, season_logs: pl.DataFrame) -> pl.DataFrame:
        teams = sorted(season_logs.get_column("team").unique().to_list())
        ratings, _ = snapshot_fit(prior_games, teams, previous, season_prior, PRIOR_HORIZON_GAMES)
        return ratings

    return snapshot


def full_season_ratings(history: PriorHistory, season: int) -> pl.DataFrame:
    """Return a season's published full-season ratings: previous penalties and the prior."""
    logs = history.game_logs(season)
    previous = history.cross_validated_fit(season - 1) if season > history.first_season else None
    means = snapshot_prior(history.season_prior(season), logs)
    return fit_team_ratings_with_previous_penalties(logs, previous, scrimmage_prior=means).ratings


@dataclass
class VariantRun:
    """One variant's walk-forward: its rows, feature rows, snapshots and full-season ratings."""

    name: str
    rows: dict[int, pl.DataFrame]
    features: pl.DataFrame
    snapshots: dict[tuple[int, int], pl.DataFrame]
    full_season: dict[int, pl.DataFrame]
    penalties: dict[int, tuple[float, float]]
    information_checked: int = 0
    seconds: float = 0.0


def run_variant(
    name: str, rows: dict[int, pl.DataFrame], seasons: Sequence[int], *, check_information: bool
) -> VariantRun:
    """Rate every week of ``seasons`` on the games before it, with one variant's rows."""
    start = time.perf_counter()
    history = PriorHistory(lambda season: rows[season], first_season=FIRST_SEASON)
    frames: list[pl.DataFrame] = []
    snapshots: dict[tuple[int, int], pl.DataFrame] = {}
    full_season: dict[int, pl.DataFrame] = {}
    penalties: dict[int, tuple[float, float]] = {}
    information = 0
    for season in seasons:
        logs = history.game_logs(season)
        snapshot = season_snapshot(history, season)

        def record(week: int, ratings: pl.DataFrame, season: int = season) -> None:
            snapshots[(season, week)] = ratings

        frames.append(build_snapshot_feature_rows(logs, season, name, snapshot, record))
        if check_information:
            information += check_information_set(logs, snapshot)
        full_season[season] = full_season_ratings(history, season)
        if season > history.first_season:
            fit = history.cross_validated_fit(season - 1)
            penalties[season] = (fit.scrimmage_lambda, fit.special_teams_lambda)
    return VariantRun(
        name=name,
        rows=rows,
        features=pl.concat(frames),
        snapshots=snapshots,
        full_season=full_season,
        penalties=penalties,
        information_checked=information,
        seconds=time.perf_counter() - start,
    )


# --- integrity checks ------------------------------------------------------------------------

MATCH_TOLERANCE = 1e-9
ZERO_TOLERANCE = 1e-12
# The source validation's figures (docs/validation-report.md at 9081bce), to three decimals.
SOURCE_VALIDATION = {
    OVERALL: (5297, 10.575),
    "weeks 5-7": (1141, 10.567),
    "weeks 8+": (4156, 10.577),
}


class IntegrityError(ValueError):
    """Raised when an integrity check fails; nothing about the variants is read then."""


def _rating_gap(left: pl.DataFrame, right: pl.DataFrame) -> tuple[float, int]:
    """Return the largest gap over the four rating columns, teams in both, and their count."""
    joined = left.select("team", *TEAM_RATING_COLUMNS).join(
        right.select("team", *TEAM_RATING_COLUMNS), on="team", suffix="_other"
    )
    gaps = [
        common.as_float(joined.select((pl.col(c) - pl.col(f"{c}_other")).abs().max()).item())
        for c in TEAM_RATING_COLUMNS
    ]
    return max(gaps, default=0.0), joined.height


def check_production(
    today: VariantRun, ratings_dir: Path, ledger: common.InputLedger
) -> dict[str, Any]:
    """Check 1: today's full-season fits and snapshots are the production rating's.

    Raises:
        IntegrityError: If a full-season rating or a snapshot differs beyond its tolerance.

    """
    season_gap = snapshot_gap = first_gap = 0.0
    compared_snapshots = 0
    for season, ratings in today.full_season.items():
        path = ratings_dir / f"{season}_ratings.parquet"
        ledger.record(path)
        published = pl.read_parquet(path)
        gap, teams = _rating_gap(ratings, published)
        if teams != published.height or gap > MATCH_TOLERANCE:
            msg = f"check 1: {season} full-season ratings differ by {gap:.3g} ({teams} teams)"
            raise IntegrityError(msg)
        season_gap = max(season_gap, gap)
        history_path = ratings_dir / f"{season}_ratings_by_week.parquet"
        ledger.record(history_path)
        by_week = pl.read_parquet(history_path)
        weeks = sorted(by_week.get_column("week").unique().to_list())
        for (snap_season, week), snapshot in today.snapshots.items():
            earlier = [value for value in weeks if value < week]
            if snap_season != season or not earlier:
                continue
            if season == FIRST_SEASON:
                logs = today.rows[season].filter(pl.col("week") < week)
                gap, _ = _rating_gap(snapshot, fit_team_ratings(logs).ratings)
                first_gap = max(first_gap, gap)
                if gap > ZERO_TOLERANCE:
                    msg = f"check 1: {season} week {week} differs from its own fit by {gap:.3g}"
                    raise IntegrityError(msg)
                continue
            gap, _ = _rating_gap(snapshot, by_week.filter(pl.col("week") == earlier[-1]))
            if gap > MATCH_TOLERANCE:
                msg = f"check 1: {season} week {week} differs from the history by {gap:.3g}"
                raise IntegrityError(msg)
            snapshot_gap = max(snapshot_gap, gap)
            compared_snapshots += 1
    return {
        "full_season_gap": season_gap,
        "snapshots_compared": compared_snapshots,
        "snapshot_gap": snapshot_gap,
        "first_season_gap": first_gap,
    }


def score(predictions: pl.DataFrame, baseline: str) -> dict[str, tuple[int, float, float]]:
    """Return a baseline's games, MAE and RMSE, overall and by band."""
    rows = predictions.filter(pl.col("baseline") == baseline)
    result: dict[str, tuple[int, float, float]] = {}
    for band in (OVERALL, *(name for name, _, _ in BANDS)):
        errors = band_rows(rows, band).get_column("error").to_numpy()
        result[band] = (
            (int(errors.size), float(np.abs(errors).mean()), float(np.sqrt((errors**2).mean())))
            if errors.size
            else (0, math.nan, math.nan)
        )
    return result


def check_source_validation(predictions: pl.DataFrame) -> dict[str, tuple[int, float, float]]:
    """Check 2: today's walk-forward reproduces the source validation's games and MAE.

    Raises:
        IntegrityError: If a band's game count or MAE (to three decimals) differs.

    """
    scores = score(predictions, TODAY)
    for band, (games, mae) in SOURCE_VALIDATION.items():
        ours = scores[band]
        if ours[0] != games or round(ours[1], 3) != mae:
            msg = f"check 2: {band} gives {ours[0]} games at {ours[1]:.4f}, not {games} at {mae}"
            raise IntegrityError(msg)
    return scores


def _moved_sums(flags: pl.DataFrame, condition: pl.Expr) -> pl.DataFrame:
    """Return each team-game's count and EPA over the plays meeting ``condition``."""
    return (
        flags.filter(condition)
        .group_by("season", "game_id", pl.col("posteam").alias("team"), maintain_order=True)
        .agg(pl.len().alias("_count"), pl.col("epa").fill_null(0.0).sum().alias("_epa"))
    )


def check_variant_rows(inputs: TeamInputs, runs: Mapping[str, VariantRun]) -> dict[str, Any]:
    """Checks 3 and 4: each variant's rows are today's plus and less its moved plays, recounted.

    Raises:
        IntegrityError: If today's written-out rule misses the adapter's rows, a count or EPA
            sum differs, another column moves, or the wiped runs and passes overlap.

    """
    epa_gap = 0.0
    moved: dict[str, dict[str, float]] = {}
    for season, base in inputs.team_games.items():
        flags = inputs.flags[season]
        rewritten = team_rows_with_snaps(base, flags, VARIANTS[TODAY].snap)
        if not rewritten.equals(base):
            msg = f"check 4: today's written-out snaps differ from the adapter's in {season}"
            raise IntegrityError(msg)
        overlap = flags.filter(pl.col("wiped_run") & pl.col("wiped_pass")).height
        if overlap:
            msg = f"check 3: {overlap} plays of {season} are both a wiped run and a wiped pass"
            raise IntegrityError(msg)
        today = team_game_rows(base)
        for name, run in runs.items():
            if name == TODAY:
                continue
            snap = VARIANTS[name].snap
            removed = _moved_sums(flags, pl.col("today_snap") & ~snap)
            added = _moved_sums(flags, ~pl.col("today_snap") & snap)
            expected = (
                today.with_columns(pl.lit(season, dtype=pl.Int64).alias("season"))
                .join(removed, on=list(_GAME_KEYS), how="left", maintain_order="left")
                .join(
                    added,
                    on=list(_GAME_KEYS),
                    how="left",
                    suffix="_added",
                    maintain_order="left",
                )
                .with_columns(pl.col("^_.*$").fill_null(0))
            )
            ours = run.rows[season]
            snaps_ok = (
                ours.get_column("offensive_snaps")
                == expected.get_column("offensive_snaps")
                - expected.get_column("_count")
                + expected.get_column("_count_added")
            ).all()
            recounted = (
                expected.get_column("offensive_epa")
                - expected.get_column("_epa")
                + expected.get_column("_epa_added")
            )
            gap = common.as_float((ours.get_column("offensive_epa") - recounted).abs().max())
            others = [c for c in today.columns if c not in {"offensive_snaps", "offensive_epa"}]
            if (
                not snaps_ok
                or gap > MATCH_TOLERANCE
                or not ours.select(others).equals(today.select(others))
            ):
                msg = f"check 3: {name} rows of {season} are not today's with its plays moved"
                raise IntegrityError(msg)
            epa_gap = max(epa_gap, gap)
            totals = moved.setdefault(name, {"removed": 0, "removed_epa": 0.0, "added": 0})
            totals["removed"] += int(removed.get_column("_count").sum())
            totals["removed_epa"] += float(removed.get_column("_epa").sum())
            totals["added"] += int(added.get_column("_count").sum())
            totals.setdefault("added_epa", 0.0)
            totals["added_epa"] += float(added.get_column("_epa").sum())
    return {"epa_gap": epa_gap, "moved": moved}


def check_penalties(runs: Mapping[str, VariantRun]) -> int:
    """Check 5: every variant's season penalties equal a separate fit of its previous season.

    Raises:
        IntegrityError: If a penalty differs.

    """
    checked = 0
    for run in runs.values():
        for season, used in run.penalties.items():
            fit = fit_team_ratings(run.rows[season - 1])
            if used != (fit.scrimmage_lambda, fit.special_teams_lambda):
                msg = f"check 5: {run.name}'s {season} penalties {used} are not its own"
                raise IntegrityError(msg)
            checked += 1
    return checked


def check_margin_rows(predictions: pl.DataFrame, features: pl.DataFrame) -> int:
    """Check 6, second part: every margin model saw exactly the baseline's earlier rows.

    Raises:
        IntegrityError: If a prediction's training-row count is not the earlier rows' count.

    """
    counts = (
        features.group_by("baseline", "season", "week")
        .len()
        .sort("baseline", "season", "week")
        .with_columns((pl.col("len").cum_sum().over("baseline") - pl.col("len")).alias("_earlier"))
    )
    joined = predictions.join(counts, on=["baseline", "season", "week"], how="left")
    wrong = joined.filter(pl.col("training_row_count") != pl.col("_earlier")).height
    if wrong:
        msg = f"check 6: {wrong} predictions trained on rows that are not all earlier"
        raise IntegrityError(msg)
    return joined.height


# --- results and extras ----------------------------------------------------------------------

ZERO_CHANGE = Variant("zero_change", "no play moved (an integrity check)", _TODAY)


@dataclass(frozen=True)
class TestSpec:
    """One of the two tests: its variants, order of simplicity and outputs."""

    title: str
    decision_variants: tuple[str, ...]
    extra_variants: tuple[str, ...]
    order: tuple[str, ...]
    protocol: str
    report: str
    output_json: str


TESTS = {
    "kneel-spike": TestSpec(
        title="Kneel-downs and spikes in the team rating",
        decision_variants=("without_kneel_spike",),
        extra_variants=("without_kneels", "without_spikes"),
        order=KNEEL_SPIKE_ORDER,
        protocol="protocols/kneel_spike.md",
        report="KNEEL_SPIKE_TEST.md",
        output_json="kneel_spike_test.json",
    ),
    "penalty-wiped": TestSpec(
        title="Penalty-wiped plays in the team rating",
        decision_variants=("every_wiped_snap", "no_wiped_snap", "wiped_passes_only"),
        extra_variants=(),
        order=PENALTY_ORDER,
        protocol="protocols/penalty_wiped.md",
        report="PENALTY_WIPED_TEST.md",
        output_json="penalty_wiped_test.json",
    ),
}


def compare_variant(
    predictions: pl.DataFrame,
    variant: str,
    seasons: Sequence[int],
    draws: np.ndarray,
    confidence: float,
) -> Comparison:
    """Return a variant's season-bootstrap comparison with today's, overall and by band."""
    differences = paired_differences(predictions, variant)
    return Comparison(
        overall=common.season_bootstrap(differences, seasons, draws, confidence),
        bands={
            name: common.season_bootstrap(band_rows(differences, name), seasons, draws, confidence)
            for name, _, _ in BANDS
        },
    )


def compare_by_game(predictions: pl.DataFrame, variant: str, confidence: float) -> Comparison:
    """Return the same comparison from a bootstrap that resamples single games (descriptive)."""
    differences = paired_differences(predictions, variant)

    def interval(rows: pl.DataFrame) -> Interval:
        return common.game_bootstrap(
            rows.get_column("difference").to_numpy(), confidence=confidence
        )

    return Comparison(
        overall=interval(differences),
        bands={name: interval(band_rows(differences, name)) for name, _, _ in BANDS},
    )


def mean_slopes(predictions: pl.DataFrame, baseline: str) -> dict[str, float]:
    """Return a baseline's mean fitted margin slope ``k``, overall and by band."""
    rows = predictions.filter(pl.col("baseline") == baseline)
    return {
        band: common.as_float(band_rows(rows, band).get_column("fitted_k").mean())
        for band in (OVERALL, *(name for name, _, _ in BANDS))
    }


def moved_by_season(inputs: TeamInputs, variant: str) -> list[dict[str, Any]]:
    """Return, per season, the plays a variant removes from and adds to today's, with EPA."""
    snap = VARIANTS[variant].snap
    rows: list[dict[str, Any]] = []
    for season, flags in inputs.flags.items():
        removed = flags.filter(pl.col("today_snap") & ~snap)
        added = flags.filter(~pl.col("today_snap") & snap)
        rows.append(
            {
                "season": season,
                "removed": removed.height,
                "removed_epa": float(removed.get_column("epa").fill_null(0.0).sum()),
                "added": added.height,
                "added_epa": float(added.get_column("epa").fill_null(0.0).sum()),
            }
        )
    return rows


def moved_by_penalty_side(inputs: TeamInputs) -> list[dict[str, Any]]:
    """Return the wiped runs and passes by the penalized side, with their EPA (1999-2025)."""
    flags = pl.concat(list(inputs.flags.values()), how="diagonal_relaxed")
    if "penalty_team" not in flags.columns:
        return []
    side = (
        pl.when(pl.col("penalty_team") == pl.col("posteam"))
        .then(pl.lit("offense"))
        .when(pl.col("penalty_team") == pl.col("defteam"))
        .then(pl.lit("defense"))
        .otherwise(pl.lit("neither recorded"))
    )
    rows: list[dict[str, Any]] = []
    for kind in ("wiped_run", "wiped_pass"):
        summary = (
            flags.filter(pl.col(kind))
            .with_columns(side.alias("side"))
            .group_by("side")
            .agg(pl.len().alias("plays"), pl.col("epa").fill_null(0.0).sum().alias("epa"))
            .sort("side")
        )
        rows.extend({"kind": kind, **row} for row in summary.iter_rows(named=True))
    return rows


def year_over_year(run: VariantRun) -> dict[str, float]:
    """Return the Pearson correlation of full-season ``team_rating`` over consecutive seasons."""
    pairs: list[tuple[float, float]] = []
    seasons = sorted(run.full_season)
    for season, following in itertools.pairwise(seasons):
        joined = (
            run.full_season[season]
            .select("team", "team_rating")
            .join(
                run.full_season[following].select("team", "team_rating"), on="team", suffix="_next"
            )
        )
        pairs.extend(joined.select("team_rating", "team_rating_next").iter_rows())
    values = np.array(pairs, dtype=np.float64)
    return {"pairs": float(len(pairs)), "pearson": float(np.corrcoef(values.T)[0, 1])}


def season_ranks(run: VariantRun, season: int) -> pl.DataFrame:
    """Return a season's full-season ratings with each team's rank on ``team_rating``."""
    return (
        run.full_season[season]
        .select("team", "offense_rating", "defense_rating", "team_rating")
        .with_columns(
            pl.col("team_rating").rank("min", descending=True).cast(pl.Int64).alias("rank")
        )
        .sort("rank", "team")
    )


def feature_impact(run: VariantRun, today: VariantRun) -> dict[str, dict[str, float]]:
    """Return how far a variant moves each team's pre-week ratings from today's, all snapshots.

    Week 1 has no earlier game, so its snapshot is the harness's zeros, with no rating columns to
    compare; it is skipped.
    """
    frames = [
        run.snapshots[key]
        .select("team", *TEAM_RATING_COLUMNS)
        .join(today.snapshots[key].select("team", *TEAM_RATING_COLUMNS), on="team", suffix="_today")
        for key in sorted(today.snapshots)
        if key in run.snapshots
        and set(TEAM_RATING_COLUMNS) <= set(today.snapshots[key].columns)
        and set(TEAM_RATING_COLUMNS) <= set(run.snapshots[key].columns)
    ]
    joined = pl.concat(frames)
    impact: dict[str, dict[str, float]] = {}
    for column in ("team_rating", "offense_rating", "defense_rating"):
        moved = (pl.col(column) - pl.col(f"{column}_today")).abs()
        impact[column] = {
            "rows": float(joined.height),
            "rows_moved": float(joined.filter(moved > MATCH_TOLERANCE).height),
            "largest": common.as_float(joined.select(moved.max()).item()),
            "mean": common.as_float(joined.select(moved.mean()).item()),
        }
    return impact


def check_zero_change(zero: VariantRun, today: VariantRun, predictions: pl.DataFrame) -> float:
    """Check 4, end to end: with no play moved, the features and predictions are today's.

    Raises:
        IntegrityError: If a rating gap or a prediction differs beyond ``ZERO_TOLERANCE``.

    """
    keys = ["season", "week", "game_id"]
    gaps = []
    for column, left, right in (
        ("rating_diff", zero.features, today.features),
        (
            "predicted_margin",
            predictions.filter(pl.col("baseline") == zero.name),
            predictions.filter(pl.col("baseline") == TODAY),
        ),
    ):
        joined = left.select(*keys, column).join(
            right.select(*keys, column), on=keys, suffix="_today"
        )
        if joined.height != left.height or left.height != right.height:
            msg = "check 4: the zero-change run predicts other games than today's"
            raise IntegrityError(msg)
        gaps.append(
            common.as_float(
                joined.select((pl.col(column) - pl.col(f"{column}_today")).abs().max()).item()
            )
        )
    if max(gaps) > ZERO_TOLERANCE:
        msg = f"check 4: the zero-change run differs from today's by {max(gaps):.3g}"
        raise IntegrityError(msg)
    return max(gaps)


def _integrity_lines(integrity: Mapping[str, Any]) -> list[str]:
    """Return the integrity checks' results as bullets."""
    production = integrity["production"]
    validation = integrity["source_validation"]
    rows = integrity["variant_rows"]
    moved = "; ".join(
        f"{_label(name)}: {totals['removed']:,} plays out ({totals['removed_epa']:+,.1f} EPA), "
        f"{totals['added']:,} in ({totals.get('added_epa', 0.0):+,.1f} EPA)"
        for name, totals in rows["moved"].items()
    )
    information = ", ".join(
        f"{_label(name)} {count:,}" for name, count in integrity["information_snapshots"].items()
    )
    return [
        (
            "1. Today's rating is the production rating: the full-season ratings of 1999-2025 "
            "equal the candidate build's `{season}_ratings.parquet` (largest gap "
            f"{production['full_season_gap']:.1e}); {production['snapshots_compared']:,} "
            "snapshots of 2000-2025 equal the published history of the week before (largest gap "
            f"{production['snapshot_gap']:.1e}); the 1999 snapshots equal separate "
            "cross-validated fits of the same games (largest gap "
            f"{production['first_season_gap']:.1e})."
        ),
        "2. The ported harness reproduces the source's validation: "
        + "; ".join(
            f"{band} {games:,} games, MAE {mae:.3f}" for band, (games, mae, _) in validation.items()
        )
        + " (the source: 5,297 at 10.575, 1,141 at 10.567, 4,156 at 10.577).",
        (
            "3. Every variant's rows are today's with its plays moved, recounted from the "
            f"play-by-play (largest EPA gap {rows['epa_gap']:.1e}); special-teams columns, "
            "margins and team-games unchanged; no play is both a wiped run and a wiped pass. "
            f"Moved: {moved}."
        ),
        (
            "4. With no play moved, the rows equal the adapter's exactly and the features and "
            f"predictions equal today's (largest gap {integrity['zero_change_gap']:.1e})."
        ),
        (
            f"5. {integrity['penalty_seasons']:,} season penalties equal separate "
            "cross-validations of each variant's previous season."
        ),
        (
            f"6. Snapshots unchanged when every later game is replaced: {information}; all "
            f"{integrity['margin_rows']:,} predictions trained on earlier rows only."
        ),
        "7. Every input file's SHA-256 is recorded (Provenance).",
    ]


@dataclass(frozen=True)
class ReportParts:
    """Everything the report shows, gathered once the integrity checks have passed."""

    integrity: Mapping[str, Any]
    comparisons: Mapping[str, Comparison]
    decision: Decision
    confidence: float
    scores: Mapping[str, Mapping[str, tuple[int, float, float]]]
    slopes: Mapping[str, Mapping[str, float]]
    extras: Mapping[str, Any]


def _report_body(spec: TestSpec, parts: ReportParts) -> list[str]:
    """Return the report's headline, decision, results, checks and extras."""
    integrity, comparisons, decision = parts.integrity, parts.comparisons, parts.decision
    confidence, scores, slopes, extras = parts.confidence, parts.scores, parts.slopes, parts.extras
    games, today_mae, _ = scores[TODAY][OVERALL]
    headline = [
        f"- {_label(name)}: MAE {scores[name][OVERALL][1]:.3f} against today's {today_mae:.3f} "
        f"over {games:,} games; paired difference {_interval_text(comparison.overall)}."
        for name, comparison in comparisons.items()
    ]
    lines = [
        (
            f"Pre-registered in `{spec.protocol}` before any variant was scored. Next-game margin "
            f"MAE, prediction weeks {START_WEEK} and later of {FIRST_SEASON}-{LAST_SEASON}, the "
            "ported team rating rebuilt each week from the season's earlier games for every "
            "variant."
        ),
        "",
        "## Headline",
        "",
        *headline,
        "",
        "## Decision",
        "",
        *_decision_lines(decision, spec, confidence),
        "",
        "## Paired differences from today's rating",
        "",
        (
            f"MAE of the variant minus today's (negative favors the variant); {confidence:.2%} "
            "percentile intervals from 10,000 resamples of whole seasons (27 clusters), seed 0, "
            "the same draws for every variant and band."
        ),
        "",
        *_comparison_lines(comparisons),
        "",
        "## Integrity checks (all passed before any result was read)",
        "",
        *_integrity_lines(integrity),
        "",
        "## Descriptive extras (not decision inputs)",
        "",
        "MAE, RMSE and the mean fitted margin slope `k`:",
        "",
        *_score_lines(scores, slopes),
        "",
        f"Single-game bootstrap of the same differences ({confidence:.2%}):",
        "",
        *_comparison_lines(extras["by_game"]),
    ]
    if extras["season"]:
        lines += [
            "",
            "Halves of the decision variant, season bootstrap at the same level:",
            "",
            *_comparison_lines(extras["season"]),
        ]
    lines += ["", "Plays moved, by season (count and EPA):", ""]
    for name, rows in extras["moved"].items():
        lines += [
            f"{_label(name)}:",
            "",
            *_table(rows, ("season", "removed", "removed_epa", "added", "added_epa")),
            "",
        ]
    if extras["wiped_sets"]:
        lines += [
            (
                "Penalty-wiped plays by set, 1999-2025, beside the phase-1 parity check's figures "
                "(every `no_play` play, both teams known or not: 8,848 with the rush flag at "
                "-5,660.46 EPA and 27,906 with the pass flag at +11,142.27). A wiped run is a "
                "`no_play` play today's snap rule counts, whichever flag makes it one (a play can "
                "carry several):"
            ),
            "",
            *_table(extras["wiped_sets"], ("set", "plays", "epa")),
            "",
        ]
    if extras["sides"]:
        lines += [
            "Penalty-wiped plays by the penalized side, 1999-2025:",
            "",
            *_table(extras["sides"], ("kind", "side", "plays", "epa")),
            "",
        ]
    lines += [
        "Team year-over-year Pearson of full-season `team_rating` (consecutive seasons, by team): "
        + "; ".join(
            f"{_label(name)} {values['pearson']:.3f} ({int(values['pairs']):,} pairs)"
            for name, values in extras["stability"].items()
        )
        + ".",
        "",
        (
            "Feature-value impact against today's, over every pre-week snapshot (team-weeks moved "
            "beyond 1e-9, largest and mean absolute change, points per game):"
        ),
        "",
    ]
    for name, impact in extras["impact"].items():
        lines.append(
            f"- {_label(name)}: "
            + "; ".join(
                f"`{column}` {int(values['rows_moved']):,} of {int(values['rows']):,}, largest "
                f"{values['largest']:.3f}, mean {values['mean']:.3f}"
                for column, values in impact.items()
            )
            + "."
        )
    today_ranks = dict(extras["ranks"][TODAY].select("team", "rank").iter_rows())
    lines += ["", f"{LAST_SEASON} full-season ranks that differ from today's:", ""]
    for name, frame in extras["ranks"].items():
        if name == TODAY:
            continue
        moved_teams = [
            f"{team_code} {today_ranks[team_code]}->{rank}"
            for team_code, rank in frame.select("team", "rank").iter_rows()
            if today_ranks[team_code] != rank
        ]
        lines.append(f"- {_label(name)}: {', '.join(moved_teams) or 'none'}.")
    return lines


# --- the report ------------------------------------------------------------------------------


def _interval_text(interval: Interval) -> str:
    """Format a paired difference and its interval."""
    lower, upper = common.bound_text(interval.lower), common.bound_text(interval.upper)
    return f"{interval.estimate:+.3f} ({lower} to {upper})"


def _label(name: str) -> str:
    """Return a variant's plain name."""
    return VARIANTS[name].label


def _comparison_lines(comparisons: Mapping[str, Comparison]) -> list[str]:
    """Return a table of each variant's paired differences, overall and by band."""
    bands = [OVERALL, *(name for name, _, _ in BANDS)]
    lines = ["| variant | " + " | ".join(bands) + " |", "| --- |" + " --- |" * len(bands)]
    for name, comparison in comparisons.items():
        cells = [_interval_text(comparison.overall)] + [
            _interval_text(comparison.bands[band]) for band in bands[1:]
        ]
        lines.append(f"| {_label(name)} | " + " | ".join(cells) + " |")
    return lines


def _score_lines(
    scores: Mapping[str, Mapping[str, tuple[int, float, float]]],
    slopes: Mapping[str, Mapping[str, float]],
) -> list[str]:
    """Return a table of each variant's games, MAE, RMSE and mean margin slope by band."""
    lines = [
        "| variant | band | games | MAE | RMSE | mean k |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, by_band in scores.items():
        for band, (games, mae, rmse) in by_band.items():
            lines.append(
                f"| {_label(name)} | {band} | {games:,} | {mae:.3f} | {rmse:.3f} | "
                f"{slopes[name][band]:.3f} |"
            )
    return lines


def _decision_lines(decision: Decision, spec: TestSpec, confidence: float) -> list[str]:
    """Return the rule's outcome in words, with every interval that excludes zero."""
    if decision.qualifying:
        outcome = (
            "qualifying: "
            + ", ".join(_label(name) for name in decision.qualifying)
            + f"; the recommendation is **{_label(decision.recommended)}** (the lowest MAE)."
        )
    else:
        outcome = (
            "no variant qualified, so the test cannot tell the variants from today's rating; the "
            f"recommendation is **{_label(decision.recommended)}**, the first definition in the "
            "order of simplicity that is not significantly worse. The order: "
            + "; ".join(_label(name) for name in spec.order)
            + "."
        )
    excluding = "; ".join(
        f"{_label(name)}, {band}: {direction}" for name, band, direction in decision.excluding_zero
    )
    guard = ", ".join(_label(name) for name in decision.band_guard) or "none"
    return [
        f"Rule as written ({confidence:.2%} season-bootstrap intervals): {outcome}",
        "",
        f"- Intervals that exclude zero, either direction: {excluding or 'none'}.",
        f"- Overall below zero but disqualified by a week band: {guard}.",
        "- The decision goes to the user; adoption is held (it changes every `ratings` column).",
    ]


def _table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[str]:
    """Return a Markdown table of ``rows`` over ``columns``, floats to three decimals."""

    def cell(value: object, column: str) -> str:
        if isinstance(value, float):
            return f"{value:,.3f}"
        if isinstance(value, int):
            return str(value) if column == "season" else f"{value:,}"
        return str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    lines.extend("| " + " | ".join(cell(row[c], c) for c in columns) + " |" for row in rows)
    return lines


def _provenance(cache_dir: Path, ratings_dir: Path, ledger: common.InputLedger) -> dict[str, Any]:
    """Return what reproduces the run: the code's and protocol's hashes and every input's."""
    files = [HERE / "team_plays_test.py", HERE / "rating_tests_common.py"]
    return {
        "nfl_predictor_version": metadata.version("nfl-predictor"),
        "code_sha256": {path.name: common.sha256_file(path) for path in files},
        "cache_dir": str(cache_dir),
        "ratings_dir": str(ratings_dir),
        "input_fingerprint": ledger.fingerprint(),
        "inputs": ledger.files,
    }


def run_test(test: str, cache_dir: Path, ratings_dir: Path, out_dir: Path) -> int:
    """Run one test end to end and write its report; return the exit code."""
    spec = TESTS[test]
    seasons = list(range(FIRST_SEASON, LAST_SEASON + 1))
    ledger = common.InputLedger()
    inputs = load_team_inputs(cache_dir, seasons, ledger)
    names = (TODAY, *spec.decision_variants, *spec.extra_variants)
    with threadpool_limits(limits=1, user_api="blas"):
        runs = {
            name: run_variant(
                name, variant_rows(inputs, VARIANTS[name]), seasons, check_information=True
            )
            for name in names
        }
        zero = run_variant(
            ZERO_CHANGE.name, variant_rows(inputs, ZERO_CHANGE), seasons, check_information=False
        )
    features = pl.concat([run.features for run in (*runs.values(), zero)])
    predictions = evaluate_feature_rows(features, START_WEEK)
    report = out_dir / spec.report
    lines = [f"# {spec.title}: test result", ""]
    try:
        integrity: dict[str, Any] = {
            "production": check_production(runs[TODAY], ratings_dir, ledger),
            "source_validation": check_source_validation(predictions),
            "variant_rows": check_variant_rows(inputs, runs),
            "zero_change_gap": check_zero_change(zero, runs[TODAY], predictions),
            "penalty_seasons": check_penalties(runs),
            "information_snapshots": {name: run.information_checked for name, run in runs.items()},
            "margin_rows": check_margin_rows(predictions, features),
        }
    except IntegrityError as error:
        lines += [f"Integrity check failed, so no result is read: {error}", ""]
        report.write_text("\n".join(common.wrap_markdown(lines)).rstrip("\n") + "\n")
        sys.stderr.write(f"{error}\n")
        return 1
    draws = common.season_draws(len(seasons))
    confidence = common.comparison_confidence(len(spec.decision_variants))
    comparisons = {
        name: compare_variant(predictions, name, seasons, draws, confidence)
        for name in spec.decision_variants
    }
    scores = {name: score(predictions, name) for name in names}
    slopes = {name: mean_slopes(predictions, name) for name in names}
    decision = decide(comparisons, {name: scores[name][OVERALL][1] for name in names}, spec.order)
    extras_by_game = {
        name: compare_by_game(predictions, name, confidence)
        for name in (*spec.decision_variants, *spec.extra_variants)
    }
    extras_season = {
        name: compare_variant(predictions, name, seasons, draws, confidence)
        for name in spec.extra_variants
    }
    moved = {
        name: moved_by_season(inputs, name)
        for name in (*spec.decision_variants, *spec.extra_variants)
    }
    stability = {name: year_over_year(run) for name, run in runs.items()}
    impact = {name: feature_impact(run, runs[TODAY]) for name, run in runs.items() if name != TODAY}
    ranks = {name: season_ranks(run, LAST_SEASON) for name, run in runs.items()}
    sides = moved_by_penalty_side(inputs) if test == "penalty-wiped" else []
    wiped_sets = wiped_totals(inputs) if test == "penalty-wiped" else []
    lines += _report_body(
        spec,
        ReportParts(
            integrity=integrity,
            comparisons=comparisons,
            decision=decision,
            confidence=confidence,
            scores=scores,
            slopes=slopes,
            extras={
                "by_game": extras_by_game,
                "season": extras_season,
                "moved": moved,
                "stability": stability,
                "impact": impact,
                "ranks": ranks,
                "sides": sides,
                "wiped_sets": wiped_sets,
            },
        ),
    )
    provenance = _provenance(cache_dir, ratings_dir, ledger)
    lines += [
        "",
        "## Provenance",
        "",
        f"- nfl-predictor {provenance['nfl_predictor_version']}; code SHA-256: "
        + ", ".join(f"`{name}` {digest[:12]}" for name, digest in provenance["code_sha256"].items())
        + ".",
        (
            f"- Inputs: {len(ledger.files)} files, fingerprint `{provenance['input_fingerprint']}` "
            f"(every file's SHA-256 is in `{spec.output_json}`), from `{cache_dir}` and "
            f"`{ratings_dir}`, read only, downloads blocked."
        ),
        f'- Command: `VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python '
        f".agents/sos_port/team_plays_test.py {test} --cache-dir {cache_dir} --ratings-dir "
        f"{ratings_dir}`; seconds per variant: "
        + ", ".join(f"{_safe_label(run.name)} {run.seconds:.0f}" for run in (*runs.values(), zero))
        + ".",
        "",
    ]
    report.write_text("\n".join(common.wrap_markdown(lines)).rstrip("\n") + "\n")
    payload = {
        "test": test,
        "integrity": integrity,
        "decision": {
            "recommended": decision.recommended,
            "qualifying": decision.qualifying,
            "band_guard": decision.band_guard,
            "excluding_zero": decision.excluding_zero,
            "confidence": confidence,
        },
        "comparisons": {name: _comparison_json(c) for name, c in comparisons.items()},
        "scores": scores,
        "slopes": slopes,
        "extras": {
            "by_game": {name: _comparison_json(c) for name, c in extras_by_game.items()},
            "season": {name: _comparison_json(c) for name, c in extras_season.items()},
            "moved": moved,
            "stability": stability,
            "impact": impact,
            "ranks": {name: frame.to_dicts() for name, frame in ranks.items()},
            "sides": sides,
            "wiped_sets": wiped_sets,
        },
        "provenance": provenance,
    }
    common.write_json_record(out_dir / spec.output_json, payload, _json_default)
    return 0


def _safe_label(name: str) -> str:
    """Return a variant's plain name, the zero-change check's included."""
    return ZERO_CHANGE.label if name == ZERO_CHANGE.name else _label(name)


def _comparison_json(comparison: Comparison) -> dict[str, list[float]]:
    """Return a comparison's intervals as lists of estimate, lower and upper."""
    return {
        OVERALL: [comparison.overall.estimate, comparison.overall.lower, comparison.overall.upper],
        **{
            band: [interval.estimate, interval.lower, interval.upper]
            for band, interval in comparison.bands.items()
        },
    }


def _json_default(value: object) -> object:
    """Return a JSON-friendly form of a value ``json`` cannot write."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("test", choices=sorted(TESTS), help="Which test to run.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="nflreadpy cache, read only.")
    parser.add_argument(
        "--ratings-dir", type=Path, required=True, help="Published ratings tables, read only."
    )
    parser.add_argument("--out-dir", type=Path, default=HERE, help="Where the report goes.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested test."""
    args = _parse_args(argv)
    return run_test(args.test, args.cache_dir, args.ratings_dir, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
