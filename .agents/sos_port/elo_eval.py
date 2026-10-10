"""Score the own Elo's components on the 2007-2025 regular season and write the report.

Run from the repository root (a worktree is fine):

    VIRTUAL_ENV="$PWD/.venv" .venv/bin/python .agents/sos_port/elo_eval.py

It writes ``ELO_EVAL.md`` and ``ELO_EVAL.json`` beside this file and nothing else. It reads,
read-only: the main checkout's cached nflverse schedules and regular-season play-by-play
(``--data-dir``, by default the ``data/`` of the checkout that owns this repository's git
directory), that checkout's ``qb_elos.csv`` and ``meta_data.csv``, FiveThirtyEight's
``nfl-elo-game`` game file (``--fivethirtyeight``) and a downloaded copy of nfelosrs's
``wt_ratings.csv`` (``--win-totals``). Every input file's SHA-256 is recorded in the report.

What it does, in order (the rule is ``decisions/elo.md``, entry 1, written before this ran):

1. Checks against 538's published numbers: the framework seeded with 538's 1999 preseason
   ratings, run on this repository's schedules, against 538's ``elo_prob1``; the rest, travel
   and home-field constants inferred from 538's quarterback-model probabilities; the starter
   rule against the starters in ``qb_elos.csv``; and the VALUE adjustments against 538's.
2. The seeding decision and the forward selection of components, each a paired comparison of
   per-game Brier and log loss with a 5,000-resample game bootstrap (seed 42).
3. Diagnostics that decide nothing: each component alone, the recommended settings against the
   ``qb_elos.csv`` probabilities, week windows, seasons, playoff games, the rolling home field
   and the fitted win-total weights.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nfl_predictor import constants
from nfl_predictor.ml import metrics
from nfl_predictor.utils.polars import (
    elo_game_context,
    elo_quarterbacks,
    elo_ratings,
    elo_win_totals,
    qb_stats,
)

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "ELO_EVAL.md"
OUT_JSON = HERE / "ELO_EVAL.json"

FIRST_SEASON = 1999
LAST_SEASON = 2025
EVAL_FIRST_SEASON = 2007
EVAL_LAST_SEASON = 2025
# The walk-forward's bootstrap (walk_forward.BOOTSTRAP_SAMPLES) and the seed the rule names.
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 42
# The components in the rule's order, by their EloSettings field names.
STEP_ORDER = ("rest", "travel", "playoff_boost", "rolling_home_field", "quarterback", "win_totals")
COMPONENT_NAMES = {
    "rest": "rest and bye",
    "travel": "travel (55-point base plus 4 per 1,000 miles)",
    "playoff_boost": "playoff boost",
    "rolling_home_field": "rolling home field",
    "quarterback": "quarterback adjustment (VALUE)",
    "win_totals": "win-total pull",
}
SCHEDULE_COLUMNS = (
    "game_id",
    "season",
    "week",
    "game_type",
    "date",
    "home_abbr",
    "away_abbr",
    "home_score",
    "away_score",
    "neutral",
    "stadium_id",
)
WEEK_WINDOWS = (("week 1", 1, 1), ("week 2", 2, 2), ("weeks 3-18", 3, 18))
# 538's game file ends with the 2020 season.
FIVETHIRTYEIGHT_LAST_SEASON = 2020
# A probability gap above this is listed as a mismatch in the framework check.
PROB_GAP_FLAG = 1e-4
# A published quarterback adjustment below this marks a backup or debut start.
BACKUP_ADJUSTMENT = -100.0
# 538's game-level terms count as reproduced by its 2019 constants within this many points.
FIXED_MODEL_TOLERANCE = 0.5
# A published rolling value counts as reproduced within this many Elo points.
ROLLING_TOLERANCE = 0.01
# 538's stated preseason team rating: this share of the end-of-season one, the rest the starter's.
PRESEASON_CARRYOVER = 1.0 / 3.0


def _main_checkout_data() -> Path:
    """Return the ``data/`` directory of the checkout that owns this repository's git directory."""
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
        cwd=HERE,
    ).stdout.strip()
    return Path(common).parent / "data"


def _git_head() -> str:
    """Return the commit this script runs at."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE
    ).stdout.strip()


def _sha256(path: Path) -> str:
    """Return a file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_of_files(paths: list[Path]) -> str:
    """Return one SHA-256 over several files, in the given order (names and contents)."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Inputs:
    """Everything the evaluation reads, and the record of where it came from."""

    schedule: pl.DataFrame
    plays: pl.DataFrame
    qb_elos: pl.DataFrame
    identity: pl.DataFrame
    fivethirtyeight: pl.DataFrame
    win_totals: pl.DataFrame
    provenance: dict[str, Any]


def _season_files(data_dir: Path, pattern: str) -> list[Path]:
    """Return one cache file per season of the evaluation, in season order."""
    paths = [
        data_dir / "cache" / "nflreadpy" / pattern.format(season=season)
        for season in range(FIRST_SEASON, LAST_SEASON + 1)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        msg = f"missing cache files: {missing}"
        raise FileNotFoundError(msg)
    return paths


def _canonical(column: str) -> pl.Expr:
    """Return a team-code column mapped to this repository's canonical codes."""
    return pl.col(column).replace(constants.ALIAS_TO_CANONICAL)


def load_inputs(data_dir: Path, fivethirtyeight_path: Path, win_totals_path: Path) -> Inputs:
    """Read every input, read-only, and record its hash."""
    schedule_paths = _season_files(data_dir, "schedule_{season}.parquet")
    schedule = pl.concat(
        [pl.read_parquet(path).select(SCHEDULE_COLUMNS) for path in schedule_paths],
        how="vertical_relaxed",
    ).with_columns(
        pl.col("season").cast(pl.Int64),
        pl.col("week").cast(pl.Int64),
        pl.col("neutral").cast(pl.Int64),
    )
    play_paths = _season_files(data_dir, "pbp_{season}_reg.parquet")
    plays = pl.concat([pl.read_parquet(path) for path in play_paths], how="diagonal_relaxed")
    qb_elos_path = data_dir / "qb_elos.csv"
    qb_elos = (
        pl.read_csv(qb_elos_path, infer_schema_length=None)
        .filter(
            pl.col("week").is_not_null() & pl.col("season").is_between(FIRST_SEASON, LAST_SEASON)
        )
        .with_columns(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            _canonical("team1").alias("home_abbr"),
            _canonical("team2").alias("away_abbr"),
        )
    )
    meta_path = data_dir / "meta_data.csv"
    identity = qb_stats.load_qb_identity(meta_path)
    fivethirtyeight = pl.read_csv(fivethirtyeight_path, infer_schema_length=None).with_columns(
        _canonical("team1").alias("home_abbr"),
        _canonical("team2").alias("away_abbr"),
        pl.col("date").str.to_date(),
    )
    win_totals = elo_win_totals.parse_win_totals(win_totals_path.read_bytes())
    provenance = {
        "git_commit": _git_head(),
        "data_dir": str(data_dir),
        "schedule_files_sha256": _sha256_of_files(schedule_paths),
        "play_by_play_files_sha256": _sha256_of_files(play_paths),
        "qb_elos": {"path": str(qb_elos_path), "sha256": _sha256(qb_elos_path)},
        "meta_data": {"path": str(meta_path), "sha256": _sha256(meta_path)},
        "fivethirtyeight_games": {
            "path": str(fivethirtyeight_path),
            "sha256": _sha256(fivethirtyeight_path),
        },
        "win_totals": {"path": str(win_totals_path), "sha256": _sha256(win_totals_path)},
        "schedule_games": schedule.height,
        "play_rows": plays.height,
    }
    return Inputs(schedule, plays, qb_elos, identity, fivethirtyeight, win_totals, provenance)


@dataclass(frozen=True)
class Window:
    """The scored games and their outcomes, in one fixed order."""

    games: pl.DataFrame
    outcome: np.ndarray


def evaluation_window(schedule: pl.DataFrame) -> Window:
    """Return every completed regular-season game of 2007-2025 and its home-win outcome."""
    games = (
        schedule.filter(
            (pl.col("game_type") == "REG")
            & pl.col("season").is_between(EVAL_FIRST_SEASON, EVAL_LAST_SEASON)
            & pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
        )
        .select("game_id", "season", "week", "home_abbr", "away_abbr", "home_score", "away_score")
        .sort("game_id")
    )
    outcome = (games["home_score"] > games["away_score"]).cast(pl.Int64).to_numpy()
    return Window(games, outcome)


def aligned(window: Window, frame: pl.DataFrame, column: str) -> np.ndarray:
    """Return ``column`` of ``frame`` for the window's games, in the window's order."""
    joined = window.games.select("game_id").join(
        frame.select("game_id", column), on="game_id", how="left", maintain_order="left"
    )
    values = joined[column].cast(pl.Float64).to_numpy()
    if np.isnan(values).any():
        msg = f"{int(np.isnan(values).sum())} window games have no {column}"
        raise ValueError(msg)
    return values


def summary(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Return the Brier score and log loss of one set of probabilities."""
    squared, log_loss = metrics.probability_losses_per_row(outcome, probability)
    return {
        "games": len(outcome),
        "brier": float(squared.mean()),
        "log_loss": float(log_loss.mean()),
    }


def paired(outcome: np.ndarray, candidate: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    """Return candidate-minus-current Brier and log loss, with 95% game-bootstrap intervals."""
    cand_sq, cand_ll = metrics.probability_losses_per_row(outcome, candidate)
    curr_sq, curr_ll = metrics.probability_losses_per_row(outcome, current)
    sq_diff = cand_sq - curr_sq
    ll_diff = cand_ll - curr_ll
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_games = len(outcome)
    brier_draws = np.empty(BOOTSTRAP_SAMPLES)
    log_loss_draws = np.empty(BOOTSTRAP_SAMPLES)
    for draw in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, n_games, size=n_games)
        brier_draws[draw] = sq_diff[indices].mean()
        log_loss_draws[draw] = ll_diff[indices].mean()
    return {
        "games": n_games,
        "candidate_brier": float(cand_sq.mean()),
        "current_brier": float(curr_sq.mean()),
        "brier_diff": float(sq_diff.mean()),
        "brier_ci": [
            float(np.percentile(brier_draws, 2.5)),
            float(np.percentile(brier_draws, 97.5)),
        ],
        "candidate_log_loss": float(cand_ll.mean()),
        "current_log_loss": float(curr_ll.mean()),
        "log_loss_diff": float(ll_diff.mean()),
        "log_loss_ci": [
            float(np.percentile(log_loss_draws, 2.5)),
            float(np.percentile(log_loss_draws, 97.5)),
        ],
    }


def passes_rule(comparison: dict[str, Any]) -> bool:
    """Return the rule's verdict: Brier interval below zero, log-loss interval not above it."""
    return comparison["brier_ci"][1] < 0.0 and not comparison["log_loss_ci"][0] > 0.0


def settings_label(settings: elo_ratings.EloSettings) -> str:
    """Return the plain name of a setting: the framework plus its components."""
    parts = [COMPONENT_NAMES[name] for name in STEP_ORDER if getattr(settings, name)]
    return "framework" + "".join(f" + {part}" for part in parts)


def settings_dict(settings: elo_ratings.EloSettings) -> dict[str, bool]:
    """Return a setting's switches."""
    return dataclasses.asdict(settings)


@dataclass
class Runner:
    """Runs the own Elo on the evaluation's inputs, caching each setting's ratings."""

    inputs: Inputs
    window: Window
    qb_adjustments: pl.DataFrame
    seeds: dict[str, float] | None = None
    cache: dict[tuple[tuple[str, bool], ...], pl.DataFrame] = dataclasses.field(
        default_factory=dict
    )
    win_total_weights: dict[int, float] | None = None

    def weights(self, settings: elo_ratings.EloSettings) -> dict[int, float]:
        """Return the win-total weights fitted for ``settings``' other components."""
        return elo_ratings.fit_win_total_weights(
            self.inputs.schedule,
            settings=settings,
            inputs=self._inputs(settings, with_win_totals=True),
        )

    def _inputs(
        self, settings: elo_ratings.EloSettings, *, with_win_totals: bool
    ) -> elo_ratings.EloInputs:
        return elo_ratings.EloInputs(
            initial_ratings=self.seeds,
            qb_adjustments=self.qb_adjustments if settings.quarterback else None,
            win_totals=self.inputs.win_totals if with_win_totals else None,
        )

    def rated(self, settings: elo_ratings.EloSettings) -> pl.DataFrame:
        """Return every game's ratings for ``settings`` (the win-total weights fitted)."""
        key = tuple(sorted(settings_dict(settings).items()))
        if key not in self.cache:
            inputs = self._inputs(settings, with_win_totals=settings.win_totals)
            if settings.win_totals:
                weights = self.weights(settings)
                self.win_total_weights = weights
                inputs = dataclasses.replace(inputs, win_total_weights=weights)
            self.cache[key] = elo_ratings.compute_elo_ratings(
                self.inputs.schedule, settings=settings, inputs=inputs
            )
        return self.cache[key]

    def probability(self, settings: elo_ratings.EloSettings) -> np.ndarray:
        """Return the window games' home win probabilities for ``settings``."""
        return aligned(self.window, self.rated(settings), "home_win_prob")


def fivethirtyeight_seeds(games: pl.DataFrame) -> dict[str, float]:
    """Return 538's rating of each team before its first 1999 game (its 1999 preseason rating)."""
    first_season = games.filter(pl.col("season") == FIRST_SEASON).with_row_index("row")
    sides = pl.concat(
        [
            first_season.select(
                "row", pl.col("home_abbr").alias("team"), pl.col("elo1").alias("elo")
            ),
            first_season.select(
                "row", pl.col("away_abbr").alias("team"), pl.col("elo2").alias("elo")
            ),
        ]
    ).sort("row")
    first = sides.unique("team", keep="first", maintain_order=True).sort("team")
    return dict(zip(first["team"].to_list(), first["elo"].to_list(), strict=True))


def _framework_gaps(rated: pl.DataFrame, published: pl.DataFrame) -> pl.DataFrame:
    """Return the rated games matched to 538's file, with the probability and rating gaps."""
    keys = ["season", "date", "home_abbr", "away_abbr"]
    return rated.join(published, on=keys, how="inner").with_columns(
        (pl.col("home_win_prob") - pl.col("published_prob")).abs().alias("prob_gap"),
        pl.max_horizontal(
            (pl.col("home_rating") - pl.col("published_home")).abs(),
            (pl.col("away_rating") - pl.col("published_away")).abs(),
        ).alias("rating_gap"),
    )


def _gap_periods(joined: pl.DataFrame) -> list[dict[str, Any]]:
    """Return the gaps by period: 538 published three-decimal ratings before 2015."""
    periods = []
    for label, first, last in (
        ("1999-2014 (538 published ratings to 3 decimals)", 1999, 2014),
        ("2015-2020 (538 published full precision)", 2015, 2020),
    ):
        part = joined.filter(pl.col("season").is_between(first, last))
        periods.append(
            {
                "period": label,
                "games": part.height,
                "max_prob_gap": float(part["prob_gap"].max() or 0.0),
                "median_prob_gap": float(part["prob_gap"].median() or 0.0),
                "max_rating_gap": float(part["rating_gap"].max() or 0.0),
                "games_prob_gap_over_1e-4": int((part["prob_gap"] > PROB_GAP_FLAG).sum()),
            }
        )
    return periods


def _aligned_to_538(
    schedule: pl.DataFrame, published: pl.DataFrame
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Return the schedule with 538's home side and neutral flag wherever 538 lists the game.

    Also returns the ids of the games whose home and away sides were swapped, and of those
    whose neutral flag differs after any swap.
    """
    keys = ["season", "date", "home_abbr", "away_abbr"]
    direct = published.select(*keys, "published_neutral")
    swapped = published.select(
        "season",
        "date",
        pl.col("home_abbr").alias("away_abbr"),
        pl.col("away_abbr").alias("home_abbr"),
        pl.col("published_neutral").alias("swapped_neutral"),
    )
    joined = schedule.join(direct, on=keys, how="left", maintain_order="left").join(
        swapped, on=keys, how="left", maintain_order="left"
    )
    swap = pl.col("published_neutral").is_null() & pl.col("swapped_neutral").is_not_null()
    flag = pl.coalesce("published_neutral", "swapped_neutral")
    swaps = joined.filter(swap)["game_id"].to_list()
    flags = joined.filter(flag.is_not_null() & (flag != pl.col("neutral")))["game_id"].to_list()
    aligned_schedule = joined.with_columns(
        pl.when(swap).then(pl.col("away_abbr")).otherwise(pl.col("home_abbr")).alias("home_abbr"),
        pl.when(swap).then(pl.col("home_abbr")).otherwise(pl.col("away_abbr")).alias("away_abbr"),
        pl.when(swap)
        .then(pl.col("away_score"))
        .otherwise(pl.col("home_score"))
        .alias("home_score"),
        pl.when(swap)
        .then(pl.col("home_score"))
        .otherwise(pl.col("away_score"))
        .alias("away_score"),
        pl.coalesce(flag, pl.col("neutral")).cast(pl.Int64).alias("neutral"),
    ).drop("published_neutral", "swapped_neutral")
    return aligned_schedule, swaps, flags


def check_framework(inputs: Inputs, seeds: dict[str, float]) -> dict[str, Any]:
    """Compare the seeded framework on this repository's schedules with 538's published file.

    Run twice: on the schedules as they are, and with every game 538 lists given 538's home side
    and neutral-site flag, which isolates the games the two sources record differently.
    """
    published = inputs.fivethirtyeight.filter(
        pl.col("season").is_between(FIRST_SEASON, FIVETHIRTYEIGHT_LAST_SEASON)
    ).select(
        "season",
        "date",
        "home_abbr",
        "away_abbr",
        pl.col("elo1").alias("published_home"),
        pl.col("elo2").alias("published_away"),
        pl.col("elo_prob1").alias("published_prob"),
        pl.col("neutral").alias("published_neutral"),
    )
    schedule = inputs.schedule
    seeded = elo_ratings.EloInputs(initial_ratings=seeds)

    def rate(frame: pl.DataFrame) -> pl.DataFrame:
        rated = elo_ratings.compute_elo_ratings(frame, inputs=seeded)
        return rated.drop("home_abbr", "away_abbr").join(
            frame.select("game_id", "date", "home_abbr", "away_abbr"), on="game_id"
        )

    as_is = _framework_gaps(rate(schedule), published)
    aligned_schedule, swaps, flags = _aligned_to_538(schedule, published)
    aligned = _framework_gaps(rate(aligned_schedule), published)
    worst = as_is.sort("prob_gap", descending=True).head(5)
    return {
        "published_games_1999_2020": published.height,
        "matched_games": as_is.height,
        "aligned_matched_games": aligned.height,
        "home_side_swaps": swaps,
        "neutral_flag_differences": flags,
        "periods": _gap_periods(as_is),
        "periods_aligned_to_538": _gap_periods(aligned),
        "largest_gaps": [
            {
                "game_id": row["game_id"],
                "prob": row["home_win_prob"],
                "published_prob": row["published_prob"],
                "published_neutral": row["published_neutral"],
            }
            for row in worst.iter_rows(named=True)
        ],
    }


def _implied_diff(probability: pl.Expr) -> pl.Expr:
    """Return the Elo difference a probability implies: -400 * log10(1 / p - 1)."""
    return -elo_ratings.LOGISTIC_SCALE * (1.0 / probability - 1.0).log10()


def check_published_adjustments(inputs: Inputs) -> list[dict[str, Any]]:
    """Fit 538's home field, travel and bye terms from its quarterback-model probabilities.

    For each game, the published probability implies an Elo difference; dividing playoff games
    by 1.2 and subtracting the published base ratings and quarterback adjustments leaves the
    game-level terms. Non-neutral games are regressed on a home intercept, the visitor's extra
    miles (in thousands) and the bye difference.
    """
    schedule = inputs.schedule
    context = (
        schedule.select(
            "game_id", "season", "week", "home_abbr", "away_abbr", "neutral", "game_type"
        )
        .join(elo_game_context.travel_miles(schedule), on="game_id")
        .join(elo_game_context.bye_weeks(schedule), on="game_id")
    )
    published = inputs.qb_elos.select(
        "season",
        "week",
        "home_abbr",
        "away_abbr",
        "qbelo1_pre",
        "qbelo2_pre",
        "qb1_adj",
        "qb2_adj",
        "qbelo_prob1",
        pl.col("playoff").is_not_null().alias("playoff"),
    )
    joined = (
        context.join(published, on=["season", "week", "home_abbr", "away_abbr"], how="inner")
        .with_columns(
            (
                _implied_diff(pl.col("qbelo_prob1"))
                / pl.when(pl.col("playoff")).then(elo_ratings.PLAYOFF_MULTIPLIER).otherwise(1.0)
                - (
                    pl.col("qbelo1_pre")
                    - pl.col("qbelo2_pre")
                    + pl.col("qb1_adj")
                    - pl.col("qb2_adj")
                )
            ).alias("game_terms"),
            ((pl.col("away_miles") - pl.col("home_miles")) / 1000.0).alias("extra_kilomiles"),
            (pl.col("home_bye").cast(pl.Float64) - pl.col("away_bye").cast(pl.Float64)).alias(
                "bye_diff"
            ),
        )
        .filter(pl.col("game_terms").is_not_null() & pl.col("extra_kilomiles").is_not_null())
    )
    groups = (
        ("2007-2018, published in 2019 for past seasons", 2007, 2018),
        ("2019, live", 2019, 2019),
        ("2021, live", 2021, 2021),
    )
    fixed_model = pl.when(pl.col("neutral") == 0).then(
        elo_game_context.TRAVEL_BASE_HOME_FIELD
        + 1000.0 * elo_game_context.TRAVEL_ELO_PER_MILE * pl.col("extra_kilomiles")
    ).otherwise(0.0) + elo_game_context.BYE_ELO * pl.col("bye_diff")
    joined = joined.with_columns((pl.col("game_terms") - fixed_model).abs().alias("fixed_gap"))
    fits = []
    for label, first, last in groups:
        part = joined.filter(pl.col("season").is_between(first, last) & (pl.col("neutral") == 0))
        design = np.column_stack(
            [
                np.ones(part.height),
                part["extra_kilomiles"].to_numpy(),
                part["bye_diff"].to_numpy(),
            ]
        )
        response = part["game_terms"].to_numpy()
        coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        neutral = joined.filter(
            pl.col("season").is_between(first, last)
            & (pl.col("neutral") == 1)
            & (pl.col("game_type") != "SB")
        )
        fits.append(
            {
                "seasons": label,
                "games": part.height,
                "home_field": float(coefficients[0]),
                "elo_per_1000_miles": float(coefficients[1]),
                "elo_per_bye": float(coefficients[2]),
                "fixed_model_median_abs_gap": float(part["fixed_gap"].median() or 0.0),
                "fixed_model_share_within_half_point": float(
                    (part["fixed_gap"] <= FIXED_MODEL_TOLERANCE).mean() or 0.0
                ),
                "neutral_games": neutral.height,
                "neutral_share_within_half_point": float(
                    (neutral["fixed_gap"] <= FIXED_MODEL_TOLERANCE).mean() or 0.0
                )
                if neutral.height
                else math.nan,
            }
        )
    return fits


def check_published_rolling_updates(inputs: Inputs) -> dict[str, Any]:
    """Test candidate rolling rates against 538's published quarterback and team values.

    538's own seasons (2007-2022) in ``qb_elos.csv`` give, per side, the starter's value before
    and after the game, his game value and his adjustment over the team, all in Elo points, so
    the team's rating is ``value - adjustment``. A rate fits when the next published value is
    ``(1 - rate) * before + rate * game value``.
    """
    published = inputs.qb_elos.filter(pl.col("season").is_between(2007, 2022)).with_row_index("row")
    sides = pl.concat(
        [
            published.select(
                "row",
                "season",
                pl.col(f"{side}_abbr").alias("team"),
                pl.col(f"qb{number}_value_pre").alias("before"),
                pl.col(f"qb{number}_adj").alias("adjustment"),
                pl.col(f"qb{number}_game_value").alias("game"),
                pl.col(f"qb{number}_value_post").alias("after"),
            )
            for side, number in (("home", 1), ("away", 2))
        ]
    ).sort("team", "row")
    sides = sides.with_columns(
        (pl.col("before") - pl.col("adjustment")).alias("team_before"),
    ).with_columns(
        pl.col("team_before").shift(-1).over("team").alias("team_next"),
        pl.col("season").shift(-1).over("team").alias("next_season"),
    )
    quarterback = sides.filter(pl.col("game").is_not_null() & pl.col("after").is_not_null())
    team = sides.filter(
        pl.col("game").is_not_null()
        & pl.col("team_next").is_not_null()
        & (pl.col("next_season") == pl.col("season"))
    )
    ends = sides.with_columns(
        (0.95 * pl.col("team_before") + 0.05 * pl.col("game")).alias("team_end"),
        pl.col("season").shift(1).over("team").alias("previous_season"),
        (0.95 * pl.col("team_before") + 0.05 * pl.col("game"))
        .shift(1)
        .over("team")
        .alias("previous_end"),
    )
    starts = ends.filter(
        (pl.col("previous_season") == pl.col("season") - 1) & pl.col("previous_end").is_not_null()
    )
    stated = (
        PRESEASON_CARRYOVER * starts["previous_end"] + (1 - PRESEASON_CARRYOVER) * starts["before"]
    )
    preseason_gap = (starts["team_before"] - stated).abs()
    result: dict[str, Any] = {
        "quarterback": [],
        "team": [],
        "team_preseason": {
            "season_starts": starts.height,
            "median_abs_gap": float(preseason_gap.median() or 0.0),
            "share_within_0.01": float((preseason_gap <= ROLLING_TOLERANCE).mean() or 0.0),
        },
    }
    for rate in (0.1, 0.05):
        gap = (
            quarterback["after"] - ((1 - rate) * quarterback["before"] + rate * quarterback["game"])
        ).abs()
        result["quarterback"].append(
            {
                "rate": rate,
                "updates": quarterback.height,
                "median_abs_gap": float(gap.median() or 0.0),
                "share_within_0.01": float((gap <= ROLLING_TOLERANCE).mean() or 0.0),
            }
        )
        gap = (team["team_next"] - ((1 - rate) * team["team_before"] + rate * team["game"])).abs()
        result["team"].append(
            {
                "rate": rate,
                "updates": team.height,
                "median_abs_gap": float(gap.median() or 0.0),
                "share_within_0.01": float((gap <= ROLLING_TOLERANCE).mean() or 0.0),
            }
        )
    return result


def check_starters(inputs: Inputs, box: pl.DataFrame) -> dict[str, Any]:
    """Compare play-by-play starter rules with the starters ``qb_elos.csv`` names."""
    passing = qb_stats.aggregate_qb_game_stats(inputs.plays)
    most_dropbacks = (
        passing.sort(
            ["season", "week", "team_abbr", "dropbacks", "qb_id"],
            descending=[False, False, False, True, False],
        )
        .unique(["season", "week", "team_abbr"], keep="first", maintain_order=True)
        .select("season", "week", "team_abbr", pl.col("qb_id").alias("most_dropbacks"))
    )
    first_passer = box.filter(pl.col("starter")).select(
        "season", "week", "team_abbr", pl.col("qb_id").alias("first_passer")
    )
    published = pl.concat(
        [
            inputs.qb_elos.select(
                "season",
                "week",
                pl.col(f"{side}_abbr").alias("team_abbr"),
                pl.col(name).alias("qb_name"),
            )
            for side, name in (("home", "qb1"), ("away", "qb2"))
        ]
    ).join(inputs.identity, on="qb_name", how="left")
    joined = (
        published.join(first_passer, on=["season", "week", "team_abbr"], how="inner")
        .join(most_dropbacks, on=["season", "week", "team_abbr"], how="inner")
        .filter(pl.col("qb_id").is_not_null())
    )
    result: dict[str, Any] = {"team_games_with_a_mapped_starter": joined.height}
    for label, first, last in (("1999-2025", 1999, 2025), ("2007-2025", 2007, 2025)):
        part = joined.filter(pl.col("season").is_between(first, last))
        result[label] = {
            "team_games": part.height,
            "first_passer_agrees": float((part["first_passer"] == part["qb_id"]).mean() or 0.0),
            "most_dropbacks_agrees": float((part["most_dropbacks"] == part["qb_id"]).mean() or 0.0),
        }
    return result


def check_quarterback_adjustments(
    inputs: Inputs, adjustments: pl.DataFrame
) -> list[dict[str, Any]]:
    """Compare the VALUE adjustments with the published ones (538's, then nfeloqb's)."""
    joined = (
        inputs.schedule.filter(pl.col("game_type") == "REG")
        .select("game_id", "season", "week", "home_abbr", "away_abbr")
        .join(adjustments, on="game_id")
        .join(
            inputs.qb_elos.select("season", "week", "home_abbr", "away_abbr", "qb1_adj", "qb2_adj"),
            on=["season", "week", "home_abbr", "away_abbr"],
        )
    )
    rows = []
    for label, first, last in (
        ("2007-2022, 538's", 2007, 2022),
        ("2023-2025, nfeloqb's", 2023, 2025),
    ):
        part = joined.filter(pl.col("season").is_between(first, last))
        mine = np.concatenate([part["home_qb_adj"].to_numpy(), part["away_qb_adj"].to_numpy()])
        theirs = np.concatenate([part["qb1_adj"].to_numpy(), part["qb2_adj"].to_numpy()])
        keep = np.isfinite(mine) & np.isfinite(theirs)
        mine, theirs = mine[keep], theirs[keep]
        backups = theirs < BACKUP_ADJUSTMENT
        rows.append(
            {
                "seasons": label,
                "team_games": int(keep.sum()),
                "correlation": float(np.corrcoef(mine, theirs)[0, 1]),
                "mean_own": float(mine.mean()),
                "mean_published": float(theirs.mean()),
                "sd_own": float(mine.std()),
                "sd_published": float(theirs.std()),
                "mean_abs_gap": float(np.abs(mine - theirs).mean()),
                "published_below_minus_100": int(backups.sum()),
                "own_mean_there": float(mine[backups].mean()) if backups.any() else math.nan,
                "published_mean_there": float(theirs[backups].mean())
                if backups.any()
                else math.nan,
            }
        )
    return rows


def seeding_decision(runners: dict[str, Runner]) -> dict[str, Any]:
    """Compare 538's 1999 seeds with every team at 1505, on the framework, under the rule."""
    framework = elo_ratings.FRAMEWORK
    seeded = runners["seeded"].probability(framework)
    plain = runners["plain"].probability(framework)
    comparison = paired(runners["plain"].window.outcome, seeded, plain)
    return {
        "candidate": "538's 1999 preseason ratings",
        "current": "every team at 1505",
        **comparison,
        "max_window_prob_gap": float(np.abs(seeded - plain).max()),
        "kept": passes_rule(comparison),
    }


def forward_selection(runner: Runner) -> dict[str, Any]:
    """Add the components in the rule's order, keeping each one that passes."""
    kept = elo_ratings.FRAMEWORK
    market_free = kept
    steps = []
    outcome = runner.window.outcome
    for name in STEP_ORDER:
        if name == "win_totals":
            market_free = kept
        candidate = dataclasses.replace(kept, **{name: True})
        comparison = paired(outcome, runner.probability(candidate), runner.probability(kept))
        verdict = passes_rule(comparison)
        steps.append(
            {
                "component": name,
                "label": COMPONENT_NAMES[name],
                "candidate": settings_label(candidate),
                "current": settings_label(kept),
                "candidate_settings": settings_dict(candidate),
                "current_settings": settings_dict(kept),
                **comparison,
                "kept": verdict,
            }
        )
        if verdict:
            kept = candidate
    return {"steps": steps, "market_free": market_free, "anchored": kept}


def single_components(runner: Runner) -> list[dict[str, Any]]:
    """Return each component added alone to the framework (a diagnostic; decides nothing)."""
    framework = elo_ratings.FRAMEWORK
    base = runner.probability(framework)
    rows = []
    for name in STEP_ORDER:
        candidate = dataclasses.replace(framework, **{name: True})
        comparison = paired(runner.window.outcome, runner.probability(candidate), base)
        rows.append({"component": name, "label": COMPONENT_NAMES[name], **comparison})
    return rows


def reference_probabilities(inputs: Inputs, window: Window) -> dict[str, np.ndarray]:
    """Return ``qb_elos.csv``'s classic and quarterback-model probabilities for the window."""
    keyed = window.games.select("game_id", "season", "week", "home_abbr", "away_abbr").join(
        inputs.qb_elos.select(
            "season", "week", "home_abbr", "away_abbr", "elo_prob1", "qbelo_prob1"
        ),
        on=["season", "week", "home_abbr", "away_abbr"],
        how="left",
    )
    return {
        "qb_elos.csv elo_prob1 (classic)": aligned(window, keyed, "elo_prob1"),
        "qb_elos.csv qbelo_prob1 (quarterback model)": aligned(window, keyed, "qbelo_prob1"),
    }


def against_references(
    outcome: np.ndarray, finals: dict[str, np.ndarray], references: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Return each recommended setting minus each reference, paired on the window's games."""
    return [
        {"own": own_name, "reference": ref_name, **paired(outcome, own, reference)}
        for own_name, own in finals.items()
        for ref_name, reference in references.items()
    ]


def breakdowns(window: Window, probabilities: dict[str, np.ndarray]) -> dict[str, Any]:
    """Return Brier and log loss by week window and by season for each set of probabilities."""
    weeks = window.games["week"].to_numpy()
    seasons = window.games["season"].to_numpy()
    by_window = {
        name: {
            label: summary(
                window.outcome[(weeks >= first) & (weeks <= last)],
                p[(weeks >= first) & (weeks <= last)],
            )
            for label, first, last in WEEK_WINDOWS
        }
        for name, p in probabilities.items()
    }
    by_season = {
        name: {
            str(season): summary(window.outcome[seasons == season], p[seasons == season])["brier"]
            for season in range(EVAL_FIRST_SEASON, EVAL_LAST_SEASON + 1)
        }
        for name, p in probabilities.items()
    }
    return {"by_week_window": by_window, "brier_by_season": by_season}


def playoff_diagnostic(runner: Runner, kept: elo_ratings.EloSettings) -> dict[str, Any]:
    """Return the kept setting's playoff-game scores with and without the playoff boost."""
    playoff_games = runner.inputs.schedule.filter(
        (pl.col("game_type") != "REG")
        & pl.col("season").is_between(EVAL_FIRST_SEASON, EVAL_LAST_SEASON)
        & pl.col("home_score").is_not_null()
    ).select("game_id", "home_score", "away_score")
    outcome = (playoff_games["home_score"] > playoff_games["away_score"]).cast(pl.Int64).to_numpy()
    window = Window(playoff_games, outcome)
    results = {}
    for label, boost in (("without the boost", False), ("with the boost", True)):
        settings = dataclasses.replace(kept, playoff_boost=boost)
        results[label] = summary(outcome, aligned(window, runner.rated(settings), "home_win_prob"))
    return {"base": settings_label(dataclasses.replace(kept, playoff_boost=False)), **results}


def rolling_home_field_values(schedule: pl.DataFrame, *, travel: bool) -> dict[str, float | None]:
    """Return the rolling home field at each season's week 1."""
    weekly = elo_game_context.rolling_home_field(schedule, travel=travel).filter(
        pl.col("week") == 1
    )
    return {
        str(season): value
        for season, value in weekly.sort("season").select("season", "home_field").iter_rows()
    }


def pre_kickoff_starters(
    runner: Runner, previous: Runner, selection: dict[str, Any]
) -> dict[str, Any]:
    """Score the presets with every game rated for the team's previous starter.

    A completed game's starter is the passer on its first dropback, known only at kickoff.
    ``previous`` rates every game with the team's last starter instead, as known before kickoff,
    while its ratings still roll in the actual starter's game. A diagnostic: the rule reads the
    forward selection, not this.
    """
    outcome = runner.window.outcome
    presets = {"market-free": selection["market_free"], "win-total": selection["anchored"]}
    rows = []
    for label, settings in presets.items():
        actual = runner.probability(settings)
        pre_kickoff = previous.probability(settings)
        rows.append(
            {
                "preset": f"{label}: {settings_label(settings)}",
                "actual_starter": summary(outcome, actual),
                "previous_starter": summary(outcome, pre_kickoff),
                "previous_minus_actual": paired(outcome, pre_kickoff, actual),
            }
        )
    step = next(s for s in selection["steps"] if s["component"] == "quarterback")
    candidate = elo_ratings.EloSettings(**step["candidate_settings"])
    current = elo_ratings.EloSettings(**step["current_settings"])
    gain = paired(outcome, previous.probability(candidate), previous.probability(current))
    return {
        "presets": rows,
        "quarterback_step": {
            "candidate": settings_label(candidate),
            "current": settings_label(current),
            "with_actual_starters": {
                key: step[key] for key in ("brier_diff", "brier_ci", "log_loss_diff", "log_loss_ci")
            },
            "with_previous_starters": gain,
        },
    }


def evaluate(data_dir: Path, fivethirtyeight_path: Path, win_totals_path: Path) -> dict[str, Any]:
    """Run every check, the seeding decision, the selection and the diagnostics."""
    inputs = load_inputs(data_dir, fivethirtyeight_path, win_totals_path)
    window = evaluation_window(inputs.schedule)
    box = elo_quarterbacks.quarterback_box_scores(inputs.plays)
    adjustments = elo_quarterbacks.value_qb_adjustments(inputs.schedule, box)
    previous_adjustments = elo_quarterbacks.value_qb_adjustments(
        inputs.schedule, box, pre_game_starter="last_starter"
    )
    seeds = fivethirtyeight_seeds(inputs.fivethirtyeight)
    runners = {
        "plain": Runner(inputs, window, adjustments),
        "seeded": Runner(inputs, window, adjustments, seeds=seeds),
    }
    seeding = seeding_decision(runners)
    runner = runners["seeded"] if seeding["kept"] else runners["plain"]
    previous = Runner(inputs, window, previous_adjustments, seeds=runner.seeds)
    selection = forward_selection(runner)
    market_free = selection["market_free"]
    anchored = selection["anchored"]
    finals = {f"market-free: {settings_label(market_free)}": runner.probability(market_free)}
    if anchored != market_free:
        finals[f"anchored: {settings_label(anchored)}"] = runner.probability(anchored)
    references = reference_probabilities(inputs, window)
    weights = runner.weights(market_free)
    return {
        "provenance": inputs.provenance,
        "window": {
            "description": "completed regular-season games, 2007-2025",
            "games": window.games.height,
            "home_wins": int(window.outcome.sum()),
        },
        "checks": {
            "framework_against_538": check_framework(inputs, seeds),
            "published_adjustments": check_published_adjustments(inputs),
            "starters": check_starters(inputs, box),
            "rolling_updates": check_published_rolling_updates(inputs),
            "quarterback_adjustments": check_quarterback_adjustments(inputs, adjustments),
        },
        "seeds_538_1999": seeds,
        "seeding": seeding,
        "selection": {
            "steps": selection["steps"],
            "market_free": settings_dict(market_free),
            "market_free_label": settings_label(market_free),
            "anchored": settings_dict(anchored),
            "anchored_label": settings_label(anchored),
        },
        "single_components": single_components(runner),
        "pre_kickoff_starters": pre_kickoff_starters(
            runner,
            previous,
            {
                "market_free": market_free,
                "anchored": anchored,
                "steps": selection["steps"],
            },
        ),
        "summaries": {
            name: summary(window.outcome, p) for name, p in {**finals, **references}.items()
        },
        "against_references": against_references(window.outcome, finals, references),
        "breakdowns": breakdowns(window, {**finals, **references}),
        "playoff_games": playoff_diagnostic(runner, anchored),
        "rolling_home_field_week_1": rolling_home_field_values(
            inputs.schedule, travel=market_free.travel
        ),
        "win_total_weights": {str(season): weight for season, weight in sorted(weights.items())},
    }


def _paragraph(*parts: str) -> list[str]:
    """Return one paragraph, wrapped at 100 columns, and the blank line after it."""
    wrapped = textwrap.wrap(
        " ".join(parts), width=100, break_long_words=False, break_on_hyphens=False
    )
    return [*wrapped, ""]


def _number(value: float, digits: int = 5) -> str:
    """Return a number with a fixed count of decimals."""
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def _signed(value: float, digits: int = 5) -> str:
    """Return a signed number with a fixed count of decimals."""
    return f"{value:+.{digits}f}"


def _interval(bounds: list[float], digits: int = 5) -> str:
    """Return an interval as ``[low, high]``."""
    return f"[{_signed(bounds[0], digits)}, {_signed(bounds[1], digits)}]"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    """Return a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return [*lines, ""]


def _comparison_cells(comparison: dict[str, Any]) -> list[str]:
    """Return the Brier and log-loss difference cells of a paired comparison."""
    return [
        _signed(comparison["brier_diff"]),
        _interval(comparison["brier_ci"]),
        _signed(comparison["log_loss_diff"]),
        _interval(comparison["log_loss_ci"]),
    ]


def _selection_lines(result: dict[str, Any]) -> list[str]:
    """Return the seeding and forward-selection sections."""
    seeding = result["seeding"]
    selection = result["selection"]
    lines = [
        "## Seeding",
        "",
        "538's 1999 preseason ratings against every team at 1505, both on the framework.",
        "Difference is seeds minus 1505 on the window's games, with 95% game-bootstrap intervals.",
        "",
    ]
    lines += _table(
        [
            "candidate",
            "games",
            "Brier diff",
            "95% interval",
            "log-loss diff",
            "95% interval",
            "kept",
        ],
        [
            [
                seeding["candidate"],
                str(seeding["games"]),
                f"{seeding['brier_diff']:+.2e}",
                "[{:+.2e}, {:+.2e}]".format(*seeding["brier_ci"]),
                f"{seeding['log_loss_diff']:+.2e}",
                "[{:+.2e}, {:+.2e}]".format(*seeding["log_loss_ci"]),
                "yes" if seeding["kept"] else "no",
            ]
        ],
    )
    lines += [
        (
            "Largest change in a window probability from the seeds: "
            f"{_number(seeding['max_window_prob_gap'], 6)}."
        ),
        "",
        "## Forward selection",
        "",
        "Each candidate is the components kept so far plus one; difference is candidate minus",
        "current, per game, on the window's games.",
        "",
    ]
    rows = [
        [
            step["label"],
            _number(step["candidate_brier"]),
            _number(step["current_brier"]),
            *_comparison_cells(step),
            "yes" if step["kept"] else "no",
        ]
        for step in selection["steps"]
    ]
    lines += _table(
        [
            "component",
            "candidate Brier",
            "current Brier",
            "Brier diff",
            "95% interval",
            "log-loss diff",
            "95% interval",
            "kept",
        ],
        rows,
    )
    lines += _paragraph(f"Market-free recommendation: {selection['market_free_label']}.")
    lines += _paragraph(f"Anchored recommendation: {selection['anchored_label']}.")
    return lines


def _reference_lines(result: dict[str, Any]) -> list[str]:
    """Return the summaries, the comparisons with qb_elos.csv and the breakdowns."""
    lines = ["## Recommended settings and the qb_elos.csv probabilities", ""]
    lines += _table(
        ["probabilities", "games", "Brier", "log loss"],
        [
            [name, str(s["games"]), _number(s["brier"]), _number(s["log_loss"])]
            for name, s in result["summaries"].items()
        ],
    )
    lines += ["Own minus reference, per game, on the same games:", ""]
    lines += _table(
        ["own", "reference", "Brier diff", "95% interval", "log-loss diff", "95% interval"],
        [
            [row["own"], row["reference"], *_comparison_cells(row)]
            for row in result["against_references"]
        ],
    )
    windows = result["breakdowns"]["by_week_window"]
    lines += ["### By week window", ""]
    lines += _table(
        ["probabilities", *[f"{label} Brier" for label, _, _ in WEEK_WINDOWS]],
        [
            [name, *[_number(windows[name][label]["brier"]) for label, _, _ in WEEK_WINDOWS]]
            for name in windows
        ],
    )
    seasons = result["breakdowns"]["brier_by_season"]
    names = list(seasons)
    lines += ["### Brier score by season", ""]
    lines += _table(
        ["season", *names],
        [
            [str(season), *[_number(seasons[name][str(season)]) for name in names]]
            for season in range(EVAL_FIRST_SEASON, EVAL_LAST_SEASON + 1)
        ],
    )
    return lines


def _diagnostic_lines(result: dict[str, Any]) -> list[str]:
    """Return the single-component, playoff, home-field and win-total diagnostics."""
    lines = [
        "## Each component alone on the framework",
        "",
        "A diagnostic: the rule reads the forward selection above, not this table.",
        "",
    ]
    lines += _table(
        ["component", "Brier diff", "95% interval", "log-loss diff", "95% interval"],
        [[row["label"], *_comparison_cells(row)] for row in result["single_components"]],
    )
    starters = result["pre_kickoff_starters"]
    lines += ["## Starters known before kickoff", ""]
    lines += _paragraph(
        "A diagnostic: the rule reads the forward selection above, not this. A completed game's",
        "starter is the passer on its first dropback, known only at kickoff. Here every game is",
        "rated for the team's previous starter instead, as known before kickoff, while the",
        "ratings still roll in the actual starter's game. Difference is previous minus actual.",
    )
    lines += _table(
        [
            "preset",
            "actual-starter Brier",
            "previous-starter Brier",
            "Brier diff",
            "95% interval",
            "log-loss diff",
            "95% interval",
        ],
        [
            [
                row["preset"],
                _number(row["actual_starter"]["brier"]),
                _number(row["previous_starter"]["brier"]),
                *_comparison_cells(row["previous_minus_actual"]),
            ]
            for row in starters["presets"]
        ],
    )
    step = starters["quarterback_step"]
    lines += _paragraph(
        f"The quarterback step ({step['candidate']} against {step['current']}), candidate minus",
        "current:",
    )
    lines += _table(
        ["starters", "Brier diff", "95% interval", "log-loss diff", "95% interval"],
        [
            ["actual (the rule's step)", *_comparison_cells(step["with_actual_starters"])],
            ["previous", *_comparison_cells(step["with_previous_starters"])],
        ],
    )
    playoff = result["playoff_games"]
    lines += ["## Playoff games, 2007-2025", ""]
    lines += _paragraph(
        f"The anchored recommendation's components without the playoff boost ({playoff['base']}),",
        "and with it. A diagnostic: the boost is judged on the regular-season window.",
    )
    lines += _table(
        ["setting", "games", "Brier", "log loss"],
        [
            [
                label,
                str(playoff[label]["games"]),
                _number(playoff[label]["brier"]),
                _number(playoff[label]["log_loss"]),
            ]
            for label in ("without the boost", "with the boost")
        ],
    )
    lines += ["## Rolling home field at week 1", "", "Elo points; null in the first season.", ""]
    lines += _table(
        ["season", "home field"],
        [
            [season, "null" if value is None else _number(value, 2)]
            for season, value in result["rolling_home_field_week_1"].items()
        ],
    )
    lines += [
        "## Win-total weights",
        "",
        "Fitted per season on the earlier win-total seasons, for a pull added to the market-free",
        "recommendation.",
        "",
    ]
    lines += _table(
        ["season", "weight"],
        [[season, _number(weight, 2)] for season, weight in result["win_total_weights"].items()],
    )
    return lines


def _check_lines(result: dict[str, Any]) -> list[str]:
    """Return the checks against 538's published numbers."""
    checks = result["checks"]
    framework = checks["framework_against_538"]
    lines = [
        "## Checks against FiveThirtyEight's published numbers",
        "",
        "### The framework on this repository's schedules",
        "",
        "Seeded with 538's 1999 preseason ratings and run on the cached nflverse schedules,",
        "against `elo_prob1` and the pre-game ratings in 538's `nfl_games.csv`, matched on",
        f"season, date and teams: {framework['matched_games']} of",
        f"{framework['published_games_1999_2020']} published 1999-2020 games matched.",
        "",
    ]
    period_header = [
        "period",
        "games",
        "max prob gap",
        "median prob gap",
        "max rating gap",
        "gaps over 1e-4",
    ]

    def period_rows(periods: list[dict[str, Any]]) -> list[list[str]]:
        return [
            [
                period["period"],
                str(period["games"]),
                f"{period['max_prob_gap']:.2e}",
                f"{period['median_prob_gap']:.2e}",
                f"{period['max_rating_gap']:.2e}",
                str(period["games_prob_gap_over_1e-4"]),
            ]
            for period in periods
        ]

    lines += _table(period_header, period_rows(framework["periods"]))
    lines += ["Largest gaps:", ""]
    lines += _table(
        ["game", "own", "538", "538 neutral flag"],
        [
            [
                gap["game_id"],
                _number(gap["prob"], 6),
                _number(gap["published_prob"], 6),
                str(gap["published_neutral"]),
            ]
            for gap in framework["largest_gaps"]
        ],
    )
    lines += _paragraph(
        f"The two sources record {len(framework['home_side_swaps'])} games with home and away",
        "swapped:",
        ", ".join(f"`{game}`" for game in framework["home_side_swaps"]) + ".",
        f"After those swaps {len(framework['neutral_flag_differences'])} games carry a different",
        "neutral-site flag:",
        ", ".join(f"`{game}`" for game in framework["neutral_flag_differences"]) + ".",
        "The same run with 538's home side and flag on every game 538 lists",
        f"({framework['aligned_matched_games']} matched):",
    )
    lines += _table(period_header, period_rows(framework["periods_aligned_to_538"]))
    lines += [
        "### 538's game-level terms, read from its quarterback-model probabilities",
        "",
        "The published probability's implied difference (playoffs divided by 1.2) minus the",
        "published ratings and quarterback adjustments leaves 538's game-level terms. For",
        "non-neutral games they are regressed on a home intercept, the visitor's extra miles",
        "(thousands) and the bye difference, and compared with 538's stated constants.",
        "",
    ]
    lines += _table(
        [
            "seasons",
            "games",
            "least-squares home field",
            "per 1,000 miles",
            "per bye",
            "538 constants: median gap",
            "within 0.5",
            "neutral games within 0.5",
        ],
        [
            [
                fit["seasons"],
                str(fit["games"]),
                _number(fit["home_field"], 2),
                _number(fit["elo_per_1000_miles"], 2),
                _number(fit["elo_per_bye"], 2),
                _number(fit["fixed_model_median_abs_gap"], 3),
                _number(fit["fixed_model_share_within_half_point"], 3),
                f"{_number(fit['neutral_share_within_half_point'], 3)} of {fit['neutral_games']}",
            ]
            for fit in checks["published_adjustments"]
        ],
    )
    lines += [
        "538 constants: 55 points at home plus 4 per 1,000 extra visitor miles plus 25 per bye,",
        "the model `elo_game_context` implements; at a neutral site (not the Super Bowl) only the",
        "bye term. The least-squares columns are pulled by a few large manual adjustments (for",
        "example the 250-point penalty for a team resting starters), the gap columns are not.",
        "",
    ]
    starters = checks["starters"]
    lines += [
        "### Starters",
        "",
        "Share of team-games whose play-by-play starter is the one `qb_elos.csv` names (mapped",
        "to GSIS ids through `meta_data.csv`).",
        "",
    ]
    lines += _table(
        ["seasons", "team-games", "first passer", "most dropbacks"],
        [
            [
                label,
                str(starters[label]["team_games"]),
                _number(starters[label]["first_passer_agrees"], 4),
                _number(starters[label]["most_dropbacks_agrees"], 4),
            ]
            for label in ("1999-2025", "2007-2025")
        ],
    )
    rolling = checks["rolling_updates"]
    lines += [
        "### 538's rolling updates",
        "",
        "538's seasons 2007-2022: how often the next published value is the candidate rolling",
        "update of the published one (within 0.01 Elo points). A quarterback's value moves with",
        "his own starts; the team's (value minus adjustment) with each game of the season.",
        "",
    ]
    lines += _table(
        ["rating", "rate", "updates", "median abs gap", "within 0.01"],
        [
            [
                name,
                _number(row["rate"], 2),
                str(row["updates"]),
                _number(row["median_abs_gap"], 4),
                _number(row["share_within_0.01"], 3),
            ]
            for name in ("quarterback", "team")
            for row in rolling[name]
        ],
    )
    preseason = rolling["team_preseason"]
    lines += _paragraph(
        "At a team's first game of a season, 538's stated rule (a third of the previous",
        "season's end-of-season team value, two thirds the starter's preseason value) against",
        f"the published team value: {preseason['season_starts']} season starts, median gap",
        f"{_number(preseason['median_abs_gap'], 2)} Elo points, within 0.01 for",
        f"{_number(preseason['share_within_0.01'], 3)} of them.",
    )
    lines += [
        "### Quarterback adjustments",
        "",
        "This repository's VALUE adjustments against the published `qb1_adj` and `qb2_adj`,",
        "regular season, in Elo points.",
        "",
    ]
    lines += _table(
        [
            "seasons",
            "team-games",
            "correlation",
            "mean own / published",
            "SD own / published",
            "mean abs gap",
            "published below -100: own / published mean",
        ],
        [
            [
                row["seasons"],
                str(row["team_games"]),
                _number(row["correlation"], 3),
                f"{_number(row['mean_own'], 2)} / {_number(row['mean_published'], 2)}",
                f"{_number(row['sd_own'], 2)} / {_number(row['sd_published'], 2)}",
                _number(row["mean_abs_gap"], 2),
                (
                    f"{row['published_below_minus_100']}: "
                    f"{_number(row['own_mean_there'], 1)} / "
                    f"{_number(row['published_mean_there'], 1)}"
                ),
            ]
            for row in checks["quarterback_adjustments"]
        ],
    )
    return lines


def markdown(result: dict[str, Any]) -> str:
    """Return the report."""
    provenance = result["provenance"]
    window = result["window"]
    lines = [
        "# Own Elo evaluation",
        "",
        "Generated by `.agents/sos_port/elo_eval.py`; do not edit by hand. Reproduce from the",
        "repository root with:",
        "",
        "```bash",
        'VIRTUAL_ENV="$PWD/.venv" .venv/bin/python .agents/sos_port/elo_eval.py',
        "```",
        "",
        "The rule that reads these numbers is `decisions/elo.md`, entry 1, written before the",
        "first run. Every probability is the variant's own pre-game home win probability.",
        "",
        f"Window: {window['description']}, {window['games']} games, {window['home_wins']} home",
        "wins (a tie is not one). Intervals are 95% percentile intervals over",
        f"{BOOTSTRAP_SAMPLES} resamples of the games (seed {BOOTSTRAP_SEED}).",
        "",
        "## Inputs",
        "",
    ]
    lines += _table(
        ["input", "detail"],
        [
            ["code commit", f"`{provenance['git_commit']}`"],
            ["data directory", f"`{provenance['data_dir']}`"],
            [
                "schedule cache files (1999-2025) SHA-256",
                f"`{provenance['schedule_files_sha256']}`",
            ],
            [
                "play-by-play cache files (1999-2025) SHA-256",
                f"`{provenance['play_by_play_files_sha256']}`",
            ],
            ["qb_elos.csv SHA-256", f"`{provenance['qb_elos']['sha256']}`"],
            ["meta_data.csv SHA-256", f"`{provenance['meta_data']['sha256']}`"],
            ["538 nfl_games.csv SHA-256", f"`{provenance['fivethirtyeight_games']['sha256']}`"],
            ["nfelosrs wt_ratings.csv SHA-256", f"`{provenance['win_totals']['sha256']}`"],
        ],
    )
    lines += _selection_lines(result)
    lines += _reference_lines(result)
    lines += _diagnostic_lines(result)
    lines += _check_lines(result)
    lines += ["## 538's 1999 preseason ratings", ""]
    lines += _table(
        ["team", "rating"],
        [[team, _number(rating, 3)] for team, rating in result["seeds_538_1999"].items()],
    )
    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> int:
    """Run the evaluation and write both reports."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--fivethirtyeight",
        type=Path,
        default=Path("~/scratch/elo_research/nfl-elo-game/data/nfl_games.csv").expanduser(),
    )
    parser.add_argument(
        "--win-totals",
        type=Path,
        default=Path("~/scratch/elo_research/wt_ratings_nfelosrs.csv").expanduser(),
    )
    args = parser.parse_args()
    data_dir = args.data_dir or _main_checkout_data()
    result = evaluate(data_dir, args.fivethirtyeight, args.win_totals)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
