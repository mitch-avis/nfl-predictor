"""Score the four candidate power rankings on next-week margins, as pre-registered.

The design and decision rule are the main checkout's ``models/step6/rankings/HYPOTHESIS.md``
(read-only; its SHA-256 is checked against the one recorded beside it). Run from the repository
root with this checkout's environment:

    VIRTUAL_ENV="$PWD/.venv" .venv/bin/python .agents/sos_port/rankings_eval.py \
        --build-dir ~/scratch/step6/data \
        --run-dir /home/mitch/workspace/nfl-predictor/models/step6/ratings_anchored_seed42

``--build-dir`` is a dataset build's data directory: its ``completed_games_ml.csv`` (the scored
games, their neutral-site flags and the ported ratings' pre-week feature values),
``strength_snapshots.csv`` (the composite) and ``ratings/{season}_ratings_by_week.parquet`` (the
ported ratings of teams on a bye). ``--run-dir`` is the anchored ported-ratings walk-forward on
seed 42 (a run directory whose ``metadata.json`` names its fold checkpoints, or the checkpoint
directory); without it the model rating and the blend are reported as pending and no decision is
made. Given again (seed 42 first, then seed 7), the report adds a supplement: the model rating fit
to each game's predicted margin averaged over the runs, and the blend rebuilt on it, reported
beside the seed-42 result, which alone is the rule's input. The report records whether the
checkout had uncommitted changes to tracked files. ``--ratings-from-snapshot`` is for development:
it builds the ported ratings' weekly history from the read-only nflreadpy cache snapshot (as
``adapter_parity.py`` loads it) in a temporary directory instead of reading ``--build-dir``'s.
``--out-dir`` (default: beside this script) is where ``RANKINGS_EVAL.md`` and
``RANKINGS_EVAL.json`` go; nothing else is written.

The candidates (``nfl_predictor.reporting.rating_sources`` and ``power_rankings``), each one number
per team per week known before the week's games:

1. the ported team rating (the reference): the game rows' pre-week ``ratings_team`` values, with
   the published weekly history for teams on a bye;
2. the current composite in points through the week's own SRS;
3. the model rating: a ridge fit of team strengths and a home term to the walk-forward's pre-game
   predicted margins of the season's earlier games;
4. the blend ``w * (1) + (1 - w) * (3)``, ``w`` chosen each season on earlier seasons.

The score (``nfl_predictor.reporting.ranking_evaluation``): every regular-season game from week 2
in 2008-2025 (2007 warms up the home term and the blend weight); predicted margin
``rating[home] - rating[away] + h`` with ``h`` each candidate's least-squares home term over the
earlier seasons' scored games (zero at neutral sites); mean absolute error; paired
candidate-minus-reference intervals, 95%, over games (10,000 resamples, seed 42) and over seasons;
windows weeks 2-4, 5-8, 9-18 and all. Rank correlations and top-5 differences are reported, never
selected on.
"""

from __future__ import annotations

import os

# One BLAS and Polars thread unless the caller set the counts, so the ratings fits and the model
# rating's ridge solves are reproducible bit for bit (as adapter_parity.py runs).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import snapshot_cache  # a sibling module, importable once its dir is on the path
from ratings_source import (  # a sibling module, importable once its dir is on the path
    _common_dir,
    _git_dir,
    repo_root,
    sha256_file,
    source_head,
)

from nfl_predictor.ratings.pbp_inputs import team_games_from_pbp
from nfl_predictor.ratings.season import BootstrapSettings, build_team_season
from nfl_predictor.ratings.team_prior import PriorHistory
from nfl_predictor.reporting import ranking_evaluation as evaluation
from nfl_predictor.reporting import rating_sources
from nfl_predictor.utils.polars import loaders

HERE = Path(__file__).resolve().parent
OUT_MD = "RANKINGS_EVAL.md"
OUT_JSON = "RANKINGS_EVAL.json"
WARM_UP_SEASON = 2007
SCORED_SEASONS = tuple(range(2008, 2026))
RATED_SEASONS = (WARM_UP_SEASON, *SCORED_SEASONS)
RESAMPLES = 10_000
SEED = 42
# The walk-forward seed whose predictions the rule reads (the first --run-dir).
PREREGISTERED_SEED = 42
HYPOTHESIS_SHA256 = "44e3a33c6d8a8fc37cf6237543eed830f8cd53459f3e24f0684dfdafe4a17f40"
FIRST_PBP_SEASON = 1999
# The candidates in the hypothesis's order, with their plain names; the first is the reference.
CANDIDATES = ("ported", "composite", "model", "blend")
REFERENCE = CANDIDATES[0]
NAMES = {
    "ported": "the ported team rating (candidate 1, the reference)",
    "composite": "the current composite in points (candidate 2)",
    "model": "the model rating (candidate 3)",
    "blend": "the blend of the ported and model ratings (candidate 4)",
}
SHORT_NAMES = {
    "ported": "ported rating",
    "composite": "composite",
    "model": "model rating",
    "blend": "blend",
}
GAME_COLUMNS = (
    "game_id",
    "season",
    "week",
    "game_type",
    "away_abbr",
    "home_abbr",
    "away_score",
    "home_score",
    "neutral",
)
PORTED_FEATURES = tuple(f"{side}_{rating_sources.PORTED_FEATURE}" for side in ("away", "home"))
MISSING_EXAMPLES = 10


def main_checkout() -> Path:
    """Return the main checkout of this repository (the parent of the common git directory)."""
    return _common_dir(_git_dir(repo_root())).parent


@dataclass
class Inputs:
    """Everything the score reads, with where it came from."""

    games: pl.DataFrame
    teams_by_season: dict[int, list[str]]
    ratings: dict[str, pl.DataFrame]
    ported_check: dict[str, Any]
    run: dict[str, Any] | None
    files: dict[str, str] = field(default_factory=dict)


def _hash(path: Path, files: dict[str, str]) -> Path:
    """Record ``path``'s SHA-256 in ``files`` and return it."""
    files[str(path)] = sha256_file(path)
    return path


def load_games(path: Path, files: dict[str, str]) -> pl.DataFrame:
    """Return the build's regular-season games of the rated seasons, every week, with ratings."""
    header = pl.read_csv(path, n_rows=0).columns
    missing = [column for column in GAME_COLUMNS if column not in header]
    if missing:
        message = f"{path} lacks {missing}"
        raise SystemExit(message)
    features = [column for column in PORTED_FEATURES if column in header]
    overrides = {
        "season": pl.Int64,
        "week": pl.Int64,
        "away_score": pl.Float64,
        "home_score": pl.Float64,
        "neutral": pl.Int64,
        **dict.fromkeys(features, pl.Float64),
    }
    return (
        pl.read_csv(
            _hash(path, files),
            columns=[*GAME_COLUMNS, *features],
            schema_overrides=overrides,
            infer_schema_length=None,
        )
        .filter(pl.col("season").is_in(RATED_SEASONS) & (pl.col("game_type") == "REG"))
        .sort("season", "week", "game_id")
    )


def season_teams(games: pl.DataFrame) -> dict[int, list[str]]:
    """Return each season's teams on its regular-season schedule."""
    teams: dict[int, list[str]] = {}
    for rows in games.partition_by("season", maintain_order=True):
        sides = pl.concat([rows.get_column("away_abbr"), rows.get_column("home_abbr")])
        teams[int(rows.get_column("season")[0])] = sorted(sides.unique().to_list())
    return teams


def load_ported(
    games: pl.DataFrame, ratings_dir: Path, files: dict[str, str]
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Return the ported ratings: game-row values first, the weekly history for byes."""
    by_week = [
        rating_sources.ported_ratings_from_by_week(
            pl.read_parquet(_hash(ratings_dir / f"{season}_ratings_by_week.parquet", files)),
            season=season,
        )
        for season in RATED_SEASONS
    ]
    from_by_week = pl.concat(by_week)
    has_features = all(column in games.columns for column in PORTED_FEATURES)
    from_games = (
        rating_sources.ported_ratings_from_games(games)
        if has_features
        else pl.DataFrame(schema=rating_sources.RATING_SCHEMA)
    )
    combined = rating_sources.combine_ported_ratings(from_games, from_by_week)
    check = {
        "game_row_values": has_features,
        "team_weeks_from_game_rows": from_games.height,
        "team_weeks_from_weekly_history_only": combined.ratings.height - from_games.height,
        "team_weeks_in_both": combined.compared,
        "team_weeks_that_disagree": combined.disagreements,
        "tolerance": 1e-9,
    }
    return combined.ratings, check


def load_composite(path: Path, files: dict[str, str]) -> pl.DataFrame:
    """Return the composite in points for every week of the rated seasons."""
    snapshots = pl.read_csv(
        _hash(path, files),
        columns=["season", "week", "team_abbr", "adj_strength_composite", "adj_srs"],
        schema_overrides={
            "season": pl.Int64,
            "week": pl.Int64,
            "adj_strength_composite": pl.Float64,
            "adj_srs": pl.Float64,
        },
        infer_schema_length=None,
    ).filter(pl.col("season").is_in(RATED_SEASONS))
    return evaluation.composite_ratings_by_week(snapshots)


def load_run(run_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Return a run's predictions in the rated seasons, and its provenance."""
    read = rating_sources.read_fold_predictions(run_dir)
    predictions = read.predictions.filter(pl.col("season").is_in(RATED_SEASONS))
    config = read.metadata.get("config") or {}
    provenance = {
        "path": str(run_dir),
        "checkpoint_dir": str(read.checkpoint_dir),
        "folds": read.folds,
        "predicted_games": read.predictions.height,
        "predicted_games_in_rated_seasons": predictions.height,
        "seasons": sorted(set(read.predictions.get_column("season").to_list())),
        "dataset_hash": read.metadata.get("dataset_hash"),
        "git_commit": read.metadata.get("git_commit_hash"),
        "random_seed": config.get("random_seed"),
        "disabled_feature_groups": config.get("disabled_feature_groups"),
        "market_anchor": config.get("market_anchor"),
        "ridge_lambda": rating_sources.MODEL_RATING_RIDGE_LAMBDA,
    }
    return predictions, provenance


def code_changes() -> list[str]:
    """Return the tracked files of this checkout with uncommitted changes (read-only git)."""
    git = shutil.which("git")
    if git is None:
        message = "git is not on PATH; it records whether the code had uncommitted changes"
        raise SystemExit(message)
    status = subprocess.run(
        [git, "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    return [line[3:] for line in status.stdout.splitlines() if line.strip()]


def snapshot_team_games(files: dict[str, str]) -> dict[int, pl.DataFrame]:
    """Return every season's team-game rows from 1999 to the last rated season, from the snapshot.

    The snapshot ``adapter_parity.py`` reads is turned into this repository's cache files in a
    temporary directory with downloads blocked, and every season is loaded through the loaders
    and adapted (``team_games_from_pbp``).
    """
    snapshot_cache.block_downloads()
    snapshot = snapshot_cache.find_snapshot_files()
    files.update({f"snapshot:{name}": digest for name, digest in snapshot.hashes.items()})
    seasons = list(range(FIRST_PBP_SEASON, RATED_SEASONS[-1] + 1))
    rows: dict[int, pl.DataFrame] = {}
    with tempfile.TemporaryDirectory(prefix="rankings_eval_") as cache:
        snapshot_cache.write_cache(snapshot, Path(cache), seasons)
        options = {"cache_dir": Path(cache), "current_season": snapshot_cache.CURRENT_SEASON}
        for season in seasons:
            plays = loaders.load_pbp([season], **options)
            schedule = loaders.load_schedule([season], **options)
            rows[season] = team_games_from_pbp(plays, schedule).drop("season")
    return rows


def build_ratings_from_snapshot(directory: Path, files: dict[str, str]) -> None:
    """Write every rated season's ``ratings_by_week`` table, fit on the nflreadpy snapshot.

    For development without a dataset build: the team-game rows of
    :func:`snapshot_team_games` and the ported team tables of ``build_team_season`` (the
    penalties and the prior need every earlier season). The rank ranges are not read, so their
    bootstrap draws two resamples.
    """
    rows = snapshot_team_games(files)
    history = PriorHistory(lambda season: rows[season])
    for season in RATED_SEASONS:
        tables = build_team_season(
            season, rows[season], history, bootstrap=BootstrapSettings(resamples=2)
        )
        tables.ratings_by_week.write_parquet(directory / f"{season}_ratings_by_week.parquet")
        sys.stdout.write(f"{season}: ported weekly history built from the snapshot\n")
        sys.stdout.flush()


@dataclass
class Scored:
    """Each candidate's error on every scored game, and what its score was fit with."""

    errors: pl.DataFrame
    names: list[str]
    home_terms: dict[str, dict[int, float]]
    blend_choices: dict[int, evaluation.BlendChoice]
    missing: dict[str, list[str]]
    common: pl.DataFrame


def score_candidates(inputs: Inputs) -> Scored:
    """Return every available candidate's absolute error on every scored game of 2008-2025."""
    games = evaluation.scored_games(inputs.games, seasons=RATED_SEASONS)
    scored_rows = pl.col("season").is_in(SCORED_SEASONS)
    errors = games.filter(scored_rows).select("game_id", "season", "week")
    differences: dict[str, pl.Series] = {}
    home_terms: dict[str, dict[int, float]] = {}
    for name in ("ported", "composite", "model"):
        if name not in inputs.ratings:
            continue
        with_difference = evaluation.rating_differences(games, inputs.ratings[name])
        differences[name] = with_difference.get_column("difference")
        home_terms[name] = evaluation.home_terms_by_season(with_difference, SCORED_SEASONS)
        scored = evaluation.absolute_errors(with_difference.filter(scored_rows), home_terms[name])
        errors = errors.join(
            scored.select("game_id", pl.col("error").alias(name)), on="game_id", how="left"
        )
    choices: dict[int, evaluation.BlendChoice] = {}
    if "model" in differences:
        both = games.with_columns(
            differences["ported"].alias("ported_difference"),
            differences["model"].alias("model_difference"),
        )
        choices = evaluation.choose_blend_weights(both, SCORED_SEASONS)
        home_terms["blend"] = {season: choice.home_term for season, choice in choices.items()}
        blended = evaluation.blend_differences(both.filter(scored_rows), choices)
        scored = evaluation.absolute_errors(blended, home_terms["blend"])
        errors = errors.join(
            scored.select("game_id", pl.col("error").alias("blend")), on="game_id", how="left"
        )
    names = [name for name in CANDIDATES if name in errors.columns]
    missing = {
        name: errors.filter(pl.col(name).is_null()).get_column("game_id").to_list()
        for name in names
    }
    return Scored(
        errors=errors,
        names=names,
        home_terms=home_terms,
        blend_choices=choices,
        missing=missing,
        common=errors.drop_nulls(names),
    )


def window_results(scored: Scored) -> list[dict[str, Any]]:
    """Return each window's errors and every candidate's paired comparison with the reference."""
    results: list[dict[str, Any]] = []
    for label, first, last in evaluation.WINDOWS:
        in_window = pl.col("week") >= first
        if last is not None:
            in_window &= pl.col("week") <= last
        rows = scored.common.filter(in_window)
        comparisons: dict[str, Any] = {}
        for name in scored.names:
            if name == REFERENCE:
                continue
            paired = scored.common.select(
                "season",
                "week",
                pl.col(name).alias("candidate_error"),
                pl.col(REFERENCE).alias("reference_error"),
            )
            comparison = evaluation.paired_comparison(
                paired, first_week=first, last_week=last, resamples=RESAMPLES, seed=SEED
            )
            comparisons[name] = {
                "games": comparison.games,
                "mae": comparison.candidate_mae,
                "reference_mae": comparison.reference_mae,
                "difference": comparison.game_interval[0],
                "game_interval": list(comparison.game_interval[1:]),
                "season_interval": list(comparison.season_interval[1:]),
            }
        results.append(
            {
                "window": label,
                "first_week": first,
                "last_week": last,
                "games": rows.height,
                "mae": {name: float(rows.get_column(name).mean() or 0.0) for name in scored.names},
                "against_reference": comparisons,
            }
        )
    return results


def season_results(scored: Scored) -> list[dict[str, Any]]:
    """Return each candidate's mean absolute error per season, every scored week."""
    table = (
        scored.common.group_by("season")
        .agg(pl.len().alias("games"), *(pl.col(name).mean() for name in scored.names))
        .sort("season")
    )
    return table.to_dicts()


def agreement_results(inputs: Inputs, scored: Scored) -> dict[str, list[dict[str, Any]]]:
    """Return how closely each candidate's weekly order follows the reference's, by window."""
    weeks = scored.common.select("season", "week").unique().sort("season", "week").rows()
    week_pairs = [(int(season), int(week)) for season, week in weeks]
    candidates = {
        name: inputs.ratings[name] for name in ("composite", "model") if name in inputs.ratings
    }
    if scored.blend_choices:
        candidates["blend"] = pl.concat(
            rating_sources.blend_ratings(
                inputs.ratings["ported"].filter(pl.col("season") == season),
                inputs.ratings["model"].filter(pl.col("season") == season),
                choice.weight,
            )
            for season, choice in sorted(scored.blend_choices.items())
        )
    reference = inputs.ratings[REFERENCE]
    results: dict[str, list[dict[str, Any]]] = {}
    for name, ratings in candidates.items():
        table = evaluation.rank_agreement(ratings, reference, week_pairs)
        summaries = []
        for label, first, last in evaluation.WINDOWS:
            in_window = pl.col("week") >= first
            if last is not None:
                in_window &= pl.col("week") <= last
            rows = table.filter(in_window)
            spearman = rows.get_column("spearman").to_numpy()
            summaries.append(
                {
                    "window": label,
                    "weeks": rows.height,
                    "mean_spearman": float(np.nanmean(spearman)) if rows.height else None,
                    "lowest_spearman": float(np.nanmin(spearman)) if rows.height else None,
                    "weeks_top_5_differs": int(rows.get_column("top_differs").sum()),
                    "share_top_5_differs": float(rows.get_column("top_differs").mean() or 0.0),
                    "fewest_teams": int(rows.get_column("teams").min() or 0),
                }
            )
        results[name] = summaries
    return results


def decision(scored: Scored, windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered rule, or say why it cannot be applied yet."""
    if "model" not in scored.names:
        return {
            "status": "pending",
            "reason": (
                "the model rating and the blend need the anchored ported-ratings walk-forward "
                "on seed 42 (--run-dir); the rule is applied only when all four are scored"
            ),
        }
    gaps = {name: len(ids) for name, ids in scored.missing.items() if ids}
    if gaps:
        return {
            "status": "not applied",
            "reason": f"games without a prediction from some candidate: {gaps}",
        }
    overall = next(window for window in windows if window["last_week"] is None)
    verdicts = [
        evaluation.CandidateVerdict(
            name=name,
            mae=result["mae"],
            reference_mae=result["reference_mae"],
            game_high=result["game_interval"][1],
            season_high=result["season_interval"][1],
        )
        for name, result in overall["against_reference"].items()
    ]
    chosen = evaluation.decide(verdicts, reference=REFERENCE)
    return {
        "status": "applied",
        "winner": chosen.winner,
        "beat_the_reference": list(chosen.qualified),
    }


def _wrap(text: str, prefix: str = "") -> list[str]:
    return textwrap.wrap(
        text,
        width=100,
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def _interval(bounds: list[float]) -> str:
    return f"[{bounds[0]:+.3f}, {bounds[1]:+.3f}]"


def _decision_lines(outcome: dict[str, Any]) -> list[str]:
    if outcome["status"] == "pending":
        return _wrap(f"Pending: {outcome['reason']}.")
    if outcome["status"] == "not applied":
        return _wrap(f"Not applied: {outcome['reason']}; this goes to the user with the numbers.")
    winner = outcome["winner"]
    beat = outcome["beat_the_reference"]
    if winner == REFERENCE:
        text = (
            "No candidate has a lower all-weeks error with both intervals below zero, so the "
            f"reference stays: {NAMES[REFERENCE]}."
        )
    else:
        text = (
            f"{NAMES[winner].capitalize()} replaces the reference: the lowest all-weeks error "
            f"among the candidates whose game and season intervals both lie below zero "
            f"({', '.join(SHORT_NAMES[name] for name in beat)})."
        )
    return _wrap(text)


def _window_table(window: dict[str, Any], names: list[str]) -> list[str]:
    """Return one window's table: each candidate's error and its paired comparison."""
    lines = [
        (
            "| candidate | games | MAE | minus the reference | 95% interval, games | "
            "95% interval, seasons |"
        ),
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for name in names:
        mae = window["mae"][name]
        if name == REFERENCE:
            lines.append(f"| {SHORT_NAMES[name]} | {window['games']:,} | {mae:.3f} | | | |")
            continue
        result = window["against_reference"][name]
        lines.append(
            f"| {SHORT_NAMES[name]} | {result['games']:,} | {mae:.3f} | "
            f"{_signed(result['difference'])} | {_interval(result['game_interval'])} | "
            f"{_interval(result['season_interval'])} |"
        )
    return lines


def _weight_runs(weights: list[tuple[int, float]]) -> str:
    """Return the seasons' weights as runs of equal weight: ``0.7 in 2008-2009, 0.5 in 2010``."""
    runs: list[tuple[float, int, int]] = []
    for season, weight in sorted(weights):
        if runs and runs[-1][0] == weight and runs[-1][2] == season - 1:
            runs[-1] = (weight, runs[-1][1], season)
        else:
            runs.append((weight, season, season))
    return ", ".join(
        f"{weight:.1f} in {first}" + (f"-{last}" if last != first else "")
        for weight, first, last in runs
    )


def _supplement_lines(supplement: dict[str, Any], names: list[str]) -> list[str]:
    """Return the supplement: the model rating on predictions averaged over several seeds."""
    seeds = ", ".join(str(seed) for seed in supplement["seeds"])
    overall = next(window for window in supplement["windows"] if window["last_week"] is None)
    weights = _weight_runs(
        [(int(season), weight) for season, weight in supplement["blend_weights"].items()]
    )
    return [
        "",
        f"## Supplement: the model rating on seeds {seeds} (not the rule's input)",
        "",
        *_wrap(
            "Reported beside the result above, which the rule reads: the model rating fit to "
            "each game's predicted margin averaged over the walk-forward runs of these seeds "
            "(in this order), and the blend rebuilt on it with its own weights; the ported "
            "rating and the composite are unchanged. All scored weeks, "
            f"{supplement['scored_games']:,} games."
        ),
        "",
        *_window_table(overall, names),
        "",
        *_wrap(f"Blend weight on the ported rating by season: {weights}."),
    ]


def markdown(report: dict[str, Any]) -> str:
    """Return the Markdown report of ``report`` (the JSON record)."""
    names = report["candidates"]
    lines = [
        "# Power rankings: the candidate ratings on next-week margins",
        "",
        *_wrap(
            f"Generated by `.agents/sos_port/rankings_eval.py` on {report['generated']}; do not "
            f"edit by hand. `{OUT_JSON}` beside it holds every number below and the SHA-256 of "
            "every file read."
        ),
        "",
        *_wrap(
            "Design and rule: the main checkout's `models/step6/rankings/HYPOTHESIS.md`, "
            f"SHA-256 `{report['hypothesis']['sha256']}` (the recorded one). Code: this "
            f"repository at `{report['code_commit']}`"
            + (
                f" with uncommitted changes to {len(report['code_changes'])} tracked "
                f"file{'s' if len(report['code_changes']) != 1 else ''}"
                if report["code_dirty"]
                else ", no uncommitted change to a tracked file"
            )
            + f". Build: `{report['build_dir']}`"
            + (
                ", with the ported weekly history rebuilt from the nflreadpy cache snapshot "
                "(development only)."
                if report["ratings_from_snapshot"]
                else "."
            ),
            "- ",
        ),
    ]
    run = report["run"]
    lines += _wrap(
        (
            f"Model predictions: `{run['path']}` ({run['folds']} folds in "
            f"`{run['checkpoint_dir']}`, seed {run['random_seed']}, dataset "
            f"`{run['dataset_hash']}`, commit `{run['git_commit']}`)."
        )
        if run
        else "Model predictions: none given, so the model rating and the blend are pending.",
        "- ",
    )
    lines += _wrap(
        f"Scored: {report['scored_games']:,} regular-season games of {SCORED_SEASONS[0]}-"
        f"{SCORED_SEASONS[-1]} from week 2 that every candidate rates; {WARM_UP_SEASON} fits "
        f"the first home term and blend weight only. Intervals: {RESAMPLES:,} resamples, seed "
        f"{SEED}, over games and over seasons; a difference is candidate minus the reference.",
        "- ",
    )
    lines += ["", "## Decision (governing window: all weeks)", ""]
    lines += _decision_lines(report["decision"])
    if "note" in report["decision"]:
        lines += ["", *_wrap(f"Note: {report['decision']['note']}.")]
    for window in report["windows"]:
        lines += ["", f"## {window['window'].capitalize()}", "", *_window_table(window, names)]
    if report["supplement"] is not None:
        lines += _supplement_lines(report["supplement"], names)
    lines += [
        "",
        "## Rank agreement with the reference (reported, never selected on)",
        "",
        *_wrap(
            "Per week, over the teams both rate: the Spearman correlation of the two orders, "
            "and whether the two top-five sets differ."
        ),
        "",
        "| candidate | window | weeks | mean Spearman | lowest | weeks with another top 5 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, summaries in report["agreement"].items():
        for summary in summaries:
            lines.append(
                f"| {SHORT_NAMES[name]} | {summary['window']} | {summary['weeks']} | "
                f"{summary['mean_spearman']:.3f} | {summary['lowest_spearman']:.3f} | "
                f"{summary['weeks_top_5_differs']} ({summary['share_top_5_differs']:.0%}) |"
            )
    lines += ["", "## By season (all scored weeks)", ""]
    lines += [
        "| season | games | " + " | ".join(SHORT_NAMES[name] for name in names) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in names) + " |",
    ]
    for row in report["seasons"]:
        values = " | ".join(f"{row[name]:.3f}" for name in names)
        lines.append(f"| {row['season']} | {row['games']} | {values} |")
    lines += ["", "## Home terms and blend weights", ""]
    lines += _wrap(
        "Each candidate's home term for a season is fit by least squares on the scored "
        "non-neutral games of every earlier season from 2007. The blend's weight on the ported "
        "rating is chosen on the same games, with its own home term, a tie going to the larger "
        "weight."
    )
    header = "| season | " + " | ".join(SHORT_NAMES[name] for name in names)
    blend = report["blend_choices"]
    lines += [
        "",
        header + (" | blend weight |" if blend else " |"),
        "| --- | " + " | ".join("---:" for _ in names) + (" | ---: |" if blend else " |"),
    ]
    for season in SCORED_SEASONS:
        terms = " | ".join(f"{report['home_terms'][name][str(season)]:.2f}" for name in names)
        weight = f" | {blend[str(season)]['weight']:.1f} |" if blend else " |"
        lines.append(f"| {season} | {terms}{weight}")
    check = report["ported_check"]
    lines += ["", "## Checks", ""]
    lines += _wrap(
        f"Ported ratings: {check['team_weeks_from_game_rows']:,} team-weeks from the game rows' "
        f"pre-week values and {check['team_weeks_from_weekly_history_only']:,} from the weekly "
        f"history alone (teams on a bye); {check['team_weeks_in_both']:,} rated by both, "
        f"{check['team_weeks_that_disagree']} differing by more than {check['tolerance']:g}.",
        "- ",
    )
    for name, count in report["missing_games"].items():
        examples = report["missing_examples"][name]
        detail = f" (first: {', '.join(examples)})" if examples else ""
        lines += _wrap(
            f"{SHORT_NAMES[name].capitalize()}: {count} scored games without a rating{detail}.",
            "- ",
        )
    lines += _wrap(f"Files read: {len(report['files'])}, with their SHA-256 in `{OUT_JSON}`.", "- ")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Score the candidates and write the reports."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "A walk-forward run (or its checkpoint directory). The first is the rule's input; "
            "with more, a supplement scores the model rating on each game's predicted margin "
            "averaged over the runs, in the order given."
        ),
    )
    parser.add_argument("--ratings-from-snapshot", action="store_true")
    parser.add_argument(
        "--hypothesis",
        type=Path,
        default=main_checkout() / "models" / "step6" / "rankings" / "HYPOTHESIS.md",
    )
    parser.add_argument("--out-dir", type=Path, default=HERE)
    arguments = parser.parse_args(argv)
    hypothesis_sha256 = sha256_file(arguments.hypothesis)
    if hypothesis_sha256 != HYPOTHESIS_SHA256:
        message = f"{arguments.hypothesis} changed: SHA-256 {hypothesis_sha256}"
        raise SystemExit(message)

    files: dict[str, str] = {}
    games = load_games(arguments.build_dir / "completed_games_ml.csv", files)
    teams = season_teams(games)
    with tempfile.TemporaryDirectory(prefix="rankings_eval_ratings_") as directory:
        ratings_dir = arguments.build_dir / "ratings"
        if arguments.ratings_from_snapshot:
            ratings_dir = Path(directory)
            build_ratings_from_snapshot(ratings_dir, files)
        ported, ported_check = load_ported(games, ratings_dir, files)
    ratings = {
        "ported": ported,
        "composite": load_composite(arguments.build_dir / "strength_snapshots.csv", files),
    }
    runs = [load_run(path) for path in arguments.run_dir]
    run = runs[0][1] if runs else None
    if runs:
        ratings["model"] = rating_sources.model_ratings_by_week(runs[0][0], teams_by_season=teams)
    inputs = Inputs(games, teams, ratings, ported_check, run, files)

    scored = score_candidates(inputs)
    windows = window_results(scored)
    outcome = decision(scored, windows)
    if run is not None and run["random_seed"] != PREREGISTERED_SEED:
        outcome["note"] = (
            f"the rule's input is the run with seed {run['random_seed']}, not the pre-registered "
            f"seed {PREREGISTERED_SEED}"
        )
    supplement = None
    if len(runs) > 1:
        averaged = rating_sources.average_predictions([predictions for predictions, _ in runs])
        supplement_ratings = {
            **ratings,
            "model": rating_sources.model_ratings_by_week(averaged, teams_by_season=teams),
        }
        supplement_scored = score_candidates(
            Inputs(games, teams, supplement_ratings, ported_check, run, files)
        )
        supplement = {
            "seeds": [provenance["random_seed"] for _, provenance in runs],
            "scored_games": supplement_scored.common.height,
            "missing_games": {name: len(ids) for name, ids in supplement_scored.missing.items()},
            "windows": window_results(supplement_scored),
            "blend_weights": {
                str(season): choice.weight
                for season, choice in supplement_scored.blend_choices.items()
            },
        }
    changes = code_changes()
    report = {
        "generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "hypothesis": {"path": str(arguments.hypothesis), "sha256": hypothesis_sha256},
        "code_commit": source_head(repo_root()),
        "code_dirty": bool(changes),
        "code_changes": changes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS",)},
        },
        "build_dir": str(arguments.build_dir),
        "ratings_from_snapshot": arguments.ratings_from_snapshot,
        "run": run,
        "runs": [provenance for _, provenance in runs],
        "supplement": supplement,
        "parameters": {
            "warm_up_season": WARM_UP_SEASON,
            "scored_seasons": list(SCORED_SEASONS),
            "first_scored_week": evaluation.FIRST_SCORED_WEEK,
            "resamples": RESAMPLES,
            "seed": SEED,
            "blend_weights": list(evaluation.BLEND_WEIGHTS),
            "model_rating_ridge_lambda": rating_sources.MODEL_RATING_RIDGE_LAMBDA,
        },
        "candidates": scored.names,
        "reference": REFERENCE,
        "scored_games": scored.common.height,
        "missing_games": {name: len(ids) for name, ids in scored.missing.items()},
        "missing_examples": {name: ids[:MISSING_EXAMPLES] for name, ids in scored.missing.items()},
        "decision": outcome,
        "windows": windows,
        "seasons": season_results(scored),
        "agreement": agreement_results(inputs, scored),
        "home_terms": {
            name: {str(season): term for season, term in terms.items()}
            for name, terms in scored.home_terms.items()
        },
        "blend_choices": {
            str(season): {
                "weight": choice.weight,
                "home_term": choice.home_term,
                "earlier_games": choice.games,
                "earlier_mae": choice.mae,
            }
            for season, choice in scored.blend_choices.items()
        },
        "ported_check": ported_check,
        "files": files,
    }
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    (arguments.out_dir / OUT_JSON).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (arguments.out_dir / OUT_MD).write_text(markdown(report))
    overall = next(window for window in windows if window["last_week"] is None)
    summary = ", ".join(f"{SHORT_NAMES[name]} {overall['mae'][name]:.3f}" for name in scored.names)
    sys.stdout.write(
        f"all weeks, {overall['games']} games: {summary}; decision {outcome['status']}\n"
        f"wrote {arguments.out_dir / OUT_MD}\n"
    )


if __name__ == "__main__":
    main()
