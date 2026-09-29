"""Paired comparison of walk-forward runs, rescored from their fold checkpoints.

A walk-forward run saves every finished week as ``fold_<season>_w<week>.joblib`` under its
checkpoint directory; each file holds that week's prediction frame. This module rescores those
frames directly (never a run's ``metrics_report.json``) and compares a candidate with a reference
game by game, which is how a reviewer checks a walk-forward result independently.

Definitions, per game, on the deterministic probability ``p`` (``deterministic_home_win_prob``)
against ``actual_home_win`` (ties are coded 0 and scored that way):

- Brier ``(p - y)^2``; log loss with ``p`` clipped to ``[1e-15, 1 - 1e-15]``.
- Pick accuracy: home is picked when ``p > 0.5`` and away when ``p < 0.5``; a pick is correct only
  when that side won outright, so ties are incorrect for both sides.
- Margin and total absolute error against ``actual_margin`` and ``actual_total``.
- Confidence-pool points: within each (season, week), games are ranked ``1..N`` by ``|p - 0.5|``
  rounded to 12 decimals (``CONFIDENCE_DECIMALS`` in ``nfl_predictor.ml.metrics``), ascending,
  with equal rounded confidences broken by ``game_id`` order, and a correct pick scores its rank.
  Rounding makes mathematically equal confidences (a home favorite and a home underdog by the
  same spread) tie almost always instead of being ordered by floating-point noise; a pair whose
  noise straddles a 12-decimal rounding boundary can still differ. Reported as the window's
  total.
- The market view is ``market_home_win_prob`` from the same rows.

Paired differences are candidate minus reference. Loss columns are bootstrapped over games (5,000
resamples by default, numpy ``default_rng(0)``, 2.5 and 97.5 percentiles of the resampled mean);
pool points over (season, week) blocks, on the window's summed difference. Given several
candidate/reference pairs (one per seed, same seed paired with same seed), the per-game (or
per-week) differences are averaged over the pairs before bootstrapping, so seed-to-seed variance
is folded into the estimate.

Windows: week 1, week 2, weeks 3-18, and all weeks.

The stability view splits every window by season with the same definitions: one row per season
(and, for a single run, an all-seasons row), for each run and for the paired contrast. A row with
fewer than two games has no game-bootstrap interval, and a row with fewer than two weeks (week 1
or week 2 of one season) has no pool-points interval, because resampling a single unit only
returns it. ``stability_report`` builds the same view for one run from its prediction frame, which
is how ``nfl-predictor backtest`` adds it to ``metrics_report.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd

from nfl_predictor import constants
from nfl_predictor.ml.metrics import confidence_ranks

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_RESAMPLES = 5000
DEFAULT_BOOTSTRAP_SEED = 0

# (label, first week, last week or None for no upper bound)
WINDOWS: tuple[tuple[str, int, int | None], ...] = (
    ("week 1", 1, 1),
    ("week 2", 2, 2),
    ("weeks 3-18", 3, 18),
    ("all weeks", 1, None),
)
LOSS_COLUMNS = ("brier", "log_loss", "margin_ae", "total_ae", "correct")
CONTRAST_COLUMNS = (*LOSS_COLUMNS, "pool")
COLUMN_LABELS = {
    "brier": "det Brier",
    "log_loss": "det log loss",
    "correct": "pick acc",
    "margin_ae": "margin MAE",
    "total_ae": "total MAE",
    "pool": "pool pts",
    "market_brier": "market Brier",
}
REQUIRED_COLUMNS = (
    "game_id",
    "season",
    "week",
    "deterministic_home_win_prob",
    "market_home_win_prob",
    "actual_home_win",
    "actual_margin",
    "actual_total",
    "predicted_margin",
    "predicted_total",
)
RUN_COLUMNS = ("brier", "log_loss", "correct", "margin_ae", "total_ae", "pool", "market_brier")
ALL_SEASONS = "all seasons"
STABILITY_HEADING = "## Stability by season"
# An interval needs at least this many resampling units (games, or weeks for pool points).
_MIN_RESAMPLING_UNITS = 2
_PROB_CLIP = 1e-15
_HEADS = ("margin_model", "total_model")
# Metadata config keys left out of the configuration difference: where the run's files live, and
# long records that follow from the settings compared anyway.
_IGNORED_CONFIG_KEYS = ("checkpoint", "feature_list", "eval_window", "data_path", "out_json")


@dataclass(frozen=True)
class RunInput:
    """One walk-forward run to compare: its fold checkpoints and, when known, its metadata."""

    label: str
    checkpoint_dir: Path
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class LoadedRun:
    """A run's per-game scores and the provenance read from its fold checkpoints."""

    run: RunInput
    games: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)


def resolve_run(path: Path) -> RunInput:
    """Return the run at ``path``: a run directory with ``metadata.json``, or a checkpoint dir.

    Raises:
        ValueError: If ``path`` is neither.

    """
    metadata_path = path / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = (metadata.get("config") or {}).get("checkpoint") or {}
        if not checkpoint.get("dir"):
            msg = f"{metadata_path} names no walk-forward checkpoint directory"
            raise ValueError(msg)
        return RunInput(path.name, Path(checkpoint["dir"]), metadata)
    if any(path.glob("fold_*.joblib")):
        return RunInput(path.name, path)
    msg = f"{path} is neither a walk-forward run directory nor a checkpoint directory"
    raise ValueError(msg)


def _fold_files(checkpoint_dir: Path) -> list[Path]:
    """Return a checkpoint directory's fold files in order, or raise when it has none."""
    files = sorted(checkpoint_dir.glob("fold_*.joblib"))
    if not files:
        msg = f"no fold checkpoints in {checkpoint_dir}"
        raise ValueError(msg)
    return files


def load_run(run: RunInput) -> LoadedRun:
    """Read a run's fold checkpoints into per-game scores, with provenance checks.

    Raises:
        ValueError: If the checkpoints lack a column the comparison needs, or repeat a game.

    """
    frames: list[pd.DataFrame] = []
    best_iterations: set[tuple[str, Any]] = set()
    early_stopped = False
    calibration_methods: set[str] = set()
    files = _fold_files(run.checkpoint_dir)
    for path in files:
        payload = joblib.load(path)
        metrics = payload.get("metrics") or {}
        for head in _HEADS:
            best_iterations.add((head, metrics.get(f"{head}.best_iteration")))
            early_stopped = early_stopped or bool(metrics.get(f"{head}.early_stopped"))
        frame = payload["predictions"]
        if "calibration_method" in frame.columns:
            calibration_methods.update(str(value) for value in frame["calibration_method"])
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    missing = [column for column in REQUIRED_COLUMNS if column not in predictions.columns]
    if missing:
        msg = f"{run.label}: fold checkpoints lack {missing}"
        raise ValueError(msg)
    if predictions["game_id"].duplicated().any():
        msg = f"{run.label}: fold checkpoints repeat a game"
        raise ValueError(msg)
    metadata = run.metadata or {}
    config = metadata.get("config") or {}
    provenance = {
        "checkpoint_dir": str(run.checkpoint_dir),
        "folds": len(files),
        "games": len(predictions),
        "dataset_hash": metadata.get("dataset_hash"),
        "git_commit": metadata.get("git_commit_hash"),
        "random_seed": config.get("random_seed"),
        "best_iteration": sorted(
            f"{head}={value}" for head, value in best_iterations if value is not None
        ),
        "early_stopped": early_stopped,
        "calibration_methods": sorted(calibration_methods),
    }
    return LoadedRun(run, per_game_scores(predictions), provenance)


def per_game_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the per-game losses and pool points of one run, sorted by ``game_id``."""
    frame = predictions.sort_values("game_id").reset_index(drop=True)
    p = frame["deterministic_home_win_prob"].to_numpy(float)
    y = frame["actual_home_win"].to_numpy(float)
    margin = frame["actual_margin"].to_numpy(float)
    even = constants.EVEN_ODDS_PROBABILITY
    clipped = np.clip(p, _PROB_CLIP, 1 - _PROB_CLIP)
    games = pd.DataFrame(
        {
            "game_id": frame["game_id"],
            "season": frame["season"].astype(int),
            "week": frame["week"].astype(int),
            "p": p,
            "brier": (p - y) ** 2,
            "log_loss": -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)),
            "margin_ae": np.abs(frame["predicted_margin"].to_numpy(float) - margin),
            "total_ae": np.abs(
                frame["predicted_total"].to_numpy(float) - frame["actual_total"].to_numpy(float)
            ),
            "correct": (((p > even) & (margin > 0)) | ((p < even) & (margin < 0))).astype(float),
            "market_brier": (frame["market_home_win_prob"].to_numpy(float) - y) ** 2,
        }
    )
    games["rank"] = confidence_ranks(
        p,
        tiebreaker=games["game_id"].to_numpy(),
        groups=(games["season"].to_numpy(), games["week"].to_numpy()),
    )
    games["pool"] = games["rank"] * games["correct"]
    return games


def window_mask(weeks: np.ndarray, first: int, last: int | None) -> np.ndarray:
    """Return which rows fall in the window ``first..last`` (``last=None``: no upper bound)."""
    mask = weeks >= first
    if last is not None:
        mask &= weeks <= last
    return mask


def bootstrap_mean(diff: np.ndarray, resamples: int, seed: int) -> tuple[float, float, float]:
    """Return the mean of ``diff`` and its 95% game-bootstrap interval."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = diff[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_sum(diff: np.ndarray, resamples: int, seed: int) -> tuple[float, float, float]:
    """Return the sum of ``diff`` and its 95% block-bootstrap interval."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    sums = diff[rng.integers(0, n, size=(resamples, n))].sum(axis=1)
    return float(diff.sum()), float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))


def _week_pool(games: pd.DataFrame) -> pd.Series:
    """Return pool points per (season, week)."""
    return games.groupby(["season", "week"])["pool"].sum()


def run_metrics(games: pd.DataFrame, mask: np.ndarray, resamples: int, seed: int) -> dict[str, Any]:
    """Return one run's window metrics, with its deterministic-minus-market Brier interval."""
    window = games[mask]
    row: dict[str, Any] = {
        column: float(window[column].mean()) for column in (*LOSS_COLUMNS, "market_brier")
    }
    row["pool"] = float(window["pool"].sum())
    row["det_minus_market_brier"] = bootstrap_mean(
        (window["brier"] - window["market_brier"]).to_numpy(), resamples, seed
    )
    return row


def paired_contrast(
    pairs: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    mask: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    """Return candidate-minus-reference estimates and intervals, averaged over the pairs."""
    result: dict[str, tuple[float, float, float]] = {}
    for column in LOSS_COLUMNS:
        diff = np.mean(
            [
                candidate[column].to_numpy()[mask] - reference[column].to_numpy()[mask]
                for candidate, reference in pairs
            ],
            axis=0,
        )
        result[column] = bootstrap_mean(diff, resamples, seed)
    week_diff = np.mean(
        [
            (_week_pool(candidate[mask]) - _week_pool(reference[mask])).to_numpy()
            for candidate, reference in pairs
        ],
        axis=0,
    )
    result["pool"] = bootstrap_sum(week_diff, resamples, seed)
    return result


def _week_count(games: pd.DataFrame, mask: np.ndarray) -> int:
    """Return how many (season, week) blocks the masked rows cover."""
    return len(games.loc[mask, ["season", "week"]].drop_duplicates())


def _drop_interval(values: Sequence[float | None], units: int) -> tuple[Any, ...]:
    """Return ``values`` unchanged, or without its interval when too few units were resampled."""
    if units < _MIN_RESAMPLING_UNITS:
        return (values[0], None, None)
    return tuple(values)


def _season_masks(games: pd.DataFrame, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Split a window mask by season, keyed by the season as text, in season order."""
    seasons = games["season"].to_numpy()
    return {
        str(season): mask & (seasons == season) for season in sorted(set(seasons[mask].tolist()))
    }


def stability_run_row(
    games: pd.DataFrame, mask: np.ndarray, resamples: int, seed: int
) -> dict[str, Any]:
    """Return one run's metrics on one stability row: ``run_metrics`` with the interval guard."""
    row = run_metrics(games, mask, resamples, seed)
    row["det_minus_market_brier"] = _drop_interval(row["det_minus_market_brier"], int(mask.sum()))
    return row


def stability_contrast(
    pairs: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    mask: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, tuple[Any, ...]]:
    """Return ``paired_contrast`` on one stability row, with the interval guard."""
    contrast: dict[str, tuple[Any, ...]] = dict(paired_contrast(pairs, mask, resamples, seed))
    games = int(mask.sum())
    for column in LOSS_COLUMNS:
        contrast[column] = _drop_interval(contrast[column], games)
    contrast["pool"] = _drop_interval(contrast["pool"], _week_count(pairs[0][0], mask))
    return contrast


def stability_report(
    predictions: pd.DataFrame,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return one run's metrics per window, for all seasons and for each season.

    ``predictions`` is a walk-forward prediction frame (the fold checkpoints' rows); the metrics
    are the ones ``compare`` reports for a run, from ``per_game_scores`` and ``run_metrics``.
    """
    games = per_game_scores(predictions)
    weeks = games["week"].to_numpy()
    windows: dict[str, dict[str, dict[str, Any]]] = {}
    for label, first, last in WINDOWS:
        mask = window_mask(weeks, first, last)
        if not mask.any():
            continue
        rows = {ALL_SEASONS: mask, **_season_masks(games, mask)}
        windows[label] = {
            season: {
                "games": int(season_mask.sum()),
                **stability_run_row(games, season_mask, resamples, seed),
            }
            for season, season_mask in rows.items()
        }
    return {"resamples": resamples, "bootstrap_seed": seed, "windows": windows}


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested config dicts into dotted keys."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def config_differences(
    candidate: dict[str, Any] | None, reference: dict[str, Any] | None
) -> dict[str, tuple[Any, Any]]:
    """Return the metadata config settings that differ between two runs."""
    if not candidate or not reference:
        return {}
    left = _flatten(candidate.get("config") or {})
    right = _flatten(reference.get("config") or {})
    differences: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(left) | set(right)):
        if key.split(".")[0] in _IGNORED_CONFIG_KEYS:
            continue
        if left.get(key) != right.get(key):
            differences[key] = (left.get(key), right.get(key))
    for key in ("dataset_hash", "git_commit_hash"):
        if candidate.get(key) != reference.get(key):
            differences[key] = (candidate.get(key), reference.get(key))
    return differences


def compare_runs(
    candidates: Sequence[LoadedRun],
    references: Sequence[LoadedRun],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compare candidate runs with reference runs, pairing them in order.

    Raises:
        ValueError: If the counts differ, no run is given, or the runs cover different games.

    """
    if not candidates or len(candidates) != len(references):
        msg = "give one reference run per candidate run (same seeds, same order)"
        raise ValueError(msg)
    runs = [*candidates, *references]
    base = runs[0].games
    for run in runs[1:]:
        if not np.array_equal(run.games["game_id"].to_numpy(), base["game_id"].to_numpy()):
            msg = (
                f"{run.run.label} and {runs[0].run.label} cover different games; "
                "a paired comparison needs the same games"
            )
            raise ValueError(msg)
    market_view_identical = all(
        np.array_equal(run.games["market_brier"].to_numpy(), base["market_brier"].to_numpy())
        for run in runs[1:]
    )
    pairs = [(c.games, r.games) for c, r in zip(candidates, references, strict=True)]
    weeks = base["week"].to_numpy()
    windows: dict[str, Any] = {}
    for label, first, last in WINDOWS:
        mask = window_mask(weeks, first, last)
        if not mask.any():
            continue
        windows[label] = {
            "games": int(mask.sum()),
            "runs": {run.run.label: run_metrics(run.games, mask, resamples, seed) for run in runs},
            "contrast": paired_contrast(pairs, mask, resamples, seed),
            "seasons": {
                season: {
                    "games": int(season_mask.sum()),
                    "runs": {
                        run.run.label: stability_run_row(run.games, season_mask, resamples, seed)
                        for run in runs
                    },
                    "contrast": stability_contrast(pairs, season_mask, resamples, seed),
                }
                for season, season_mask in _season_masks(base, mask).items()
            },
        }
    return {
        "candidates": [run.run.label for run in candidates],
        "references": [run.run.label for run in references],
        "resamples": resamples,
        "bootstrap_seed": seed,
        "provenance": {run.run.label: run.provenance for run in runs},
        "config_differences": {
            f"{c.run.label} vs {r.run.label}": {
                key: list(values)
                for key, values in config_differences(c.run.metadata, r.run.metadata).items()
            }
            for c, r in zip(candidates, references, strict=True)
        },
        "market_view_identical": market_view_identical,
        "windows": windows,
    }


def _format_interval(values: Sequence[float | None], column: str) -> str:
    """Format an estimate and interval; ``*`` marks an interval that excludes zero.

    An interval left out for too few resampling units prints as ``[n/a]``.
    """
    estimate, low, high = values
    digits = 1 if column == "pool" else 4 if column in ("margin_ae", "total_ae", "correct") else 5
    if low is None or high is None:
        return f"{estimate:+.{digits}f} [n/a]"
    flag = " *" if low > 0 or high < 0 else ""
    return f"{estimate:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]{flag}"


def _run_cells(row: dict[str, Any]) -> list[str]:
    """Format one run's metric cells, in ``RUN_COLUMNS`` order, then the market interval."""
    return [
        f"{row['brier']:.5f}",
        f"{row['log_loss']:.5f}",
        f"{row['correct']:.4f}",
        f"{row['margin_ae']:.4f}",
        f"{row['total_ae']:.4f}",
        f"{row['pool']:.0f}",
        f"{row['market_brier']:.5f}",
        _format_interval(row["det_minus_market_brier"], "brier"),
    ]


def _table_header(leading: Sequence[str], columns: Sequence[str], *, market: bool) -> list[str]:
    """Return a Markdown table header: the leading labels, the metric columns, the interval."""
    cells = [*leading, *(COLUMN_LABELS[c] for c in columns)]
    if market:
        cells.append("det - market Brier [95%]")
    return ["| " + " | ".join(cells) + " |", "| --- " * len(cells) + "|"]


def _stability_preamble() -> list[str]:
    """Return the stability section's heading and its one-line reading guide."""
    return [
        STABILITY_HEADING,
        "",
        (
            "Each window split by season, with the same columns and definitions. [n/a] = too "
            "few games (weeks, for pool pts) to resample."
        ),
    ]


def format_stability(stability: dict[str, Any]) -> list[str]:
    """Return one run's stability view as Markdown: per window, all seasons then each season."""
    lines = _stability_preamble()
    for label, rows in stability["windows"].items():
        lines += [
            "",
            f"### {label}",
            "",
            *_table_header(("season", "games"), RUN_COLUMNS, market=True),
        ]
        for season, row in rows.items():
            lines.append(f"| {season} | {row['games']} | " + " | ".join(_run_cells(row)) + " |")
    return lines


def _format_comparison_stability(report: dict[str, Any]) -> list[str]:
    """Return the comparison's per-season run rows and paired contrasts, window by window."""
    lines = _stability_preamble()
    for label, window in report["windows"].items():
        seasons = window["seasons"]
        lines += [
            "",
            f"### {label}",
            "",
            *_table_header(("run", "season", "games"), RUN_COLUMNS, market=True),
        ]
        for run_label in window["runs"]:
            for season, entry in seasons.items():
                cells = _run_cells(entry["runs"][run_label])
                lines.append(
                    f"| {run_label} | {season} | {entry['games']} | " + " | ".join(cells) + " |"
                )
        lines += [
            "",
            *_table_header(("contrast", "season", "games"), CONTRAST_COLUMNS, market=False),
        ]
        for season, entry in seasons.items():
            cells = [_format_interval(entry["contrast"][c], c) for c in CONTRAST_COLUMNS]
            lines.append(
                f"| candidate - reference | {season} | {entry['games']} | "
                + " | ".join(cells)
                + " |"
            )
    return lines


def format_report(report: dict[str, Any]) -> list[str]:
    """Return the comparison as Markdown lines: provenance, then one table per window."""
    lines = [
        f"Candidate: {', '.join(report['candidates'])}",
        f"Reference: {', '.join(report['references'])}",
        (
            f"Paired differences are candidate minus reference, averaged over "
            f"{len(report['candidates'])} pair(s); {report['resamples']} resamples, "
            f"bootstrap seed {report['bootstrap_seed']}. * = the 95% interval excludes zero. "
            "Loss columns: negative favors the candidate; pick acc and pool pts: positive does."
        ),
        "",
        "## Provenance",
    ]
    for label, provenance in report["provenance"].items():
        lines.append(f"- {label}: {json.dumps(provenance, default=str)}")
    for pair, differences in report["config_differences"].items():
        lines.append(f"- config differences, {pair}: {json.dumps(differences, default=str)}")
    if not report["market_view_identical"]:
        lines.append("- warning: the market view differs between runs on the same games")
    for label, window in report["windows"].items():
        lines += [
            "",
            f"## {label} ({window['games']} games)",
            "",
            *_table_header(("run",), RUN_COLUMNS, market=True),
        ]
        for run_label, row in window["runs"].items():
            lines.append(f"| {run_label} | " + " | ".join(_run_cells(row)) + " |")
        lines += [
            "",
            *_table_header(("contrast",), CONTRAST_COLUMNS, market=False),
            "| candidate - reference | "
            + " | ".join(_format_interval(window["contrast"][c], c) for c in CONTRAST_COLUMNS)
            + " |",
        ]
    return [*lines, "", *_format_comparison_stability(report)]
