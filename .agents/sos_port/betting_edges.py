"""Split the betting report's moneyline edges into the model's part and the market's own gap.

Run from the repository root (a worktree is fine):

    VIRTUAL_ENV="$PWD/.venv" .venv/bin/python .agents/sos_port/betting_edges.py

It writes ``BETTING_EDGES.md`` beside this file and nothing else. It reads, read-only, the fold
checkpoints of the adopted-settings reference, seed 42 (``--checkpoint-dir``, by default the main
checkout's ``models/wf_checkpoints/df20b775abf7f2657d13``): each fold's ``predictions`` frame and
its ``metrics["floor_sigma"]``, that week's sigma; every fold's fingerprint must begin
with the directory's name. It also reads the reference's market-lines metadata
(``--market-lines-metadata``) to find the seasons whose moneylines are all real pairs rather than
derived from the spread. The rule is ``decisions/betting_edges.md``, entry 1, written before this
ran; it reads all scored games, and the other groups decide nothing.

Per game, on the home side, with ``sigma`` the week's floor sigma: ``p_model`` is
``Phi(predicted_margin / sigma)``, ``p_ml`` the no-vig moneyline probability (the betting
report's own conversion) and ``p_spread`` is ``Phi(-home_spread / sigma)``. The report's edge,
``p_model - p_ml``, is the sum of ``beyond = p_model - p_spread`` (what the model adds beyond the
spread) and ``gap = p_spread - p_ml`` (the curve-versus-moneyline gap, with no information from
the model's margin).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl_predictor.reporting.betting_report import _moneyline_to_implied_prob, _novig_pair
from nfl_predictor.utils.logger import log

FIRST_SEASON = 2007
LAST_SEASON = 2025
EXPECTED_FOLDS = 328
FLOOR_TOLERANCE = 1e-12
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 0
INTERVAL_LEVEL = 0.95
# The rule recommends basing the labels on ``beyond`` when ``|gap| > |beyond|`` in more than this
# share of the top decile of ``|edge|``.
RULE_SHARE = 0.5
RUN_DIR = "models/step4_reference/adopted_seed42"
DEFAULT_CHECKPOINT_DIR = Path(
    "/home/mitch/workspace/nfl-predictor/models/wf_checkpoints/df20b775abf7f2657d13"
)
DEFAULT_METADATA_PATH = Path(
    "/home/mitch/workspace/nfl-predictor/models/step4_reference/inputs/"
    "adopted_market_lines_metadata.json"
)
REPORT_PATH = Path(__file__).resolve().parent / "BETTING_EDGES.md"
ALL_GAMES = "All scored games"
# The per-season counts of moneylines derived from the spread in the market-lines metadata.
_DERIVED_MONEYLINE_COUNTS = ("moneylines_from_map", "moneylines_from_fixed_conversion")
_FOLD_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "home_moneyline",
    "away_moneyline",
    "home_spread",
    "predicted_margin",
    "deterministic_home_win_prob",
    "actual_home_win",
]


def load_fold_games(checkpoint_dir: Path, expected_folds: int = EXPECTED_FOLDS) -> pd.DataFrame:
    """Return every fold's games, each with its week's ``floor_sigma`` from the fold's metrics.

    Raises:
        ValueError: If the directory does not hold exactly ``expected_folds`` fold checkpoints, or
            a fold's fingerprint does not begin with the directory's name (another run's fold).

    """
    paths = sorted(checkpoint_dir.glob("fold_*.joblib"))
    if len(paths) != expected_folds:
        msg = f"Expected {expected_folds} fold checkpoints in {checkpoint_dir}, found {len(paths)}."
        raise ValueError(msg)
    frames: list[pd.DataFrame] = []
    for path in paths:
        payload = joblib.load(path)
        # A walk-forward names its checkpoint directory after the start of the run's fingerprint.
        if not str(payload.get("fingerprint")).startswith(checkpoint_dir.name):
            msg = (
                f"{path} has fingerprint {payload.get('fingerprint')!r}, which the checkpoint "
                f"directory's name {checkpoint_dir.name!r} does not abbreviate."
            )
            raise ValueError(msg)
        frame = payload["predictions"].loc[:, _FOLD_COLUMNS].copy()
        frame["floor_sigma"] = float(payload["metrics"]["floor_sigma"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def check_floor_reproduction(games: pd.DataFrame, tolerance: float = FLOOR_TOLERANCE) -> float:
    """Return the largest difference between ``Phi(margin / sigma)`` and the fold's floor.

    Raises:
        ValueError: If the largest difference exceeds ``tolerance``.

    """
    margin = games["predicted_margin"].to_numpy(dtype=float)
    floor = norm.cdf(margin / games["floor_sigma"].to_numpy(dtype=float))
    stored = games["deterministic_home_win_prob"].to_numpy(dtype=float)
    difference = float(np.max(np.abs(floor - stored)))
    if not difference <= tolerance:
        msg = (
            "Phi(predicted_margin / floor_sigma) does not reproduce the fold's deterministic "
            f"probability: largest difference {difference:.3e}, tolerance {tolerance:.0e}."
        )
        raise ValueError(msg)
    return difference


def decompose(games: pd.DataFrame) -> pd.DataFrame:
    """Return the scored games with their probabilities and the edge's two parts.

    The added columns are ``p_model``, ``p_ml``, ``p_spread``, ``edge``, ``gap`` and ``beyond``.
    Scored games are the 2007-2025 regular-season games with both moneylines and a spread.
    """
    in_window = games["season"].between(FIRST_SEASON, LAST_SEASON) & games["game_type"].eq("REG")
    frame = games.loc[in_window].copy()
    home_raw = frame["home_moneyline"].map(_moneyline_to_implied_prob)
    away_raw = frame["away_moneyline"].map(_moneyline_to_implied_prob)
    frame["p_ml"] = [
        _novig_pair(home, away)[0] for home, away in zip(home_raw, away_raw, strict=True)
    ]
    sigma = frame["floor_sigma"].to_numpy(dtype=float)
    frame["p_model"] = norm.cdf(frame["predicted_margin"].to_numpy(dtype=float) / sigma)
    frame["p_spread"] = norm.cdf(-frame["home_spread"].to_numpy(dtype=float) / sigma)
    scored = np.isfinite(frame["p_ml"].to_numpy(dtype=float)) & np.isfinite(
        frame["p_spread"].to_numpy(dtype=float)
    )
    frame = frame.loc[scored].copy()
    frame["edge"] = frame["p_model"] - frame["p_ml"]
    frame["gap"] = frame["p_spread"] - frame["p_ml"]
    frame["beyond"] = frame["p_model"] - frame["p_spread"]
    return frame.reset_index(drop=True)


def top_decile(decomposed: pd.DataFrame) -> pd.DataFrame:
    """Return the ``ceil(n / 10)`` games with the largest ``|edge|``."""
    count = math.ceil(len(decomposed) / 10)
    return decomposed.loc[decomposed["edge"].abs().nlargest(count).index]


def paired_bootstrap_interval(
    differences: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Return the percentile interval of the mean per-game difference, resampling games."""
    rng = np.random.default_rng(seed)
    count = len(differences)
    means = np.empty(resamples)
    for index in range(resamples):
        means[index] = differences[rng.integers(0, count, size=count)].mean()
    tail = (1.0 - INTERVAL_LEVEL) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(low), float(high)


@dataclass(frozen=True)
class EdgeSummary:
    """The measurement's numbers; ``recommend_beyond`` is the rule's outcome."""

    games: int
    mean_abs_edge: float
    mean_abs_gap: float
    mean_abs_beyond: float
    top_decile_games: int
    share_gap_dominates: float
    share_gap_sign_matches_edge: float
    corr_edge_gap: float
    corr_edge_beyond: float
    brier_moneyline: float
    brier_spread: float
    brier_difference: float
    brier_difference_interval: tuple[float, float]

    @property
    def recommend_beyond(self) -> bool:
        """Whether ``|gap| > |beyond|`` in more than half of the top decile of ``|edge|``."""
        return self.share_gap_dominates > RULE_SHARE


def summarize(decomposed: pd.DataFrame) -> EdgeSummary:
    """Return the rule's share, the diagnostics and the paired Brier comparison."""
    top = top_decile(decomposed)
    outcome = decomposed["actual_home_win"].to_numpy(dtype=float)
    moneyline_loss = (outcome - decomposed["p_ml"].to_numpy(dtype=float)) ** 2
    spread_loss = (outcome - decomposed["p_spread"].to_numpy(dtype=float)) ** 2
    differences = spread_loss - moneyline_loss
    edge = decomposed["edge"].to_numpy(dtype=float)
    return EdgeSummary(
        games=len(decomposed),
        mean_abs_edge=float(np.abs(edge).mean()),
        mean_abs_gap=float(decomposed["gap"].abs().mean()),
        mean_abs_beyond=float(decomposed["beyond"].abs().mean()),
        top_decile_games=len(top),
        share_gap_dominates=float((top["gap"].abs() > top["beyond"].abs()).mean()),
        share_gap_sign_matches_edge=float((np.sign(top["gap"]) == np.sign(top["edge"])).mean()),
        corr_edge_gap=float(np.corrcoef(edge, decomposed["gap"].to_numpy(dtype=float))[0, 1]),
        corr_edge_beyond=float(np.corrcoef(edge, decomposed["beyond"].to_numpy(dtype=float))[0, 1]),
        brier_moneyline=float(moneyline_loss.mean()),
        brier_spread=float(spread_loss.mean()),
        brier_difference=float(differences.mean()),
        brier_difference_interval=paired_bootstrap_interval(differences),
    )


def real_moneyline_seasons(metadata_path: Path) -> list[int]:
    """Return the scored seasons none of whose moneylines were derived from the spread.

    A season qualifies when the market-lines metadata counts no moneyline from the fitted
    spread-to-moneyline map and none from the fixed conversion. The metadata holds per-season
    counts, not per-game flags, so a season with any derived moneyline is left out whole.
    """
    seasons = json.loads(metadata_path.read_text(encoding="utf-8"))["seasons"]
    return sorted(
        int(season)
        for season, counts in seasons.items()
        if FIRST_SEASON <= int(season) <= LAST_SEASON
        and all(int(counts[key]) == 0 for key in _DERIVED_MONEYLINE_COUNTS)
    )


def summary_groups(
    decomposed: pd.DataFrame, real_seasons: list[int]
) -> list[tuple[str, EdgeSummary]]:
    """Return the summary of all scored games, then of the real-moneyline seasons and the last.

    The first group is the one the rule reads; a group without games is left out.
    """
    real_names = ", ".join(str(season) for season in real_seasons)
    groups = [
        (ALL_GAMES, decomposed),
        (
            f"Real moneyline pairs ({real_names})",
            decomposed[decomposed["season"].isin(real_seasons)],
        ),
        (f"{LAST_SEASON} alone", decomposed[decomposed["season"] == LAST_SEASON]),
    ]
    return [(name, summarize(group)) for name, group in groups if len(group) > 0]


RECOMMEND_BEYOND = (
    "base the confidence and action labels on `beyond`, not `edge` (held for the user)"
)
RECOMMEND_KEEP = "keep the confidence and action labels on `edge`"


_MEASURES = (
    "Scored games",
    "Mean `\\|edge\\|`",
    "Mean `\\|gap\\|`",
    "Mean `\\|beyond\\|`",
    "Top-decile games",
    "Top decile: share with `\\|gap\\| > \\|beyond\\|`",
    "Top decile: share with `sign(gap) == sign(edge)`",
    "Correlation of `edge` with `gap`",
    "Correlation of `edge` with `beyond`",
    "Brier of `p_ml`",
    "Brier of `p_spread`",
    "Brier of `p_spread` minus `p_ml` (95% interval)",
)


def _values(summary: EdgeSummary) -> list[str]:
    """Return one group's formatted values, in the order of ``_MEASURES``."""
    low, high = summary.brier_difference_interval
    return [
        f"{summary.games}",
        f"{summary.mean_abs_edge:.4f}",
        f"{summary.mean_abs_gap:.4f}",
        f"{summary.mean_abs_beyond:.4f}",
        f"{summary.top_decile_games}",
        f"{summary.share_gap_dominates:.4f}",
        f"{summary.share_gap_sign_matches_edge:.4f}",
        f"{summary.corr_edge_gap:.4f}",
        f"{summary.corr_edge_beyond:.4f}",
        f"{summary.brier_moneyline:.5f}",
        f"{summary.brier_spread:.5f}",
        f"{summary.brier_difference:+.5f} ({low:+.5f} to {high:+.5f})",
    ]


@dataclass(frozen=True)
class ReportInputs:
    """Where the numbers came from: the checkpoints, the floor check and the lines metadata."""

    checkpoint_dir: Path
    folds: int
    fold_rows: int
    floor_difference: float
    metadata_path: Path


def render_report(groups: list[tuple[str, EdgeSummary]], inputs: ReportInputs) -> str:
    """Return ``BETTING_EDGES.md``: the data, each group's numbers and the rule's outcome.

    The rule reads the first group, all scored games.
    """
    checkpoint_dir, metadata_path = inputs.checkpoint_dir, inputs.metadata_path
    folds, fold_rows, floor_difference = inputs.folds, inputs.fold_rows, inputs.floor_difference
    summary = groups[0][1]
    recommendation = RECOMMEND_BEYOND if summary.recommend_beyond else RECOMMEND_KEEP
    columns = [_values(group) for _, group in groups]
    header = "| Measure | " + " | ".join(name for name, _ in groups) + " |"
    separator = "| --- |" + " --- |" * len(groups)
    rows = [
        f"| {measure} | " + " | ".join(column[index] for column in columns) + " |"
        for index, measure in enumerate(_MEASURES)
    ]
    table = "\n".join([header, separator, *rows])
    return f"""# Betting report edges: the model's part and the spread-moneyline gap

Generated by `.agents/sos_port/betting_edges.py`; do not edit by hand. The rule is
`decisions/betting_edges.md`, entry 1, written before this ran. All quantities are on the home
side: `edge = p_model - p_ml` is the report's moneyline edge, `beyond = p_model - p_spread` is
what the model adds beyond the spread and `gap = p_spread - p_ml` is the gap between the floor's
curve at the market's spread and the no-vig moneyline, so `edge = beyond + gap`.

Reproduce from the repository root (it reads the fold checkpoints and writes only this file):

```bash
VIRTUAL_ENV="$PWD/.venv" .venv/bin/python .agents/sos_port/betting_edges.py \\
    --checkpoint-dir {checkpoint_dir} \\
    --market-lines-metadata {metadata_path}
```

## Data

- The adopted-settings reference, seed 42, run directory `{RUN_DIR}`:
  {folds} fold checkpoints and {fold_rows} game rows, read from
  `{checkpoint_dir}`.
- Scored: the {FIRST_SEASON}-{LAST_SEASON} regular-season games with both moneylines and a spread,
  each at its week's floor sigma (the fold's `metrics["floor_sigma"]`).
- Floor check: `Phi(predicted_margin / floor_sigma)` reproduces the fold's deterministic
  probability to a largest difference of `{floor_difference:.1e}`
  (tolerance `{FLOOR_TOLERANCE:.0e}`).
- The Brier interval resamples games {BOOTSTRAP_RESAMPLES:,} times (numpy seed {BOOTSTRAP_SEED},
  95% percentile interval); a tie scores 0, as in every walk-forward Brier here.
- Groups: the rule reads all scored games. The real-moneyline seasons are those whose
  moneylines the market-lines metadata counts none of as derived from the spread (by the
  fitted spread-to-moneyline map or the fixed conversion), read from
  `{metadata_path}`.
  On a game with a derived moneyline the gap is a function of the spread and the sigma, not a
  disagreement between the game's spread and its moneyline. Each group's top decile is its own.
  These groups decide nothing.

## Numbers

{table}

## The rule's outcome

`|gap| > |beyond|` in {summary.share_gap_dominates:.4f} of the top decile of `|edge|` over all
scored games; the rule's threshold is more than {RULE_SHARE}.

Recommendation: {recommendation}.
"""


def main(argv: list[str] | None = None) -> int:
    """Measure the decomposition and write the report."""
    parser = argparse.ArgumentParser(description="Split the betting report's moneyline edges.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--expected-folds", type=int, default=EXPECTED_FOLDS)
    parser.add_argument("--market-lines-metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    games = load_fold_games(args.checkpoint_dir, args.expected_folds)
    floor_difference = check_floor_reproduction(games)
    real_seasons = real_moneyline_seasons(args.market_lines_metadata)
    groups = summary_groups(decompose(games), real_seasons)
    inputs = ReportInputs(
        checkpoint_dir=args.checkpoint_dir,
        folds=args.expected_folds,
        fold_rows=len(games),
        floor_difference=floor_difference,
        metadata_path=args.market_lines_metadata,
    )
    report = render_report(groups, inputs)
    args.out.write_text(report, encoding="utf-8")
    log.info("Wrote %s (%d scored games).", args.out, groups[0][1].games)
    return 0


if __name__ == "__main__":
    sys.exit(main())
