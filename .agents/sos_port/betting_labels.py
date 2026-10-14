r"""The betting report's labels on the whole edge or on the model's part beyond the spread.

Run from the repository root (a worktree is fine):

    VIRTUAL_ENV="$PWD/.venv" PYTHONDONTWRITEBYTECODE=1 nice -n 10 .venv/bin/python \
        .agents/sos_port/betting_labels.py

It writes ``BETTING_LABELS.md`` and ``betting_labels.json`` beside this file and nothing else.
The protocol is ``protocols/betting_labels.md``, committed before any game was scored; this
script implements it. It reads, read-only, the fold checkpoints of the two seeds of the reviewed
reference (``--checkpoint-dir`` twice, seed 42 then seed 7, each checked against its manifest's
``random_seed``), the build's market-lines metadata (``--market-lines-metadata``) for the seasons
whose moneylines are all real pairs, and the earlier build's metadata
(``--earlier-market-lines-metadata``) to explain a governing-game count other than that build's.

Per game and seed, on the home side: ``edge = p_model - p_ml`` claims ``p_model``, and
``beyond = p_model - p_spread`` claims ``q_beyond = p_ml + beyond``. The instrument is
``d = (q_beyond - y)^2 - (p_model - y)^2`` with ``q_beyond`` unclipped, averaged over the two
seeds per game, then its mean over the governing games with a bootstrap over games. The rule: an
interval wholly below zero recommends basing the labels on ``beyond`` (held for the user); any
other interval keeps them on the whole edge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import betting_edges as edges
import numpy as np
import pandas as pd
import rating_tests_common as common

from nfl_predictor.reporting import betting_report as report_module
from nfl_predictor.reporting.betting_report import _edge_to_action, _edge_to_confidence_1_to_10
from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

HERE = Path(__file__).resolve().parent
SEEDS = (42, 7)
MAIN_MODELS = Path("/home/mitch/workspace/nfl-predictor/models")
DEFAULT_CHECKPOINT_DIRS = (
    MAIN_MODELS / "wf_checkpoints" / "d97fab54e7261f18e349",
    MAIN_MODELS / "wf_checkpoints" / "2f7442e89fd6ff3981f4",
)
RUN_DIRS = ("models/step6/ratings_anchored_seed42", "models/step6/ratings_anchored_seed7")
DEFAULT_METADATA_PATH = MAIN_MODELS / "step6" / "inputs" / "candidate_market_lines_metadata.json"
DEFAULT_EARLIER_METADATA_PATH = (
    MAIN_MODELS / "step4_reference" / "inputs" / "adopted_market_lines_metadata.json"
)
# The governing-game count on the earlier build (the adopted-settings reference).
EARLIER_GOVERNING_GAMES = 815
LOG_LOSS_CLIP = 1e-6
OUT_MD = "BETTING_LABELS.md"
OUT_JSON = "betting_labels.json"
SCRIPT = ".agents/sos_port/betting_labels.py"
BRANCH_BELOW = "below zero"
BRANCH_CONTAINS = "contains zero"
BRANCH_ABOVE = "above zero"
RECOMMEND_BEYOND = (
    "base the confidence score and the action label on `beyond` (the switch is held for the "
    "user; nothing in the report changes in this test)"
)
RECOMMEND_KEEP = "the labels stay on the whole edge"
_DERIVED_KEYS = ("moneylines_from_map", "moneylines_from_fixed_conversion")


# The claims and the instrument.


def add_claims(decomposed: pd.DataFrame) -> pd.DataFrame:
    """Return the decomposed games with ``q_beyond = p_ml + beyond``, the beyond scheme's claim."""
    frame = decomposed.copy()
    frame["q_beyond"] = frame["p_ml"] + frame["beyond"]
    return frame


def brier_difference(q_beyond: np.ndarray, p_model: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Return ``(q_beyond - y)^2 - (p_model - y)^2`` per game, ``q_beyond`` unclipped."""
    return (q_beyond - outcome) ** 2 - (p_model - outcome) ** 2


def _log_loss(probability: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Return each game's log loss, the probability clipped to ``[1e-6, 1 - 1e-6]``."""
    clipped = np.clip(probability, LOG_LOSS_CLIP, 1.0 - LOG_LOSS_CLIP)
    return -(outcome * np.log(clipped) + (1.0 - outcome) * np.log(1.0 - clipped))


def log_loss_difference(
    q_beyond: np.ndarray, p_model: np.ndarray, outcome: np.ndarray
) -> np.ndarray:
    """Return the log loss of ``q_beyond`` minus that of ``p_model`` per game, both clipped."""
    return _log_loss(q_beyond, outcome) - _log_loss(p_model, outcome)


def out_of_range_count(q_beyond: np.ndarray) -> int:
    """Return how many claims lie outside ``[0, 1]``."""
    return int(np.count_nonzero((q_beyond < 0.0) | (q_beyond > 1.0)))


def pair_seeds(frames: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Return one row per game with each seed's claims and differences and their seed means.

    Per seed ``s`` the columns ``p_model_s``, ``q_beyond_s``, ``edge_s``, ``beyond_s``,
    ``brier_d_s`` and ``log_d_s``; ``brier_d`` and ``log_d`` average them over the seeds.

    Raises:
        ValueError: If the seeds do not score the same games (by ``game_id``), or a seed scores a
            game twice.

    """
    seeds = list(frames)
    reference = frames[seeds[0]]
    reference_ids = set(reference["game_id"])
    for seed, frame in frames.items():
        if not frame["game_id"].is_unique:
            msg = f"Seed {seed} scores a game more than once."
            raise ValueError(msg)
        if set(frame["game_id"]) != reference_ids:
            only_here = sorted(set(frame["game_id"]) - reference_ids)[:5]
            only_there = sorted(reference_ids - set(frame["game_id"]))[:5]
            msg = (
                f"Seeds {seeds[0]} and {seed} do not score the same games: {only_here} only in "
                f"seed {seed}, {only_there} only in seed {seeds[0]} (first five of each)."
            )
            raise ValueError(msg)
    paired = (
        reference.loc[:, ["game_id", "season", "actual_home_win"]]
        .sort_values("game_id")
        .reset_index(drop=True)
    )
    outcome = paired["actual_home_win"].to_numpy(dtype=float)
    for seed, frame in frames.items():
        aligned = frame.set_index("game_id").loc[paired["game_id"]]
        if not np.array_equal(aligned["actual_home_win"].to_numpy(dtype=float), outcome):
            msg = f"Seed {seed} records another outcome for a game than seed {seeds[0]}."
            raise ValueError(msg)
        for column in ("p_model", "q_beyond", "edge", "beyond"):
            paired[f"{column}_{seed}"] = aligned[column].to_numpy(dtype=float)
        q_beyond = paired[f"q_beyond_{seed}"].to_numpy()
        p_model = paired[f"p_model_{seed}"].to_numpy()
        paired[f"brier_d_{seed}"] = brier_difference(q_beyond, p_model, outcome)
        paired[f"log_d_{seed}"] = log_loss_difference(q_beyond, p_model, outcome)
    for measure in ("brier_d", "log_d"):
        paired[measure] = paired[[f"{measure}_{seed}" for seed in seeds]].mean(axis=1)
    return paired


@dataclass(frozen=True)
class Difference:
    """A mean per-game difference over ``games`` with its bootstrap interval."""

    games: int
    mean: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies wholly on one side of zero."""
        return self.high < 0.0 or self.low > 0.0


def summarize_difference(differences: np.ndarray) -> Difference:
    """Return the mean and the 95% percentile interval of 5,000 resamples of games (seed 0)."""
    values = np.asarray(differences, dtype=float)
    low, high = edges.paired_bootstrap_interval(values)
    return Difference(games=len(values), mean=float(values.mean()), low=low, high=high)


@dataclass(frozen=True)
class RuleOutcome:
    """Where the governing interval lies, and the recommendation that follows."""

    branch: str
    recommend_beyond: bool

    @property
    def recommendation(self) -> str:
        """The recommendation's text."""
        return RECOMMEND_BEYOND if self.recommend_beyond else RECOMMEND_KEEP


def apply_rule(difference: Difference) -> RuleOutcome:
    """Return the rule's outcome: only an interval wholly below zero recommends ``beyond``.

    Raises:
        ValueError: If a bound is not finite (an empty set of games gives NaN bounds), so no
            outcome is read from an interval that does not exist.

    """
    if not (math.isfinite(difference.low) and math.isfinite(difference.high)):
        msg = (
            f"The interval's bounds must be finite, got {difference.low!r} to "
            f"{difference.high!r} over {difference.games} games."
        )
        raise ValueError(msg)
    if difference.high < 0.0:
        return RuleOutcome(branch=BRANCH_BELOW, recommend_beyond=True)
    if difference.low > 0.0:
        return RuleOutcome(branch=BRANCH_ABOVE, recommend_beyond=False)
    return RuleOutcome(branch=BRANCH_CONTAINS, recommend_beyond=False)


@dataclass(frozen=True)
class LabelChanges:
    """How many games change action label, confidence score or value side on ``beyond``."""

    games: int
    action: int
    confidence: int
    side: int


def label_changes(games: pd.DataFrame) -> LabelChanges:
    """Count the games whose labels change when they read ``beyond`` instead of ``edge``.

    The labels come from the report's own ``_edge_to_action`` and
    ``_edge_to_confidence_1_to_10``; the value side is home when the edge is at least zero, as in
    the report.
    """
    edge = games["edge"].to_numpy(dtype=float)
    beyond = games["beyond"].to_numpy(dtype=float)
    action = sum(
        _edge_to_action(old) != _edge_to_action(new) for old, new in zip(edge, beyond, strict=True)
    )
    confidence = sum(
        _edge_to_confidence_1_to_10(old) != _edge_to_confidence_1_to_10(new)
        for old, new in zip(edge, beyond, strict=True)
    )
    side = int(np.count_nonzero((edge >= 0.0) != (beyond >= 0.0)))
    return LabelChanges(games=len(games), action=action, confidence=confidence, side=side)


def top_decile(paired: pd.DataFrame, column: str, seeds: Sequence[int] = SEEDS) -> pd.DataFrame:
    """Return the ``ceil(n / 10)`` games with the largest two-seed mean of ``|column|``."""
    magnitude = paired[[f"{column}_{seed}" for seed in seeds]].abs().mean(axis=1)
    return paired.loc[magnitude.nlargest(math.ceil(len(paired) / 10)).index]


def seed_top_decile(paired: pd.DataFrame, column: str, seed: int) -> pd.DataFrame:
    """Return the ``ceil(n / 10)`` games with the largest ``|column|`` in one seed."""
    magnitude = paired[f"{column}_{seed}"].abs()
    return paired.loc[magnitude.nlargest(math.ceil(len(paired) / 10)).index]


# Loading and checking the two seeds.


@dataclass(frozen=True)
class SeedRun:
    """One seed's scored games and what was checked on its folds."""

    seed: int
    checkpoint_dir: Path
    folds: int
    rows: int
    floor_difference: float
    scored: pd.DataFrame


def check_manifest_seed(checkpoint_dir: Path, seed: int) -> None:
    """Check that the checkpoint directory's manifest records ``seed`` as its random seed.

    Raises:
        ValueError: If the manifest records another seed.

    """
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest["config"]["random_seed"]
    if recorded != seed:
        msg = f"{checkpoint_dir} holds the run of seed {recorded}, not seed {seed}."
        raise ValueError(msg)


def load_seed(checkpoint_dir: Path, seed: int, expected_folds: int) -> SeedRun:
    """Return one seed's scored games after the floor check on every one of its folds.

    Raises:
        ValueError: If the manifest names another seed, the folds are not the run's, or
            ``Phi(predicted_margin / floor_sigma)`` does not reproduce a fold's probability.

    """
    check_manifest_seed(checkpoint_dir, seed)
    games = edges.load_fold_games(checkpoint_dir, expected_folds)
    floor_difference = edges.check_floor_reproduction(games)
    return SeedRun(
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        folds=expected_folds,
        rows=len(games),
        floor_difference=floor_difference,
        scored=add_claims(edges.decompose(games)),
    )


def directory_digest(directory: Path) -> str:
    """Return one SHA-256 over each file's SHA-256 and name, in name order."""
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        digest.update(f"{common.sha256_file(path)}  {path.name}\n".encode())
    return digest.hexdigest()


def season_counts(metadata_path: Path) -> dict[int, dict[str, int]]:
    """Return each 2007-2025 season's games and derived moneylines from the metadata."""
    seasons = json.loads(metadata_path.read_text(encoding="utf-8"))["seasons"]
    return {
        int(season): {
            "games": int(counts.get("games", 0)),
            "derived": sum(int(counts[key]) for key in _DERIVED_KEYS),
        }
        for season, counts in seasons.items()
        if edges.FIRST_SEASON <= int(season) <= edges.LAST_SEASON
    }


def count_comparison(
    paired: pd.DataFrame, real_seasons: Sequence[int], metadata: Path, earlier: Path
) -> list[dict[str, Any]]:
    """Return, per season real in either build, both builds' metadata counts and the games here.

    This is what explains a governing-game count other than the earlier build's; it is computed
    before any score.
    """
    here, before = season_counts(metadata), season_counts(earlier)
    earlier_real = edges.real_moneyline_seasons(earlier)
    scored = paired.groupby("season").size()
    return [
        {
            "season": season,
            "real_here": season in real_seasons,
            "real_earlier": season in earlier_real,
            "metadata_games_here": here.get(season, {}).get("games", 0),
            "metadata_games_earlier": before.get(season, {}).get("games", 0),
            "derived_here": here.get(season, {}).get("derived", 0),
            "derived_earlier": before.get(season, {}).get("derived", 0),
            "scored_games": int(scored.get(season, 0)),
            "governing_games": int(scored.get(season, 0)) if season in real_seasons else 0,
        }
        for season in sorted(set(real_seasons) | set(earlier_real))
    ]


# Scoring.


def _difference(paired: pd.DataFrame, column: str) -> Difference:
    """Return the summary of one difference column over the given games."""
    return summarize_difference(paired[column].to_numpy(dtype=float))


def _seed_mean_loss(paired: pd.DataFrame, prefix: str, seeds: Sequence[int], loss: str) -> float:
    """Return the mean over games of the two-seed mean Brier or log loss of one claim."""
    outcome = paired["actual_home_win"].to_numpy(dtype=float)
    per_seed = []
    for seed in seeds:
        probability = paired[f"{prefix}_{seed}"].to_numpy(dtype=float)
        if loss == "brier":
            per_seed.append((probability - outcome) ** 2)
        else:
            per_seed.append(_log_loss(probability, outcome))
    return float(np.mean(per_seed, axis=0).mean())


def _seed_view(paired: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return one seed's ``edge`` and ``beyond`` columns under their plain names."""
    return pd.DataFrame({"edge": paired[f"edge_{seed}"], "beyond": paired[f"beyond_{seed}"]})


@dataclass(frozen=True)
class Scores:
    """The rule's number and outcome and every number reported beside it."""

    rule_number: Difference
    outcome: RuleOutcome
    claims: dict[str, float]
    beside: dict[str, Difference]
    by_season: dict[int, Difference]
    label_changes: dict[int, LabelChanges]
    out_of_range: dict[int, dict[str, int]]


def score(paired: pd.DataFrame, real_seasons: Sequence[int], seeds: Sequence[int]) -> Scores:
    """Return the rule's number and outcome and every number reported beside it.

    ``paired`` holds every scored game; the rule reads the games of ``real_seasons``.
    """
    governing = paired[paired["season"].isin(real_seasons)]
    others = paired[~paired["season"].isin(real_seasons)]
    rule_number = _difference(governing, "brier_d")
    claims = {
        f"{loss}_{prefix}": _seed_mean_loss(governing, prefix, seeds, loss)
        for loss in ("brier", "log_loss")
        for prefix in ("q_beyond", "p_model")
    }
    beside = {
        f"Seed {seed} alone, governing games": _difference(governing, f"brier_d_{seed}")
        for seed in seeds
    }
    beside["Every scored game"] = _difference(paired, "brier_d")
    beside["Games of the other seasons"] = _difference(others, "brier_d")
    beside["Log loss, governing games"] = _difference(governing, "log_d")
    for column in ("edge", "beyond"):
        beside[f"Top decile of `|{column}|` (two-seed mean), governing games"] = _difference(
            top_decile(governing, column, seeds), "brier_d"
        )
    for column in ("edge", "beyond"):
        for seed in seeds:
            beside[f"Top decile of `|{column}|` in seed {seed}, governing games"] = _difference(
                seed_top_decile(governing, column, seed), f"brier_d_{seed}"
            )
    return Scores(
        rule_number=rule_number,
        outcome=apply_rule(rule_number),
        claims=claims,
        beside=beside,
        by_season={
            int(season): _difference(group, "brier_d")
            for season, group in paired.groupby("season", sort=True)
        },
        label_changes={seed: label_changes(_seed_view(governing, seed)) for seed in seeds},
        out_of_range={
            seed: {
                "governing": out_of_range_count(governing[f"q_beyond_{seed}"].to_numpy()),
                "every_scored_game": out_of_range_count(paired[f"q_beyond_{seed}"].to_numpy()),
            }
            for seed in seeds
        },
    )


# The report.


def _interval_text(difference: Difference) -> str:
    """Return a difference as its mean and interval at five decimals."""
    return f"`{difference.mean:+.5f}` (`{difference.low:+.5f}` to `{difference.high:+.5f}`)"


def _table_cell(text: str) -> str:
    """Return text with its pipes escaped for a Markdown table cell."""
    return text.replace("|", "\\|")


def _yes(flag: object) -> str:
    """Return ``yes`` for a true flag, ``no`` otherwise."""
    return "yes" if flag else "no"


def _count_lines(comparison: Sequence[Mapping[str, Any]], governing_games: int) -> list[str]:
    """Return the governing-game count and the per-season comparison with the earlier build."""
    if governing_games == EARLIER_GOVERNING_GAMES:
        verdict = (
            f"The count matches the {EARLIER_GOVERNING_GAMES} governing games of the earlier "
            "build. The table gives, per season real in either build, the metadata's games and "
            "derived moneylines in each build and the games scored here."
        )
    else:
        verdict = (
            f"The count differs from the {EARLIER_GOVERNING_GAMES} governing games of the "
            "earlier build. The table gives, per season real in either build, the metadata's "
            "games and derived moneylines in each build and the games scored here."
        )
    rows = [
        f"| {row['season']} | {_yes(row['real_here'])} | {_yes(row['real_earlier'])} | "
        f"{row['metadata_games_here']} | {row['metadata_games_earlier']} | "
        f"{row['derived_here']} | {row['derived_earlier']} | {row['scored_games']} | "
        f"{row['governing_games']} |"
        for row in comparison
    ]
    return [
        verdict,
        "",
        (
            "| Season | Real here | Real earlier | Games here | Games earlier | Derived here "
            "| Derived earlier | Scored here | Governing here |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *rows,
    ]


@dataclass(frozen=True)
class RunInputs:
    """Where the numbers came from."""

    runs: Sequence[SeedRun]
    metadata_path: Path
    earlier_metadata_path: Path
    real_seasons: Sequence[int]
    comparison: Sequence[Mapping[str, Any]]
    command: Sequence[tuple[str, str]]


_BRANCH_TEXT = {
    BRANCH_BELOW: "lies wholly below zero",
    BRANCH_CONTAINS: "contains zero",
    BRANCH_ABOVE: "lies wholly above zero",
}


def _run_label(run: SeedRun, index: int) -> str:
    """Return the reference run's directory and a comma for its default checkpoints, else ``""``.

    Only the protocol's own checkpoint directories are labeled as the reference runs.
    """
    if (
        index < len(DEFAULT_CHECKPOINT_DIRS)
        and run.checkpoint_dir == (DEFAULT_CHECKPOINT_DIRS[index])
    ):
        return f"`{RUN_DIRS[index]}`, "
    return ""


def command_arguments(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Return every option of a run with its value, as the run used them.

    The checkpoint directories and both metadata files are always named; the fold count and the
    output only when they differ from their defaults.
    """
    pairs = [("--checkpoint-dir", str(directory)) for directory in args.checkpoint_dir]
    pairs += [
        ("--market-lines-metadata", str(args.market_lines_metadata)),
        ("--earlier-market-lines-metadata", str(args.earlier_market_lines_metadata)),
    ]
    if args.expected_folds != edges.EXPECTED_FOLDS:
        pairs.append(("--expected-folds", str(args.expected_folds)))
    if args.out != HERE / OUT_MD:
        pairs.append(("--out", str(args.out)))
    return pairs


_COMMAND_PREFIX = 'VIRTUAL_ENV="$PWD/.venv" PYTHONDONTWRITEBYTECODE=1 nice -n 10 .venv/bin/python'


def command_text(pairs: Sequence[tuple[str, str]]) -> str:
    """Return the run's command on one line."""
    options = " ".join(f"{flag} {shlex.quote(value)}" for flag, value in pairs)
    return f"{_COMMAND_PREFIX} {SCRIPT} {options}"


def _command_block(pairs: Sequence[tuple[str, str]]) -> list[str]:
    """Return the run's command as a fenced shell block, one option per line."""
    body = [f"    {SCRIPT}", *(f"    {flag} {shlex.quote(value)}" for flag, value in pairs)]
    return [
        "```bash",
        f"{_COMMAND_PREFIX} \\",
        *(f"{line} \\" for line in body[:-1]),
        body[-1],
        "```",
    ]


def _data_lines(inputs: RunInputs, governing_games: int) -> list[str]:
    """Return the report's data section."""
    lines = ["## Data", ""]
    lines += [
        f"- Seed {run.seed}: {_run_label(run, index)}{run.folds} fold checkpoints and "
        f"{run.rows} game rows from `{run.checkpoint_dir}` (its manifest's `random_seed` is "
        f"{run.seed}); "
        f"{len(run.scored)} scored games ({edges.FIRST_SEASON}-{edges.LAST_SEASON} regular "
        "season, both moneylines and a spread). Floor check: "
        "`Phi(predicted_margin / floor_sigma)` reproduces every fold's deterministic "
        f"probability to a largest difference of `{run.floor_difference:.1e}` (tolerance "
        f"`{edges.FLOOR_TOLERANCE:.0e}`)."
        for index, run in enumerate(inputs.runs)
    ]
    seasons_text = ", ".join(str(season) for season in inputs.real_seasons)
    return [
        *lines,
        (
            f"- Market: `{inputs.metadata_path}`; `p_ml` from the report's own "
            "`_moneyline_to_implied_prob` and `_novig_pair`."
        ),
        (
            f"- Governing games: {governing_games} games in seasons {seasons_text}, the seasons "
            "none of whose moneylines the metadata counts as derived from the spread (by the "
            "fitted map or the fixed conversion). Both seeds score the same games, by game id."
        ),
        "- Outcome: the fold's `actual_home_win`; a tie scores 0.",
        f"- The earlier build's metadata: `{inputs.earlier_metadata_path}`.",
        "",
        *_count_lines(inputs.comparison, governing_games),
    ]


def _beside_lines(scores: Scores) -> list[str]:
    """Return the section of the numbers reported beside the rule."""
    excluding = [name for name, diff in scores.beside.items() if diff.excludes_zero]
    claims = scores.claims
    return [
        "## Reported beside the rule, deciding nothing",
        "",
        (
            "Every mean is of `d`, the Brier difference, except the log-loss row, which is the "
            "paired difference of the two claims' log loss with each probability clipped to "
            "`[1e-6, 1 - 1e-6]`. A seed's row uses that seed's difference alone; every other row "
            "averages each game's difference over the two seeds. The top-decile rows rank the "
            "governing games two ways, since the edge differs by seed: by the two-seed mean of the "
            "magnitude (with the two-seed mean `d`), and by one seed's magnitude (with that seed's "
            "`d`). A decile is `ceil(n / 10)` games."
        ),
        "",
        "| Games | n | Mean (95% interval) | Excludes zero |",
        "| --- | --- | --- | --- |",
        *[
            f"| {_table_cell(name)} | {diff.games} | {_interval_text(diff)} | "
            f"{_yes(diff.excludes_zero)} |"
            for name, diff in scores.beside.items()
        ],
        "",
        (
            "On the governing games, averaged over the seeds per game: Brier of `q_beyond` "
            f"`{claims['brier_q_beyond']:.5f}` and of `p_model` `{claims['brier_p_model']:.5f}`; "
            f"log loss of `q_beyond` `{claims['log_loss_q_beyond']:.5f}` and of `p_model` "
            f"`{claims['log_loss_p_model']:.5f}`."
        ),
        "",
        "Intervals beside the rule that exclude zero: "
        + ("; ".join(excluding) if excluding else "none")
        + ". None of them decides anything.",
        "",
        "### Labels that change when they read `beyond`",
        "",
        (
            "On the governing games, through the report's own `_edge_to_action` and "
            "`_edge_to_confidence_1_to_10`; the value side is home when the edge is at least zero, "
            "as in the report."
        ),
        "",
        "| Seed | Games | Action label changes | Confidence score changes | Value side changes |",
        "| --- | --- | --- | --- | --- |",
        *[
            f"| {seed} | {change.games} | {change.action} | {change.confidence} | {change.side} |"
            for seed, change in scores.label_changes.items()
        ],
        "",
        "`q_beyond` outside `[0, 1]`, kept unclipped in the Brier term: "
        + "; ".join(
            f"seed {seed}, {counts['governing']} of the governing games and "
            f"{counts['every_scored_game']} of every scored game"
            for seed, counts in scores.out_of_range.items()
        )
        + ".",
    ]


def build_report(scores: Scores, inputs: RunInputs, provenance: Mapping[str, Any]) -> list[str]:
    """Return the lines of ``BETTING_LABELS.md``."""
    rule_number, outcome = scores.rule_number, scores.outcome
    code = ", ".join(f"`{name}` {digest[:12]}" for name, digest in provenance["code"].items())
    inputs_text = "; ".join(
        f"`{path}` {digest[:12]}" for path, digest in provenance["inputs"].items()
    )
    lines = [
        "# Betting report labels: the whole edge or the model's part beyond the spread",
        "",
        (
            f"Generated by `{SCRIPT}`; do not edit by hand. The protocol is "
            "`protocols/betting_labels.md`, committed before any game was scored. Per game and "
            "seed, on the home side, `edge = p_model - p_ml` claims `p_model` and "
            "`beyond = p_model - p_spread` claims `q_beyond = p_ml + beyond`. The instrument is "
            "`d = (q_beyond - y)^2 - (p_model - y)^2`, `q_beyond` unclipped, averaged over the two "
            "seeds per game, then its mean over the governing games with a bootstrap over games "
            f"({edges.BOOTSTRAP_RESAMPLES:,} resamples, numpy seed {edges.BOOTSTRAP_SEED}, 95% "
            "percentile interval) that resamples the games in game-id order, so the interval "
            "depends on that order. A negative `d` means the beyond claim scored better."
        ),
        "",
        (
            "Reproduce from the repository root (it reads the inputs below and writes only this "
            f"file and `{OUT_JSON}`):"
        ),
        "",
        *_command_block(inputs.command),
        "",
        *_data_lines(inputs, rule_number.games),
        "",
        "## The rule's outcome",
        "",
        (
            f"On the {rule_number.games} governing games the mean of `d` is "
            f"{_interval_text(rule_number)}: the interval {_BRANCH_TEXT[outcome.branch]}."
        ),
        "",
        f"Outcome, as the rule states it: {outcome.recommendation}.",
        "",
        *_beside_lines(scores),
        "",
        "### By season",
        "",
        "| Season | Governing | Games | Mean `d` (95% interval) |",
        "| --- | --- | --- | --- |",
        *[
            f"| {season} | {_yes(season in inputs.real_seasons)} | {diff.games} | "
            f"{_interval_text(diff)} |"
            for season, diff in scores.by_season.items()
        ],
        "",
        "## Provenance",
        "",
        (
            f"- nfl-predictor {provenance['version']} at commit `{provenance['git_commit']}`; "
            f"code SHA-256: {code}."
        ),
        (
            "- Inputs, SHA-256 (a checkpoint directory's is over each file's SHA-256 and name, in "
            f"name order): {inputs_text}. The full digests are in `{OUT_JSON}`."
        ),
        f"- Command: `{provenance['command']}`; {provenance['seconds']:.0f} seconds.",
    ]
    return common.wrap_markdown(lines)


def _record(scores: Scores, inputs: RunInputs) -> dict[str, Any]:
    """Return the JSON record's decisions and scores."""
    rule_number, outcome = scores.rule_number, scores.outcome
    return {
        "protocol": ".agents/sos_port/protocols/betting_labels.md",
        "seeds": [run.seed for run in inputs.runs],
        "bootstrap": {
            "resamples": edges.BOOTSTRAP_RESAMPLES,
            "numpy_seed": edges.BOOTSTRAP_SEED,
            "game_order": "game_id",
        },
        "runs": [
            {
                "seed": run.seed,
                "checkpoint_dir": str(run.checkpoint_dir),
                "folds": run.folds,
                "rows": run.rows,
                "scored_games": len(run.scored),
                "floor_difference": run.floor_difference,
            }
            for run in inputs.runs
        ],
        "governing": {
            "seasons": list(inputs.real_seasons),
            "games": rule_number.games,
            "earlier_build_games": EARLIER_GOVERNING_GAMES,
            "count_comparison": list(inputs.comparison),
        },
        "rule": {
            **asdict(rule_number),
            "branch": outcome.branch,
            "recommend_beyond": outcome.recommend_beyond,
            "recommendation": outcome.recommendation,
        },
        "claims": scores.claims,
        "beside": {
            name: {**asdict(diff), "excludes_zero": diff.excludes_zero}
            for name, diff in scores.beside.items()
        },
        "by_season": {str(season): asdict(diff) for season, diff in scores.by_season.items()},
        "label_changes": {
            str(seed): asdict(change) for seed, change in scores.label_changes.items()
        },
        "out_of_range": {str(seed): counts for seed, counts in scores.out_of_range.items()},
    }


def run(args: argparse.Namespace) -> int:
    """Check the inputs, score once, and write the report and its JSON record.

    Raises:
        ValueError: If the checkpoint directories are not one per seed, or any check fails.

    """
    start = time.perf_counter()
    args.checkpoint_dir = args.checkpoint_dir or list(DEFAULT_CHECKPOINT_DIRS)
    checkpoint_dirs: list[Path] = args.checkpoint_dir
    if len(checkpoint_dirs) != len(SEEDS):
        msg = f"Give one checkpoint directory per seed, in seed order {SEEDS}."
        raise ValueError(msg)
    runs = [
        load_seed(directory, seed, args.expected_folds)
        for directory, seed in zip(checkpoint_dirs, SEEDS, strict=True)
    ]
    real_seasons = edges.real_moneyline_seasons(args.market_lines_metadata)
    paired = pair_seeds({run.seed: run.scored for run in runs})
    comparison = count_comparison(
        paired, real_seasons, args.market_lines_metadata, args.earlier_market_lines_metadata
    )
    governing_games = int(paired["season"].isin(real_seasons).sum())
    if governing_games == 0:
        msg = (
            f"No governing games: the real-moneyline seasons {real_seasons} hold no scored game, "
            "so the rule has nothing to read."
        )
        raise ValueError(msg)
    log.info("Governing games: %d in seasons %s.", governing_games, real_seasons)
    scores = score(paired, real_seasons, SEEDS)
    inputs = RunInputs(
        runs=runs,
        metadata_path=args.market_lines_metadata,
        earlier_metadata_path=args.earlier_market_lines_metadata,
        real_seasons=real_seasons,
        comparison=comparison,
        command=command_arguments(args),
    )
    input_digests = {str(directory): directory_digest(directory) for directory in checkpoint_dirs}
    for path in (args.market_lines_metadata, args.earlier_market_lines_metadata):
        input_digests[str(path)] = common.sha256_file(path)
    code_paths = [
        HERE / "betting_labels.py",
        HERE / "betting_edges.py",
        HERE / "rating_tests_common.py",
        Path(report_module.__file__),
    ]
    provenance = {
        "version": metadata.version("nfl-predictor"),
        "code": {path.name: common.sha256_file(path) for path in code_paths},
        "inputs": input_digests,
        "git_commit": common.git_commit(),
        "command": command_text(inputs.command),
        "seconds": time.perf_counter() - start,
    }
    out: Path = args.out
    out.write_text("\n".join(build_report(scores, inputs, provenance)) + "\n", encoding="utf-8")
    common.write_json_record(
        out.parent / OUT_JSON, {**_record(scores, inputs), "provenance": provenance}
    )
    log.info("Wrote %s: the interval %s.", out, _BRANCH_TEXT[scores.outcome.branch])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line and run the test."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        action="append",
        help="A seed's fold checkpoints; give it twice, seed 42 then seed 7.",
    )
    parser.add_argument("--expected-folds", type=int, default=edges.EXPECTED_FOLDS)
    parser.add_argument("--market-lines-metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument(
        "--earlier-market-lines-metadata", type=Path, default=DEFAULT_EARLIER_METADATA_PATH
    )
    parser.add_argument("--out", type=Path, default=HERE / OUT_MD)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
