"""Check whether each step-4 follow-up still feeds production or the default power rankings.

The roadmap's follow-up row "6, phases 3-4 (from step 4)" in ``.agents/TODO.md`` lists ten items
left over from the feature milestones. An item may be retired only when it feeds neither:

- the production model: the feature columns a weekly run trains on, which are the dataset's
  feature range after the market setting, the pruning list, the identifier, constant and
  high-cardinality drops (``ml.feature_spec.build_feature_spec``) and the feature groups the
  weekly configuration leaves out (``config/weekly_run.yaml``, else
  ``constants.PRODUCTION_DISABLED_FEATURE_GROUPS``);
- the default power rankings: the method the weekly run ranks on (the blend of the ported team
  rating and the model rating unless the configuration sets another), and the code that method
  reads.

Read-only: it reads one dataset file (its header with Polars, its rows with pandas, as the
model does), the configuration and this repository's source, and writes nothing. A fact it
cannot establish (a missing file, a token it does not find) is printed as not found, never
filled in. Run from the repository root, pointing ``--data`` at the training file a weekly run
reads (``data/completed_games_ml.csv`` in the main checkout, written here as ``CHECKOUT``):

    .venv/bin/python .agents/sos_port/followup_check.py --data CHECKOUT/data/completed_games_ml.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl
import yaml

from nfl_predictor import constants
from nfl_predictor.cli.options import resolve_disabled_feature_groups
from nfl_predictor.ml.feature_spec import (
    FeatureSelection,
    build_feature_spec,
    resolve_feature_group_columns,
)
from nfl_predictor.weekly_run.config import DEFAULT_POWER_RANKINGS_METHOD

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "nfl_predictor"
CONFIG = ROOT / "config" / "weekly_run.yaml"
MAX_CARDINALITY_RATIO = 0.5
EXAMPLES = 4


@dataclass(frozen=True)
class Production:
    """What the production weekly run trains on and ranks with."""

    dataset_columns: list[str]
    features: list[str]
    disabled_groups: tuple[str, ...]
    ranking_method: str


@dataclass(frozen=True)
class SourceToken:
    """A string looked for in one source file, and what finding it shows."""

    path: Path
    token: str
    meaning: str


@dataclass(frozen=True)
class Item:
    """One follow-up, the feature columns it concerns and the source facts that bear on it."""

    name: str
    column_markers: tuple[str, ...]
    tokens: tuple[SourceToken, ...]


def read_config(path: Path) -> dict[str, object]:
    """Return the weekly configuration as a mapping (empty when the file has no keys)."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def production_settings(data: Path, config_path: Path) -> Production:
    """Return the production feature columns, disabled groups and ranking method."""
    config = read_config(config_path)
    market_mode = str(config.get("wf_market_mode", "hybrid"))
    without_lines = market_mode == "none"
    explicit = config.get("disable_feature_groups")
    groups = resolve_disabled_feature_groups(
        [str(group) for group in explicit] if isinstance(explicit, list) else None,
        without_game_lines=without_lines,
    )
    columns = pl.scan_csv(data).collect_schema().names()
    dropped = set(resolve_feature_group_columns(columns, groups, warn_unmatched=False))
    # The model's own path is pandas (``weekly_run.stage1`` reads the file this way), and the
    # feature spec takes a pandas frame, so the constant and cardinality drops see what it sees.
    frame = pd.read_csv(data, dtype={"game_id": str}, low_memory=False)
    kept = frame[[column for column in columns if column not in dropped]]
    selection = FeatureSelection(
        include_market=not without_lines,
        max_cardinality_ratio=MAX_CARDINALITY_RATIO,
        market_transform=not without_lines,
    )
    spec = build_feature_spec(kept, selection)
    method = config.get("power_rankings_method")
    return Production(
        dataset_columns=columns,
        features=list(spec.feature_columns),
        disabled_groups=groups,
        ranking_method=str(method) if method else DEFAULT_POWER_RANKINGS_METHOD,
    )


def matching(columns: list[str], markers: tuple[str, ...]) -> list[str]:
    """Return the columns that contain any of ``markers``."""
    return [column for column in columns if any(marker in column for marker in markers)]


def describe_columns(label: str, columns: list[str]) -> str:
    """Return one line naming how many columns matched and a few of them."""
    shown = ", ".join(columns[:EXAMPLES])
    more = f", ... ({len(columns) - EXAMPLES} more)" if len(columns) > EXAMPLES else ""
    return f"  {label}: {len(columns)}" + (f" ({shown}{more})" if columns else "")


def describe_token(token: SourceToken) -> str:
    """Return one line saying whether ``token`` is in its file."""
    relative = token.path.relative_to(ROOT)
    if not token.path.is_file():
        return f"  NOT FOUND: file {relative} does not exist ({token.meaning})"
    found = token.token in token.path.read_text(encoding="utf-8")
    state = "found" if found else "NOT FOUND"
    return f"  {state}: `{token.token}` in {relative} ({token.meaning})"


def blend_reads_snapshot() -> list[str]:
    """Return the ported-rating and blend source files that mention the strength snapshot."""
    files = [SRC / "reporting" / "rating_sources.py", *sorted((SRC / "ratings").rglob("*.py"))]
    return [
        str(path.relative_to(ROOT))
        for path in files
        if "strength_snapshot" in path.read_text(encoding="utf-8")
        or "PRIOR_BLEND_GAMES" in path.read_text(encoding="utf-8")
    ]


def items() -> list[Item]:
    """Return the ten follow-ups with their column markers and source tokens."""
    polars_dir = SRC / "utils" / "polars"
    strength = tuple(constants.FEATURE_GROUP_COLUMN_MARKERS["strength"])
    qb_family = tuple(constants.FEATURE_GROUP_COLUMN_MARKERS["qb"])
    return [
        Item(
            "Unused PBP_COUNT_COLUMNS (raw counts as features; the rates they feed)",
            tuple(constants.PBP_COUNT_COLUMNS),
            (
                SourceToken(
                    SRC / "constants.py",
                    "a count no rate reads is not carried",
                    "the count list is limited to counts a published rate reads",
                ),
            ),
        ),
        Item(
            "strength_games_played_diff",
            ("strength_games_played_diff",),
            (
                SourceToken(
                    SRC / "constants.py",
                    '"strength_games_played_diff",',
                    "the column is on a constants list (pruning or the no-diff list)",
                ),
            ),
        ),
        Item("adj_* scale drift (the strength group's columns)", strength, ()),
        Item("sos_played_raw null in weeks 1-2", ("sos_played_raw",), ()),
        Item(
            "The early-season blend's weeks 3-18 margin MAE cost (K, PRIOR_BLEND_GAMES)",
            (),
            (
                SourceToken(
                    polars_dir / "week_rows.py",
                    "stat_prior_blend_games",
                    "every season-to-date stat is blended with the prior season by K",
                ),
                SourceToken(
                    polars_dir / "week_rows.py",
                    "strength_prior_blend_games",
                    "the strength snapshot is blended with its prior by K",
                ),
            ),
        ),
        Item(
            "_attach_qb_features double request (the quarterback family's columns)",
            qb_family,
            (
                SourceToken(
                    SRC / "data_collection.py",
                    "requested_seasons",
                    "the quarterback history skips seasons the run already requested",
                ),
            ),
        ),
        Item(
            "Scramble attribution (the quarterback family's columns)",
            qb_family,
            (
                SourceToken(
                    polars_dir / "qb_stats.py",
                    "rusher_player_id",
                    "a scramble is credited to the rusher",
                ),
                SourceToken(
                    polars_dir / "qb_stats.py",
                    "def _ratio(",
                    "the item's other part: a further copy of the null-safe ratio is still here",
                ),
            ),
        ),
        Item(
            "The QB identity chain (the quarterback family's and the Elo's columns)",
            (*qb_family, *constants.FEATURE_GROUP_COLUMN_MARKERS["elo"]),
            (
                SourceToken(
                    SRC / "constants.py",
                    '"away_qb_id",',
                    "the schedule's starter ids are among the downloaded schedule columns",
                ),
            ),
        ),
        Item("The early-season strength prior's weight (the strength snapshot)", strength, ()),
        Item(
            "The next-opponent identity columns",
            tuple(constants.FEATURE_GROUP_COLUMN_MARKERS["next_opponent_identity"]),
            (
                SourceToken(
                    SRC / "constants.py",
                    '"away_next_opponent_abbr",',
                    "the identity pair is on a constants list (pruned from the model)",
                ),
            ),
        ),
    ]


def report(production: Production) -> list[str]:
    """Return the report lines: the production settings, then one block per follow-up."""
    lines = [
        f"Dataset columns: {len(production.dataset_columns)}",
        f"Production feature columns: {len(production.features)}",
        f"Production disabled groups: {', '.join(production.disabled_groups) or 'none'}",
        f"Default power-ranking method: {production.ranking_method}",
        (
            "Ported-rating and blend sources that mention the strength snapshot or "
            f"PRIOR_BLEND_GAMES: {', '.join(blend_reads_snapshot()) or 'none'}"
        ),
        "",
    ]
    for item in items():
        lines.append(item.name)
        if item.column_markers:
            in_data = matching(production.dataset_columns, item.column_markers)
            in_model = matching(production.features, item.column_markers)
            lines.append(describe_columns("columns in the dataset", in_data))
            lines.append(describe_columns("columns among production features", in_model))
        lines.extend(describe_token(token) for token in item.tokens)
        lines.append("")
    return lines


def main() -> None:
    """Parse the arguments and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="the weekly run's training CSV")
    parser.add_argument("--config", type=Path, default=CONFIG, help="the weekly configuration")
    args = parser.parse_args()
    lines = report(production_settings(args.data, args.config))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
