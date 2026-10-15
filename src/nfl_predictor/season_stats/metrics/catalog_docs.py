"""Generate the human-readable stats catalogs beside this package from the metric registry.

The registry is the single source of truth for the season tables' columns, so the catalogs are
rendered from it rather than maintained by hand. ``python -m
nfl_predictor.season_stats.metrics.catalog_docs`` rewrites both files, and a test fails whenever
the committed copies drift from the registry.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from nfl_predictor.season_stats.metrics.catalog import get_registry
from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    from nfl_predictor.season_stats.metrics.registry import MetricRegistry
    from nfl_predictor.season_stats.metrics.schema import Entity, MetricDef

CATALOG_PATHS: dict[Entity, str] = {
    "team": "src/nfl_predictor/season_stats/stats-catalog.md",
    "qb": "src/nfl_predictor/season_stats/qb-stats-catalog.md",
}
_TITLES: dict[Entity, str] = {"team": "NFL Team Stats Catalog", "qb": "NFL QB Stats Catalog"}
_REPO_ROOT = Path(__file__).resolve().parents[4]
_LINE_LIMIT = 100
REGENERATE_COMMAND = "python -m nfl_predictor.season_stats.metrics.catalog_docs"
_GENERATED_LINES = (
    "Generated from the metric registry (`src/nfl_predictor/season_stats/metrics/`) by",
    f"`.venv/bin/{REGENERATE_COMMAND}`; do not edit it by hand.",
)
_SOURCES = (
    ("PBP", "Computed by this project from nflverse play-by-play (`load_pbp`)."),
    (
        "PBP +TS",
        (
            "Computed from play-by-play and also published in nflverse weekly team stats, which "
            "serves as a cross-check."
        ),
    ),
    ("PLS", "nflverse weekly player stats (`load_player_stats`)."),
    ("SCH", "nflverse schedules and final scores (`load_schedules`)."),
    ("SNP", "nflverse snap counts (`load_snap_counts`, 2013 onward; the 2012 file is empty)."),
    ("D", "Derived by this project from the other columns, including every rating."),
)
_INTRO = (
    "Every column below is regular season only. Data files and the analyst app add a prefix or "
    "suffix to these base names: `opp_` (the team's head-to-head-excluded opponent profile), "
    "`qopp_` (what the defenses a quarterback faced allowed to other passers), `_per_game`, "
    "`_total`, the per-snap denominators `_per_offensive_snap` and `_per_defensive_snap`, "
    "rank-range percentiles `_q025` through `_q975`, and, in the analyst app's garbage-time "
    "filter view, `filtered_` (refit on the plays the filter keeps) and `_change` (filtered "
    "minus unfiltered). "
    "Shapes: `count` totals, `rate` ratios with their own denominator, `avg` per-event "
    "averages, `max` the largest single value (such as the longest play), `score` model outputs "
    "on their own scale, `flag` booleans, and `id` identity fields."
)


def _paragraph(text: str) -> str:
    """Wrap prose to the Markdown line limit."""
    return textwrap.fill(text, width=_LINE_LIMIT)


def _cell(text: str) -> str:
    """Escape a value for a Markdown table cell."""
    return text.replace("|", "\\|")


def _description(metric: MetricDef) -> str:
    """Return the description plus any duplicate, formula, or note annotation."""
    parts = [metric.description]
    if metric.duplicate_of:
        parts.append(f"Same value as `{metric.duplicate_of}`.")
    if metric.formula:
        parts.append(f"Formula: {metric.formula}")
    if metric.note:
        parts.append(metric.note)
    return " ".join(parts)


def _metric_table(metrics: list[MetricDef]) -> list[str]:
    """Return one Markdown table for a list of metrics."""
    lines = [
        "| Column | Label | Shape | Per | Since | Source | Description |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| `{metric.name}` | {_cell(metric.label)} | {metric.shape} | "
        f"{_cell(metric.denominator or '-')} | {metric.since or '-'} | {metric.source} | "
        f"{_cell(_description(metric))} |"
        for metric in metrics
    )
    return lines


def render_catalog(entity: Entity, registry: MetricRegistry | None = None) -> str:
    """Render one entity's stats catalog as Markdown from ``registry`` (default: the project's)."""
    registry = registry if registry is not None else get_registry()
    metrics = [metric for metric in registry.metrics.values() if metric.entity == entity]
    companion = Path(CATALOG_PATHS["qb" if entity == "team" else "team"]).name
    lines = [
        f"# {_TITLES[entity]}",
        "",
        *_GENERATED_LINES,
        f"Companion catalog: [{companion}]({companion}).",
        "",
        _paragraph(_INTRO),
        "",
        "## Sources",
        "",
        "| Code | Meaning |",
        "| --- | --- |",
        *(f"| {code} | {meaning} |" for code, meaning in _SOURCES),
    ]
    for category in registry.categories(entity):
        in_category = [metric for metric in metrics if metric.category == category.name]
        if not in_category:
            continue
        lines.extend(["", f"## {category.name}", "", _paragraph(category.description)])
        sections: list[tuple[str | None, list[MetricDef]]] = [
            (None, [metric for metric in in_category if metric.subcategory is None]),
            *(
                (sub, [metric for metric in in_category if metric.subcategory == sub])
                for sub in category.subcategories
            ),
        ]
        for subcategory, section_metrics in sections:
            if not section_metrics:
                continue
            lines.append("")
            if subcategory is not None:
                lines.extend([f"### {category.name}: {subcategory}", ""])
            lines.extend(_metric_table(section_metrics))
    return "\n".join(lines) + "\n"


def write_catalogs(repo_root: Path = _REPO_ROOT) -> list[Path]:
    """Rewrite both catalogs from the registry and return their paths."""
    written: list[Path] = []
    for entity, relative_path in CATALOG_PATHS.items():
        path = repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_catalog(entity), encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    """Rewrite both stats catalogs from the metric registry and log each file written.

    A developer's tool, run as ``REGENERATE_COMMAND``; it takes no options and is not a command
    of ``nfl-predictor``.
    """
    for path in write_catalogs():
        log.info("Wrote %s", path)


__all__ = ["CATALOG_PATHS", "REGENERATE_COMMAND", "main", "render_catalog", "write_catalogs"]


if __name__ == "__main__":
    main()
