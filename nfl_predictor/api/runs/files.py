"""Typed locations of every artifact a run directory can hold.

``scripts/weekly_run.py`` names its per-week outputs ``season_{S}_week_{WW}_*`` and its power
rankings ``*_season_{S}_week_{WW}.csv`` (stamped one week earlier than the predictions), so the
file names are discovered by glob rather than assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from nfl_predictor.ml.artifacts import resolve_run_paths

STAGES: tuple[str, ...] = ("wf_compare", "train", "predictions", "reports")
WEEK_FILE_RE = re.compile(
    r"season_(\d{4})_week_(\d{2})_(predictions|confidence_picks|betting_report)\.csv$"
)
POWER_FILE_RE = re.compile(
    r"(power_rankings|projected_standings|projected_division_standings)_season_(\d{4})_week_(\d{2})\.csv$"
)


@dataclass(frozen=True)
class RunFiles:
    """Resolved artifact paths for one run directory (paths may not exist)."""

    run_id: str
    run_dir: Path
    model: Path
    metadata: Path
    metrics: Path
    feature_importance: Path
    wf_compare_csv: Path
    wf_best: Path
    shap_report: Path
    betting_xlsx: Path
    leakage_audit: Path
    predictions: Path | None = None
    picks: Path | None = None
    betting_csv: Path | None = None
    power_rankings: Path | None = None
    standings: Path | None = None
    division_standings: Path | None = None
    season: int | None = None
    week: int | None = None
    power_season: int | None = None
    power_week: int | None = None
    stage_markers: dict[str, Path] = field(default_factory=dict)

    @property
    def stages(self) -> dict[str, bool]:
        """Return which orchestration stages have a completion marker."""
        return {stage: self.stage_markers[stage].is_file() for stage in STAGES}

    @property
    def complete(self) -> bool:
        """Return whether the orchestrated run finished its reports stage."""
        return self.stage_markers["reports"].is_file()

    def presence(self) -> dict[str, bool]:
        """Return a name -> exists map for every optional artifact."""
        candidates: dict[str, Path | None] = {
            "model": self.model,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "wf_compare": self.wf_compare_csv,
            "wf_best": self.wf_best,
            "shap_report": self.shap_report,
            "betting_xlsx": self.betting_xlsx,
            "leakage_audit": self.leakage_audit,
            "predictions": self.predictions,
            "picks": self.picks,
            "betting_csv": self.betting_csv,
            "power_rankings": self.power_rankings,
            "standings": self.standings,
            "division_standings": self.division_standings,
        }
        return {name: path is not None and path.is_file() for name, path in candidates.items()}


def _newest_week_file(run_dir: Path, kind: str) -> tuple[Path, int, int] | None:
    """Return the ``(path, season, week)`` of the latest ``season_*_week_*_{kind}.csv``."""
    best: tuple[int, int, Path] | None = None
    for path in run_dir.glob(f"season_*_week_*_{kind}.csv"):
        match = WEEK_FILE_RE.search(path.name)
        if match is None:
            continue
        key = (int(match.group(1)), int(match.group(2)), path)
        if best is None or key[:2] > best[:2]:
            best = key
    if best is None:
        return None
    return best[2], best[0], best[1]


def _newest_power_file(run_dir: Path, kind: str) -> tuple[Path, int, int] | None:
    """Return the ``(path, season, week)`` of the latest ``{kind}_season_*_week_*.csv``."""
    best: tuple[int, int, Path] | None = None
    for path in run_dir.glob(f"{kind}_season_*_week_*.csv"):
        match = POWER_FILE_RE.search(path.name)
        if match is None:
            continue
        key = (int(match.group(2)), int(match.group(3)), path)
        if best is None or key[:2] > best[:2]:
            best = key
    if best is None:
        return None
    return best[2], best[0], best[1]


def resolve_run_files(run_dir: Path) -> RunFiles:
    """Discover every artifact in ``run_dir``."""
    run_id = run_dir.name
    paths = resolve_run_paths(run_id, run_dir=run_dir)
    predictions = _newest_week_file(run_dir, "predictions")
    picks = _newest_week_file(run_dir, "confidence_picks")
    betting = _newest_week_file(run_dir, "betting_report")
    power = _newest_power_file(run_dir, "power_rankings")
    standings = _newest_power_file(run_dir, "projected_standings")
    division = _newest_power_file(run_dir, "projected_division_standings")
    season_week = predictions or picks or betting
    return RunFiles(
        run_id=run_id,
        run_dir=run_dir,
        model=paths.model_path,
        metadata=paths.metadata_path,
        metrics=paths.metrics_path,
        feature_importance=paths.feature_importance_path,
        wf_compare_csv=run_dir / "wf_compare.csv",
        wf_best=run_dir / "wf_best.json",
        shap_report=run_dir / "shap_report.json",
        betting_xlsx=run_dir / "betting_report.xlsx",
        leakage_audit=run_dir / "leakage_audit.json",
        predictions=predictions[0] if predictions else None,
        picks=picks[0] if picks else None,
        betting_csv=betting[0] if betting else None,
        power_rankings=power[0] if power else None,
        standings=standings[0] if standings else None,
        division_standings=division[0] if division else None,
        season=season_week[1] if season_week else None,
        week=season_week[2] if season_week else None,
        power_season=power[1] if power else None,
        power_week=power[2] if power else None,
        stage_markers={stage: run_dir / f"{stage}_state.json" for stage in STAGES},
    )
