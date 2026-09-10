"""Read model metadata, metrics, feature importance, calibration, and walk-forward comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from nfl_predictor.api.readers.cache import cached
from nfl_predictor.api.registry import project
from nfl_predictor.api.registry.model import WF_COMPARE_COLUMNS
from nfl_predictor.api.runs.files import RunFiles
from nfl_predictor.api.schemas.common import TablePayload

TOP_FEATURES = 40
METADATA_KEYS = (
    "created_at", "run_id", "git_commit_hash", "dataset_hash", "library_versions", "params",
    "tuned_params", "early_stopping", "optuna_summary", "splits",
)  # fmt: skip


def _dict(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}


def _opt_dict(value: Any) -> dict[str, Any] | None:
    """Return ``value`` when it is a dict, else ``None``."""
    return value if isinstance(value, dict) else None


def _opt_list(value: Any) -> list[Any] | None:
    """Return ``value`` when it is a list, else ``None``."""
    return value if isinstance(value, list) else None


def _read_json(path: Path) -> Any:
    """Parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> Any:
    """Return the cached parsed JSON at ``path``."""
    return cached(path, _read_json)


def metadata_summary(path: Path) -> dict[str, Any]:
    """Return the metadata block without the (long) feature list, plus its length and config."""
    raw = _opt_dict(load_json(path))
    if raw is None:
        return {}
    out: dict[str, Any] = {key: raw.get(key) for key in METADATA_KEYS}
    features = _opt_list(raw.get("feature_list"))
    out["feature_count"] = len(features) if features is not None else None
    out["config"] = _opt_dict(raw.get("config"))
    return out


def metrics_summary(path: Path) -> dict[str, Any]:
    """Return holdout, pool, missing-data, and walk-forward blocks from a metrics report."""
    raw = _opt_dict(load_json(path))
    if raw is None:
        return {}
    metrics = _dict(raw.get("metrics"))
    inner = _dict(metrics.get("metrics"))
    return {
        "kind": metrics.get("kind"),
        "holdout": _opt_dict(inner.get("holdout")),
        "pool": _opt_dict(metrics.get("pool")),
        "missing_data": _opt_dict(metrics.get("missing_data")),
        "overall": _opt_dict(metrics.get("overall")),
        "per_season": _opt_list(metrics.get("per_season")),
        "per_week": _opt_list(metrics.get("per_week")),
        "summary_table": _opt_list(metrics.get("summary_table")),
        "metric_strategy": _opt_dict(raw.get("metric_strategy")),
        "calibration": _opt_dict(raw.get("calibration")),
    }


def feature_importance(path: Path, top: int = TOP_FEATURES) -> list[dict[str, Any]]:
    """Return the top features by combined gain.

    Each row is ``{feature, gain, weight, margin_gain, total_gain}``.
    """
    base = _opt_dict(_dict(load_json(path)).get("base_features"))
    if base is None:
        return []
    names = _opt_list(base.get("feature_names"))
    combined = _dict(base.get("combined"))
    margin = _dict(base.get("margin"))
    total = _dict(base.get("total"))
    gains = _opt_list(combined.get("gain"))
    if names is None or gains is None:
        return []

    def _at(block: dict[str, Any], key: str, index: int) -> float | None:
        values = block.get(key)
        if isinstance(values, list) and index < len(values):
            value = values[index]
            return float(value) if isinstance(value, int | float) else None
        return None

    rows: list[dict[str, Any]] = [
        {
            "feature": str(name),
            "gain": float(gain),
            "weight": _at(combined, "weight", i),
            "margin_gain": _at(margin, "gain", i),
            "total_gain": _at(total, "gain", i),
        }
        for i, (name, gain) in enumerate(zip(names, gains, strict=False))
        if isinstance(gain, int | float)
    ]
    rows.sort(key=lambda r: float(r["gain"]), reverse=True)
    return rows[:top]


def wf_compare(path: Path) -> TablePayload:
    """Return the candidate comparison table ranked by Brier then log loss."""
    frame: pl.DataFrame = cached(path, lambda p: pl.read_csv(p, infer_schema_length=10000))
    if "rank" in frame.columns:
        frame = frame.rename({"rank": "wf_rank"}).sort("wf_rank")
    elif {"brier", "log_loss"} <= set(frame.columns):
        frame = frame.sort(["brier", "log_loss"]).with_row_index("wf_rank", offset=1)
    return project(frame, WF_COMPARE_COLUMNS)


def best_candidate_calibration(files: RunFiles) -> dict[str, Any] | None:
    """Return the reliability bins of the winning walk-forward candidate of a weekly run.

    ``weekly_run.py`` stores one JSON per candidate under ``wf_compare/`` with a
    ``metrics.reliability`` list; ``wf_best.json`` names the winner by ``candidate_key``.
    """
    if not files.wf_best.is_file():
        return None
    key = _dict(load_json(files.wf_best)).get("candidate_key")
    if not key:
        return None
    directory = files.run_dir / "wf_compare"
    if not directory.is_dir():
        return None
    for candidate in sorted(directory.glob("wf_candidate_*.json")):
        try:
            payload = load_json(candidate)
        except ValueError:
            continue
        if not isinstance(payload, dict) or payload.get("candidate_key") != key:
            continue
        bins = _opt_list(_dict(payload.get("metrics")).get("reliability"))
        if bins is not None:
            return {"bin_count": len(bins), "bins": bins, "source": candidate.name}
    return None


def model_payload(files: RunFiles) -> dict[str, Any]:
    """Assemble everything the Model page shows for one run."""
    metrics = metrics_summary(files.metrics) if files.metrics.is_file() else {}
    calibration = metrics.get("calibration") or best_candidate_calibration(files)
    return {
        "metadata": metadata_summary(files.metadata) if files.metadata.is_file() else {},
        "metrics": metrics,
        "feature_importance": feature_importance(files.feature_importance)
        if files.feature_importance.is_file()
        else [],
        "calibration": calibration,
        "wf_compare": wf_compare(files.wf_compare_csv) if files.wf_compare_csv.is_file() else None,
        "wf_best": load_json(files.wf_best) if files.wf_best.is_file() else None,
        "shap_available": files.shap_report.is_file(),
    }
