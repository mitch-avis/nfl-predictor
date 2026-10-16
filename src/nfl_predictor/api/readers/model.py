"""Read model metadata, metrics, feature importance, calibration, and walk-forward comparisons."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import polars as pl

from nfl_predictor.api.readers.cache import cached
from nfl_predictor.api.registry import project
from nfl_predictor.api.registry.model import WF_COMPARE_COLUMNS
from nfl_predictor.utils.narrowing import as_list, as_str_dict

if TYPE_CHECKING:
    from pathlib import Path

    from nfl_predictor.api.runs.files import RunFiles
    from nfl_predictor.api.schemas.common import TablePayload

TOP_FEATURES = 40
METADATA_KEYS = (
    "created_at", "run_id", "git_commit_hash", "dataset_hash", "library_versions", "params",
    "tuned_params", "early_stopping", "optuna_summary", "splits",
)  # fmt: skip


def _dict(value: object) -> dict[str, Any]:
    """Return ``value`` when it is a dict, else an empty dict."""
    record = as_str_dict(value)
    return {} if record is None else record


def _opt_dict(value: object) -> dict[str, Any] | None:
    """Return ``value`` when it is a dict, else ``None``."""
    return as_str_dict(value)


def _opt_list(value: object) -> list[Any] | None:
    """Return ``value`` when it is a list, else ``None``."""
    return as_list(value)


def _read_json(path: Path) -> object:
    """Parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> object:
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
        "windows": _opt_list(metrics.get("windows")),
        "summary_table": _opt_list(metrics.get("summary_table")),
        "metric_strategy": _opt_dict(raw.get("metric_strategy")),
        "calibration": _opt_dict(raw.get("calibration")),
    }


IMPORTANCE_MEASURES: tuple[tuple[str, str], ...] = (
    ("mean_abs_shap", "mean_abs_shap"),
    ("total_gain", "total_gain"),
    ("gain", "summed_average_gain"),
)
"""``(key in feature_importance.json, measure reported to the page)``, in order of preference.

Current files record mean absolute SHAP (in points) and total gain per base feature. Files
written before SHAP was recorded fall back to total gain. The oldest record only XGBoost's
average gain per split summed over encoded columns and heads, reported as
``summed_average_gain`` so the page can say the ranking favors many-category features.
"""


def _empty_importance() -> dict[str, Any]:
    """Return the importance payload for a run without usable importance."""
    return {"measure": None, "rows": [], "margin_rows": []}


def _value_at(block: dict[str, Any], key: str, index: int) -> float | None:
    """Return ``block[key][index]`` as a float when it is a number, else ``None``."""
    values = as_list(block.get(key))
    if values is not None and index < len(values):
        value = values[index]
        return float(value) if isinstance(value, int | float) else None
    return None


def _ranked_rows(
    names: list[Any],
    ranking: dict[str, Any],
    heads: tuple[dict[str, Any], dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    """Return one row per feature with a number under ``ranking[key]``, largest value first.

    ``value`` and ``splits`` come from ``ranking`` (the combined block or one head's block);
    ``margin_value`` and ``total_value`` from the two heads.
    """
    margin, total = heads
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        value = _value_at(ranking, key, index)
        if value is None:
            continue
        rows.append(
            {
                "feature": str(name),
                "value": value,
                "margin_value": _value_at(margin, key, index),
                "total_value": _value_at(total, key, index),
                "splits": _value_at(ranking, "weight", index),
            }
        )
    rows.sort(key=lambda row: float(row["value"]), reverse=True)
    return rows


def feature_importance(path: Path, top: int = TOP_FEATURES) -> dict[str, Any]:
    """Return the top base features by the best importance measure the file records.

    The payload is ``{measure, rows, margin_rows}``; each row is
    ``{feature, value, margin_value, total_value, splits}``. In ``rows`` the features rank by
    the measure over both heads, ``value`` is that sum and ``splits`` the split count over both
    heads. ``margin_rows`` ranks every feature by the margin head alone, so ``value`` is the
    margin head's measure and ``splits`` its split count; it is empty when the file records no
    margin head. ``margin_value``/``total_value`` are the measure per head in both lists.
    """
    base = _opt_dict(_dict(load_json(path)).get("base_features"))
    if base is None:
        return _empty_importance()
    names = _opt_list(base.get("feature_names"))
    combined = _dict(base.get("combined"))
    margin = _dict(base.get("margin"))
    total = _dict(base.get("total"))
    found = next(
        (
            (key, measure)
            for key, measure in IMPORTANCE_MEASURES
            if _opt_list(combined.get(key)) is not None
        ),
        None,
    )
    if names is None or found is None:
        return _empty_importance()
    key, measure = found
    heads = (margin, total)
    return {
        "measure": measure,
        "rows": _ranked_rows(names, combined, heads, key)[:top],
        "margin_rows": _ranked_rows(names, margin, heads, key)[:top],
    }


def wf_compare(path: Path) -> TablePayload:
    """Return the candidate comparison table ranked by the deterministic instrument."""
    frame: pl.DataFrame = cached(path, lambda p: pl.read_csv(p, infer_schema_length=10000))
    if "rank" in frame.columns:
        frame = frame.rename({"rank": "wf_rank"}).sort("wf_rank")
    elif {"deterministic_brier", "deterministic_log_loss"} <= set(frame.columns):
        frame = frame.sort(["deterministic_brier", "deterministic_log_loss"]).with_row_index(
            "wf_rank", offset=1
        )
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
            payload = _opt_dict(load_json(candidate))
        except ValueError:
            continue
        if payload is None or payload.get("candidate_key") != key:
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
        else _empty_importance(),
        "calibration": calibration,
        "wf_compare": wf_compare(files.wf_compare_csv) if files.wf_compare_csv.is_file() else None,
        "wf_best": load_json(files.wf_best) if files.wf_best.is_file() else None,
        "shap_available": files.shap_report.is_file(),
    }
