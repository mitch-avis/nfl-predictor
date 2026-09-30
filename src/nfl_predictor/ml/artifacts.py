"""Artifact helpers for reproducible runs.

This module centralizes writing run-scoped artifacts:
- model checkpoint (joblib)
- metadata.json
- metrics_report.json

The goal is to make training/backtest outputs self-describing and comparable.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from nfl_predictor import constants
from nfl_predictor.utils.logger import log


@dataclass(frozen=True)
class RunPaths:
    """Resolved artifact locations for a run."""

    run_id: str
    run_dir: Path
    model_path: Path
    metadata_path: Path
    metrics_path: Path
    feature_importance_path: Path


def now_utc_iso() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 fingerprint of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None


def stable_short_hash(payload: object) -> str:
    """Return a stable short hash for a JSON-serializable payload."""
    encoded = json.dumps(to_jsonable(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:8]


def to_jsonable(value: object) -> JsonValue:
    """Convert a nested payload into JSON-serializable Python primitives.

    This avoids brittle failures when payloads contain NumPy/Polars/Pandas scalar types
    (e.g., numpy.int64) or other objects that the stdlib JSON encoder can't handle.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    # A class object, a dataclass included, is recorded by name; asdict() needs an instance.
    if isinstance(value, Path | type):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [to_jsonable(v) for v in value]
    return _convert_by_protocol(value)


def _convert_by_protocol(value: object) -> JsonValue:
    """Convert a value through the first of its ``item``/``tolist``/``isoformat`` that works.

    NumPy, pandas and Polars scalars implement ``item()``; arrays and series raise
    ``ValueError`` from it when they hold more than one element and fall through to
    ``tolist()``. Dates and timestamps implement ``isoformat()``. Anything else is its ``str``.
    """
    for method_name in ("item", "tolist"):
        method = getattr(value, method_name, None)
        if callable(method):
            with contextlib.suppress(ValueError):
                return to_jsonable(method())
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        with contextlib.suppress(ValueError):
            return str(isoformat())
    return str(value)


def generate_run_id(prefix: str, dataset_hash: str, config: dict[str, Any]) -> str:
    """Generate a run id using timestamp + dataset/config hash."""
    created = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    short_hash = stable_short_hash({"dataset_hash": dataset_hash, "config": config})
    return f"{prefix}_{created}_{short_hash}"


MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics_report.json"
FEATURE_IMPORTANCE_FILENAME = "feature_importance.json"


def resolve_run_paths(run_id: str, run_dir: Path | None = None) -> RunPaths:
    """Resolve the artifact paths of a run, under ``models/<run_id>`` unless ``run_dir``."""
    base = run_dir if run_dir is not None else (Path(constants.ROOT_DIR) / "models" / run_id)
    return RunPaths(
        run_id=run_id,
        run_dir=base,
        model_path=base / MODEL_FILENAME,
        metadata_path=base / METADATA_FILENAME,
        metrics_path=base / METRICS_FILENAME,
        feature_importance_path=base / FEATURE_IMPORTANCE_FILENAME,
    )


def git_commit_hash() -> str | None:
    """Return current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(constants.ROOT_DIR),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def library_versions() -> dict[str, str | None]:
    """Return versions of key libraries for reproducibility."""
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": _module_version("numpy"),
        "pandas": _module_version("pandas"),
        "polars": _module_version("polars"),
        "scipy": _module_version("scipy"),
        "sklearn": _module_version("sklearn"),
        "xgboost": _module_version("xgboost"),
        "optuna": _module_version("optuna"),
    }


@dataclass(frozen=True, kw_only=True)
class TrainedModelDetails:
    """What a fitted model records in its metadata beyond the run's identity.

    ``xgb_device`` is the concrete XGBoost device the model's heads trained on (``cpu`` or
    ``cuda``), or ``mixed`` when a head fell back to another device. ``floor_sigma`` is the
    model's recorded floor sigma (``FloorSigma.to_dict``): the value its probabilities use, the
    week it was estimated for, its pool's size and seasons, whether it fell back to the
    constant, and the reference runs it came from.
    """

    feature_list: list[str] | None = None
    splits: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    tuned_params: dict[str, Any] | None = None
    early_stopping: dict[str, Any] | None = None
    optuna_summary: dict[str, Any] | None = None
    xgb_device: str | None = None
    floor_sigma: dict[str, Any] | None = None


def build_metadata(
    *,
    created_at: str,
    run_id: str,
    dataset_hash: str,
    config: dict[str, Any],
    details: TrainedModelDetails | None = None,
) -> dict[str, Any]:
    """Build a metadata payload meeting the repo's artifact contract.

    Without ``details`` (no fitted model), every model field is recorded as ``None``.
    """
    details = details or TrainedModelDetails()
    return {
        "created_at": created_at,
        "run_id": run_id,
        "git_commit_hash": git_commit_hash(),
        "dataset_hash": dataset_hash,
        "library_versions": library_versions(),
        "config": config,
        "feature_list": details.feature_list,
        "splits": details.splits,
        "params": details.params,
        "tuned_params": details.tuned_params,
        "early_stopping": details.early_stopping,
        "optuna_summary": details.optuna_summary,
        "xgb_device": details.xgb_device,
        "floor_sigma": details.floor_sigma,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to disk with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = to_jsonable(payload)
    path.write_text(json.dumps(safe_payload, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote %s", path)


def save_model(path: Path, model: object) -> None:
    """Persist a model artifact via joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("Saved model checkpoint to %s", path)
