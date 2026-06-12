"""Artifact helpers for reproducible runs.

This module centralizes writing run-scoped artifacts:
- model checkpoint (joblib)
- metadata.json
- metrics_report.json

The goal is to make training/backtest outputs self-describing and comparable.
"""

from __future__ import annotations

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


def stable_short_hash(payload: Any) -> str:
    """Return a stable short hash for a JSON-serializable payload."""
    encoded = json.dumps(_to_jsonable(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:8]


def _to_jsonable(value: Any) -> Any:
    """Convert a nested payload into JSON-serializable Python primitives.

    This avoids brittle failures when payloads contain NumPy/Polars/Pandas scalar types
    (e.g., numpy.int64) or other objects that the stdlib JSON encoder can't handle.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        # dataclasses.is_dataclass() returns True for both instances and classes.
        # dataclasses.asdict() only accepts instances.
        if isinstance(value, type):
            return str(value)
        return _to_jsonable(asdict(value))

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    # NumPy / Polars scalars typically implement .item(); ndarrays often implement .tolist().
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_jsonable(tolist())
        except Exception:
            pass

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            pass

    return str(value)


def generate_run_id(prefix: str, dataset_hash: str, config: dict[str, Any]) -> str:
    """Generate a run id using timestamp + dataset/config hash."""
    created = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    short_hash = stable_short_hash({"dataset_hash": dataset_hash, "config": config})
    return f"{prefix}_{created}_{short_hash}"


def resolve_run_paths(
    run_id: str,
    run_dir: Path | None = None,
    model_filename: str = "model.joblib",
    metadata_filename: str = "metadata.json",
    metrics_filename: str = "metrics_report.json",
    feature_importance_filename: str = "feature_importance.json",
) -> RunPaths:
    """Resolve default artifact paths for a given run id."""
    base = run_dir if run_dir is not None else (Path(constants.ROOT_DIR) / "models" / run_id)
    return RunPaths(
        run_id=run_id,
        run_dir=base,
        model_path=base / model_filename,
        metadata_path=base / metadata_filename,
        metrics_path=base / metrics_filename,
        feature_importance_path=base / feature_importance_filename,
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


def build_metadata(
    *,
    created_at: str,
    run_id: str,
    dataset_hash: str,
    config: dict[str, Any],
    feature_list: list[str] | None = None,
    splits: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    tuned_params: dict[str, Any] | None = None,
    early_stopping: dict[str, Any] | None = None,
    optuna_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a metadata payload meeting the repo's artifact contract."""
    payload: dict[str, Any] = {
        "created_at": created_at,
        "run_id": run_id,
        "git_commit_hash": git_commit_hash(),
        "dataset_hash": dataset_hash,
        "library_versions": library_versions(),
        "config": config,
        "feature_list": feature_list,
        "splits": splits,
        "params": params,
        "tuned_params": tuned_params,
        "early_stopping": early_stopping,
        "optuna_summary": optuna_summary,
    }
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to disk with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _to_jsonable(payload)
    path.write_text(json.dumps(safe_payload, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote %s", path)


def save_model(path: Path, model: Any) -> None:
    """Persist a model artifact via joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("Saved model checkpoint to %s", path)
