"""Stable fingerprint helpers for resumable workflows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from nfl_predictor.ml import artifacts


def dataset_fingerprint(path: Path) -> dict[str, Any]:
    """Return a stable dataset fingerprint payload."""
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "sha256": artifacts.sha256_file(path),
    }


def stable_fingerprint(payload: dict[str, Any]) -> str:
    """Return a full SHA-256 fingerprint for a payload."""
    encoded = json.dumps(to_jsonable(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def wf_run_fingerprint(
    dataset_fp: dict[str, Any],
    wf_args: dict[str, Any],
    *,
    code_version: str | None = None,
) -> str:
    """Build a walk-forward run fingerprint from dataset + args."""
    payload = {
        "dataset_sha256": dataset_fp.get("sha256"),
        "wf_args": wf_args,
        "code_version": code_version,
    }
    return stable_fingerprint(payload)


def to_jsonable(value: Any) -> Any:
    """Convert nested payloads to JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return to_jsonable(tolist())
        except Exception:
            pass

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            pass

    return str(value)
