"""Stable fingerprint helpers for resumable workflows."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from nfl_predictor.ml import artifacts

if TYPE_CHECKING:
    from pathlib import Path


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
    encoded = json.dumps(artifacts.to_jsonable(payload), sort_keys=True).encode("utf-8")
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
