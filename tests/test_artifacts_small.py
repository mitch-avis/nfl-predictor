"""Unit tests for small, deterministic helpers in nfl_predictor.ml.artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from nfl_predictor.ml import artifacts


def test_now_utc_iso_parses_as_isoformat() -> None:
    """now_utc_iso returns an ISO timestamp string."""
    value = artifacts.now_utc_iso()
    # Should be parseable by stdlib; tolerate timezone suffix.
    parsed = value.replace("Z", "+00:00")
    assert parsed
    # datetime.fromisoformat is strict, so this catches malformed strings.
    datetime.fromisoformat(parsed)


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    """sha256_file matches hashlib's result for the same bytes."""
    path = tmp_path / "payload.bin"
    payload = b"hello world\n"
    path.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert artifacts.sha256_file(path) == expected


def test_stable_short_hash_is_order_invariant() -> None:
    """stable_short_hash is stable under dict key reordering."""
    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}
    assert artifacts.stable_short_hash(a) == artifacts.stable_short_hash(b)


def test_write_json_serializes_paths(tmp_path: Path) -> None:
    """write_json converts Path values to strings and writes valid JSON."""
    out = tmp_path / "out.json"
    artifacts.write_json(out, {"path": tmp_path})

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["path"] == str(tmp_path)
