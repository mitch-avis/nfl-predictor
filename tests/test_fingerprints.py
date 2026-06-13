"""Tests for resumable workflow fingerprint helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from nfl_predictor.utils import fingerprints


class _ItemValue:
    """Test helper exposing an ``item()`` method for scalar coercion."""

    def __init__(self, value: object) -> None:
        """Store the wrapped test value."""
        self._value = value

    def item(self) -> object:
        """Return the wrapped value."""
        return self._value


class _FallbackToList:
    """Test helper whose ``item()`` fails before ``tolist()`` succeeds."""

    def item(self) -> object:
        """Raise to force the next fallback branch."""
        raise RuntimeError("item failed")

    def tolist(self) -> list[int]:
        """Return a list payload."""
        return [1, 2, 3]


class _OnlyToList:
    """Test helper exposing only a ``tolist()`` method."""

    def tolist(self) -> list[int]:
        """Return a list payload."""
        return [4, 5, 6]


class _FallbackIsoformat:
    """Test helper whose ``isoformat()`` branch should be used."""

    def item(self) -> object:
        """Raise to force the next fallback branch."""
        raise RuntimeError("item failed")

    def tolist(self) -> list[int]:
        """Raise to force the next fallback branch."""
        raise RuntimeError("tolist failed")

    def isoformat(self) -> str:
        """Return an ISO-8601 timestamp string."""
        return datetime(2026, 6, 12, 18, 0, 0, tzinfo=UTC).isoformat()


class _OnlyIsoformat:
    """Test helper exposing only an ``isoformat()`` method."""

    def isoformat(self) -> str:
        """Return an ISO-8601 timestamp string."""
        return datetime(2026, 6, 12, 19, 0, 0, tzinfo=UTC).isoformat()


class _FallbackString:
    """Test helper that falls back all the way to ``str()``."""

    def item(self) -> object:
        """Raise to force the next fallback branch."""
        raise RuntimeError("item failed")

    def tolist(self) -> list[int]:
        """Raise to force the next fallback branch."""
        raise RuntimeError("tolist failed")

    def isoformat(self) -> str:
        """Raise to force the final ``str()`` fallback."""
        raise RuntimeError("isoformat failed")

    def __str__(self) -> str:
        """Return the string fallback representation."""
        return "fallback-value"


class _PlainStringValue:
    """Test helper with no protocol methods beyond ``__str__``."""

    def __str__(self) -> str:
        """Return the string fallback representation."""
        return "plain-fallback"


def test_dataset_fingerprint_records_hash_and_file_metadata(tmp_path: Path) -> None:
    """dataset_fingerprint should capture path, size, mtime, and SHA-256."""
    path = tmp_path / "dataset.csv"
    path.write_text("team,score\nBUF,24\n", encoding="utf-8")

    result = fingerprints.dataset_fingerprint(path)
    stat = path.stat()

    assert result["path"] == str(path)
    assert result["size"] == stat.st_size
    assert result["mtime"] == float(stat.st_mtime)
    assert isinstance(result["sha256"], str)
    assert len(result["sha256"]) == 64


def test_stable_fingerprint_normalizes_key_order_and_paths() -> None:
    """stable_fingerprint should ignore dict key order after JSON normalization."""
    left = {"path": Path("data.csv"), "values": (1, 2), "nested": {1: True}}
    right = {"nested": {1: True}, "values": [1, 2], "path": Path("data.csv")}

    assert fingerprints.stable_fingerprint(left) == fingerprints.stable_fingerprint(right)


def test_wf_run_fingerprint_depends_on_sha_args_and_code_version() -> None:
    """wf_run_fingerprint should ignore non-hash dataset metadata but react to inputs."""
    base_dataset = {"sha256": "deadbeef", "path": "one.csv", "size": 10, "mtime": 1.0}
    same_hash_dataset = {"sha256": "deadbeef", "path": "two.csv", "size": 99, "mtime": 2.0}
    wf_args = {"eval_last_n_seasons": 3, "resume": True}

    first = fingerprints.wf_run_fingerprint(base_dataset, wf_args, code_version="v1")
    second = fingerprints.wf_run_fingerprint(same_hash_dataset, dict(wf_args), code_version="v1")
    changed_args = fingerprints.wf_run_fingerprint(
        base_dataset,
        {"eval_last_n_seasons": 4, "resume": True},
        code_version="v1",
    )
    changed_code = fingerprints.wf_run_fingerprint(base_dataset, wf_args, code_version="v2")

    assert first == second
    assert first != changed_args
    assert first != changed_code


def test_to_jsonable_covers_protocol_fallback_chain() -> None:
    """to_jsonable should walk through item, tolist, isoformat, and str fallbacks."""
    assert fingerprints.to_jsonable(_ItemValue(Path("nested.csv"))) == "nested.csv"
    assert fingerprints.to_jsonable(_FallbackToList()) == [1, 2, 3]
    assert fingerprints.to_jsonable(_OnlyToList()) == [4, 5, 6]
    assert fingerprints.to_jsonable(_FallbackIsoformat()) == "2026-06-12T18:00:00+00:00"
    assert fingerprints.to_jsonable(_OnlyIsoformat()) == "2026-06-12T19:00:00+00:00"
    assert fingerprints.to_jsonable(_FallbackString()) == "fallback-value"
    assert fingerprints.to_jsonable(_PlainStringValue()) == "plain-fallback"
