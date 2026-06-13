"""Unit tests for small, deterministic helpers in nfl_predictor.ml.artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from nfl_predictor.ml import artifacts


@dataclass(frozen=True)
class _PayloadDataclass:
    """Small dataclass payload used to exercise JSON coercion branches."""

    path: Path
    count: int


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


def test_to_jsonable_handles_dataclasses_and_fallback_protocols() -> None:
    """_to_jsonable should normalize dataclasses and protocol-like helper objects."""
    dataclass_payload = _PayloadDataclass(path=Path("data.csv"), count=3)

    assert artifacts._to_jsonable(dataclass_payload) == {"path": "data.csv", "count": 3}
    assert artifacts._to_jsonable(_PayloadDataclass) == str(_PayloadDataclass)
    assert artifacts._to_jsonable(_ItemValue(Path("nested.csv"))) == "nested.csv"
    assert artifacts._to_jsonable(_FallbackToList()) == [1, 2, 3]
    assert artifacts._to_jsonable(_OnlyToList()) == [4, 5, 6]
    assert artifacts._to_jsonable(_FallbackIsoformat()) == "2026-06-12T18:00:00+00:00"
    assert artifacts._to_jsonable(_OnlyIsoformat()) == "2026-06-12T19:00:00+00:00"
    assert artifacts._to_jsonable(_FallbackString()) == "fallback-value"
    assert artifacts._to_jsonable(_PlainStringValue()) == "plain-fallback"


def test_generate_run_id_uses_frozen_timestamp_and_stable_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_run_id should combine the prefix, timestamp, and stable short hash."""

    class _FrozenDateTime:
        """Test double for the datetime class used by generate_run_id."""

        @classmethod
        def now(cls, tz: object) -> datetime:
            """Return a fixed UTC timestamp."""
            assert tz is artifacts.UTC
            return datetime(2026, 6, 12, 18, 1, 2, tzinfo=UTC)

    monkeypatch.setattr(artifacts, "datetime", _FrozenDateTime)

    config = {"calibration": "platt", "holdout_seasons": 1}
    expected_hash = artifacts.stable_short_hash({"dataset_hash": "abc123", "config": config})

    assert (
        artifacts.generate_run_id("weekly", "abc123", config)
        == f"weekly_20260612_180102_{expected_hash}"
    )


def test_resolve_run_paths_defaults_to_repo_models_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """resolve_run_paths should default under ``ROOT_DIR/models/<run_id>``."""
    monkeypatch.setattr(artifacts.constants, "ROOT_DIR", tmp_path)

    paths = artifacts.resolve_run_paths("weekly_run")

    assert paths.run_dir == tmp_path / "models" / "weekly_run"
    assert paths.model_path == paths.run_dir / "model.joblib"
    assert paths.metadata_path == paths.run_dir / "metadata.json"
    assert paths.metrics_path == paths.run_dir / "metrics_report.json"
    assert paths.feature_importance_path == paths.run_dir / "feature_importance.json"


def test_git_commit_hash_handles_success_failure_and_missing_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """git_commit_hash should tolerate git failures and missing executables."""
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="abc123\n"),
    )
    assert artifacts.git_commit_hash() == "abc123"

    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="ignored\n"),
    )
    assert artifacts.git_commit_hash() is None

    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="\n"),
    )
    assert artifacts.git_commit_hash() is None

    def _raise_os_error(*_args: object, **_kwargs: object) -> SimpleNamespace:
        """Raise an OSError to emulate a missing git executable."""
        raise OSError("git unavailable")

    monkeypatch.setattr(artifacts.subprocess, "run", _raise_os_error)
    assert artifacts.git_commit_hash() is None


def test_module_version_library_versions_and_metadata_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Version helpers and metadata assembly should remain deterministic under mocks."""
    fake_module = ModuleType("fake_versioned_module")
    fake_module.__dict__["__version__"] = "fake-1.0"
    monkeypatch.setitem(sys.modules, "fake_versioned_module", fake_module)

    assert artifacts._module_version("fake_versioned_module") == "fake-1.0"
    assert artifacts._module_version("missing_module") is None

    monkeypatch.setattr(artifacts, "_module_version", lambda name: f"{name}-1.0")

    versions = artifacts.library_versions()
    assert versions["numpy"] == "numpy-1.0"
    assert versions["pandas"] == "pandas-1.0"
    assert versions["python"]
    assert versions["python_executable"]

    monkeypatch.setattr(artifacts, "git_commit_hash", lambda: "commit123")
    monkeypatch.setattr(artifacts, "library_versions", lambda: {"python": "3.14-test"})

    metadata = artifacts.build_metadata(
        created_at="2026-06-12T18:00:00+00:00",
        run_id="weekly_20260612",
        dataset_hash="dataset123",
        config={"market_mode": "anchor"},
        feature_list=["feature_a"],
        splits={"holdout_seasons": [2025]},
        params={"max_depth": 4},
        tuned_params={"eta": 0.1},
        early_stopping={"best_iteration": 12},
        optuna_summary={"best_value": 0.2},
    )

    assert metadata["git_commit_hash"] == "commit123"
    assert metadata["library_versions"] == {"python": "3.14-test"}
    assert metadata["optuna_summary"] == {"best_value": 0.2}


def test_write_json_serializes_paths(tmp_path: Path) -> None:
    """write_json converts Path values to strings and writes valid JSON."""
    out = tmp_path / "out.json"
    artifacts.write_json(out, {"path": tmp_path})

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["path"] == str(tmp_path)
