"""Unit tests for small, deterministic helpers in nfl_predictor.ml.artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from nfl_predictor.ml import artifacts

if TYPE_CHECKING:
    from collections.abc import Callable


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
        msg = "item failed"
        raise ValueError(msg)

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
        msg = "item failed"
        raise ValueError(msg)

    def tolist(self) -> list[int]:
        """Raise to force the next fallback branch."""
        msg = "tolist failed"
        raise ValueError(msg)

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
        msg = "item failed"
        raise ValueError(msg)

    def tolist(self) -> list[int]:
        """Raise to force the next fallback branch."""
        msg = "tolist failed"
        raise ValueError(msg)

    def isoformat(self) -> str:
        """Raise to force the final ``str()`` fallback."""
        msg = "isoformat failed"
        raise ValueError(msg)

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
    # Act
    value = artifacts.now_utc_iso()

    # Assert
    # Should be parseable by stdlib; tolerate timezone suffix.
    parsed = value.replace("Z", "+00:00")
    assert parsed
    # datetime.fromisoformat is strict, so this catches malformed strings.
    assert isinstance(datetime.fromisoformat(parsed), datetime)


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    """sha256_file matches hashlib's result for the same bytes."""
    # Arrange
    path = tmp_path / "payload.bin"
    payload = b"hello world\n"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    # Act
    digest = artifacts.sha256_file(path)

    # Assert
    assert digest == expected


def test_stable_short_hash_is_order_invariant() -> None:
    """stable_short_hash is stable under dict key reordering."""
    # Arrange
    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}

    # Act
    hashes = {artifacts.stable_short_hash(payload) for payload in (a, b)}

    # Assert
    assert len(hashes) == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (_PayloadDataclass(path=Path("data.csv"), count=3), {"path": "data.csv", "count": 3}),
        (_PayloadDataclass, str(_PayloadDataclass)),
        (_ItemValue(Path("nested.csv")), "nested.csv"),
        (_FallbackToList(), [1, 2, 3]),
        (_OnlyToList(), [4, 5, 6]),
        (_FallbackIsoformat(), "2026-06-12T18:00:00+00:00"),
        (_OnlyIsoformat(), "2026-06-12T19:00:00+00:00"),
        (_FallbackString(), "fallback-value"),
        (_PlainStringValue(), "plain-fallback"),
    ],
    ids=[
        "dataclass",
        "dataclass-type",
        "item",
        "item-then-tolist",
        "tolist",
        "isoformat-fallback",
        "isoformat",
        "str-fallback",
        "plain-str",
    ],
)
def test_to_jsonable_handles_dataclasses_and_fallback_protocols(
    value: object, expected: object
) -> None:
    """to_jsonable should normalize dataclasses and protocol-like helper objects."""
    # Act
    result = artifacts.to_jsonable(value)

    # Assert
    assert result == expected


def test_generate_run_id_uses_frozen_timestamp_and_stable_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_run_id should combine the prefix, timestamp, and stable short hash."""

    # Arrange
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

    # Act
    run_id = artifacts.generate_run_id("weekly", "abc123", config)

    # Assert
    assert run_id == f"weekly_20260612_180102_{expected_hash}"


def test_resolve_run_paths_defaults_to_repo_models_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """resolve_run_paths should default under ``MODELS_PATH/<run_id>``."""
    # Arrange
    monkeypatch.setattr(artifacts.constants, "MODELS_PATH", tmp_path / "models")

    # Act
    paths = artifacts.resolve_run_paths("weekly_run")

    # Assert
    assert paths.run_dir == tmp_path / "models" / "weekly_run"
    assert paths.model_path == paths.run_dir / "model.joblib"
    assert paths.metadata_path == paths.run_dir / "metadata.json"
    assert paths.metrics_path == paths.run_dir / "metrics_report.json"
    assert paths.feature_importance_path == paths.run_dir / "feature_importance.json"


def _completed(returncode: int, stdout: str) -> Callable[..., SimpleNamespace]:
    """Return a stand-in for ``subprocess.run`` that finishes with ``returncode`` and ``stdout``."""
    return lambda *_args, **_kwargs: SimpleNamespace(returncode=returncode, stdout=stdout)


def _raise_os_error(*_args: object, **_kwargs: object) -> SimpleNamespace:
    """Raise an OSError to emulate a missing git executable."""
    msg = "git unavailable"
    raise OSError(msg)


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        (_completed(0, "abc123\n"), "abc123"),
        (_completed(1, "ignored\n"), None),
        (_completed(0, "\n"), None),
        (_raise_os_error, None),
    ],
    ids=["success", "failure", "empty-output", "missing-git"],
)
def test_git_commit_hash_handles_success_failure_and_missing_git(
    monkeypatch: pytest.MonkeyPatch, run: Callable[..., SimpleNamespace], expected: str | None
) -> None:
    """git_commit_hash should tolerate git failures and missing executables."""
    # Arrange
    monkeypatch.setattr(artifacts.subprocess, "run", run)

    # Act
    commit = artifacts.git_commit_hash()

    # Assert
    assert commit == expected


@pytest.mark.parametrize(
    ("module", "expected"),
    [("fake_versioned_module", "fake-1.0"), ("missing_module", None)],
    ids=["versioned", "missing"],
)
def test_module_version_reads_the_version_of_an_importable_module(
    monkeypatch: pytest.MonkeyPatch, module: str, expected: str | None
) -> None:
    """_module_version returns a module's ``__version__``, or None when it cannot import it."""
    # Arrange
    fake_module = ModuleType("fake_versioned_module")
    fake_module.__dict__["__version__"] = "fake-1.0"
    monkeypatch.setitem(sys.modules, "fake_versioned_module", fake_module)

    # Act
    version = artifacts._module_version(module)

    # Assert
    assert version == expected


def test_library_versions_report_each_library_and_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """library_versions should remain deterministic under a mocked version lookup."""
    # Arrange
    monkeypatch.setattr(artifacts, "_module_version", lambda name: f"{name}-1.0")

    # Act
    versions = artifacts.library_versions()

    # Assert
    assert versions["numpy"] == "numpy-1.0"
    assert versions["pandas"] == "pandas-1.0"
    assert versions["python"]
    assert versions["python_executable"]


def test_build_metadata_records_the_commit_versions_and_optuna_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata assembly should remain deterministic under mocks."""
    # Arrange
    monkeypatch.setattr(artifacts, "git_commit_hash", lambda: "commit123")
    monkeypatch.setattr(artifacts, "library_versions", lambda: {"python": "3.14-test"})
    details = artifacts.TrainedModelDetails(
        feature_list=["feature_a"],
        splits={"holdout_seasons": [2025]},
        params={"max_depth": 4},
        tuned_params={"eta": 0.1},
        early_stopping={"best_iteration": 12},
        optuna_summary={"best_value": 0.2},
    )

    # Act
    metadata = artifacts.build_metadata(
        created_at="2026-06-12T18:00:00+00:00",
        run_id="weekly_20260612",
        dataset_hash="dataset123",
        config={"market_mode": "anchor"},
        details=details,
    )

    # Assert
    assert metadata["git_commit_hash"] == "commit123"
    assert metadata["library_versions"] == {"python": "3.14-test"}
    assert metadata["optuna_summary"] == {"best_value": 0.2}
    assert metadata["xgb_device"] is None


def test_build_metadata_records_the_xgb_device() -> None:
    """The artifact contract names the device the model trained on."""
    # Act
    metadata = artifacts.build_metadata(
        created_at="2026-09-27T00:00:00+00:00",
        run_id="train_20260927",
        dataset_hash="dataset123",
        config={},
        details=artifacts.TrainedModelDetails(xgb_device="cuda"),
    )

    # Assert
    assert metadata["xgb_device"] == "cuda"


def test_write_json_serializes_paths(tmp_path: Path) -> None:
    """write_json converts Path values to strings and writes valid JSON."""
    # Arrange
    out = tmp_path / "out.json"

    # Act
    artifacts.write_json(out, {"path": tmp_path})

    # Assert
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["path"] == str(tmp_path)


class _BrokenItem:
    """Test helper whose ``item()`` fails with an unexpected error."""

    def item(self) -> object:
        """Raise an error the fallback chain must not hide."""
        msg = "item broke"
        raise RuntimeError(msg)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(np.int64(7), 7), (np.array([1, 2]), [1, 2])],
    ids=["scalar", "array"],
)
def test_to_jsonable_converts_numpy_values(value: object, expected: object) -> None:
    """A NumPy scalar becomes its Python value and an array falls back to a list."""
    # Act
    result = artifacts.to_jsonable(value)

    # Assert
    assert result == expected


def test_to_jsonable_raises_unexpected_errors() -> None:
    """Only the multi-element ``ValueError`` falls through; other errors surface."""
    # Act & Assert
    with pytest.raises(RuntimeError, match="item broke"):
        artifacts.to_jsonable(_BrokenItem())
