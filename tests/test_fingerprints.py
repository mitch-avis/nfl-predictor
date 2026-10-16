"""Tests for resumable workflow fingerprint helpers."""

from __future__ import annotations

from pathlib import Path

from nfl_predictor.utils import fingerprints


def test_dataset_fingerprint_records_hash_and_file_metadata(tmp_path: Path) -> None:
    """dataset_fingerprint should capture path, size, mtime, and SHA-256."""
    # Arrange
    path = tmp_path / "dataset.csv"
    path.write_text("team,score\nBUF,24\n", encoding="utf-8")
    stat = path.stat()

    # Act
    result = fingerprints.dataset_fingerprint(path)

    # Assert
    assert result["path"] == str(path)
    assert result["size"] == stat.st_size
    assert result["mtime"] == float(stat.st_mtime)
    assert isinstance(result["sha256"], str)
    assert len(result["sha256"]) == 64


def test_stable_fingerprint_normalizes_key_order_and_paths() -> None:
    """stable_fingerprint should ignore dict key order after JSON normalization."""
    # Arrange
    left = {"path": Path("data.csv"), "values": (1, 2), "nested": {1: True}}
    right = {"nested": {1: True}, "values": [1, 2], "path": Path("data.csv")}
    expected = fingerprints.stable_fingerprint(right)

    # Act
    result = fingerprints.stable_fingerprint(left)

    # Assert
    assert result == expected


def test_wf_run_fingerprint_ignores_non_hash_dataset_metadata() -> None:
    """wf_run_fingerprint should ignore non-hash dataset metadata."""
    # Arrange
    base_dataset = {"sha256": "deadbeef", "path": "one.csv", "size": 10, "mtime": 1.0}
    same_hash_dataset = {"sha256": "deadbeef", "path": "two.csv", "size": 99, "mtime": 2.0}
    wf_args = {"eval_last_n_seasons": 3, "resume": True}
    first = fingerprints.wf_run_fingerprint(base_dataset, wf_args, code_version="v1")

    # Act
    second = fingerprints.wf_run_fingerprint(same_hash_dataset, dict(wf_args), code_version="v1")

    # Assert
    assert first == second


def test_wf_run_fingerprint_depends_on_wf_args() -> None:
    """wf_run_fingerprint should react to changed walk-forward arguments."""
    # Arrange
    base_dataset = {"sha256": "deadbeef", "path": "one.csv", "size": 10, "mtime": 1.0}
    wf_args = {"eval_last_n_seasons": 3, "resume": True}
    first = fingerprints.wf_run_fingerprint(base_dataset, wf_args, code_version="v1")

    # Act
    changed_args = fingerprints.wf_run_fingerprint(
        base_dataset,
        {"eval_last_n_seasons": 4, "resume": True},
        code_version="v1",
    )

    # Assert
    assert first != changed_args


def test_wf_run_fingerprint_depends_on_code_version() -> None:
    """wf_run_fingerprint should react to a changed code version."""
    # Arrange
    base_dataset = {"sha256": "deadbeef", "path": "one.csv", "size": 10, "mtime": 1.0}
    wf_args = {"eval_last_n_seasons": 3, "resume": True}
    first = fingerprints.wf_run_fingerprint(base_dataset, wf_args, code_version="v1")

    # Act
    changed_code = fingerprints.wf_run_fingerprint(base_dataset, wf_args, code_version="v2")

    # Assert
    assert first != changed_code
