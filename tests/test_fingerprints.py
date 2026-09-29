"""Tests for resumable workflow fingerprint helpers."""

from __future__ import annotations

from pathlib import Path

from nfl_predictor.utils import fingerprints


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
