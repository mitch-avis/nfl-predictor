"""Snapshot helpers for the characterization tests.

A characterization test runs real code on a fixed input and compares its outputs with files
committed under ``tests/fixtures/``. Text and integers must match exactly; floats must match to
a relative ``1e-6``, which absorbs last-digit differences between machines but not a changed
prediction. Running the tests with ``NFLP_UPDATE_SNAPSHOTS=1`` rewrites the snapshots instead
of comparing; commit the rewrite with the change that caused it.
"""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UPDATE_ENV = "NFLP_UPDATE_SNAPSHOTS"
FLOAT_RTOL = 1e-6
FLOAT_ATOL = 1e-9


def updating() -> bool:
    """Return whether this run rewrites snapshots instead of comparing with them."""
    return os.environ.get(UPDATE_ENV) == "1"


def assert_frames_match(name: str, actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Compare two tables: exact for text and integers, close for floats."""
    assert list(actual.columns) == list(expected.columns), f"{name}: columns differ"
    assert len(actual) == len(expected), f"{name}: row count differs"
    for column in expected.columns:
        left, right = actual[column], expected[column]
        if pd.api.types.is_float_dtype(right) or pd.api.types.is_float_dtype(left):
            np.testing.assert_allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=FLOAT_RTOL,
                atol=FLOAT_ATOL,
                equal_nan=True,
                err_msg=f"{name}: column {column} differs",
            )
        else:
            assert left.astype(str).tolist() == right.astype(str).tolist(), (
                f"{name}: column {column} differs"
            )


def assert_json_match(name: str, actual: Any, expected: Any) -> None:
    """Compare JSON values: exact except for floats, which are compared closely."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{name}: type differs"
        assert sorted(actual) == sorted(expected), f"{name}: keys differ"
        for key, value in expected.items():
            assert_json_match(f"{name}.{key}", actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{name}: type differs"
        assert len(actual) == len(expected), f"{name}: length differs"
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_json_match(f"{name}[{index}]", left, right)
    elif isinstance(expected, float) or isinstance(actual, float):
        if expected is None or actual is None:
            assert actual == expected, f"{name}: {actual!r} != {expected!r}"
            return
        both_nan = math.isnan(float(actual)) and math.isnan(float(expected))
        assert both_nan or math.isclose(
            float(actual), float(expected), rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL
        ), f"{name}: {actual} != {expected}"
    else:
        assert actual == expected, f"{name}: {actual!r} != {expected!r}"


def check_csv(name: str, frame: pd.DataFrame, snapshot: Path) -> None:
    """Rewrite ``snapshot`` from ``frame`` when updating, else compare with it."""
    if updating():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(snapshot, index=False)
        return
    assert snapshot.exists(), f"missing snapshot {snapshot}; run with {UPDATE_ENV}=1"
    # Round-trip through CSV so both sides carry the dtypes a CSV read gives them.
    actual = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
    assert_frames_match(name, actual, pd.read_csv(snapshot))


def check_json(name: str, payload: Any, snapshot: Path) -> None:
    """Rewrite ``snapshot`` from ``payload`` when updating, else compare with it."""
    if updating():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    assert snapshot.exists(), f"missing snapshot {snapshot}; run with {UPDATE_ENV}=1"
    assert_json_match(name, payload, json.loads(snapshot.read_text(encoding="utf-8")))
