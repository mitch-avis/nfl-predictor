"""Tests for the betting pipeline script.

These tests are intentionally lightweight and only validate that the script can be imported
and that --dry-run exits successfully.
"""

from __future__ import annotations

import sys

from scripts import betting_pipeline


def test_betting_pipeline_dry_run_exits_successfully() -> None:
    """The betting pipeline script should support a dry run without heavy work."""

    old_argv = sys.argv
    try:
        sys.argv = [
            "betting_pipeline.py",
            "--dry-run",
            "--run-id",
            "test_betting_pipeline",
        ]
        exit_code = betting_pipeline.main()
    finally:
        sys.argv = old_argv
    assert isinstance(exit_code, int)
    assert exit_code == 0
