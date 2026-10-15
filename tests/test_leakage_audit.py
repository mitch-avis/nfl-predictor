"""Tests for leakage audit tooling."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.ml import leakage_audit


def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "feat1": [0.1, 0.2, 0.3],
            "feat2": [1.0, 1.1, 1.2],
            "home_moneyline": [-110, -120, 130],
            "away_score": [17, 21, 14],
            "home_score": [20, 17, 24],
            # Intentionally leaked feature (a target copy) for audit tests.
            "away_score_leak": [17, 21, 14],
        }
    )


def test_leakage_audit_detects_target_in_features() -> None:
    """Audit should fail if a target copy is in the feature range."""
    # Arrange
    df = _tiny_df()

    config = leakage_audit.LeakageAuditConfig(
        feature_start="feat1",
        feature_end="away_score_leak",  # intentionally includes a leaked target copy
        include_market=True,
        market_transform=False,
    )

    # Act
    report = leakage_audit.run_leakage_audit(df, config)

    # Assert
    assert report["ok"] is False
    assert any("nearly equals" in msg for msg in report["failures"])


def test_leakage_audit_runs_on_fixture() -> None:
    """Audit should pass when targets are excluded from the feature range."""
    # Arrange
    df = _tiny_df()

    config = leakage_audit.LeakageAuditConfig(
        feature_start="feat1",
        feature_end="home_moneyline",
        include_market=True,
        market_transform=False,
    )

    # Act
    report = leakage_audit.run_leakage_audit(df, config)

    # Assert
    assert "ok" in report
    assert "flagged_columns" in report
    # Should pass: targets are not in feature range.
    assert report["ok"] is True


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.array([1.0, np.nan]), np.array([2.0, 3.0])),
        (np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])),
    ],
)
def test_safe_corr_and_equality_rate_handle_edge_cases(left: np.ndarray, right: np.ndarray) -> None:
    """The correlation helper returns `None` for degenerate inputs."""
    # Act
    corr = leakage_audit._safe_corr(left, right)

    # Assert
    assert corr is None


def test_equality_rate_degenerate_input_is_none() -> None:
    """The equality-rate helper returns `None` for degenerate inputs."""
    # Act
    rate = leakage_audit._equality_rate(np.array([np.nan]), np.array([1.0]))

    # Assert
    assert rate is None


def test_equality_rate_scores_valid_arrays() -> None:
    """The equality-rate helper scores valid arrays."""
    # Act
    rate = leakage_audit._equality_rate(np.array([1.0, 2.0]), np.array([1.0, 3.0]))

    # Assert
    assert rate == 0.5


def test_leakage_audit_flags_suspicious_names_and_near_perfect_correlation(
    monkeypatch,
) -> None:
    """Flags heuristic names and fails for near-perfect label-derived features."""
    # Arrange
    df = pd.DataFrame(
        {
            "away_score": [10, 14, 21],
            "home_score": [20, 17, 24],
            "winner_hint": [1, 0, 1],
            "total_copy": [30, 31, 45],
        }
    )

    monkeypatch.setattr(
        leakage_audit.ml_model,
        "get_target_columns",
        lambda _df: ("away_score", "home_score"),
    )
    monkeypatch.setattr(
        leakage_audit.ml_model,
        "build_feature_spec",
        lambda *_args, **_kwargs: SimpleNamespace(feature_columns=["winner_hint", "total_copy"]),
    )

    # Act
    report = leakage_audit.run_leakage_audit(df, leakage_audit.LeakageAuditConfig())

    # Assert
    assert report["ok"] is False
    assert any("Near-perfect correlation" in message for message in report["failures"])
    assert any(item["reason"] == "suspicious_name" for item in report["flagged_columns"])


def test_write_report_writes_sorted_json(tmp_path) -> None:
    """Persists the JSON report to disk using a stable sort order."""
    # Arrange
    report = {"b": 2, "a": 1}
    out_path = tmp_path / "report.json"

    # Act
    leakage_audit.write_report(report, out_path)

    # Assert
    assert json.loads(out_path.read_text(encoding="utf-8")) == report
    assert out_path.read_text(encoding="utf-8").splitlines()[1].strip().startswith('"a"')


def test_cli_main_creates_missing_report_directory(tmp_path, monkeypatch) -> None:
    """The CLI writes its report even when the output directory does not exist yet."""
    # Arrange
    from nfl_predictor.cli import leakage_audit as leakage_audit_cli

    data_path = tmp_path / "games.csv"
    _tiny_df().to_csv(data_path, index=False)
    out_path = tmp_path / "missing" / "nested" / "leakage_audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "leakage-audit",
            "--data-path",
            str(data_path),
            "--out-json",
            str(out_path),
            "--feature-start",
            "feat1",
            "--feature-end",
            "home_moneyline",
        ],
    )

    # Act
    exit_code = leakage_audit_cli.main()

    # Assert
    assert exit_code == 0
    assert json.loads(out_path.read_text())["ok"] is True
