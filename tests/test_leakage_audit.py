"""Tests for leakage audit tooling."""

from __future__ import annotations

import pandas as pd

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
    df = _tiny_df()

    config = leakage_audit.LeakageAuditConfig(
        feature_start="feat1",
        feature_end="away_score_leak",  # intentionally includes a leaked target copy
        include_market=True,
        market_transform=False,
    )

    report = leakage_audit.run_leakage_audit(df, config)

    assert report["ok"] is False
    assert any("nearly equals" in msg for msg in report["failures"])


def test_leakage_audit_runs_on_fixture() -> None:
    """Audit should pass when targets are excluded from the feature range."""
    df = _tiny_df()

    config = leakage_audit.LeakageAuditConfig(
        feature_start="feat1",
        feature_end="home_moneyline",
        include_market=True,
        market_transform=False,
    )

    report = leakage_audit.run_leakage_audit(df, config)

    assert "ok" in report
    assert "flagged_columns" in report
    # Should pass: targets are not in feature range.
    assert report["ok"] is True
