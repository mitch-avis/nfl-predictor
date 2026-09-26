"""Tests for the paired bootstrap of probability metrics against the market.

The reference below is the original implementation: one Brier and one log-loss call per
resample through ``metrics.probability_summary``. The production bootstrap must return the
same interval bounds from the same random draws, whatever it computes internally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.ml import metrics as metrics_utils
from nfl_predictor.ml import walk_forward


def _reference_bootstrap(
    frame: pd.DataFrame, *, prefix: str, n_samples: int, seed: int
) -> dict[str, float]:
    """Resample games with replacement and recompute both metrics per resample."""
    model_prob = frame["model"].to_numpy(dtype=float)
    market_prob = frame["market"].to_numpy(dtype=float)
    valid = np.isfinite(model_prob) & np.isfinite(market_prob)
    actual_home_win = frame.loc[valid, "actual_home_win"].to_numpy(dtype=int)
    actual_margin = frame.loc[valid, "actual_margin"].to_numpy(dtype=float)
    model_prob, market_prob = model_prob[valid], market_prob[valid]
    rng = np.random.default_rng(seed)
    n_rows = len(actual_home_win)
    brier = np.empty(n_samples)
    log_loss = np.empty(n_samples)
    for index in range(n_samples):
        rows = rng.integers(0, n_rows, size=n_rows)
        model = metrics_utils.probability_summary(
            actual_home_win[rows], actual_margin[rows], model_prob[rows]
        )
        market = metrics_utils.probability_summary(
            actual_home_win[rows], actual_margin[rows], market_prob[rows]
        )
        brier[index] = model["brier"] - market["brier"]
        log_loss[index] = model["log_loss"] - market["log_loss"]
    return {
        f"{prefix}_brier_vs_market_ci_low": float(np.nanpercentile(brier, 2.5)),
        f"{prefix}_brier_vs_market_ci_high": float(np.nanpercentile(brier, 97.5)),
        f"{prefix}_log_loss_vs_market_ci_low": float(np.nanpercentile(log_loss, 2.5)),
        f"{prefix}_log_loss_vs_market_ci_high": float(np.nanpercentile(log_loss, 97.5)),
    }


def _games(n_rows: int, seed: int) -> pd.DataFrame:
    """Random games, with probabilities at and beyond both ends and a few missing values."""
    rng = np.random.default_rng(seed)
    margin = rng.normal(2.0, 13.0, size=n_rows).round()
    model = rng.uniform(0.0, 1.0, size=n_rows)
    market = np.clip(model + rng.normal(0.0, 0.08, size=n_rows), 0.01, 0.99)
    model[:4] = [0.0, 1.0, -0.2, 1.3]
    market[4] = np.nan
    return pd.DataFrame(
        {
            "model": model,
            "market": market,
            "actual_margin": margin,
            "actual_home_win": (margin > 0).astype(int),
        }
    )


@pytest.mark.parametrize(("n_rows", "seed"), [(48, 0), (720, 7), (1615, 42)])
def test_bootstrap_matches_the_per_resample_metric_calls(n_rows: int, seed: int) -> None:
    """Same draws, same metrics: every interval bound equals the reference to 1e-12."""
    frame = _games(n_rows, seed)

    actual = walk_forward._bootstrap_probability_differences(
        frame,
        model_column="model",
        market_column="market",
        prefix="deterministic",
        n_samples=400,
        seed=seed,
    )

    expected = _reference_bootstrap(frame, prefix="deterministic", n_samples=400, seed=seed)
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=1e-12, abs=1e-15), key


def test_bootstrap_returns_nothing_without_samples_or_columns() -> None:
    """No resamples, a missing column or no valid row gives an empty result."""
    frame = _games(20, 1)
    kwargs = {"prefix": "p", "seed": 0}
    assert (
        walk_forward._bootstrap_probability_differences(
            frame, model_column="model", market_column="market", n_samples=0, **kwargs
        )
        == {}
    )
    assert (
        walk_forward._bootstrap_probability_differences(
            frame, model_column="model", market_column="absent", n_samples=10, **kwargs
        )
        == {}
    )
    frame["market"] = np.nan
    assert (
        walk_forward._bootstrap_probability_differences(
            frame, model_column="model", market_column="market", n_samples=10, **kwargs
        )
        == {}
    )
