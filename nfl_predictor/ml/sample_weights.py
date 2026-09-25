"""Sample-weight helpers for training and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_postseason_sample_weight(
    df: pd.DataFrame,
    *,
    include_postseason: bool,
    postseason_weight: float,
) -> np.ndarray | None:
    """Compute per-row sample weights with optional postseason upweighting.

    When `game_type` exists and `include_postseason` is True, any row whose
    `game_type` is not REG is treated as postseason and assigned `postseason_weight`.
    Regular season rows keep weight 1.0.

    Returns None when weights are unnecessary (all ones).
    """
    if postseason_weight <= 0:
        raise ValueError("postseason_weight must be positive.")

    if not include_postseason:
        return None
    if "game_type" not in df.columns:
        return None

    game_type = df["game_type"].astype(str).str.upper()
    is_postseason = game_type != "REG"
    if not bool(is_postseason.any()):
        return None

    weights = np.ones(len(df), dtype=float)
    weights[is_postseason.to_numpy()] = float(postseason_weight)
    if np.allclose(weights, 1.0):
        return None
    return weights


def compute_recency_sample_weight(
    df: pd.DataFrame,
    *,
    half_life_seasons: float | None = None,
) -> np.ndarray | None:
    """Compute exponential recency weights by season: ``0.5 ** (age / half_life_seasons)``.

    ``age`` is the number of seasons before the newest season in ``df``. Returns ``None``
    when no half-life is given, ``df`` is empty, or every weight is 1.
    """
    if half_life_seasons is None:
        return None
    if half_life_seasons <= 0:
        raise ValueError("half_life_seasons must be positive.")
    if df.empty:
        return None

    if "season" not in df.columns:
        raise ValueError("season column required for recency weighting.")
    season_series = pd.to_numeric(df["season"], errors="coerce")
    if season_series.isna().any():
        raise ValueError("season column contains non-numeric values.")
    season_int = season_series.astype(int)

    age = season_int.max() - season_int
    weights = 0.5 ** (age.to_numpy() / float(half_life_seasons))
    if np.allclose(weights, 1.0):
        return None
    return weights.astype(float)


def combine_sample_weights(*weights: np.ndarray | None) -> np.ndarray | None:
    """Combine multiple weight vectors by multiplication."""
    active = [w for w in weights if w is not None]
    if not active:
        return None

    combined = np.ones_like(active[0], dtype=float)
    for weight in active:
        combined = combined * weight

    if np.allclose(combined, 1.0):
        return None
    return combined
