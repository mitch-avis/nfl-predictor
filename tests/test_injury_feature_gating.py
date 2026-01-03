"""Tests for disabling injury features for in-season prediction.

NFLverse participation/injury data for in-progress seasons does not update during the season.
The ML pipeline should default to disabling injury features when prediction inputs have no
in-season injury data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_predictor import constants, ml_model

# pylint: disable=protected-access


def test_auto_disables_injury_features_when_prediction_all_null() -> None:
    """Auto mode disables injuries when prediction columns are entirely null."""

    predict_df = pd.DataFrame({c: [np.nan, np.nan] for c in constants.INJURY_FEATURE_COLUMNS})
    predict_df["season"] = [2025, 2025]

    enabled = ml_model._should_enable_injury_features(
        None,
        predict_df,
        current_season=2025,
    )

    assert enabled is False


def test_explicit_flag_overrides_auto_disabling() -> None:
    """Explicit user flag should override auto behavior."""

    predict_df = pd.DataFrame({c: [np.nan] for c in constants.INJURY_FEATURE_COLUMNS})
    predict_df["season"] = [2025]

    assert ml_model._should_enable_injury_features(True, predict_df, current_season=2025) is True
    assert ml_model._should_enable_injury_features(False, predict_df, current_season=2025) is False


def test_drop_injury_feature_columns_removes_only_injury_fields() -> None:
    """Dropping should remove injury columns and keep others intact."""

    df = pd.DataFrame(
        {
            "away_abbr": ["BUF"],
            "home_abbr": ["MIA"],
            "some_feature": [1.0],
            constants.INJURY_FEATURE_COLUMNS[0]: [0.5],
        }
    )

    out, dropped = ml_model._drop_injury_feature_columns(df)

    assert constants.INJURY_FEATURE_COLUMNS[0] in dropped
    assert constants.INJURY_FEATURE_COLUMNS[0] not in out.columns
    assert out["some_feature"].tolist() == [1.0]
