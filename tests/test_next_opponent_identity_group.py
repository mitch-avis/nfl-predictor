"""Tests for the ablation group that drops the next opponent's team identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from nfl_predictor import constants
from nfl_predictor.ml import ml_model_xgb_utils, walk_forward
from nfl_predictor.ml.feature_spec import (
    FeatureSelection,
    apply_feature_spec,
    build_feature_spec,
    build_preprocessor,
)
from nfl_predictor.utils.polars.finalize import build_final_column_order

GROUP = "next_opponent_identity"
PAIR = ["away_next_opponent_abbr", "home_next_opponent_abbr"]
ROWS = 256
TEAMS = sorted(set(constants.ALIAS_TO_CANONICAL.values()))
STRING_FEATURES = {
    "stadium_surface": ("grass", "fieldturf", "a_turf"),
    "stadium_type": ("outdoors", "dome", "retractable"),
}


def _schema_frame() -> pd.DataFrame:
    """Return a frame with every published column, typed the way the real dataset loads.

    Inside the model's feature range the real dataset has four string columns: the stadium
    surface and type, and the next-opponent pair. Everything else there is numeric.
    """
    rng = np.random.default_rng(7)
    data: dict[str, object] = {}
    for column in build_final_column_order():
        if column in PAIR:
            data[column] = [TEAMS[int(i) % len(TEAMS)] for i in rng.integers(0, 10**6, ROWS)]
        elif column in STRING_FEATURES:
            choices = STRING_FEATURES[column]
            data[column] = [choices[int(i)] for i in rng.integers(0, len(choices), ROWS)]
        else:
            data[column] = rng.normal(size=ROWS)
    frame = pd.DataFrame(data)
    frame["season"] = 2024
    frame["week"] = 3
    frame["game_type"] = "REG"
    return frame


def _frame_after_group_drop(
    monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame, groups: tuple[str, ...]
) -> pd.DataFrame:
    """Return the games a walk-forward run trains on after it drops ``groups``."""
    captured: dict[str, pd.DataFrame] = {}

    def capture(df: pd.DataFrame, *, include_postseason: bool = False) -> pd.DataFrame:
        captured["df"] = df
        msg = "stop after the drop"
        raise RuntimeError(msg)

    monkeypatch.setattr(walk_forward, "filter_regular_season", capture)
    config = walk_forward.WalkForwardConfig(
        disabled_feature_groups=groups, xgb_params_overrides={"device": "cpu"}
    )
    with pytest.raises(RuntimeError, match="stop after the drop"):
        walk_forward.run_walk_forward_backtest(frame, config)
    return captured["df"]


def _feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the encoded matrix a walk-forward week fits on, with its column names."""
    spec = build_feature_spec(
        frame,
        FeatureSelection(
            include_market=True,
            max_cardinality_ratio=walk_forward.WalkForwardConfig().max_cardinality_ratio,
            market_transform=True,
        ),
    )
    preprocessor = build_preprocessor(spec, for_tree=True)
    matrix = ml_model_xgb_utils.fit_transform_matrix(preprocessor, apply_feature_spec(frame, spec))
    dense = sparse.csr_matrix(matrix).toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    return pd.DataFrame(dense, columns=names)


def test_group_resolves_to_exactly_the_next_opponent_pair() -> None:
    """The group names the two identity columns of the published schema and nothing else."""
    order = build_final_column_order()

    assert walk_forward.resolve_feature_group_columns(order, [GROUP]) == PAIR
    for column in ("away_next_opponent_win_pct", "home_next_opponent_win_pct"):
        assert column in order


def test_group_is_disjoint_from_every_other_group() -> None:
    """No other ablation group drops either identity column."""
    order = build_final_column_order()

    for other in constants.FEATURE_GROUP_COLUMN_MARKERS:
        if other != GROUP:
            assert not set(walk_forward.resolve_feature_group_columns(order, [other])) & set(PAIR)


def test_feature_matrix_is_unchanged_when_the_group_is_not_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no group disabled, the run fits on exactly the matrix the full frame gives."""
    frame = _schema_frame()

    after = _frame_after_group_drop(monkeypatch, frame, ())

    pd.testing.assert_frame_equal(after, frame)
    pd.testing.assert_frame_equal(_feature_matrix(after), _feature_matrix(frame))


def test_disabling_the_group_removes_only_the_pair_and_its_one_hot_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled, the group takes out the pair's encoded columns and leaves the rest identical."""
    frame = _schema_frame()
    full = _feature_matrix(frame)

    after = _frame_after_group_drop(monkeypatch, frame, (GROUP,))
    ablated = _feature_matrix(after)

    assert set(frame.columns) - set(after.columns) == set(PAIR)
    removed = [name for name in full.columns if name not in set(ablated.columns)]
    pair_encodings = [name for name in full.columns if any(f"__{c}_" in name for c in PAIR)]
    assert removed == pair_encodings
    assert len(pair_encodings) == 2 * len(TEAMS)
    assert "num__away_next_opponent_win_pct" in ablated.columns
    pd.testing.assert_frame_equal(ablated, full[list(ablated.columns)])
