"""Tests for feature specification helpers."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from nfl_predictor import constants
from nfl_predictor.ml import feature_spec
from nfl_predictor.utils.logger import log
from tests.fakes import returning

_RANGE_FRAME_COLUMNS = ["meta", "feat_a", "feat_b", "feat_c", "tail"]


def test_get_feature_range_columns_returns_the_range_and_its_neighbours() -> None:
    """Feature range resolution returns the range and the columns before and after it."""
    # Arrange
    df = pd.DataFrame(columns=_RANGE_FRAME_COLUMNS)

    # Act
    feature_range, metadata_columns, post_feature_columns = feature_spec._get_feature_range_columns(
        df,
        "feat_a",
        "feat_c",
    )

    # Assert
    assert feature_range == ["feat_a", "feat_b", "feat_c"]
    assert metadata_columns == ["meta"]
    assert post_feature_columns == ["tail"]


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("missing", "feat_c", "Expected feature range columns"),
        ("feat_c", "feat_a", "occurs after"),
    ],
)
def test_get_feature_range_columns_rejects_missing_or_reversed_bounds(
    start: str, end: str, message: str
) -> None:
    """Feature range resolution rejects a missing bound and a start after the end."""
    # Arrange
    df = pd.DataFrame(columns=_RANGE_FRAME_COLUMNS)

    # Act & Assert
    with pytest.raises(ValueError, match=message):
        feature_spec._get_feature_range_columns(df, start, end)


def test_implied_prob_from_moneyline_accepts_series() -> None:
    """Moneyline conversion should work for Series inputs."""
    # Act
    series_probs = feature_spec._implied_prob_from_moneyline(pd.Series([-150, 200, 0]))

    # Assert
    assert series_probs[0] == pytest.approx(150 / 250)
    assert series_probs[1] == pytest.approx(100 / 300)
    assert np.isnan(series_probs[2])


def test_implied_prob_from_moneyline_accepts_arrays() -> None:
    """Moneyline conversion should work for ndarray inputs."""
    # Act
    array_probs = feature_spec._implied_prob_from_moneyline(np.array([-120, 150]))

    # Assert
    assert array_probs[0] == pytest.approx(120 / 220)
    assert array_probs[1] == pytest.approx(100 / 250)


def test_add_market_transforms_derives_missing_columns() -> None:
    """Market transforms should derive missing helper columns from spread, total, and moneyline."""
    # Arrange
    home_spread_df = pd.DataFrame(
        {
            "home_spread": [-3.5],
            "total_line": [46.5],
            "home_moneyline": [-150],
            "away_moneyline": [130],
        }
    )

    # Act
    transformed_home = feature_spec.add_market_transforms(home_spread_df)

    # Assert
    assert transformed_home["market_home_margin"].iloc[0] == pytest.approx(3.5)
    assert transformed_home["market_total_line"].iloc[0] == pytest.approx(46.5)
    assert transformed_home["home_market_prob"].iloc[0] == pytest.approx(150 / 250)
    assert transformed_home["away_market_prob"].iloc[0] == pytest.approx(100 / 230)


def test_add_market_transforms_derives_missing_columns_without_overwriting_existing() -> None:
    """Market transforms keep a market column that is already present."""
    # Arrange
    away_spread_df = pd.DataFrame(
        {
            "away_spread": [2.5],
            "market_home_margin": [9.0],
        }
    )

    # Act
    transformed_away = feature_spec.add_market_transforms(away_spread_df)

    # Assert
    assert transformed_away["market_home_margin"].iloc[0] == pytest.approx(9.0)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        pytest.param(
            pd.DataFrame({"feat": [1]}), "spread/total columns are missing", id="no_columns"
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "home_spread": [np.nan],
                    "total_line": [44.5],
                }
            ),
            "contains missing values",
            id="missing_values",
        ),
    ],
)
def test_get_market_baseline_validates_columns_and_missing_values(
    frame: pd.DataFrame, message: str
) -> None:
    """Market baseline extraction should reject missing or NaN-derived market values."""
    # Act & Assert
    with pytest.raises(ValueError, match=message):
        feature_spec.get_market_baseline(frame)


def test_get_market_baseline_returns_margin_and_total() -> None:
    """Market baseline extraction returns the home margin and the total line."""
    # Arrange
    frame = pd.DataFrame(
        {
            "home_spread": [-3.0],
            "total_line": [45.5],
        }
    )

    # Act
    baseline_margin, baseline_total = feature_spec.get_market_baseline(frame)

    # Assert
    assert baseline_margin.tolist() == [3.0]
    assert baseline_total.tolist() == [45.5]


def _id_constant_df() -> pd.DataFrame:
    """Return a frame with an identifier, a constant and a kept column."""
    return pd.DataFrame(
        {
            "team_id": ["a", "b"],
            "constant_flag": [1, 1],
            "keep": [10, 11],
        }
    )


def test_drop_identifier_columns_drops_an_identifier() -> None:
    """The identifier helper drops an identifier column."""
    # Arrange
    id_constant_df = _id_constant_df()

    # Act
    dropped_id_df, id_columns = feature_spec._drop_identifier_columns(id_constant_df)

    # Assert
    assert id_columns == ["team_id"]
    assert "team_id" not in dropped_id_df.columns


def test_drop_constant_columns_drops_a_constant_column() -> None:
    """The constant helper drops a column with one value."""
    # Arrange
    id_constant_df = _id_constant_df()

    # Act
    dropped_constant_df, constant_columns = feature_spec._drop_constant_columns(id_constant_df)

    # Assert
    assert constant_columns == ["constant_flag"]
    assert "constant_flag" not in dropped_constant_df.columns


def test_drop_high_cardinality_columns_drops_a_high_cardinality_column() -> None:
    """The high-cardinality helper drops a column above the ratio."""
    # Arrange
    high_card_df = pd.DataFrame(
        {
            "category": ["x", "y"],
            "keep": [10, 11],
        }
    )

    # Act
    dropped_high_card_df, high_card_columns = feature_spec._drop_high_cardinality_columns(
        high_card_df,
        max_cardinality_ratio=0.5,
    )

    # Assert
    assert high_card_columns == ["category"]
    assert "category" not in dropped_high_card_df.columns


def test_drop_identifier_columns_passes_through_without_identifiers() -> None:
    """The identifier helper keeps a frame without identifiers."""
    # Arrange
    frame = pd.DataFrame({"keep": [1]})

    # Act
    no_id_df, no_id_columns = feature_spec._drop_identifier_columns(frame)

    # Assert
    assert no_id_columns == []
    assert list(no_id_df.columns) == ["keep"]


def test_drop_constant_columns_passes_through_varying_columns() -> None:
    """The constant helper keeps a column whose values vary."""
    # Arrange
    frame = pd.DataFrame({"keep": [1, 2]})

    # Act
    no_constant_df, no_constant_columns = feature_spec._drop_constant_columns(frame)

    # Assert
    assert no_constant_columns == []
    assert list(no_constant_df.columns) == ["keep"]


def test_drop_high_cardinality_columns_passes_through_low_cardinality() -> None:
    """The high-cardinality helper keeps a column under the ratio."""
    # Arrange
    frame = pd.DataFrame({"category": ["x", "x"]})

    # Act
    no_high_card_df, no_high_card_columns = feature_spec._drop_high_cardinality_columns(
        frame,
        max_cardinality_ratio=0.75,
    )

    # Assert
    assert no_high_card_columns == []
    assert list(no_high_card_df.columns) == ["category"]


def test_build_feature_spec_handles_market_transform_pruning_and_drop_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature-spec construction should track market, pruning, ID, constant, and high-card drops."""
    # Arrange
    monkeypatch.setattr(log, "debug", returning(None))
    monkeypatch.setattr(constants, "RESULT_COLUMNS", ["home_score"])
    monkeypatch.setattr(constants, "PRUNED_FEATURE_COLUMNS", ["feat_pruned"])
    df = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "feat_pruned": [1.0, 2.0, 3.0],
            "feat_keep": [10.0, 11.0, 12.0],
            "game_id": ["g1", "g2", "g3"],
            "constant_flag": [1, 1, 1],
            "high_card": ["a", "b", "c"],
            "home_spread": [-3.5, -2.5, -1.5],
            "total_line": [45.5, 46.0, 46.5],
            "home_moneyline": [-150, -130, -110],
            "away_moneyline": [130, 120, 100],
            "home_score": [21, 24, 27],
        }
    )

    # Act
    spec = feature_spec.build_feature_spec(
        df,
        feature_spec.FeatureSelection(
            include_market=False,
            max_cardinality_ratio=0.5,
            feature_start="feat_pruned",
            feature_end="away_moneyline",
            market_transform=True,
        ),
    )

    # Assert
    assert spec.feature_columns == ["feat_keep"]
    assert spec.numeric_columns == ["feat_keep"]
    assert spec.categorical_columns == []
    assert spec.id_columns == ["game_id"]
    assert spec.constant_columns == ["constant_flag"]
    assert spec.high_cardinality_columns == ["high_card"]
    assert spec.market_columns == [
        "market_home_margin",
        "market_total_line",
        "home_market_prob",
        "away_market_prob",
    ]
    assert "feat_pruned" in spec.dropped_columns
    assert "home_score" in spec.dropped_columns
    assert "home_spread" in spec.dropped_columns
    assert "market_home_margin" in spec.dropped_columns


def _pruning_spec(
    monkeypatch: pytest.MonkeyPatch, *, disable_pruning: bool
) -> feature_spec.FeatureSpec:
    """Build the spec over three feature columns, one of them on the pruned list."""
    monkeypatch.setattr(log, "debug", returning(None))
    monkeypatch.setattr(constants, "PRUNED_FEATURE_COLUMNS", ["feat_pruned"])
    df = pd.DataFrame(
        {
            "meta": [1, 2],
            "feat_pruned": [3.0, 4.0],
            "feat_kept": [5.0, 7.0],
            "home_spread": [-3.0, -2.5],
        }
    )
    return feature_spec.build_feature_spec(
        df,
        feature_spec.FeatureSelection(
            include_market=True,
            max_cardinality_ratio=0.9,
            feature_start="feat_pruned",
            feature_end="home_spread",
            disable_pruning=disable_pruning,
        ),
    )


def test_build_feature_spec_drops_pruned_features_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pruned features are dropped by default."""
    # Act
    spec = _pruning_spec(monkeypatch, disable_pruning=False)

    # Assert
    assert spec.feature_columns == ["feat_kept", "home_spread"]


def test_build_feature_spec_disable_pruning_keeps_pruned_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pruned features are kept when pruning is disabled."""
    # Act
    spec = _pruning_spec(monkeypatch, disable_pruning=True)

    # Assert
    assert spec.feature_columns == [
        "feat_pruned",
        "feat_kept",
        "home_spread",
    ]


def test_apply_feature_spec_adds_market_transforms_and_reindexes_missing_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Applying a feature spec should derive market columns and fill missing columns with NaN."""
    # Arrange
    monkeypatch.setattr(log, "debug", returning(None))
    spec = feature_spec.FeatureSpec(
        feature_columns=["market_home_margin", "missing_feature"],
        categorical_columns=[],
        numeric_columns=["market_home_margin", "missing_feature"],
        dropped_columns=[],
        id_columns=[],
        constant_columns=[],
        high_cardinality_columns=[],
        feature_start="feat_a",
        feature_end="feat_b",
        metadata_columns=[],
        post_feature_columns=[],
        market_columns=["market_home_margin"],
    )
    df = pd.DataFrame({"away_spread": [2.5]})

    # Act
    applied = feature_spec.apply_feature_spec(df, spec)

    # Assert
    assert list(applied.columns) == ["market_home_margin", "missing_feature"]
    assert applied["market_home_margin"].iloc[0] == pytest.approx(2.5)
    assert np.isnan(applied["missing_feature"].iloc[0])


def _num_cat_spec() -> feature_spec.FeatureSpec:
    """Return a spec with one numeric and one categorical feature."""
    return feature_spec.FeatureSpec(
        feature_columns=["num_feature", "cat_feature"],
        categorical_columns=["cat_feature"],
        numeric_columns=["num_feature"],
        dropped_columns=[],
        id_columns=[],
        constant_columns=[],
        high_cardinality_columns=[],
        feature_start="num_feature",
        feature_end="cat_feature",
        metadata_columns=[],
        post_feature_columns=[],
        market_columns=[],
    )


def test_build_preprocessor_builds_a_sparse_tree_preprocessor() -> None:
    """A tree preprocessor is a column transformer that keeps sparse output."""
    # Arrange
    spec = _num_cat_spec()

    # Act
    tree_preprocessor = feature_spec.build_preprocessor(spec, for_tree=True)

    # Assert
    assert isinstance(tree_preprocessor, ColumnTransformer)
    assert tree_preprocessor.sparse_threshold == 1.0


def test_build_preprocessor_scales_numeric_features_for_dense_models() -> None:
    """A dense preprocessor ends its numeric pipeline with a scaler."""
    # Arrange
    spec = _num_cat_spec()

    # Act
    dense_preprocessor = feature_spec.build_preprocessor(spec, for_tree=False)

    # Assert
    numeric_pipeline = next(
        transformer for name, transformer, _cols in dense_preprocessor.transformers if name == "num"
    )
    assert isinstance(numeric_pipeline, Pipeline)
    assert numeric_pipeline.steps[-1][0] == "scaler"


def test_build_preprocessor_passes_the_legacy_sparse_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older OneHotEncoder gets the legacy ``sparse`` keyword."""
    # Arrange
    spec = _num_cat_spec()
    legacy_signature = SimpleNamespace(parameters={"sparse": object()})

    class _LegacyOneHot:
        """Test double that records the legacy sparse kwarg."""

        def __init__(self, **kwargs: object) -> None:
            """Store the kwargs passed by build_preprocessor."""
            self.kwargs = kwargs

    monkeypatch.setattr(inspect, "signature", returning(legacy_signature))
    monkeypatch.setattr(feature_spec, "OneHotEncoder", _LegacyOneHot)

    # Act
    legacy_preprocessor = feature_spec.build_preprocessor(spec, for_tree=False)

    # Assert
    legacy_categorical_pipeline = next(
        transformer
        for name, transformer, _cols in legacy_preprocessor.transformers
        if name == "cat"
    )
    assert isinstance(legacy_categorical_pipeline, Pipeline)
    legacy_encoder = legacy_categorical_pipeline.steps[-1][1]
    assert isinstance(legacy_encoder, _LegacyOneHot)
    assert legacy_encoder.kwargs["sparse"] is False


def test_build_preprocessor_rejects_an_empty_spec() -> None:
    """A spec without feature columns cannot build a preprocessor."""
    # Arrange
    empty_spec = feature_spec.FeatureSpec(
        feature_columns=[],
        categorical_columns=[],
        numeric_columns=[],
        dropped_columns=[],
        id_columns=[],
        constant_columns=[],
        high_cardinality_columns=[],
        feature_start="feat_a",
        feature_end="feat_b",
        metadata_columns=[],
        post_feature_columns=[],
        market_columns=[],
    )

    # Act & Assert
    with pytest.raises(ValueError, match="No feature columns available"):
        feature_spec.build_preprocessor(empty_spec)


def test_constant_columns_ignore_missing_values() -> None:
    """A column is constant when its non-missing values agree, and an all-missing one too."""
    # Arrange
    frame = pd.DataFrame(
        {
            "one_value_and_gaps": [np.nan, 1.0, 1.0],
            "all_missing": [np.nan, np.nan, np.nan],
            "one_string": ["a", "a", None],
            "int_and_float": [1, 1.0, 1],
            "varies": [1.0, np.nan, 2.0],
            "strings_vary": ["a", "b", "a"],
        }
    )

    # Act
    kept, constant_columns = feature_spec._drop_constant_columns(frame)

    # Assert
    assert constant_columns == ["one_value_and_gaps", "all_missing", "one_string", "int_and_float"]
    assert list(kept.columns) == ["varies", "strings_vary"]
