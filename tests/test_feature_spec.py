"""Tests for feature specification helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from nfl_predictor.ml import feature_spec


def test_get_feature_range_columns_supports_success_and_error_paths() -> None:
    """Feature range resolution should validate presence and ordering."""
    df = pd.DataFrame(columns=["meta", "feat_a", "feat_b", "feat_c", "tail"])

    feature_range, metadata_columns, post_feature_columns = feature_spec._get_feature_range_columns(
        df,
        "feat_a",
        "feat_c",
    )

    assert feature_range == ["feat_a", "feat_b", "feat_c"]
    assert metadata_columns == ["meta"]
    assert post_feature_columns == ["tail"]

    with pytest.raises(ValueError, match="Expected feature range columns"):
        feature_spec._get_feature_range_columns(df, "missing", "feat_c")

    with pytest.raises(ValueError, match="occurs after"):
        feature_spec._get_feature_range_columns(df, "feat_c", "feat_a")


def test_implied_prob_from_moneyline_accepts_series_and_arrays() -> None:
    """Moneyline conversion should work for Series and ndarray inputs."""
    series_probs = feature_spec._implied_prob_from_moneyline(pd.Series([-150, 200, 0]))
    array_probs = feature_spec._implied_prob_from_moneyline(np.array([-120, 150]))

    assert series_probs[0] == pytest.approx(150 / 250)
    assert series_probs[1] == pytest.approx(100 / 300)
    assert np.isnan(series_probs[2])
    assert array_probs[0] == pytest.approx(120 / 220)
    assert array_probs[1] == pytest.approx(100 / 250)


def test_add_market_transforms_derives_missing_columns_without_overwriting_existing() -> None:
    """Market transforms should derive missing helper columns from spread, total, and moneyline."""
    home_spread_df = pd.DataFrame(
        {
            "home_spread": [-3.5],
            "total_line": [46.5],
            "home_moneyline": [-150],
            "away_moneyline": [130],
        }
    )
    transformed_home = feature_spec._add_market_transforms(home_spread_df)

    assert transformed_home["market_home_margin"].iloc[0] == pytest.approx(3.5)
    assert transformed_home["market_total_line"].iloc[0] == pytest.approx(46.5)
    assert transformed_home["home_market_prob"].iloc[0] == pytest.approx(150 / 250)
    assert transformed_home["away_market_prob"].iloc[0] == pytest.approx(100 / 230)

    away_spread_df = pd.DataFrame(
        {
            "away_spread": [2.5],
            "market_home_margin": [9.0],
        }
    )
    transformed_away = feature_spec._add_market_transforms(away_spread_df)

    assert transformed_away["market_home_margin"].iloc[0] == pytest.approx(9.0)


def test_get_market_baseline_validates_columns_and_missing_values() -> None:
    """Market baseline extraction should reject missing or NaN-derived market values."""
    with pytest.raises(ValueError, match="spread/total columns are missing"):
        feature_spec.get_market_baseline(pd.DataFrame({"feat": [1]}))

    with pytest.raises(ValueError, match="contains missing values"):
        feature_spec.get_market_baseline(
            pd.DataFrame(
                {
                    "home_spread": [np.nan],
                    "total_line": [44.5],
                }
            )
        )

    baseline_margin, baseline_total = feature_spec.get_market_baseline(
        pd.DataFrame(
            {
                "home_spread": [-3.0],
                "total_line": [45.5],
            }
        )
    )

    assert baseline_margin.tolist() == [3.0]
    assert baseline_total.tolist() == [45.5]


def test_drop_helpers_cover_drop_and_noop_paths() -> None:
    """Identifier, constant, and high-cardinality helpers should support drop and passthrough."""
    id_constant_df = pd.DataFrame(
        {
            "team_id": ["a", "b"],
            "constant_flag": [1, 1],
            "keep": [10, 11],
        }
    )
    high_card_df = pd.DataFrame(
        {
            "category": ["x", "y"],
            "keep": [10, 11],
        }
    )

    dropped_id_df, id_columns = feature_spec._drop_identifier_columns(id_constant_df)
    assert id_columns == ["team_id"]
    assert "team_id" not in dropped_id_df.columns

    dropped_constant_df, constant_columns = feature_spec._drop_constant_columns(id_constant_df)
    assert constant_columns == ["constant_flag"]
    assert "constant_flag" not in dropped_constant_df.columns

    dropped_high_card_df, high_card_columns = feature_spec._drop_high_cardinality_columns(
        high_card_df,
        max_cardinality_ratio=0.5,
    )
    assert high_card_columns == ["category"]
    assert "category" not in dropped_high_card_df.columns

    no_id_df, no_id_columns = feature_spec._drop_identifier_columns(pd.DataFrame({"keep": [1]}))
    assert no_id_columns == []
    assert list(no_id_df.columns) == ["keep"]

    no_constant_df, no_constant_columns = feature_spec._drop_constant_columns(
        pd.DataFrame({"keep": [1, 2]})
    )
    assert no_constant_columns == []
    assert list(no_constant_df.columns) == ["keep"]

    no_high_card_df, no_high_card_columns = feature_spec._drop_high_cardinality_columns(
        pd.DataFrame({"category": ["x", "x"]}),
        max_cardinality_ratio=0.75,
    )
    assert no_high_card_columns == []
    assert list(no_high_card_df.columns) == ["category"]


def test_build_feature_spec_handles_market_transform_pruning_and_drop_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature-spec construction should track market, pruning, ID, constant, and high-card drops."""
    monkeypatch.setattr(feature_spec.log, "debug", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(feature_spec.constants, "RESULT_COLUMNS", ["home_score"])
    monkeypatch.setattr(feature_spec.constants, "PRUNED_FEATURE_COLUMNS", ["feat_pruned"])

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

    spec = feature_spec._build_feature_spec(
        df,
        include_market=False,
        max_cardinality_ratio=0.5,
        feature_start="feat_pruned",
        feature_end="away_moneyline",
        market_transform=True,
    )

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


def test_build_feature_spec_supports_market_only_and_disable_pruning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Market-only selection should keep market columns.

    Disable-pruning should preserve candidate pruned features.
    """
    monkeypatch.setattr(feature_spec.log, "debug", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(feature_spec.constants, "PRUNED_FEATURE_COLUMNS", ["feat_pruned"])

    df = pd.DataFrame(
        {
            "meta": [1, 2],
            "feat_pruned": [3.0, 4.0],
            "home_spread": [-3.0, -2.5],
            "away_moneyline": [130, 120],
        }
    )

    with pytest.raises(ValueError, match="Market-only model requested"):
        feature_spec._build_feature_spec(
            pd.DataFrame({"meta": [1], "feat": [2.0]}),
            include_market=True,
            max_cardinality_ratio=0.9,
            feature_start="feat",
            feature_end="feat",
            market_only=True,
        )

    market_only_spec = feature_spec._build_feature_spec(
        df,
        include_market=True,
        max_cardinality_ratio=0.9,
        feature_start="feat_pruned",
        feature_end="away_moneyline",
        market_only=True,
        disable_pruning=True,
    )

    assert market_only_spec.feature_columns == ["home_spread", "away_moneyline"]
    assert market_only_spec.market_columns == ["home_spread", "away_moneyline"]


def test_apply_feature_spec_adds_market_transforms_and_reindexes_missing_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Applying a feature spec should derive market columns and fill missing columns with NaN."""
    monkeypatch.setattr(feature_spec.log, "debug", lambda *_args, **_kwargs: None)
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

    applied = feature_spec._apply_feature_spec(df, spec)

    assert list(applied.columns) == ["market_home_margin", "missing_feature"]
    assert applied["market_home_margin"].iloc[0] == pytest.approx(2.5)
    assert np.isnan(applied["missing_feature"].iloc[0])


def test_build_preprocessor_supports_tree_dense_and_empty_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preprocessor construction should support tree, dense, legacy, and empty-spec paths."""
    spec = feature_spec.FeatureSpec(
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

    tree_preprocessor = feature_spec._build_preprocessor(spec, for_tree=True)
    assert isinstance(tree_preprocessor, ColumnTransformer)
    assert tree_preprocessor.sparse_threshold == 1.0

    dense_preprocessor = feature_spec._build_preprocessor(spec, for_tree=False)
    numeric_pipeline = next(
        transformer for name, transformer, _cols in dense_preprocessor.transformers if name == "num"
    )
    assert numeric_pipeline.steps[-1][0] == "scaler"

    legacy_signature = SimpleNamespace(parameters={"sparse": object()})

    class _LegacyOneHot:
        """Test double that records the legacy sparse kwarg."""

        def __init__(self, **kwargs: object) -> None:
            """Store the kwargs passed by _build_preprocessor."""
            self.kwargs = kwargs

    monkeypatch.setattr(feature_spec.inspect, "signature", lambda _obj: legacy_signature)
    monkeypatch.setattr(feature_spec, "OneHotEncoder", _LegacyOneHot)
    legacy_preprocessor = feature_spec._build_preprocessor(spec, for_tree=False)
    legacy_categorical_pipeline = next(
        transformer
        for name, transformer, _cols in legacy_preprocessor.transformers
        if name == "cat"
    )
    assert legacy_categorical_pipeline.steps[-1][1].kwargs["sparse"] is False

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
    with pytest.raises(ValueError, match="No feature columns available"):
        feature_spec._build_preprocessor(empty_spec)
