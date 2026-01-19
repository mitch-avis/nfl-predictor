"""Optional SHAP analysis for trained models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

from nfl_predictor.ml import artifacts, feature_importance, ml_model_core
from nfl_predictor.utils.logger import log


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional SHAP analysis for model features.")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a saved model.joblib artifact.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="CSV dataset for feature extraction (same schema used for training).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output path for the SHAP report (JSON).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Max rows to sample for SHAP (default: 500).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for row sampling.",
    )
    parser.add_argument(
        "--target",
        choices=["margin", "total", "away", "home"],
        default="margin",
        help="Target model head to analyze (margin/total or away/home).",
    )
    parser.add_argument(
        "--component",
        choices=["team", "market"],
        default="team",
        help="For blended models, select the team or market component.",
    )
    return parser.parse_args()


def _import_shap() -> Optional[Any]:
    """Return the shap module when available; otherwise None."""
    try:
        import shap  # type: ignore
    except ImportError:
        return None
    return shap


def _select_model_component(
    model: Any,
    *,
    component: str,
) -> tuple[Any, str]:
    """Resolve the base model component to analyze."""
    if isinstance(model, ml_model_core.BlendedMarginTotalModel):
        if component == "market" and model.market_model is not None:
            return model.market_model, "blend_market"
        return model.team_model, "blend_team"
    if isinstance(model, ml_model_core.MarginTotalModel):
        return model, "margin_total"
    if isinstance(model, ml_model_core.ScoreModel):
        return model, "score"
    raise ValueError(f"Unsupported model type: {type(model).__name__}")


def _resolve_head(model: Any, target: str) -> tuple[Any, str]:
    """Resolve the specific model head to analyze."""
    if isinstance(model, ml_model_core.MarginTotalModel):
        if target == "margin":
            return model.margin_model, "margin"
        if target == "total":
            return model.total_model, "total"
        raise ValueError("Margin/total models support --target margin|total.")
    if isinstance(model, ml_model_core.ScoreModel):
        if target == "away":
            return model.away_model, "away"
        if target == "home":
            return model.home_model, "home"
        raise ValueError("Score models support --target away|home.")
    raise ValueError("Unsupported model type for SHAP target resolution.")


def main() -> int:
    """Run optional SHAP analysis; requires the `shap` library."""

    args = _parse_args()

    shap = _import_shap()
    if shap is None:
        log.warning("SHAP is not installed. Install with: .venv/bin/pip install shap (optional).")
        return 2

    if not args.model_path.exists():
        raise FileNotFoundError(f"Missing model: {args.model_path}")
    if not args.data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data_path}")

    model = joblib.load(args.model_path)
    base_model, model_kind = _select_model_component(model, component=args.component)
    head_model, target = _resolve_head(base_model, args.target)

    df = pd.read_csv(args.data_path)
    feature_df = ml_model_core._apply_feature_spec(df, base_model.feature_spec)
    x_matrix = base_model.preprocessor.transform(feature_df)

    sample_size = max(int(args.sample_size), 1)
    if x_matrix.shape[0] > sample_size:
        rng = np.random.default_rng(int(args.random_seed))
        indices = rng.choice(x_matrix.shape[0], size=sample_size, replace=False)
        x_matrix = x_matrix[indices]

    if isinstance(x_matrix, spmatrix):
        to_dense = getattr(x_matrix, "toarray", None)
        if to_dense is None:
            to_dense = getattr(x_matrix, "todense", None)
        if callable(to_dense):
            x_matrix = np.asarray(to_dense())
        else:
            raise TypeError("Unsupported sparse matrix type for SHAP conversion.")

    feature_names = feature_importance.resolve_feature_names(
        base_model.preprocessor,
        head_model,
    )

    explainer = shap.TreeExplainer(head_model)
    shap_values = explainer.shap_values(x_matrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    mean_abs = np.abs(shap_values).mean(axis=0)

    if len(feature_names) != len(mean_abs):
        log.debug(
            "Feature name count mismatch for SHAP (names=%d, shap=%d); using f0..",
            len(feature_names),
            len(mean_abs),
        )
        feature_names = [f"f{i}" for i in range(len(mean_abs))]

    rows = [
        {"feature": name, "mean_abs_shap": float(value)}
        for name, value in zip(feature_names, mean_abs, strict=False)
    ]
    rows.sort(key=lambda row: row["mean_abs_shap"], reverse=True)

    output_path = (
        args.output_path
        if args.output_path is not None
        else args.model_path.parent / "shap_report.json"
    )
    payload = {
        "model_kind": model_kind,
        "component": args.component,
        "target": target,
        "sample_size": int(x_matrix.shape[0]),
        "rows": rows,
    }
    artifacts.write_json(output_path, payload)
    log.info("Wrote SHAP report to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
