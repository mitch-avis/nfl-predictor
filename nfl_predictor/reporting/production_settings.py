"""Settings versus production: where a walk-forward run differs from the weekly run.

The benchmark is meant to measure what production does, so every walk-forward report lists
each model-affecting setting where the run's resolved configuration differs from the production
weekly run's, for both halves of that run: stage 1's walk-forward and the final fit the week's
picks come from. The production side is read through the weekly run's own parser, config file
(``config/weekly_run.yaml`` unless ``--config`` names another) and helpers, so it follows any
change to them.

Each half gets three lists: ``differences`` (a setting both sides have, with different
values), ``run_only`` and ``production_only`` (a setting only one side has; the final fit's
holdout and tuning options have no walk-forward counterpart, for example). Two groups are
listed apart and are never differences: ``scope`` (which seasons and weeks are scored, the
seed, the files a run reads and writes) and ``runtime`` (thread count, log verbosity and the
quantile models, none of which changes a scored prediction). Market settings are compared
after resolving them against the dataset's columns, as both sides do before training, and the
XGBoost parameters are compared as trained: the shared defaults with each side's overrides.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_predictor.ml import ml_model_core, walk_forward
from nfl_predictor.weekly_run import config as weekly_config
from nfl_predictor.weekly_run import stage1

SECTION_KEY = "settings_versus_production"
HEADING = "Settings versus production"

_STAGES = {"stage1": "stage 1 walk-forward", "final_fit": "final fit"}
# XGBoost parameters that change no scored prediction, and the one that is the run's seed.
_XGB_RUNTIME_PARAMS = ("n_jobs", "verbosity")
_XGB_SEED_PARAM = "random_state"
# Walk-forward settings that choose what is scored, not how the model is trained.
_WALK_FORWARD_SCOPE = (
    "eval_seasons",
    "eval_last_n_seasons",
    "wf_start_week",
    "exclude_incomplete_seasons",
)


def _xgb_settings(params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split trained XGBoost parameters into model settings and runtime settings."""
    model: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    for name, value in sorted(params.items()):
        if name == _XGB_SEED_PARAM:
            continue
        target = runtime if name in _XGB_RUNTIME_PARAMS else model
        target[f"xgb.{name}"] = value
    return model, runtime


def _market_settings(
    frame: pd.DataFrame, include_market: bool, market_transform: bool | None, market_anchor: bool
) -> dict[str, Any]:
    """Return the market settings a fit on ``frame`` resolves to."""
    include, transform, anchor = walk_forward.resolve_market_settings(
        frame, include_market, market_transform, market_anchor
    )
    return {"include_market": include, "market_transform": transform, "market_anchor": anchor}


def _walk_forward_settings(
    config: walk_forward.WalkForwardConfig, frame: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Return a walk-forward configuration's model, scope and runtime settings."""
    # The run's seed is XGBoost's unless the overrides name one, as the walk-forward trains.
    overrides = {_XGB_SEED_PARAM: config.random_seed, **(config.xgb_params_overrides or {})}
    params = {**ml_model_core.DEFAULT_XGB_PARAMS, **overrides}
    xgb_model, xgb_runtime = _xgb_settings(params)
    model = {
        "calibration": config.calibration,
        "include_postseason": bool(config.include_postseason),
        "recency_half_life_seasons": config.recency_half_life_seasons,
        **_market_settings(
            frame, config.include_market, config.market_transform, config.market_anchor
        ),
        "max_cardinality_ratio": config.max_cardinality_ratio,
        "feature_start": config.feature_start,
        "feature_end": config.feature_end,
        "disable_pruning": bool(config.disable_pruning),
        "disabled_feature_groups": list(config.disabled_feature_groups),
        **xgb_model,
    }
    scope = {name: getattr(config, name) for name in _WALK_FORWARD_SCOPE}
    scope["eval_seasons"] = list(config.eval_seasons) if config.eval_seasons else None
    scope["random_seed"] = params[_XGB_SEED_PARAM]
    runtime = {"include_quantiles": bool(config.include_quantiles), **xgb_runtime}
    return {"model": model, "scope": scope, "runtime": runtime}


def _final_fit_settings(
    train_config: Mapping[str, Any], optuna_config: ml_model_core.OptunaConfig
) -> dict[str, dict[str, Any]]:
    """Return the weekly final fit's model, scope and runtime settings."""
    overrides: dict[str, Any] = {"device": optuna_config.device}
    if optuna_config.tree_method and optuna_config.tree_method != "auto":
        overrides["tree_method"] = optuna_config.tree_method
    if optuna_config.xgb_n_jobs is not None:
        overrides["n_jobs"] = optuna_config.xgb_n_jobs
    params = {**ml_model_core.DEFAULT_XGB_PARAMS, **overrides}
    xgb_model, xgb_runtime = _xgb_settings(params)
    model: dict[str, Any] = {
        name: train_config[name]
        for name in (
            "calibration",
            "include_postseason",
            "recency_half_life_seasons",
            "include_market",
            "market_transform",
            "market_anchor",
            "max_cardinality_ratio",
            "feature_start",
            "feature_end",
            "holdout_seasons",
            "postseason_weight",
        )
    }
    model["tune"] = bool(optuna_config.enabled)
    if optuna_config.enabled:
        model["tune_objective"] = optuna_config.objective
    model.update(xgb_model)
    return {
        "model": model,
        "scope": {"random_seed": params.get(_XGB_SEED_PARAM)},
        "runtime": xgb_runtime,
    }


def _compare(run: Mapping[str, Any], production: Mapping[str, Any]) -> dict[str, Any]:
    """Return the settings that differ, and those on one side only."""
    shared = sorted(set(run) & set(production))
    return {
        "differences": [
            {"setting": name, "run": run[name], "production": production[name]}
            for name in shared
            if run[name] != production[name]
        ],
        "run_only": {name: run[name] for name in sorted(set(run) - set(production))},
        "production_only": {name: production[name] for name in sorted(set(production) - set(run))},
    }


def settings_versus_production(
    run_config: walk_forward.WalkForwardConfig,
    frame: pd.DataFrame,
    *,
    production_argv: Sequence[str] = (),
    run_extras: Mapping[str, Any] | None = None,
    run_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the settings-versus-production section for a walk-forward run.

    Args:
        run_config: The run's configuration, with its XGBoost device already resolved.
        frame: The run's dataset, or any frame with its columns; market settings resolve
            against it on both sides.
        production_argv: Weekly-run options that stand in for production's command line; none
            reads the shipped config file with the code defaults.
        run_extras: Model-affecting settings the run applies outside its configuration (a
            column ablation, for example); production never applies them.
        run_scope: Scope settings the run records outside its configuration (paths).

    Returns:
        The section, or ``{"unavailable": reason}`` when the weekly configuration does not
        load, so a finished run still writes its report.

    """
    try:
        args = weekly_config._parse_args(list(production_argv))
    except (OSError, RuntimeError, ValueError) as error:
        return {"unavailable": f"the production weekly configuration does not load: {error}"}
    except SystemExit:
        # The weekly parser exits on option combinations it rejects (the ranking options).
        return {"unavailable": "the production weekly configuration does not parse"}
    weekly_config.apply_run_defaults(args)
    production_stage1 = stage1.production_walk_forward_config(**weekly_config.stage1_options(args))
    optuna_config, train_config = weekly_config.final_fit_options(args, frame, run_dir=None)

    run = _walk_forward_settings(run_config, frame)
    run["model"].update(run_extras or {})
    run["scope"].update(run_scope or {})
    stage1_settings = _walk_forward_settings(production_stage1, frame)
    final_fit = _final_fit_settings(train_config, optuna_config)
    return {
        "production_config": str(args.config) if args.config is not None else None,
        "stage1": _compare(run["model"], stage1_settings["model"]),
        "final_fit": _compare(run["model"], final_fit["model"]),
        "scope": {
            "run": run["scope"],
            "stage1": stage1_settings["scope"],
            "final_fit": final_fit["scope"],
        },
        "runtime": {
            "run": run["runtime"],
            "stage1": stage1_settings["runtime"],
            "final_fit": final_fit["runtime"],
        },
    }


def _recorded_config(config: Mapping[str, Any]) -> walk_forward.WalkForwardConfig:
    """Rebuild a run's walk-forward configuration from its recorded metadata config.

    The recorded market settings are the resolved ones, which resolve to themselves again.
    """
    names = {item.name for item in fields(walk_forward.WalkForwardConfig)}
    values = {name: config[name] for name in names if name in config}
    if values.get("disabled_feature_groups") is not None:
        values["disabled_feature_groups"] = tuple(values["disabled_feature_groups"])
    return walk_forward.WalkForwardConfig(**values)


def section_for_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the section for a finished run from its ``metadata.json`` payload.

    A run that recorded no metadata, whose dataset is no longer at its recorded path, or whose
    recorded configuration no longer loads (a retired setting) gets ``{"unavailable": reason}``.
    """
    config = (metadata or {}).get("config")
    if not config:
        return {"unavailable": "the run recorded no metadata config"}
    data_path = Path(str(config.get("data_path") or ""))
    if not config.get("data_path") or not data_path.is_file():
        return {"unavailable": f"the run's dataset is not at its recorded path ({data_path})"}
    try:
        run_config = _recorded_config(config)
    except (TypeError, ValueError) as error:
        return {"unavailable": f"the recorded configuration no longer loads: {error}"}
    checkpoint = config.get("checkpoint") or {}
    return settings_versus_production(
        run_config,
        pd.read_csv(data_path, nrows=0),
        run_extras={"disable_trend_features": True}
        if config.get("disable_trend_features")
        else None,
        run_scope={"data_path": str(data_path), "checkpoint_dir": checkpoint.get("dir")},
    )


def _value(value: Any) -> str:
    """Return a setting's value as JSON, so ``None`` and strings read unambiguously."""
    return json.dumps(value, default=str)


def _pairs(settings: Mapping[str, Any]) -> str:
    """Return ``name value`` pairs, comma-separated."""
    return ", ".join(f"{name} {_value(value)}" for name, value in settings.items())


def format_section(section: Mapping[str, Any]) -> list[str]:
    """Return the section as log lines: a heading, then one line per list."""
    if "unavailable" in section:
        return [f"{HEADING}: unavailable ({section['unavailable']})"]
    source = section.get("production_config") or "the code defaults"
    lines = [f"{HEADING} (the weekly run with {source}):"]
    for key, label in _STAGES.items():
        stage = section[key]
        differences = stage["differences"]
        if differences:
            details = "; ".join(
                f"{row['setting']}: run {_value(row['run'])}, "
                f"production {_value(row['production'])}"
                for row in differences
            )
        else:
            details = "no model-affecting differences"
        lines.append(f"- {label}: {details}")
        if stage["run_only"]:
            lines.append(f"- {label}, only in this run: {_pairs(stage['run_only'])}")
        if stage["production_only"]:
            lines.append(f"- {label}, only in production: {_pairs(stage['production_only'])}")
    for group, label in (
        ("scope", "scope, not a difference"),
        ("runtime", "runtime, changes no prediction"),
    ):
        sides = section[group]
        parts = [
            f"{'this run' if side == 'run' else _STAGES[side]}: {_pairs(values)}"
            for side, values in sides.items()
            if values
        ]
        lines.append(f"- {label}: " + "; ".join(parts))
    return lines
