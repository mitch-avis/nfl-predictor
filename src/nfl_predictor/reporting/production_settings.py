"""Settings versus production: where a walk-forward run differs from the weekly run.

The benchmark is meant to measure what production does, so every walk-forward report lists
each model-affecting setting where the run's resolved configuration differs from the production
weekly run's, for both halves of that run: stage 1's walk-forward and the final fit the week's
picks come from. The production side is read through the weekly run's own parser, config file
(``config/weekly_run.yaml`` unless ``--config`` names another) and helpers, so it follows any
change to them.

Each half gets four lists. ``differences`` holds the settings whose values differ. A setting
one side applies implicitly carries that value on the side with no option for it (production
never drops pruning, feature groups or trend features; a walk-forward fold holds out no season,
never tunes and weights postseason games like any other), so a real difference lands in
``differences``. What is left on one side only goes to ``run_only`` or ``production_only``
(the tuning options when production tunes, for example), and ``not_recorded`` names the
production settings an older run did not record (its resolved XGBoost parameters). A half
reads "no model-affecting differences" only when all four are empty and the run recorded no
retired setting. Settings an older run recorded that no longer exist are listed as
``retired``, with their values. Two groups are listed apart and are never differences:
``scope`` (which seasons and weeks are scored, the seed, the files a run reads) and
``runtime`` (thread count, log verbosity and the quantile models, none of which changes a
scored prediction). Market settings are compared after resolving them against the dataset's
columns, and the XGBoost parameters are built on both sides by the resolver training uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from nfl_predictor.ml import ml_model_core, ml_model_xgb_utils, walk_forward
from nfl_predictor.weekly_run import config as weekly_config
from nfl_predictor.weekly_run import stage1

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SECTION_KEY = "settings_versus_production"
HEADING = "Settings versus production"
NO_DIFFERENCES = "no model-affecting differences"
# The config key under which a backtest records the XGBoost parameters its folds trained with.
RESOLVED_XGB_PARAMS_KEY = "xgb_params"

_STAGES = {"stage1": "stage 1 walk-forward", "final_fit": "final fit"}
_XGB_PREFIX = "xgb."
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
# Keys a backtest records in its metadata config besides the walk-forward configuration. The
# backtest writes them inline (and the walk-forward adds its resolved settings), so the list
# is pinned by a test that runs a real backtest and requires exactly these keys back.
RECORDED_CONFIG_EXTRAS = frozenset(
    {
        "checkpoint",
        "data_path",
        "disable_trend_features",
        "dropped_feature_group_columns",
        "dropped_trend_columns",
        "eval_window",
        "excluded_incomplete_seasons",
        "feature_list",
        "floor_sigma",
        "in_season_early_stopping",
        "out_json",
        "resolved_eval_seasons",
        "run_id",
        "splits",
        "xgb_device",
        RESOLVED_XGB_PARAMS_KEY,
    }
)
_TUNING_SETTINGS = {
    "tune_objective": "objective",
    "tune_trials": "n_trials",
    "tune_timeout": "timeout_seconds",
    "tune_cv_splits": "cv_splits",
    "tune_early_stopping_rounds": "early_stopping_rounds",
}


def _xgb_settings(params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split XGBoost parameters into model settings and runtime settings, leaving the seed."""
    model: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    for name, value in sorted(params.items()):
        if name == _XGB_SEED_PARAM:
            continue
        target = runtime if name in _XGB_RUNTIME_PARAMS else model
        target[f"{_XGB_PREFIX}{name}"] = value
    return model, runtime


def _market_settings(
    frame: pd.DataFrame, *, include_market: bool, market_transform: bool | None, market_anchor: bool
) -> dict[str, Any]:
    """Return the market settings a fit on ``frame`` resolves to."""
    include, transform, anchor = walk_forward.resolve_market_settings(
        frame,
        include_market=include_market,
        market_transform=market_transform,
        market_anchor=market_anchor,
    )
    return {"include_market": include, "market_transform": transform, "market_anchor": anchor}


def _walk_forward_settings(
    config: walk_forward.WalkForwardConfig,
    frame: pd.DataFrame,
    xgb_params: Mapping[str, Any],
    *,
    disable_trend_features: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return a walk-forward configuration's model, scope and runtime settings."""
    xgb_model, xgb_runtime = _xgb_settings(xgb_params)
    model = {
        "calibration": config.calibration,
        "include_postseason": bool(config.include_postseason),
        "recency_half_life_seasons": config.recency_half_life_seasons,
        **_market_settings(
            frame,
            include_market=config.include_market,
            market_transform=config.market_transform,
            market_anchor=config.market_anchor,
        ),
        "max_cardinality_ratio": config.max_cardinality_ratio,
        "feature_start": config.feature_start,
        "feature_end": config.feature_end,
        "disable_pruning": bool(config.disable_pruning),
        "disabled_feature_groups": list(config.disabled_feature_groups),
        "disable_trend_features": bool(disable_trend_features),
        **xgb_model,
    }
    scope = {name: getattr(config, name) for name in _WALK_FORWARD_SCOPE}
    scope["eval_seasons"] = list(config.eval_seasons) if config.eval_seasons else None
    scope["random_seed"] = xgb_params.get(_XGB_SEED_PARAM, config.random_seed)
    runtime = {"include_quantiles": bool(config.include_quantiles), **xgb_runtime}
    return {"model": model, "scope": scope, "runtime": runtime}


def _as_final_fit(model: Mapping[str, Any]) -> dict[str, Any]:
    """Return walk-forward model settings with the final fit's options at their fold values.

    A fold holds out no season, never tunes, and weights a postseason game like any other.
    """
    return {
        **model,
        "holdout_seasons": 0,
        "tune": False,
        "postseason_weight": 1.0 if model["include_postseason"] else None,
    }


def _final_fit_settings(
    train_config: Mapping[str, Any], optuna_config: ml_model_core.OptunaConfig
) -> dict[str, dict[str, Any]]:
    """Return the weekly final fit's model, scope and runtime settings."""
    overrides = dict(train_config["xgb_params_overrides"])
    if optuna_config.xgb_n_jobs is not None:
        overrides["n_jobs"] = optuna_config.xgb_n_jobs
    params = ml_model_xgb_utils.resolve_xgb_params(
        ml_model_core.DEFAULT_XGB_PARAMS,
        overrides=overrides,
        tree_method=optuna_config.tree_method,
        device=optuna_config.device,
    )
    xgb_model, xgb_runtime = _xgb_settings(params)
    include_postseason = bool(train_config["include_postseason"])
    model: dict[str, Any] = {
        name: train_config[name]
        for name in (
            "calibration",
            "recency_half_life_seasons",
            "include_market",
            "market_transform",
            "market_anchor",
            "max_cardinality_ratio",
            "feature_start",
            "feature_end",
            "holdout_seasons",
        )
    }
    model.update(
        {
            "include_postseason": include_postseason,
            # The weight applies only when postseason games are in training.
            "postseason_weight": train_config["postseason_weight"] if include_postseason else None,
            # The final fit has no option to drop pruning, feature groups or trend features.
            "disable_pruning": False,
            "disabled_feature_groups": [],
            "disable_trend_features": False,
            "tune": bool(optuna_config.enabled),
        }
    )
    if optuna_config.enabled:
        for name, attribute in _TUNING_SETTINGS.items():
            model[name] = getattr(optuna_config, attribute)
    model.update(xgb_model)
    return {
        "model": model,
        "scope": {"random_seed": params.get(_XGB_SEED_PARAM)},
        "runtime": xgb_runtime,
    }


def _compare(
    run: Mapping[str, Any], production: Mapping[str, Any], *, xgb_recorded: bool
) -> dict[str, Any]:
    """Return the settings that differ, those on one side only, and those not recorded."""
    shared = sorted(set(run) & set(production))
    production_only = sorted(set(production) - set(run))
    not_recorded = [
        name for name in production_only if not xgb_recorded and name.startswith(_XGB_PREFIX)
    ]
    return {
        "differences": [
            {"setting": name, "run": run[name], "production": production[name]}
            for name in shared
            if run[name] != production[name]
        ],
        "run_only": {name: run[name] for name in sorted(set(run) - set(production))},
        "production_only": {
            name: production[name] for name in production_only if name not in not_recorded
        },
        "not_recorded": not_recorded,
    }


def _production_settings(
    production_argv: Sequence[str], frame: pd.DataFrame
) -> tuple[Any, walk_forward.WalkForwardConfig, ml_model_core.OptunaConfig, dict[str, Any]]:
    """Return production's options, stage-1 configuration and final-fit settings.

    Raises:
        OSError, RuntimeError, ValueError: When the weekly configuration does not load.
        SystemExit: When the weekly parser rejects an option combination.

    """
    args = weekly_config.parse_args(production_argv)
    weekly_config.apply_run_defaults(args)
    production_stage1 = stage1.production_walk_forward_config(weekly_config.stage1_options(args))
    optuna_config, train_config = weekly_config.final_fit_options(args, frame, run_dir=None)
    return args, production_stage1, optuna_config, train_config


@dataclass(frozen=True, kw_only=True)
class RunRecord:
    """What a walk-forward run recorded about itself beside its configuration.

    Attributes:
        disable_trend_features: Whether the run dropped the trend features before training.
        xgb_params: The XGBoost parameters the run trained with; by default they are
            resolved from the run's configuration as its folds resolve them.
        xgb_recorded: ``False`` when ``xgb_params`` holds only what an older run recorded
            (its overrides), so production's other parameters are "not recorded".
        retired: Settings the run recorded that no longer exist, with their values.
        scope: Scope settings the run records outside its configuration (paths).

    """

    disable_trend_features: bool = False
    xgb_params: Mapping[str, Any] | None = None
    xgb_recorded: bool = True
    retired: Mapping[str, Any] | None = None
    scope: Mapping[str, Any] | None = None


def settings_versus_production(
    run_config: walk_forward.WalkForwardConfig,
    frame: pd.DataFrame,
    record: RunRecord | None = None,
    *,
    production_argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the settings-versus-production section for a walk-forward run.

    Args:
        run_config: The run's configuration, with its XGBoost device already resolved.
        frame: The run's dataset, or any frame with its columns; market settings resolve
            against it on both sides.
        record: What the run recorded beside its configuration; none means nothing beyond
            the defaults of `RunRecord`.
        production_argv: Weekly-run options that stand in for production's command line; none
            reads the shipped config file with the code defaults.

    Returns:
        The section, or ``{"unavailable": reason}`` when the weekly configuration does not
        load, so a finished run still writes its report.

    """
    try:
        args, production_stage1, optuna_config, train_config = _production_settings(
            production_argv, frame
        )
    except (OSError, RuntimeError, ValueError) as error:
        return {"unavailable": f"the production weekly configuration does not load: {error}"}
    except SystemExit:
        # The weekly parser exits on option combinations it rejects (the ranking options).
        return {"unavailable": "the production weekly configuration does not parse"}

    record = record or RunRecord()
    run_xgb_params = record.xgb_params
    if run_xgb_params is None:
        run_xgb_params = walk_forward.xgb_params_for_config(run_config)
    run = _walk_forward_settings(
        run_config, frame, run_xgb_params, disable_trend_features=record.disable_trend_features
    )
    run["scope"].update(record.scope or {})
    stage1_settings = _walk_forward_settings(
        production_stage1, frame, walk_forward.xgb_params_for_config(production_stage1)
    )
    final_fit = _final_fit_settings(train_config, optuna_config)
    return {
        "production_config": str(args.config) if args.config is not None else None,
        "stage1": _compare(
            run["model"], stage1_settings["model"], xgb_recorded=record.xgb_recorded
        ),
        "final_fit": _compare(
            _as_final_fit(run["model"]), final_fit["model"], xgb_recorded=record.xgb_recorded
        ),
        "retired": dict(record.retired or {}),
        "scope": {
            "run": run["scope"],
            "stage1": stage1_settings["scope"],
            "final_fit": final_fit["scope"],
            "production": {
                "data_path": str(args.data_path),
                "data_collection_args": args.data_collection_args,
            },
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


def _retired_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the recorded keys that are neither configuration fields nor known records."""
    names = {item.name for item in fields(walk_forward.WalkForwardConfig)}
    return {key: config[key] for key in sorted(config) if key not in names | RECORDED_CONFIG_EXTRAS}


def _recorded_xgb_params(config: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return the XGBoost parameters a run recorded, and whether they are complete.

    A run that recorded its resolved parameters has them all. An older run recorded only its
    overrides and, perhaps, its device; nothing else is inferred from today's defaults.
    """
    resolved = config.get(RESOLVED_XGB_PARAMS_KEY)
    if resolved:
        return dict(resolved), True
    params = dict(config.get("xgb_params_overrides") or {})
    if config.get("xgb_device"):
        params["device"] = config["xgb_device"]
    params.setdefault(_XGB_SEED_PARAM, config.get("random_seed", walk_forward.DEFAULT_RANDOM_SEED))
    return params, False


def section_for_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the section for a finished run from its ``metadata.json`` payload.

    A run that recorded no metadata, whose dataset is no longer at its recorded path, or whose
    recorded configuration no longer loads (for example a calibration method that was
    retired) gets ``{"unavailable": reason}``. Other retired keys the run recorded are listed
    under ``retired`` with their values.
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
    xgb_params, xgb_recorded = _recorded_xgb_params(config)
    checkpoint = config.get("checkpoint") or {}
    record = RunRecord(
        disable_trend_features=bool(config.get("disable_trend_features")),
        xgb_params=xgb_params,
        xgb_recorded=xgb_recorded,
        retired=_retired_settings(config),
        scope={"data_path": str(data_path), "checkpoint_dir": checkpoint.get("dir")},
    )
    return settings_versus_production(run_config, pd.read_csv(data_path, nrows=0), record)


def _value(value: object) -> str:
    """Return a setting's value as JSON, so ``None`` and strings read unambiguously."""
    return json.dumps(value, default=str)


def _pairs(settings: Mapping[str, Any]) -> str:
    """Return ``name value`` pairs, comma-separated."""
    return ", ".join(f"{name} {_value(value)}" for name, value in settings.items())


def _headline(stage: Mapping[str, Any], retired: Mapping[str, Any]) -> str:
    """Return a half's summary: its differences, or whether anything is left uncompared."""
    if stage["differences"]:
        return "; ".join(
            f"{row['setting']}: run {_value(row['run'])}, production {_value(row['production'])}"
            for row in stage["differences"]
        )
    uncompared = (
        len(stage["run_only"])
        + len(stage["production_only"])
        + len(stage["not_recorded"])
        + len(retired)
    )
    if not uncompared:
        return NO_DIFFERENCES
    return f"no compared setting differs; {uncompared} model setting(s) could not be compared"


def format_section(section: Mapping[str, Any]) -> list[str]:
    """Return the section as log lines: a heading, then one line per list."""
    if "unavailable" in section:
        return [f"{HEADING}: unavailable ({section['unavailable']})"]
    source = section.get("production_config") or "the code defaults"
    retired = section.get("retired") or {}
    lines = [f"{HEADING} (the weekly run with {source}):"]
    for key, label in _STAGES.items():
        stage = section[key]
        lines.append(f"- {label}: {_headline(stage, retired)}")
        if stage["run_only"]:
            lines.append(f"- {label}, only in this run: {_pairs(stage['run_only'])}")
        if stage["production_only"]:
            lines.append(f"- {label}, only in production: {_pairs(stage['production_only'])}")
        if stage["not_recorded"]:
            lines.append(f"- {label}, not recorded by this run: {', '.join(stage['not_recorded'])}")
    if retired:
        lines.append(f"- retired settings recorded by this run: {_pairs(retired)}")
    for group, label in (
        ("scope", "scope, not a difference"),
        ("runtime", "runtime, changes no prediction"),
    ):
        parts = [
            f"{'this run' if side == 'run' else _STAGES.get(side, side)}: {_pairs(values)}"
            for side, values in section[group].items()
            if values
        ]
        lines.append(f"- {label}: " + "; ".join(parts))
    return lines
