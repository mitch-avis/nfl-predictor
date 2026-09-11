"""The catalog of jobs the UI can launch, and the argv each one builds.

Every project entrypoint is a CLI that reads ``sys.argv`` and configures logging globally, so jobs
always run as subprocesses of the venv interpreter rather than in-process. A template declares its
parameters once; the API validates submissions against them and the frontend renders the form from
the same schema.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from nfl_predictor.api.errors import ConflictError, UnprocessableEntityError
from nfl_predictor.api.runs.indexer import RunSummary
from nfl_predictor.api.settings import Settings

ParamKind = Literal["int", "float", "str", "bool", "choice"]
WALK_FORWARD_GROUP = "walk_forward"
DATASET_GROUP = "datasets"
DEFAULT_MODEL_KIND = "margin_total"
MODEL_KINDS = ("margin_total", "blended_margin_total", "score")


@dataclass(frozen=True)
class ParamSpec:
    """One parameter of a job template, and everything the form needs to render it.

    Attributes:
        name: Parameter name as submitted and as used by the build function.
        label: Human label for the form field.
        kind: Value type used for coercion and for the input widget.
        description: Help text shown under the field.
        required: Whether a submission must provide it.
        default: Value used when the submission omits it.
        choices: Allowed values when ``kind`` is ``choice``.
        minimum: Inclusive lower bound for numeric kinds.
        maximum: Inclusive upper bound for numeric kinds.

    """

    name: str
    label: str
    kind: ParamKind
    description: str = ""
    required: bool = False
    default: Any = None
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class JobContext:
    """Everything a build function may consult when assembling its command line."""

    settings: Settings
    job_id: str
    params: Mapping[str, Any]
    run: RunSummary | None = None

    @property
    def python(self) -> str:
        """Return the interpreter every job is launched with."""
        return str(self.settings.python_path)

    @property
    def root(self) -> Path:
        """Return the repository root jobs run in."""
        return self.settings.root_dir

    def script(self, name: str) -> str:
        """Return the absolute path of ``scripts/<name>``."""
        return str(self.root / "scripts" / name)

    def data_file(self, name: str) -> str:
        """Return the absolute path of a file in the data directory."""
        return str(self.settings.data_path / name)

    def require_run(self) -> RunSummary:
        """Return the active run, or explain that one is needed."""
        if self.run is None:
            raise ConflictError(
                "This job needs an active run with a model; pin one on the Runs page.",
                code="no_active_run",
            )
        return self.run

    def config_path(self) -> Path:
        """Return the per-job config file path, creating its directory."""
        directory = self.settings.state_path / "job_configs"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{self.job_id}.json"


@dataclass(frozen=True)
class JobTemplate:
    """A launchable job: its parameters, its command line, and how it is scheduled.

    Attributes:
        id: Stable template identifier used by the API and the frontend.
        label: Display name.
        description: One-line explanation shown on the catalog card.
        category: Grouping for the catalog page.
        params: Parameters the form renders and the API validates.
        build: Callable turning a :class:`JobContext` into an argv list.
        exclusive_group: Jobs sharing a group never run at the same time.
        chain_template_id: Template launched automatically after this one succeeds.
        writes_datasets: Whether the job rewrites files under ``data/``.
        needs_active_run: Whether the job reads the active run's model.

    """

    id: str
    label: str
    description: str
    category: str
    build: Callable[[JobContext], list[str]]
    params: tuple[ParamSpec, ...] = ()
    exclusive_group: str | None = None
    chain_template_id: str | None = None
    writes_datasets: bool = False
    needs_active_run: bool = False


def _flag(argv: list[str], flag: str, value: Any) -> None:
    """Append ``flag value`` when ``value`` is set, or the boolean form of the flag."""
    if value is None:
        return
    if isinstance(value, bool):
        argv.append(flag if value else f"--no-{flag[2:]}")
        return
    argv.extend([flag, str(value)])


def _week_predict_path(ctx: JobContext, week: int) -> str:
    """Return the games-to-predict file for ``week``."""
    return str(ctx.settings.data_path / "predict" / f"week_{week:02d}_games_to_predict.csv")


def _timestamp() -> str:
    """Return a UTC timestamp usable in a run id."""
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _build_etl_full(ctx: JobContext) -> list[str]:
    """Build the full ETL rebuild command."""
    argv = [ctx.python, "-m", "nfl_predictor.data_collection"]
    _flag(argv, "--min-season", ctx.params.get("min_season"))
    _flag(argv, "--max-season", ctx.params.get("max_season"))
    _flag(argv, "--refresh-nflreadpy", ctx.params.get("refresh_nflreadpy"))
    return argv


def _build_lines_refresh(ctx: JobContext) -> list[str]:
    """Build the lines-only refresh command."""
    return [
        ctx.python,
        "-m",
        "nfl_predictor.lines_refresh",
        "--season",
        str(ctx.params["season"]),
        "--week",
        str(ctx.params["week"]),
        "--data-dir",
        str(ctx.settings.data_path),
    ]


def _build_weekly_run(ctx: JobContext) -> list[str]:
    """Write the weekly-run config file and build the command that reads it."""
    config = {key: value for key, value in ctx.params.items() if value is not None}
    config.setdefault("output_dir", str(ctx.settings.models_path))
    config.setdefault("data_path", ctx.data_file("completed_games_ml.csv"))
    week = config.pop("week", None)
    if week is not None:
        config["predict_path"] = _week_predict_path(ctx, int(week))
    path = ctx.config_path()
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return [ctx.python, ctx.script("weekly_run.py"), "--config", str(path)]


def _build_train(ctx: JobContext) -> list[str]:
    """Build the training command, writing into a fresh run directory."""
    run_id = f"train_{_timestamp()}"
    argv = [
        ctx.python,
        "-m",
        "nfl_predictor.ml_model",
        "--model-kind",
        str(ctx.params.get("model_kind", DEFAULT_MODEL_KIND)),
        "--data-path",
        ctx.data_file("completed_games_ml.csv"),
        "--run-dir",
        str(ctx.settings.models_path / run_id),
        "--run-id",
        run_id,
    ]
    _flag(argv, "--holdout-seasons", ctx.params.get("holdout_seasons"))
    _flag(argv, "--win-prob-calibration", ctx.params.get("win_prob_calibration"))
    return argv


def _build_predict(ctx: JobContext) -> list[str]:
    """Build the prediction command against the active run's model."""
    run = ctx.require_run()
    season = int(ctx.params["season"])
    week = int(ctx.params["week"])
    output = run.run_dir / f"season_{season}_week_{week:02d}_predictions.csv"
    return [
        ctx.python,
        "-m",
        "nfl_predictor.ml_model",
        "--model-in",
        str(run.run_files.model),
        "--model-kind",
        run.model_kind or DEFAULT_MODEL_KIND,
        # The CLI hashes the training dataset even when predicting, so it always needs a path
        # inside the configured data directory rather than its own default.
        "--data-path",
        ctx.data_file("completed_games_ml.csv"),
        "--predict-path",
        _week_predict_path(ctx, week),
        "--output-path",
        str(output),
        "--score-rounding",
        str(ctx.params.get("score_rounding", "nfl")),
    ]


def _build_predict_week(ctx: JobContext) -> list[str]:
    """Build the command that extracts a future week's prediction inputs."""
    argv = [
        ctx.python,
        "-m",
        "nfl_predictor.week_builder",
        "--season",
        str(ctx.params["season"]),
        "--week",
        str(ctx.params["week"]),
        "--data-dir",
        str(ctx.settings.data_path),
    ]
    if ctx.params.get("overwrite"):
        argv.append("--overwrite")
    return argv


def _build_power_rankings(ctx: JobContext) -> list[str]:
    """Build the power-rankings command writing into the active run directory."""
    run = ctx.require_run()
    return [
        ctx.python,
        ctx.script("power_rankings.py"),
        "--model-in",
        str(run.run_files.model),
        "--model-kind",
        run.model_kind or DEFAULT_MODEL_KIND,
        "--season",
        str(ctx.params["season"]),
        "--through-week",
        str(ctx.params["through_week"]),
        "--data-ml",
        ctx.data_file("all_data_ml.csv"),
        "--data-schedule",
        ctx.data_file("all_data.csv"),
        "--out-dir",
        str(run.run_dir),
    ]


def _build_betting_xlsx(ctx: JobContext) -> list[str]:
    """Build the betting-workbook command for the active run's predictions."""
    run = ctx.require_run()
    if run.run_files.predictions is None:
        raise ConflictError(
            f"Run {run.run_id!r} has no predictions to build a workbook from.",
            code="no_predictions",
        )
    return [
        ctx.python,
        ctx.script("betting_report_excel.py"),
        "--predictions",
        str(run.run_files.predictions),
        "--out",
        str(run.run_dir / "betting_report.xlsx"),
    ]


def _build_leakage_audit(ctx: JobContext) -> list[str]:
    """Build the leakage-audit command."""
    out_json = ctx.params.get("out_json") or str(ctx.settings.reports_path / "leakage_audit.json")
    return [
        ctx.python,
        ctx.script("leakage_audit.py"),
        "--data-path",
        ctx.data_file("completed_games_ml.csv"),
        "--out-json",
        str(out_json),
    ]


def _build_validate_offline(ctx: JobContext) -> list[str]:
    """Build the offline validation command."""
    return [ctx.python, ctx.script("validate_offline.py")]


def _build_validate_live(ctx: JobContext) -> list[str]:
    """Build the live validation command."""
    return [ctx.python, ctx.script("validate_live.py")]


def _build_walk_forward(ctx: JobContext) -> list[str]:
    """Build the standalone walk-forward backtest command."""
    argv = [
        ctx.python,
        ctx.script("walk_forward_backtest.py"),
        "--data-path",
        ctx.data_file("completed_games_ml.csv"),
    ]
    _flag(argv, "--eval-last-n-seasons", ctx.params.get("eval_last_n_seasons"))
    _flag(argv, "--wf-start-week", ctx.params.get("wf_start_week"))
    _flag(argv, "--out-json", ctx.params.get("out_json"))
    return argv


def _build_shap_analysis(ctx: JobContext) -> list[str]:
    """Build the SHAP analysis command for the active run's model."""
    run = ctx.require_run()
    argv = [
        ctx.python,
        ctx.script("shap_analysis.py"),
        "--model-path",
        str(run.run_files.model),
        "--data-path",
        ctx.data_file("completed_games_ml.csv"),
        "--output-path",
        str(run.run_dir / "shap_report.json"),
    ]
    _flag(argv, "--sample-size", ctx.params.get("sample_size"))
    return argv


SEASON_PARAM = ParamSpec(
    name="season",
    label="Season",
    kind="int",
    description="Season year, for example 2026.",
    required=True,
    minimum=1999,
    maximum=2100,
)
WEEK_PARAM = ParamSpec(
    name="week",
    label="Week",
    kind="int",
    description="Week number; week 1-18 in the regular season, 19-22 in the playoffs.",
    required=True,
    minimum=1,
    maximum=22,
)

TEMPLATES: tuple[JobTemplate, ...] = (
    JobTemplate(
        id="etl_full",
        label="Full ETL rebuild",
        description="Rebuild every dataset from nflverse sources. Takes several minutes.",
        category="Data",
        build=_build_etl_full,
        params=(
            ParamSpec("min_season", "First season", "int", "Leave blank for the project default."),
            ParamSpec("max_season", "Last season", "int", "Leave blank for the current season."),
            ParamSpec(
                "refresh_nflreadpy",
                "Refresh nflverse cache",
                "bool",
                "Re-download nflreadpy data instead of using the cached parquet files.",
                default=False,
            ),
        ),
        exclusive_group=DATASET_GROUP,
        writes_datasets=True,
    ),
    JobTemplate(
        id="lines_refresh",
        label="Refresh market lines",
        description=(
            "Re-read the schedule and update only the spread, total, and moneyline columns, "
            "then re-predict the week with the active model."
        ),
        category="Data",
        build=_build_lines_refresh,
        params=(SEASON_PARAM, WEEK_PARAM),
        exclusive_group=DATASET_GROUP,
        chain_template_id="predict",
        writes_datasets=True,
    ),
    JobTemplate(
        id="weekly_run",
        label="Weekly run",
        description="Compare, train, predict, and report in one orchestrated run.",
        category="Pipeline",
        build=_build_weekly_run,
        params=(
            ParamSpec(
                "week", "Week", "int", "Week to predict.", required=True, minimum=1, maximum=22
            ),
            ParamSpec("run_id", "Run id", "str", "Defaults to a timestamped id."),
            ParamSpec(
                "resume",
                "Resume",
                "bool",
                "Reuse completed stages of an existing run.",
                default=False,
            ),
            ParamSpec(
                "dry_run",
                "Dry run",
                "bool",
                "Log the plan without running any stage.",
                default=False,
            ),
            ParamSpec(
                "skip_data_refresh",
                "Skip ETL",
                "bool",
                "Do not refresh datasets first.",
                default=True,
            ),
            ParamSpec(
                "wf_eval_last_n_seasons",
                "Walk-forward seasons",
                "int",
                "Seasons in the comparison window.",
            ),
        ),
        exclusive_group=WALK_FORWARD_GROUP,
    ),
    JobTemplate(
        id="train",
        label="Train a model",
        description="Train a new model into its own run directory.",
        category="Model",
        build=_build_train,
        params=(
            ParamSpec(
                "model_kind",
                "Model kind",
                "choice",
                "Head configuration to train.",
                default=DEFAULT_MODEL_KIND,
                choices=MODEL_KINDS,
            ),
            ParamSpec(
                "holdout_seasons",
                "Holdout seasons",
                "int",
                "Most recent seasons held out of training.",
                minimum=0,
                maximum=10,
            ),
            ParamSpec(
                "win_prob_calibration",
                "Calibration",
                "choice",
                "Win-probability calibration method.",
                choices=("platt", "isotonic", "none"),
            ),
        ),
        exclusive_group=WALK_FORWARD_GROUP,
    ),
    JobTemplate(
        id="predict",
        label="Predict a week",
        description="Predict one week with the active run's model.",
        category="Model",
        build=_build_predict,
        params=(
            SEASON_PARAM,
            WEEK_PARAM,
            ParamSpec(
                "score_rounding",
                "Score rounding",
                "choice",
                "Post-processing applied to predicted scores.",
                default="nfl",
                choices=("none", "int", "half", "nfl"),
            ),
        ),
        needs_active_run=True,
    ),
    JobTemplate(
        id="predict_week",
        label="Predict a future week",
        description=(
            "Extract a future week's games from the ML dataset and predict them with the active "
            "model. The week keeps the lines, rest, and quarterbacks of the last ETL run."
        ),
        category="Model",
        build=_build_predict_week,
        params=(
            SEASON_PARAM,
            WEEK_PARAM,
            ParamSpec(
                "overwrite",
                "Rebuild the week file",
                "bool",
                "Replace an existing games-to-predict file instead of reusing it.",
                default=False,
            ),
        ),
        chain_template_id="predict",
    ),
    JobTemplate(
        id="power_rankings",
        label="Power rankings",
        description="Rate every team and project the standings into the active run.",
        category="Model",
        build=_build_power_rankings,
        params=(
            SEASON_PARAM,
            ParamSpec(
                "through_week",
                "Through week",
                "int",
                "Records are computed through this week.",
                required=True,
                minimum=0,
                maximum=22,
            ),
        ),
        needs_active_run=True,
    ),
    JobTemplate(
        id="betting_xlsx",
        label="Betting workbook",
        description="Build the Excel betting workbook from the active run's predictions.",
        category="Reports",
        build=_build_betting_xlsx,
        needs_active_run=True,
    ),
    JobTemplate(
        id="leakage_audit",
        label="Leakage audit",
        description="Scan the training dataset for features that leak the outcome.",
        category="Validation",
        build=_build_leakage_audit,
        params=(
            ParamSpec("out_json", "Report path", "str", "Defaults to reports/leakage_audit.json."),
        ),
    ),
    JobTemplate(
        id="validate_offline",
        label="Offline validation",
        description="Run the offline dataset and artifact checks. Reads only.",
        category="Validation",
        build=_build_validate_offline,
    ),
    JobTemplate(
        id="validate_live",
        label="Live validation",
        description="Check the live nflverse sources against the local datasets.",
        category="Validation",
        build=_build_validate_live,
    ),
    JobTemplate(
        id="walk_forward_backtest",
        label="Walk-forward backtest",
        description="Backtest the model week by week. Long-running.",
        category="Validation",
        build=_build_walk_forward,
        params=(
            ParamSpec(
                "eval_last_n_seasons",
                "Seasons",
                "int",
                "How many recent seasons to evaluate.",
                minimum=1,
                maximum=10,
            ),
            ParamSpec(
                "wf_start_week",
                "Start week",
                "int",
                "First week evaluated in each season.",
                minimum=1,
                maximum=18,
            ),
            ParamSpec("out_json", "Report path", "str", "Optional JSON report path."),
        ),
        exclusive_group=WALK_FORWARD_GROUP,
    ),
    JobTemplate(
        id="shap_analysis",
        label="SHAP analysis",
        description="Explain the active run's model with SHAP values.",
        category="Model",
        build=_build_shap_analysis,
        params=(
            ParamSpec(
                "sample_size",
                "Sample size",
                "int",
                "Rows sampled for the explanation.",
                default=500,
                minimum=50,
                maximum=20000,
            ),
        ),
        needs_active_run=True,
    ),
)

TEMPLATES_BY_ID: dict[str, JobTemplate] = {template.id: template for template in TEMPLATES}


def get_template(template_id: str) -> JobTemplate:
    """Return the template with ``template_id`` or raise 422."""
    template = TEMPLATES_BY_ID.get(template_id)
    if template is None:
        raise UnprocessableEntityError(f"Unknown job template {template_id!r}", code="unknown_job")
    return template


def _coerce_bool(spec: ParamSpec, value: Any) -> bool:
    """Coerce ``value`` to a bool, accepting the strings a form may send."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false", "1", "0", "yes", "no"}:
        return value.lower() in {"true", "1", "yes"}
    raise UnprocessableEntityError(f"{spec.label} must be true or false", code="invalid_param")


def _coerce_number(spec: ParamSpec, value: Any) -> int | float:
    """Coerce ``value`` to the spec's numeric type and range-check it."""
    try:
        number = int(value) if spec.kind == "int" else float(value)
    except (TypeError, ValueError) as exc:
        raise UnprocessableEntityError(
            f"{spec.label} must be a{'n integer' if spec.kind == 'int' else ' number'}",
            code="invalid_param",
        ) from exc
    if spec.minimum is not None and number < spec.minimum:
        raise UnprocessableEntityError(
            f"{spec.label} must be at least {spec.minimum:g}", code="invalid_param"
        )
    if spec.maximum is not None and number > spec.maximum:
        raise UnprocessableEntityError(
            f"{spec.label} must be at most {spec.maximum:g}", code="invalid_param"
        )
    return number


def _coerce(spec: ParamSpec, value: Any) -> Any:
    """Coerce one submitted value to the type its spec declares."""
    if spec.kind == "bool":
        return _coerce_bool(spec, value)
    if spec.kind in {"int", "float"}:
        return _coerce_number(spec, value)
    text = str(value)
    if spec.kind == "choice" and text not in spec.choices:
        raise UnprocessableEntityError(
            f"{spec.label} must be one of: {', '.join(spec.choices)}", code="invalid_param"
        )
    return text


def validate_params(template: JobTemplate, raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate a submission against ``template`` and return the coerced parameters.

    Args:
        template: Template the submission targets.
        raw: Submitted parameters; ``None`` is treated as an empty mapping.

    Returns:
        The coerced parameters, with defaults filled in.

    Raises:
        UnprocessableEntityError: If a key is unknown, a required value is missing, or a value
            does not fit its spec.

    """
    submitted = dict(raw or {})
    known = {spec.name for spec in template.params}
    unknown = sorted(set(submitted) - known)
    if unknown:
        raise UnprocessableEntityError(
            f"Unknown parameters for {template.id}: {', '.join(unknown)}", code="unknown_param"
        )
    coerced: dict[str, Any] = {}
    for spec in template.params:
        value = submitted.get(spec.name)
        if value is None or value == "":
            if spec.required:
                raise UnprocessableEntityError(f"{spec.label} is required", code="missing_param")
            if spec.default is not None:
                coerced[spec.name] = spec.default
            continue
        coerced[spec.name] = _coerce(spec, value)
    return coerced


def chained_params(template: JobTemplate, params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the parameters to launch ``template.chain_template_id`` with.

    Only parameters the chained template also declares are carried over, so a refresh of week 2
    predicts week 2 without the caller repeating itself.
    """
    if template.chain_template_id is None:
        return {}
    chained = get_template(template.chain_template_id)
    names = {spec.name for spec in chained.params}
    return {key: value for key, value in params.items() if key in names}


@dataclass(frozen=True)
class TemplateView:
    """A template rendered for the catalog endpoint."""

    id: str
    label: str
    description: str
    category: str
    exclusive_group: str | None
    chain_template_id: str | None
    writes_datasets: bool
    needs_active_run: bool
    params: list[dict[str, Any]] = field(default_factory=list)


def describe(template: JobTemplate) -> TemplateView:
    """Return the JSON-friendly description of ``template``."""
    return TemplateView(
        id=template.id,
        label=template.label,
        description=template.description,
        category=template.category,
        exclusive_group=template.exclusive_group,
        chain_template_id=template.chain_template_id,
        writes_datasets=template.writes_datasets,
        needs_active_run=template.needs_active_run,
        params=[
            {
                "name": spec.name,
                "label": spec.label,
                "kind": spec.kind,
                "description": spec.description,
                "required": spec.required,
                "default": spec.default,
                "choices": list(spec.choices),
                "minimum": spec.minimum,
                "maximum": spec.maximum,
            }
            for spec in template.params
        ],
    )
