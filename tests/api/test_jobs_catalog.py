"""Tests for the job catalog: parameter validation and the command line each template builds."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest
from tests.api import factories
from tests.cli_parsing import parse_command

from nfl_predictor.api.errors import ConflictError, UnprocessableEntityError
from nfl_predictor.api.jobs import catalog
from nfl_predictor.api.jobs.catalog import JobContext
from nfl_predictor.api.runs.indexer import RunSummary, summarize_run
from nfl_predictor.weekly_run import config as run_config

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from nfl_predictor.api.settings import Settings


@pytest.fixture
def run(settings: Settings) -> RunSummary:
    """Return a summarized weekly run in the fake models directory."""
    run_dir = factories.make_run_dir(
        settings.models_path, "weekly_2026_week_02", factories.RunSpec(week=2)
    )
    summary = summarize_run(run_dir)
    assert summary is not None
    return summary


def context(
    settings: Settings, params: dict[str, object], run: RunSummary | None = None
) -> JobContext:
    """Build a job context for a template's build function."""
    return JobContext(settings=settings, job_id="job123", params=params, run=run)


def build(
    template_id: str, settings: Settings, params: dict[str, object], run: RunSummary | None = None
) -> list[str]:
    """Validate ``params`` and build the argv for ``template_id``."""
    template = catalog.get_template(template_id)
    coerced = catalog.validate_params(template, params)
    return template.build(context(settings, coerced, run))


def test_every_template_is_describable_and_documented() -> None:
    """Templates carry a label, a description, a category, and unique ids."""
    # Act
    described = [catalog.describe(template) for template in catalog.TEMPLATES]

    # Assert
    ids = [template.id for template in catalog.TEMPLATES]
    assert len(ids) == len(set(ids))
    assert [entry.id for entry in described] == ids
    for template in catalog.TEMPLATES:
        assert template.label
        assert template.description
        assert template.category
    for entry in described:
        for param in entry.params:
            assert param["label"]
            assert param["kind"]


def test_every_command_starts_with_the_configured_interpreter(
    settings: Settings, run: RunSummary
) -> None:
    """No template shells out to a bare ``python``; all use the venv interpreter."""
    # Arrange
    defaults: dict[str, object] = {"season": 2026, "week": 2, "through_week": 1}
    submissions = [
        (
            template.id,
            {spec.name: defaults[spec.name] for spec in template.params if spec.name in defaults},
        )
        for template in catalog.TEMPLATES
    ]

    # Act
    commands = [build(template_id, settings, params, run) for template_id, params in submissions]

    # Assert
    assert {argv[0] for argv in commands} == {str(settings.python_path)}


def test_chained_template_ids_exist() -> None:
    """A template that chains a follow-up names a template that exists."""
    # Act
    chained = {
        template.chain_template_id
        for template in catalog.TEMPLATES
        if template.chain_template_id is not None
    }

    # Assert
    assert chained <= set(catalog.TEMPLATES_BY_ID)


def test_get_template_rejects_an_unknown_id() -> None:
    """An unknown template id is a 422, not a 500."""
    # Act & Assert
    with pytest.raises(UnprocessableEntityError):
        catalog.get_template("does_not_exist")


def test_validate_params_fills_defaults_and_coerces(settings: Settings) -> None:
    """Strings from a form are coerced and declared defaults are filled in."""
    # Arrange
    template = catalog.get_template("predict")

    # Act
    params = catalog.validate_params(template, {"season": "2026", "week": "2"})

    # Assert
    assert params == {"season": 2026, "week": 2, "score_rounding": "nfl"}


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"season": 2026, "week": 1, "nope": 1}, "Unknown parameters"),
        ({"season": 2026}, "required"),
        ({"season": 2026, "week": 99}, "at most"),
        ({"season": 2026, "week": "two"}, "integer"),
        ({"season": 2026, "week": [2]}, "integer"),
        ({"season": 2026, "week": 2, "score_rounding": "sideways"}, "one of"),
    ],
)
def test_validate_params_rejects_unknown_missing_and_bad_values(
    params: dict[str, object], match: str
) -> None:
    """Unknown keys, missing required values, and out-of-range numbers are all 422."""
    # Arrange
    template = catalog.get_template("predict")

    # Act & Assert
    with pytest.raises(UnprocessableEntityError, match=match):
        catalog.validate_params(template, params)


@pytest.mark.parametrize(("submitted", "expected"), [("true", True), ("no", False)])
def test_validate_params_reads_booleans_from_form_strings(
    submitted: str, *, expected: bool
) -> None:
    """A checkbox submitted as a string still becomes a boolean."""
    # Arrange
    template = catalog.get_template("etl_full")

    # Act
    params = catalog.validate_params(template, {"refresh_nflreadpy": submitted})

    # Assert
    assert params == {"refresh_nflreadpy": expected}


def test_validate_params_rejects_a_string_that_is_not_a_boolean() -> None:
    """A checkbox value that reads as neither true nor false is a 422."""
    # Arrange
    template = catalog.get_template("etl_full")

    # Act & Assert
    with pytest.raises(UnprocessableEntityError, match="true or false"):
        catalog.validate_params(template, {"refresh_nflreadpy": "maybe"})


def test_etl_template_leaves_blank_fields_off_the_command(settings: Settings) -> None:
    """Blank fields are left off the command line entirely, but the data tree is always set."""
    # Act
    argv = build("etl_full", settings, {})

    # Assert
    assert argv == [
        str(settings.python_path),
        "-m",
        "nfl_predictor",
        "data",
        "--data-dir",
        str(settings.data_path),
        "--no-refresh-nflreadpy",
    ]


def test_etl_template_passes_the_options_given(settings: Settings) -> None:
    """Fields the form fills reach the command line as their options."""
    # Act
    argv = build("etl_full", settings, {"min_season": 2015, "refresh_nflreadpy": True})

    # Assert
    assert argv[-3:] == ["--min-season", "2015", "--refresh-nflreadpy"]


def test_lines_refresh_targets_the_configured_data_directory(settings: Settings) -> None:
    """The refresh runs the lines command against the configured data tree."""
    # Act
    argv = build("lines_refresh", settings, {"season": 2026, "week": 2})

    # Assert
    assert argv[1:4] == ["-m", "nfl_predictor", "lines"]
    assert argv[-2:] == ["--data-dir", str(settings.data_path)]


def test_lines_refresh_chains_a_predict_of_the_same_week() -> None:
    """The chained predict inherits the season and week, not the refresh-only parameters."""
    # Arrange
    template = catalog.get_template("lines_refresh")

    # Act
    chained = catalog.chained_params(template, {"season": 2026, "week": 2})

    # Assert
    assert template.chain_template_id == "predict"
    assert chained == {"season": 2026, "week": 2}


def test_templates_without_a_chain_carry_no_chained_params() -> None:
    """A template with no follow-up produces no parameters for one."""
    # Arrange
    template = catalog.get_template("predict")

    # Act
    chained = catalog.chained_params(template, {"season": 2026})

    # Assert
    assert chained == {}


def test_weekly_run_passes_options_rather_than_a_config_file(settings: Settings) -> None:
    """The job never writes a config file, which would replace the shipped one."""
    # Act
    argv = build("weekly_run", settings, {"week": 2, "run_id": "weekly_2026_week_02"})

    # Assert
    assert "--config" not in argv


def test_weekly_run_layers_the_form_over_the_shipped_config(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The job passes options, so the shipped config still supplies every default it omits."""
    # Arrange
    params: dict[str, object] = {"week": 2, "run_id": "weekly_2026_week_02"}

    # Act
    _name, args = parse_built(build("weekly_run", settings, params), settings, monkeypatch)

    # Assert
    assert args.config == run_config.DEFAULT_CONFIG_PATH
    assert args.data_collection_args == "--incremental"
    assert args.run_id == "weekly_2026_week_02"
    assert args.predict_path == settings.data_path / "predict" / "week_02_games_to_predict.csv"
    assert args.output_dir == settings.models_path
    assert args.data_path == settings.data_path / "completed_games_ml.csv"
    assert args.skip_data_refresh is True
    assert args.resume is False
    assert args.dry_run is False


def test_weekly_run_form_values_override_the_shipped_config(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A value the form sets wins over the shipped config, switches turned off included."""
    # Arrange
    shipped = tmp_path / "weekly_run.yaml"
    shipped.write_text(
        "skip_data_refresh: true\ndry_run: true\nresume: true\nwf_eval_last_n_seasons: 7\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_config, "DEFAULT_CONFIG_PATH", shipped)
    params: dict[str, object] = {
        "week": 2,
        "skip_data_refresh": False,
        "dry_run": False,
        "resume": False,
        "wf_eval_last_n_seasons": 4,
    }

    # Act
    _name, args = parse_built(build("weekly_run", settings, params), settings, monkeypatch)

    # Assert
    assert args.config == shipped
    assert args.skip_data_refresh is False
    assert args.dry_run is False
    assert args.resume is False
    assert args.wf_eval_last_n_seasons == 4


@pytest.mark.parametrize("value", ["--min-season 2010", "--incremental", "--no-incremental"])
def test_weekly_run_forwards_data_collection_arguments(
    value: str, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ETL pass-through string replaces the shipped one unchanged, even as a single flag."""
    # Arrange
    params: dict[str, object] = {"week": 2, "data_collection_args": value}

    # Act
    _name, args = parse_built(build("weekly_run", settings, params), settings, monkeypatch)

    # Assert
    assert args.data_collection_args == value


def test_predict_uses_the_active_run_model(settings: Settings, run: RunSummary) -> None:
    """Prediction reads the pinned run's model and writes back into its directory."""
    # Act
    argv = build("predict", settings, {"season": 2026, "week": 2}, run)

    # Assert
    assert "--model-in" in argv
    assert argv[argv.index("--model-in") + 1] == str(run.run_files.model)
    assert argv[argv.index("--data-path") + 1] == str(settings.data_path / "completed_games_ml.csv")
    assert argv[argv.index("--output-path") + 1] == str(
        run.run_dir / "season_2026_week_02_predictions.csv"
    )


@pytest.mark.parametrize("template_id", ["predict", "power_rankings", "shap_analysis"])
def test_templates_that_need_a_run_say_so_when_none_is_pinned(
    template_id: str, settings: Settings
) -> None:
    """Building without an active run is a conflict the API can explain."""
    # Arrange
    defaults: dict[str, object] = {"season": 2026, "week": 2, "through_week": 1}
    template = catalog.get_template(template_id)
    params = {spec.name: defaults[spec.name] for spec in template.params if spec.name in defaults}

    # Act & Assert
    with pytest.raises(ConflictError, match="active run"):
        build(template_id, settings, params)


def test_the_betting_workbook_job_is_retired() -> None:
    """The Excel betting workbook is no longer built, so its job template is gone."""
    # Act & Assert
    with pytest.raises(UnprocessableEntityError, match="betting_xlsx"):
        catalog.get_template("betting_xlsx")


@pytest.mark.parametrize(
    ("template_id", "extra"), [("validate_offline", []), ("validate_live", ["--live"])]
)
def test_read_only_templates_take_no_parameters(
    template_id: str, extra: list[str], settings: Settings
) -> None:
    """The validation jobs take no parameters but read the configured data tree."""
    # Act
    argv = build(template_id, settings, {})

    # Assert
    assert argv[1:] == [
        "-m",
        "nfl_predictor",
        "validate",
        *extra,
        "--data-dir",
        str(settings.data_path),
    ]


def test_leakage_audit_defaults_its_report_path(settings: Settings) -> None:
    """The audit writes into the reports directory unless a path is given."""
    # Act
    argv = build("leakage_audit", settings, {})

    # Assert
    assert f"--out-json={settings.reports_path / 'leakage_audit.json'}" in argv


def test_leakage_audit_writes_to_a_given_report_path(settings: Settings) -> None:
    """A report path the form sets replaces the default one."""
    # Arrange
    chosen = str(settings.state_path / "audit.json")

    # Act
    argv = build("leakage_audit", settings, {"out_json": chosen})

    # Assert
    assert f"--out-json={chosen}" in argv


def test_only_dataset_and_walk_forward_jobs_are_exclusive() -> None:
    """Long or dataset-mutating jobs are serialized; the read-only ones are not."""
    # Act
    groups = {template.id: template.exclusive_group for template in catalog.TEMPLATES}

    # Assert
    assert groups["weekly_run"] == catalog.WALK_FORWARD_GROUP
    assert groups["walk_forward_backtest"] == catalog.WALK_FORWARD_GROUP
    assert groups["train"] == catalog.WALK_FORWARD_GROUP
    assert groups["etl_full"] == catalog.DATASET_GROUP
    assert groups["lines_refresh"] == catalog.DATASET_GROUP
    assert groups["validate_offline"] is None


def test_predict_week_builds_the_missing_week(settings: Settings) -> None:
    """The future-week template extracts the week without overwriting an existing file."""
    # Act
    argv = build("predict_week", settings, {"season": 2026, "week": 5})

    # Assert
    assert argv[1:4] == ["-m", "nfl_predictor", "build-week"]
    assert argv[argv.index("--data-dir") + 1] == str(settings.data_path)
    assert "--overwrite" not in argv


def test_predict_week_chains_a_predict_of_the_same_week() -> None:
    """The chained predict inherits the season and week, not the overwrite switch."""
    # Arrange
    template = catalog.get_template("predict_week")

    # Act
    chained = catalog.chained_params(template, {"season": 2026, "week": 5, "overwrite": True})

    # Assert
    assert template.chain_template_id == "predict"
    assert chained == {"season": 2026, "week": 5}


def test_predict_week_can_rebuild_an_existing_file(settings: Settings) -> None:
    """The overwrite switch is passed through as a flag."""
    # Act
    argv = build("predict_week", settings, {"season": 2026, "week": 5, "overwrite": True})

    # Assert
    assert argv[-1] == "--overwrite"


SAMPLE_PARAMS: dict[str, object] = {"season": 2026, "week": 2, "through_week": 1}


def parse_built(
    argv: list[str], settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, argparse.Namespace]:
    """Parse a built command line with the real parser of the command it launches.

    Every job runs ``<python> -m nfl_predictor <command> ...``; anything else fails here. The
    weekly job's config file is read and its keys validated, as a real run does.
    """
    assert argv[:3] == [str(settings.python_path), "-m", "nfl_predictor"], argv
    return argv[3], parse_command(argv[3], argv[4:], monkeypatch)


def _template_cases() -> list[tuple[str, dict[str, object]]]:
    """Return one submission per template, and one per value of every choice parameter."""
    cases: list[tuple[str, dict[str, object]]] = []
    for template in catalog.TEMPLATES:
        base = {spec.name: SAMPLE_PARAMS[spec.name] for spec in template.params if spec.required}
        cases.append((template.id, base))
        for spec in template.params:
            cases.extend((template.id, {**base, spec.name: choice}) for choice in spec.choices)
    return cases


@pytest.mark.parametrize(("template_id", "params"), _template_cases())
def test_every_template_builds_a_command_its_target_parses(
    template_id: str,
    params: dict[str, object],
    settings: Settings,
    run: RunSummary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each job launches the front door, and the command's own parser accepts every option."""
    # Arrange
    argv = build(template_id, settings, params, run)

    # Act & Assert
    parse_built(argv, settings, monkeypatch)


def _text_param_cases() -> list[tuple[str, str]]:
    """Return ``(template id, parameter)`` for every free-text parameter in the catalog."""
    return [
        (template.id, spec.name)
        for template in catalog.TEMPLATES
        for spec in template.params
        if spec.kind == "str"
    ]


@pytest.mark.parametrize(("template_id", "name"), _text_param_cases())
def test_a_text_value_that_starts_with_a_dash_reaches_the_command_unchanged(
    template_id: str,
    name: str,
    settings: Settings,
    run: RunSummary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A free-text value such as ``-x`` is passed as that option's value, never as an option."""
    # Arrange
    template = catalog.get_template(template_id)
    params = {spec.name: SAMPLE_PARAMS[spec.name] for spec in template.params if spec.required}
    argv = build(template_id, settings, {**params, name: "-x"}, run)

    # Act
    _name, args = parse_built(argv, settings, monkeypatch)

    # Assert
    assert str(getattr(args, name)) == "-x"


def test_etl_arguments_with_an_unclosed_quote_are_rejected_at_submit() -> None:
    """A string the data refresh could not split fails validation, not the queued job."""
    # Arrange
    template = catalog.get_template("weekly_run")

    # Act & Assert
    with pytest.raises(UnprocessableEntityError, match="ETL arguments") as raised:
        catalog.validate_params(template, {"week": 2, "data_collection_args": "--min-season '20"})
    assert raised.value.code == "invalid_param"


def test_the_train_form_offers_no_model_kind_choice() -> None:
    """``margin_total`` is the only model kind, so the train form does not ask for one."""
    # Act
    names = {spec.name for spec in catalog.get_template("train").params}

    # Assert
    assert "model_kind" not in names


@pytest.mark.parametrize("template_id", ["predict", "power_rankings"])
def test_run_based_jobs_pass_the_recorded_margin_total_kind(
    template_id: str,
    settings: Settings,
    run: RunSummary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run recorded as ``margin_total`` launches with that kind."""
    # Arrange
    pinned = dataclasses.replace(run, model_kind="margin_total")
    template = catalog.get_template(template_id)
    params = {spec.name: SAMPLE_PARAMS[spec.name] for spec in template.params if spec.required}

    # Act
    _name, args = parse_built(build(template_id, settings, params, pinned), settings, monkeypatch)

    # Assert
    assert args.model_kind == "margin_total"


@pytest.mark.parametrize("template_id", ["predict", "power_rankings"])
@pytest.mark.parametrize("recorded", ["blend", "blended_margin_total"])
def test_run_based_jobs_refuse_a_retired_blend_run(
    template_id: str, recorded: str, settings: Settings, run: RunSummary
) -> None:
    """A run holding a retired blend model cannot launch a job that loads it."""
    # Arrange
    pinned = dataclasses.replace(run, model_kind=recorded)
    template = catalog.get_template(template_id)
    params = {spec.name: SAMPLE_PARAMS[spec.name] for spec in template.params if spec.required}

    # Act & Assert
    with pytest.raises(ConflictError, match="blend model kind was retired"):
        build(template_id, settings, params, pinned)
