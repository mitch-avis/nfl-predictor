"""Tests for the job catalog: parameter validation and the command line each template builds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nfl_predictor.api.errors import ConflictError, UnprocessableEntityError
from nfl_predictor.api.jobs import catalog
from nfl_predictor.api.jobs.catalog import JobContext
from nfl_predictor.api.runs.indexer import RunSummary, summarize_run
from nfl_predictor.api.settings import Settings
from tests.api import factories


@pytest.fixture
def run(settings: Settings) -> RunSummary:
    """Return a summarized weekly run in the fake models directory."""
    run_dir = factories.make_run_dir(settings.models_path, "weekly_2026_week_02", week=2)
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
    ids = [template.id for template in catalog.TEMPLATES]
    assert len(ids) == len(set(ids))
    for template in catalog.TEMPLATES:
        assert template.label and template.description and template.category
        described = catalog.describe(template)
        assert described.id == template.id
        for param in described.params:
            assert param["label"] and param["kind"]


def test_every_command_starts_with_the_configured_interpreter(
    settings: Settings, run: RunSummary
) -> None:
    """No template shells out to a bare ``python``; all use the venv interpreter."""
    defaults: dict[str, object] = {"season": 2026, "week": 2, "through_week": 1}
    for template in catalog.TEMPLATES:
        params = {
            spec.name: defaults[spec.name] for spec in template.params if spec.name in defaults
        }
        argv = build(template.id, settings, params, run)
        assert argv[0] == str(settings.python_path)


def test_chained_template_ids_exist() -> None:
    """A template that chains a follow-up names a template that exists."""
    for template in catalog.TEMPLATES:
        if template.chain_template_id is not None:
            assert template.chain_template_id in catalog.TEMPLATES_BY_ID


def test_get_template_rejects_an_unknown_id() -> None:
    """An unknown template id is a 422, not a 500."""
    with pytest.raises(UnprocessableEntityError):
        catalog.get_template("does_not_exist")


def test_validate_params_fills_defaults_and_coerces(settings: Settings) -> None:
    """Strings from a form are coerced and declared defaults are filled in."""
    template = catalog.get_template("predict")
    params = catalog.validate_params(template, {"season": "2026", "week": "2"})
    assert params == {"season": 2026, "week": 2, "score_rounding": "nfl"}


def test_validate_params_rejects_unknown_missing_and_bad_values() -> None:
    """Unknown keys, missing required values, and out-of-range numbers are all 422."""
    template = catalog.get_template("predict")
    with pytest.raises(UnprocessableEntityError, match="Unknown parameters"):
        catalog.validate_params(template, {"season": 2026, "week": 1, "nope": 1})
    with pytest.raises(UnprocessableEntityError, match="required"):
        catalog.validate_params(template, {"season": 2026})
    with pytest.raises(UnprocessableEntityError, match="at most"):
        catalog.validate_params(template, {"season": 2026, "week": 99})
    with pytest.raises(UnprocessableEntityError, match="integer"):
        catalog.validate_params(template, {"season": 2026, "week": "two"})
    with pytest.raises(UnprocessableEntityError, match="one of"):
        catalog.validate_params(template, {"season": 2026, "week": 2, "score_rounding": "sideways"})


def test_validate_params_reads_booleans_from_form_strings() -> None:
    """A checkbox submitted as a string still becomes a boolean."""
    template = catalog.get_template("etl_full")
    assert catalog.validate_params(template, {"refresh_nflreadpy": "true"}) == {
        "refresh_nflreadpy": True
    }
    assert catalog.validate_params(template, {"refresh_nflreadpy": "no"}) == {
        "refresh_nflreadpy": False
    }
    with pytest.raises(UnprocessableEntityError, match="true or false"):
        catalog.validate_params(template, {"refresh_nflreadpy": "maybe"})


def test_etl_template_passes_only_the_options_given(settings: Settings) -> None:
    """Blank fields are left off the command line entirely."""
    assert build("etl_full", settings, {}) == [
        str(settings.python_path),
        "-m",
        "nfl_predictor.data_collection",
        "--no-refresh-nflreadpy",
    ]
    argv = build("etl_full", settings, {"min_season": 2015, "refresh_nflreadpy": True})
    assert argv[-3:] == ["--min-season", "2015", "--refresh-nflreadpy"]


def test_lines_refresh_targets_the_configured_data_directory(settings: Settings) -> None:
    """The refresh runs the module against the configured data tree."""
    argv = build("lines_refresh", settings, {"season": 2026, "week": 2})
    assert argv[1:3] == ["-m", "nfl_predictor.lines_refresh"]
    assert argv[-2:] == ["--data-dir", str(settings.data_path)]


def test_lines_refresh_chains_a_predict_of_the_same_week() -> None:
    """The chained predict inherits the season and week, not the refresh-only parameters."""
    template = catalog.get_template("lines_refresh")
    assert template.chain_template_id == "predict"
    assert catalog.chained_params(template, {"season": 2026, "week": 2}) == {
        "season": 2026,
        "week": 2,
    }


def test_templates_without_a_chain_carry_no_chained_params() -> None:
    """A template with no follow-up produces no parameters for one."""
    assert catalog.chained_params(catalog.get_template("predict"), {"season": 2026}) == {}


def test_weekly_run_writes_a_config_file(settings: Settings) -> None:
    """The weekly run is launched from a JSON config so the script validates its own keys."""
    argv = build("weekly_run", settings, {"week": 2, "run_id": "weekly_2026_week_02"})

    assert argv[1].endswith("scripts/weekly_run.py")
    assert argv[2] == "--config"
    config = json.loads(Path(argv[3]).read_text(encoding="utf-8"))
    assert config["run_id"] == "weekly_2026_week_02"
    assert config["predict_path"].endswith("week_02_games_to_predict.csv")
    assert config["output_dir"] == str(settings.models_path)
    assert config["data_path"] == str(settings.data_path / "completed_games_ml.csv")
    assert "week" not in config


def test_predict_uses_the_active_run_model(settings: Settings, run: RunSummary) -> None:
    """Prediction reads the pinned run's model and writes back into its directory."""
    argv = build("predict", settings, {"season": 2026, "week": 2}, run)

    assert "--model-in" in argv
    assert argv[argv.index("--model-in") + 1] == str(run.run_files.model)
    assert argv[argv.index("--data-path") + 1] == str(settings.data_path / "completed_games_ml.csv")
    output = argv[argv.index("--output-path") + 1]
    assert output == str(run.run_dir / "season_2026_week_02_predictions.csv")


def test_templates_that_need_a_run_say_so_when_none_is_pinned(settings: Settings) -> None:
    """Building without an active run is a conflict the API can explain."""
    defaults: dict[str, object] = {"season": 2026, "week": 2, "through_week": 1}
    for template_id in ("predict", "power_rankings", "betting_xlsx", "shap_analysis"):
        template = catalog.get_template(template_id)
        params = {
            spec.name: defaults[spec.name] for spec in template.params if spec.name in defaults
        }
        with pytest.raises(ConflictError, match="active run"):
            build(template_id, settings, params)


def test_betting_workbook_needs_predictions(settings: Settings) -> None:
    """A run without a predictions CSV cannot produce a workbook."""
    run_dir = factories.make_run_dir(settings.models_path, "train_only", kind="training")
    summary = summarize_run(run_dir)
    assert summary is not None
    with pytest.raises(ConflictError, match="predictions"):
        build("betting_xlsx", settings, {}, summary)


def test_read_only_templates_take_no_parameters(settings: Settings) -> None:
    """The validation scripts run without arguments."""
    for template_id in ("validate_offline", "validate_live"):
        argv = build(template_id, settings, {})
        assert len(argv) == 2
        assert argv[1].endswith(f"{template_id}.py")


def test_leakage_audit_defaults_its_report_path(settings: Settings) -> None:
    """The audit writes into the reports directory unless a path is given."""
    argv = build("leakage_audit", settings, {})
    assert argv[argv.index("--out-json") + 1] == str(settings.reports_path / "leakage_audit.json")
    chosen = str(settings.state_path / "audit.json")
    argv = build("leakage_audit", settings, {"out_json": chosen})
    assert argv[argv.index("--out-json") + 1] == chosen


def test_only_dataset_and_walk_forward_jobs_are_exclusive() -> None:
    """Long or dataset-mutating jobs are serialized; the read-only ones are not."""
    groups = {template.id: template.exclusive_group for template in catalog.TEMPLATES}
    assert groups["weekly_run"] == catalog.WALK_FORWARD_GROUP
    assert groups["walk_forward_backtest"] == catalog.WALK_FORWARD_GROUP
    assert groups["train"] == catalog.WALK_FORWARD_GROUP
    assert groups["etl_full"] == catalog.DATASET_GROUP
    assert groups["lines_refresh"] == catalog.DATASET_GROUP
    assert groups["validate_offline"] is None


def test_predict_week_builds_the_missing_week_then_predicts(settings: Settings) -> None:
    """The future-week template extracts the week and chains the prediction of the same week."""
    argv = build("predict_week", settings, {"season": 2026, "week": 5})

    assert argv[1:3] == ["-m", "nfl_predictor.week_builder"]
    assert argv[argv.index("--data-dir") + 1] == str(settings.data_path)
    assert "--overwrite" not in argv
    template = catalog.get_template("predict_week")
    assert template.chain_template_id == "predict"
    assert catalog.chained_params(template, {"season": 2026, "week": 5, "overwrite": True}) == {
        "season": 2026,
        "week": 5,
    }


def test_predict_week_can_rebuild_an_existing_file(settings: Settings) -> None:
    """The overwrite switch is passed through as a flag."""
    argv = build("predict_week", settings, {"season": 2026, "week": 5, "overwrite": True})
    assert argv[-1] == "--overwrite"
