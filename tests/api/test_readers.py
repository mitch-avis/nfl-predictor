"""Tests for the predictions, betting, power, model, and data-status readers."""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

from nfl_predictor.api.readers import betting, cache, data_status, model, power, predictions
from nfl_predictor.api.runs.files import resolve_run_files
from tests.api import factories

if TYPE_CHECKING:
    from pathlib import Path

    from nfl_predictor.api.db import Database


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Start every test with an empty artifact cache."""
    cache.clear()


def _write_power_weeks(directory: Path) -> tuple[Path, Path]:
    """Write rankings for weeks 0 and 1 and return ``(current, previous)``."""
    factories.write_power_csvs(directory, 2026, 1, order=["KC", "DEN", "BUF", "GB", "MIN"])
    factories.write_power_csvs(directory, 2026, 0, order=["DEN", "KC", "GB", "BUF", "MIN"])
    return (
        directory / "power_rankings_season_2026_week_01.csv",
        directory / "power_rankings_season_2026_week_00.csv",
    )


def _write_candidate_calibration(run_dir: Path) -> None:
    """Write the winning candidate's reliability bins next to two candidates that never match."""
    (run_dir / "wf_compare").mkdir()
    (run_dir / "wf_compare" / "wf_candidate_other.json").write_text(
        json.dumps({"candidate_key": "zzz"}), encoding="utf-8"
    )
    (run_dir / "wf_compare" / "wf_candidate_bad.json").write_text("nope", encoding="utf-8")
    (run_dir / "wf_compare" / "wf_candidate_a.json").write_text(
        json.dumps(
            {
                "candidate_key": "a",
                "metrics": {
                    "reliability": [
                        {
                            "bin_lower": 0,
                            "bin_upper": 1,
                            "count": 5,
                            "avg_pred": 0.5,
                            "avg_actual": 0.4,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def _write_leakage_audits(models_dir: Path, newest: str) -> None:
    """Write an old top-level audit and a newer nested one holding ``newest``."""
    (models_dir / "old_leakage_audit.json").write_text(json.dumps({"ok": False}), encoding="utf-8")
    nested = models_dir / "run"
    nested.mkdir()
    (nested / "leakage_audit.json").write_text(newest, encoding="utf-8")
    os.utime(nested / "leakage_audit.json", (2_000_000_000, 2_000_000_000))


def _wait_for_fingerprint(
    fingerprints: data_status.FingerprintCache, target: Path
) -> dict[str, Any] | None:
    """Poll ``fingerprints`` until the background hash of ``target`` lands, for up to 2.5 s."""
    for _ in range(50):
        payload = fingerprints.get(target)
        if payload is not None:
            return payload
        time.sleep(0.05)
    return fingerprints.get(target)


def test_predictions_reader_enriches_and_summarizes(tmp_path: Path) -> None:
    """Derived market columns and the summary follow the fixture games."""
    # Arrange
    path = factories.write_predictions_csv(tmp_path / "p.csv")

    # Act
    table, summary = predictions.read_predictions(path)

    # Assert
    assert "unregistered_feature" not in table.visible_columns
    by_id = {row["game_id"]: row for row in table.rows}
    den = by_id["2026_01_DEN_KC"]
    assert den["market_home_prob_novig"] == pytest.approx(0.5830, abs=1e-3)
    assert den["market_home_margin"] == 3.0
    assert den["market_favorite"] == "KC"
    assert den["agrees_with_market"] is False
    assert den["edge_home_prob"] == pytest.approx(0.4023 - 0.5830, abs=1e-3)
    nyj = by_id["2026_01_NYJ_NE"]
    assert nyj["market_favorite"] == "NE"
    assert nyj["predicted_winner"] == "NE"
    assert summary.games == 4
    assert summary.market_disagreements == 2
    assert summary.games_with_lines == 4
    assert summary.first_kickoff is not None
    assert summary.first_kickoff.startswith("2026-09-13")
    assert next(row["game_id"] for row in table.rows).startswith("2026_01_")


def test_enrich_row_without_moneylines_takes_the_favorite_from_the_spread() -> None:
    """A row without moneylines falls back to the spread for the market favorite."""
    # Arrange
    row = {
        "home_abbr": "KC",
        "away_abbr": "DEN",
        "home_spread": 2.5,
        "predicted_winner": "KC",
        "predicted_margin": 1.0,
    }

    # Act
    enriched = predictions.enrich_row(row)

    # Assert
    assert enriched["market_favorite"] == "DEN"
    assert enriched["agrees_with_market"] is False
    assert enriched["market_home_prob_novig"] is None


def test_enrich_row_without_any_line_has_no_market_favorite() -> None:
    """A row without a spread or moneylines has no market favorite to agree with."""
    # Act
    bare = predictions.enrich_row({"home_abbr": "KC", "away_abbr": "DEN"})

    # Assert
    assert bare["market_favorite"] is None
    assert bare["agrees_with_market"] is None


def test_enrich_keeps_an_empty_frame_empty() -> None:
    """Enriching a frame without rows returns a frame without rows."""
    # Act
    empty = predictions.enrich(pl.DataFrame({"game_id": []}))

    # Assert
    assert empty.height == 0


def test_summarize_an_empty_frame_has_no_average_confidence() -> None:
    """A week without games has no average confidence."""
    # Arrange
    empty = predictions.enrich(pl.DataFrame({"game_id": []}))

    # Act
    summary = predictions.summarize(empty)

    # Assert
    assert summary.avg_confidence is None


def test_read_picks_orders_most_confident_first(tmp_path: Path) -> None:
    """Picks read from a picks file come out most-confident first."""
    # Arrange
    picks_path = factories.write_picks_csv(tmp_path / "picks.csv")

    # Act
    table = predictions.read_picks(picks_path)

    # Assert
    assert [row["confidence_rank"] for row in table.rows] == [4, 3, 2, 1]


def test_picks_from_predictions_orders_most_confident_first(tmp_path: Path) -> None:
    """Picks derived from a predictions file come out most-confident first, rank first."""
    # Arrange
    pred_path = factories.write_predictions_csv(tmp_path / "p.csv")

    # Act
    derived = predictions.picks_from_predictions(pred_path)

    # Assert
    assert [row["confidence_rank"] for row in derived.rows] == [4, 3, 2, 1]
    assert derived.visible_columns[0] == "confidence_rank"


def test_betting_reader(tmp_path: Path) -> None:
    """Moneyline, spread, and total blocks are computed with the workbook formulas."""
    # Arrange
    path = factories.write_predictions_csv(tmp_path / "p.csv")

    # Act
    table = betting.read_betting(path)

    # Assert
    by_id = {row["game_id"]: row for row in table.rows}
    den = by_id["2026_01_DEN_KC"]
    assert den["model_home_prob"] == pytest.approx(0.4023)
    assert den["market_home_prob_raw"] == pytest.approx(155 / 255)
    assert den["moneyline_value_side"] == "DEN"
    assert den["moneyline_edge_prob"] == pytest.approx((1 - 0.4023) - 100 / 230, abs=1e-6)
    assert den["moneyline_action"] == "STRONG"
    assert den["moneyline_confidence_1_10"] == 10
    assert den["moneyline_ev"] == pytest.approx((1 - 0.4023) * 1.3 - 0.4023)
    assert den["model_fair_home_moneyline"] == 149
    assert den["model_fair_away_moneyline"] == -149
    assert den["spread_sigma"] == pytest.approx(34 / 2.563103, abs=1e-3)
    assert 0 < den["spread_p_home_cover"] < 1
    assert den["spread_value_side"] in {"DEN", "KC"}
    assert den["spread_edge_points"] == pytest.approx(
        den["predicted_margin_raw"] + den["home_spread"]
    )
    assert den["total_value_side"] in {"OVER", "UNDER"}
    assert den["total_edge_points"] == pytest.approx(den["predicted_total_raw"] - den["total_line"])
    assert table.column_metadata["total_action"].actionable is False
    assert table.column_metadata["moneyline_action"].actionable is True


def test_betting_row_without_market() -> None:
    """Missing lines yield PASS rows rather than errors."""
    # Act
    row = betting.betting_row({"home_abbr": "KC", "away_abbr": "DEN", "home_win_prob": 0.6})

    # Assert
    assert row["moneyline_action"] == "PASS"
    assert row["moneyline_value_side"] is None
    assert row["spread_action"] == "PASS"
    assert row["total_action"] == "PASS"
    assert row["model_fair_home_moneyline"] == -150


def test_build_betting_frame_keeps_an_empty_frame_empty() -> None:
    """A week without games builds a betting frame without rows."""
    # Act
    frame = betting.build_betting_frame(pl.DataFrame({"game_id": []}))

    # Assert
    assert frame.height == 0


def test_read_rankings_reports_movement_against_the_previous_week(tmp_path: Path) -> None:
    """Rank change is the previous rank minus the current rank."""
    # Arrange
    current, previous = _write_power_weeks(tmp_path)

    # Act
    table = power.read_rankings(current, previous)

    # Assert
    rows = {row["team_abbr"]: row for row in table.rows}
    assert rows["KC"]["rank"] == 1
    assert rows["KC"]["previous_rank"] == 2
    assert rows["KC"]["rank_change"] == 1
    assert rows["DEN"]["rank_change"] == -1
    assert rows["MIN"]["rank_change"] == 0
    assert rows["KC"]["record"] == "9-1"


def test_read_rankings_without_a_previous_week_has_no_movement(tmp_path: Path) -> None:
    """Without a previous week's rankings the rank change is empty."""
    # Arrange
    current, _previous = _write_power_weeks(tmp_path)

    # Act
    table = power.read_rankings(current, None)

    # Assert
    assert table.rows[0]["rank_change"] is None


def test_read_standings_sorts_by_projected_win_pct(tmp_path: Path) -> None:
    """Projected standings come out sorted by projected win percentage."""
    # Arrange
    _write_power_weeks(tmp_path)

    # Act
    standings = power.read_standings(tmp_path / "projected_standings_season_2026_week_01.csv")

    # Assert
    pcts = [row["projected_win_pct"] for row in standings.rows]
    assert pcts == sorted(pcts, reverse=True)


def test_read_standings_keeps_the_division_rank(tmp_path: Path) -> None:
    """Division standings keep their projected division rank column."""
    # Arrange
    _write_power_weeks(tmp_path)

    # Act
    division = power.read_standings(
        tmp_path / "projected_division_standings_season_2026_week_01.csv"
    )

    # Assert
    assert "projected_division_rank" in division.visible_columns


def test_with_record_shows_ties_only_when_nonzero() -> None:
    """Ties appear in the record string only when non-zero."""
    # Arrange
    df = pl.DataFrame(
        {"team_abbr": ["A", "B"], "wins": [3, 2], "losses": [1, 2], "ties": [1, 0], "rank": [1, 2]}
    )
    out = power.with_movement(df, None)

    # Act
    records = power._with_record(out)["record"].to_list()

    # Assert
    assert power.read_standings.__name__ == "read_standings"
    assert records == ["3-1-1", "2-2"]


def test_with_record_leaves_a_frame_without_records_alone() -> None:
    """A frame without win and loss columns gains no record column."""
    # Act
    out = power._with_record(pl.DataFrame({"x": [1]}))

    # Assert
    assert out.columns == ["x"]


def test_model_payload_assembles_a_weekly_run(tmp_path: Path) -> None:
    """Metadata, metrics, importance and the walk-forward comparison are assembled."""
    # Arrange
    weekly = factories.make_run_dir(tmp_path, "weekly")
    files = resolve_run_files(weekly)

    # Act
    payload = model.model_payload(files)

    # Assert
    assert payload["metadata"]["feature_count"] == 3
    assert payload["metadata"]["config"]["model_kind"] == "margin_total"
    assert payload["metrics"]["holdout"]["brier"] == pytest.approx(0.2192)
    assert payload["metrics"]["pool"]["weeks"] == 18
    importance = payload["feature_importance"]
    assert importance["measure"] == "mean_abs_shap"
    assert [row["feature"] for row in importance["rows"]] == [
        "away_rest",
        "away_elo_pre",
        "home_rest",
    ]
    assert importance["rows"][0] == {
        "feature": "away_rest",
        "value": 3.0,
        "margin_value": 2.5,
        "total_value": 0.5,
        "splits": 2.0,
    }
    assert payload["wf_compare"] is not None
    assert payload["wf_compare"].rows[0]["wf_rank"] == 1
    assert payload["wf_best"]["candidate_key"] == "a"
    assert payload["calibration"] is None


def test_model_payload_reads_the_winning_candidate_calibration(tmp_path: Path) -> None:
    """A weekly run's calibration comes from the candidate ``wf_best.json`` names."""
    # Arrange
    weekly = factories.make_run_dir(tmp_path, "weekly")
    _write_candidate_calibration(weekly)

    # Act
    payload = model.model_payload(resolve_run_files(weekly))

    # Assert
    assert payload["calibration"]["bin_count"] == 1
    assert payload["calibration"]["source"] == "wf_candidate_a.json"


def test_model_payload_for_a_walk_forward_run(tmp_path: Path) -> None:
    """A walk-forward run has overall metrics and calibration but no importance or comparison."""
    # Arrange
    wf = factories.make_run_dir(tmp_path, "wf", factories.RunSpec(kind="walk_forward"))

    # Act
    wf_payload = model.model_payload(resolve_run_files(wf))

    # Assert
    assert wf_payload["metrics"]["overall"]["brier"] == pytest.approx(0.2277)
    assert wf_payload["calibration"]["bin_count"] == 2
    assert wf_payload["feature_importance"] == {"measure": None, "rows": [], "margin_rows": []}
    assert wf_payload["wf_compare"] is None


def test_model_payload_for_a_training_run_has_no_calibration(tmp_path: Path) -> None:
    """A training run without a walk-forward report has no calibration."""
    # Arrange
    training = factories.make_run_dir(tmp_path, "train", factories.RunSpec(kind="training"))

    # Act
    payload = model.model_payload(resolve_run_files(training))

    # Assert
    assert payload["calibration"] is None


def test_metadata_summary_of_a_non_object_is_empty(tmp_path: Path) -> None:
    """Metadata JSON that is not an object reads as an empty summary."""
    # Arrange
    (tmp_path / "meta.json").write_text("[1, 2]", encoding="utf-8")

    # Act
    summary = model.metadata_summary(tmp_path / "meta.json")

    # Assert
    assert summary == {}


def test_metrics_summary_of_a_non_object_is_empty(tmp_path: Path) -> None:
    """Metrics JSON that is not an object reads as an empty summary."""
    # Arrange
    (tmp_path / "meta.json").write_text("[1, 2]", encoding="utf-8")

    # Act
    summary = model.metrics_summary(tmp_path / "meta.json")

    # Assert
    assert summary == {}


@pytest.mark.parametrize(
    "payload",
    [{"base_features": {"feature_names": "x"}}, {"base_features": None}],
    ids=["feature-names-not-a-list", "no-base-features"],
)
def test_feature_importance_of_an_odd_shape_is_empty(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    """Importance files without usable base features produce an empty result, not an error."""
    # Arrange
    path = tmp_path / "fi.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    # Act
    importance = model.feature_importance(path)

    # Assert
    assert importance == {"measure": None, "rows": [], "margin_rows": []}


def test_wf_compare_without_a_rank_sorts_by_brier(tmp_path: Path) -> None:
    """A comparison without a rank column is ranked by Brier, then log loss."""
    # Arrange
    (tmp_path / "wf.csv").write_text(
        "label,brier,log_loss\nb,0.3,0.8\na,0.2,0.7\n", encoding="utf-8"
    )

    # Act
    table = model.wf_compare(tmp_path / "wf.csv")

    # Assert
    assert [row["label"] for row in table.rows] == ["a", "b"]
    assert table.rows[0]["wf_rank"] == 1


def test_best_candidate_calibration_without_a_candidate_key_is_none(tmp_path: Path) -> None:
    """A ``wf_best.json`` that names no candidate has no calibration."""
    # Arrange
    run = factories.make_run_dir(tmp_path, "nobest")
    (run / "wf_best.json").write_text("{}", encoding="utf-8")

    # Act
    calibration = model.best_candidate_calibration(resolve_run_files(run))

    # Assert
    assert calibration is None


def test_best_candidate_calibration_without_candidate_files_is_none(tmp_path: Path) -> None:
    """A named candidate without a ``wf_compare/`` directory has no calibration."""
    # Arrange
    run = factories.make_run_dir(tmp_path, "nobest")
    (run / "wf_best.json").write_text(json.dumps({"candidate_key": "a"}), encoding="utf-8")

    # Act
    calibration = model.best_candidate_calibration(resolve_run_files(run))

    # Assert
    assert calibration is None


def test_feature_importance_falls_back_to_summed_average_gain_for_old_files(
    tmp_path: Path,
) -> None:
    """Files without total gain still load, ranked and labelled by their summed average gain."""
    # Arrange
    path = tmp_path / "feature_importance.json"
    path.write_text(json.dumps(factories.legacy_feature_importance_payload("old")), "utf-8")

    # Act
    importance = model.feature_importance(path)

    # Assert
    assert importance["measure"] == "summed_average_gain"
    assert [row["feature"] for row in importance["rows"]] == [
        "away_rest",
        "home_rest",
        "away_elo_pre",
    ]
    assert importance["rows"][0] == {
        "feature": "away_rest",
        "value": 3.5,
        "margin_value": 3.0,
        "total_value": 0.5,
        "splits": 2.0,
    }


def test_feature_importance_falls_back_to_total_gain_for_files_without_shap(
    tmp_path: Path,
) -> None:
    """Files written before SHAP was recorded rank and label their features by total gain."""
    # Arrange
    path = tmp_path / "feature_importance.json"
    path.write_text(json.dumps(factories.gain_feature_importance_payload("gain")), "utf-8")

    # Act
    importance = model.feature_importance(path)

    # Assert
    assert importance["measure"] == "total_gain"
    assert [row["feature"] for row in importance["rows"]] == [
        "away_elo_pre",
        "home_rest",
        "away_rest",
    ]
    assert importance["rows"][0] == {
        "feature": "away_elo_pre",
        "value": 35.0,
        "margin_value": 30.0,
        "total_value": 5.0,
        "splits": 4.0,
    }


def test_feature_importance_prefers_total_gain_when_both_measures_exist(tmp_path: Path) -> None:
    """Total gain ranks the features even when a file also carries the summed average gain."""
    # Arrange
    payload = factories.gain_feature_importance_payload("both")
    payload["base_features"]["combined"]["gain"] = [99.0, 1.0, 1.0]
    path = tmp_path / "feature_importance.json"
    path.write_text(json.dumps(payload), "utf-8")

    # Act
    importance = model.feature_importance(path, top=1)

    # Assert
    assert importance["measure"] == "total_gain"
    assert [row["feature"] for row in importance["rows"]] == ["away_elo_pre"]


def test_feature_importance_ranks_the_margin_head_over_every_feature(tmp_path: Path) -> None:
    """The margin-only ranking reaches past the combined top list to the margin head's leaders."""
    # Arrange
    payload = factories.feature_importance_payload("heads")
    base = payload["base_features"]
    base["margin"]["mean_abs_shap"] = [1.0, 2.0, 0.5]
    base["total"]["mean_abs_shap"] = [2.0, 0.0, 2.0]
    base["combined"]["mean_abs_shap"] = [3.0, 2.0, 2.5]
    path = tmp_path / "feature_importance.json"
    path.write_text(json.dumps(payload), "utf-8")

    # Act
    importance = model.feature_importance(path, top=2)

    # Assert
    assert [row["feature"] for row in importance["rows"]] == ["away_rest", "away_elo_pre"]
    assert [row["feature"] for row in importance["margin_rows"]] == ["home_rest", "away_rest"]
    assert importance["margin_rows"][0] == {
        "feature": "home_rest",
        "value": 2.0,
        "margin_value": 2.0,
        "total_value": 0.0,
        "splits": 2.0,
    }


def test_feature_importance_margin_rows_use_the_fallback_measure(tmp_path: Path) -> None:
    """A run without SHAP ranks the margin head by its total gain and its own split count."""
    # Arrange
    path = tmp_path / "feature_importance.json"
    path.write_text(json.dumps(factories.gain_feature_importance_payload("gain")), "utf-8")

    # Act
    importance = model.feature_importance(path)

    # Assert
    assert [row["feature"] for row in importance["margin_rows"]] == [
        "away_elo_pre",
        "home_rest",
        "away_rest",
    ]
    assert importance["margin_rows"][0] == {
        "feature": "away_elo_pre",
        "value": 30.0,
        "margin_value": 30.0,
        "total_value": 5.0,
        "splits": 3.0,
    }


def test_feature_importance_without_a_margin_block_has_no_margin_rows(tmp_path: Path) -> None:
    """A file that records only the combined heads still ranks them, with no margin view."""
    # Arrange
    payload = factories.gain_feature_importance_payload("combined-only")
    del payload["base_features"]["margin"]
    path = tmp_path / "feature_importance.json"
    path.write_text(json.dumps(payload), "utf-8")

    # Act
    importance = model.feature_importance(path)

    # Assert
    assert [row["feature"] for row in importance["rows"]] == [
        "away_elo_pre",
        "home_rest",
        "away_rest",
    ]
    assert importance["margin_rows"] == []


def test_wf_compare_prefers_deterministic_ranking(tmp_path: Path) -> None:
    """WF compare tables should rank by the deterministic instrument when present."""
    # Arrange
    (tmp_path / "wf.csv").write_text(
        (
            "label,brier,log_loss,deterministic_brier,deterministic_log_loss\n"
            "configured-better,0.2,0.6,0.22,0.62\n"
            "deterministic-better,0.23,0.63,0.19,0.59\n"
        ),
        encoding="utf-8",
    )

    # Act
    table = model.wf_compare(tmp_path / "wf.csv")

    # Assert
    assert [row["label"] for row in table.rows] == ["deterministic-better", "configured-better"]
    assert table.rows[0]["wf_rank"] == 1


def test_file_status_reads_rows_and_seasons(project_root: Path) -> None:
    """A dataset file reports its rows and season range."""
    # Arrange
    data = project_root / "data"
    (data / "all_data_ml.csv").write_text("season,week,x\n2024,1,1\n2025,2,2\n", encoding="utf-8")

    # Act
    status = data_status.file_status(data, "all_data_ml.csv", "d")

    # Assert
    assert status.exists
    assert status.rows == 2
    assert status.seasons == (2024, 2025)


def test_file_status_without_a_season_column_has_no_seasons(project_root: Path) -> None:
    """A dataset file without a season column has no season range."""
    # Arrange
    data = project_root / "data"
    (data / "qb_elos.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    # Act
    status = data_status.file_status(data, "qb_elos.csv", "d")

    # Assert
    assert status.seasons is None


def test_file_status_of_a_missing_file_does_not_exist(project_root: Path) -> None:
    """A missing dataset file reports that it does not exist."""
    # Act
    status = data_status.file_status(project_root / "data", "missing.csv", "d")

    # Assert
    assert not status.exists


def test_cache_coverage_lists_seasons_per_family(project_root: Path) -> None:
    """Cache coverage lists each nflreadpy family's seasons and ignores other files."""
    # Arrange
    cache_dir = project_root / "data" / "cache" / "nflreadpy"
    cache_dir.mkdir(parents=True)
    for name in (
        "pbp_2024_reg.parquet",
        "schedule_2025.parquet",
        "schedule_2024.parquet",
        "junk.txt",
    ):
        (cache_dir / name).write_bytes(b"")

    # Act
    coverage = data_status.cache_coverage(cache_dir)

    # Assert
    assert coverage == {"schedule": [2024, 2025], "pbp": [2024]}


def test_cache_coverage_of_a_missing_directory_is_empty(project_root: Path) -> None:
    """A cache directory that does not exist covers no seasons."""
    # Act
    coverage = data_status.cache_coverage(project_root / "data" / "nope")

    # Assert
    assert coverage == {"schedule": [], "pbp": []}


def test_latest_leakage_audit_without_reports_is_none(project_root: Path) -> None:
    """With no audit on disk there is no latest audit."""
    # Act
    audit = data_status.latest_leakage_audit(project_root / "models")

    # Assert
    assert audit is None


def test_latest_leakage_audit_reads_the_newest_report(project_root: Path) -> None:
    """The newest leakage audit is found in nested run folders, with its path attached."""
    # Arrange
    _write_leakage_audits(project_root / "models", json.dumps({"ok": True, "findings": []}))

    # Act
    audit = data_status.latest_leakage_audit(project_root / "models", project_root / "reports")

    # Assert
    assert audit is not None
    assert audit["ok"] is True
    assert audit["path"].endswith("run/leakage_audit.json")


@pytest.mark.parametrize("newest", ["broken", "[1]"], ids=["not-json", "not-an-object"])
def test_latest_leakage_audit_unreadable_newest_report_is_none(
    project_root: Path, newest: str
) -> None:
    """When the newest audit is not a readable object there is no latest audit."""
    # Arrange
    _write_leakage_audits(project_root / "models", newest)

    # Act
    audit = data_status.latest_leakage_audit(project_root / "models")

    # Assert
    assert audit is None


def test_unattached_files_lists_weeks(project_root: Path) -> None:
    """Prediction, games and report files outside runs are listed with their weeks."""
    # Arrange
    data = project_root / "data"
    predict = data / "predict"
    factories.write_predictions_csv(predict / "week_03_predictions.csv", 2026, 3)
    (predict / "week_04_games_to_predict.csv").write_text("game_id\nx\n", encoding="utf-8")
    (predict / "other.csv").write_text("a\n1\n", encoding="utf-8")
    (project_root / "reports" / "week_01_2026_betting.xlsx").write_bytes(b"PK")

    # Act
    items = data_status.unattached_files(data, project_root / "reports")

    # Assert
    assert [(i["name"], i["season"], i["week"]) for i in items] == [
        ("week_03_predictions.csv", 2026, 3),
        ("week_04_games_to_predict.csv", None, 4),
        ("week_01_2026_betting.xlsx", None, None),
    ]


def test_current_season_week_is_in_range() -> None:
    """The current season and week follow the ETL's calendar."""
    # Act
    season, week = data_status.current_season_week()

    # Assert
    assert season >= 2026
    assert 0 <= week <= 22


def test_fingerprint_cache_of_a_missing_file_is_none(project_root: Path, db: Database) -> None:
    """A file that does not exist has no fingerprint."""
    # Arrange
    fingerprints = data_status.FingerprintCache(db)

    # Act
    payload = fingerprints.get(project_root / "data" / "missing.csv")

    # Assert
    assert payload is None


def test_fingerprint_cache_returns_none_while_hashing(project_root: Path, db: Database) -> None:
    """The first read of a file starts the hash in the background and returns nothing yet."""
    # Arrange
    target = project_root / "data" / "all_data_ml.csv"
    target.write_text("season,week,x\n2024,1,1\n", encoding="utf-8")
    fingerprints = data_status.FingerprintCache(db)

    # Act
    payload = fingerprints.get(target)

    # Assert
    assert payload is None


def test_fingerprint_cache_hashes_in_the_background(project_root: Path, db: Database) -> None:
    """A fingerprint computed in the background is cached and read back."""
    # Arrange
    target = project_root / "data" / "all_data_ml.csv"
    target.write_text("season,week,x\n2024,1,1\n", encoding="utf-8")
    fingerprints = data_status.FingerprintCache(db)

    # Act
    payload = _wait_for_fingerprint(fingerprints, target)

    # Assert
    assert payload is not None
    assert len(payload["sha256"]) == 64


def test_fingerprint_cache_reads_a_corrupt_entry_as_none(project_root: Path, db: Database) -> None:
    """A stored fingerprint that is not JSON reads as none."""
    # Arrange
    target = project_root / "data" / "all_data_ml.csv"
    target.write_text("season,week,x\n2024,1,1\n", encoding="utf-8")
    fingerprints = data_status.FingerprintCache(db)
    db.set_value(f"{data_status.FINGERPRINT_KEY_PREFIX}{target}", "not json")

    # Act
    payload = fingerprints.get(target)

    # Assert
    assert payload is None
