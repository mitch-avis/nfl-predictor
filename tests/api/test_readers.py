"""Tests for the predictions, betting, power, model, and data-status readers."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from nfl_predictor.api.db import Database
from nfl_predictor.api.readers import betting, cache, data_status, model, power, predictions
from nfl_predictor.api.runs.files import resolve_run_files
from tests.api import factories


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Start every test with an empty artifact cache."""
    cache.clear()


def test_predictions_reader_enriches_and_summarizes(tmp_path: Path) -> None:
    """Derived market columns and the summary follow the fixture games."""
    path = factories.write_predictions_csv(tmp_path / "p.csv")
    table, summary = predictions.read_predictions(path)
    assert "unregistered_feature" not in table.visible_columns
    by_id = {row["game_id"]: row for row in table.rows}
    den = by_id["2026_01_DEN_KC"]
    assert den["market_home_prob_novig"] == pytest.approx(0.5830, abs=1e-3)
    assert den["market_home_margin"] == 3.0
    assert den["market_favorite"] == "KC"
    assert den["agrees_with_market"] is False
    assert den["edge_home_prob"] == pytest.approx(0.4023 - 0.5830, abs=1e-3)
    nyj = by_id["2026_01_NYJ_NE"]
    assert nyj["market_favorite"] == "NE" and nyj["predicted_winner"] == "NE"
    assert summary.games == 4
    assert summary.market_disagreements == 2
    assert summary.games_with_lines == 4
    assert summary.first_kickoff is not None and summary.first_kickoff.startswith("2026-09-13")
    assert [row["game_id"] for row in table.rows][0].startswith("2026_01_")


def test_predictions_reader_without_lines() -> None:
    """Rows without moneylines fall back to the spread for the market favorite, or nothing."""
    row = predictions.enrich_row(
        {
            "home_abbr": "KC",
            "away_abbr": "DEN",
            "home_spread": 2.5,
            "predicted_winner": "KC",
            "predicted_margin": 1.0,
        }
    )
    assert row["market_favorite"] == "DEN"
    assert row["agrees_with_market"] is False
    assert row["market_home_prob_novig"] is None
    bare = predictions.enrich_row({"home_abbr": "KC", "away_abbr": "DEN"})
    assert bare["market_favorite"] is None and bare["agrees_with_market"] is None
    empty = predictions.enrich(pl.DataFrame({"game_id": []}))
    assert empty.height == 0
    assert predictions.summarize(empty).avg_confidence is None


def test_picks_readers(tmp_path: Path) -> None:
    """Picks come out most-confident first from either source."""
    picks_path = factories.write_picks_csv(tmp_path / "picks.csv")
    table = predictions.read_picks(picks_path)
    assert [row["confidence_rank"] for row in table.rows] == [4, 3, 2, 1]
    pred_path = factories.write_predictions_csv(tmp_path / "p.csv")
    derived = predictions.picks_from_predictions(pred_path)
    assert [row["confidence_rank"] for row in derived.rows] == [4, 3, 2, 1]
    assert derived.visible_columns[0] == "confidence_rank"


def test_betting_reader(tmp_path: Path) -> None:
    """Moneyline, spread, and total blocks are computed with the workbook formulas."""
    path = factories.write_predictions_csv(tmp_path / "p.csv")
    table = betting.read_betting(path)
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
    row = betting.betting_row({"home_abbr": "KC", "away_abbr": "DEN", "home_win_prob": 0.6})
    assert row["moneyline_action"] == "PASS"
    assert row["moneyline_value_side"] is None
    assert row["spread_action"] == "PASS"
    assert row["total_action"] == "PASS"
    assert row["model_fair_home_moneyline"] == -150
    assert betting.build_betting_frame(pl.DataFrame({"game_id": []})).height == 0


def test_power_reader_with_movement(tmp_path: Path) -> None:
    """Rank change is previous rank minus current rank; standings sort by projected win pct."""
    factories.write_power_csvs(tmp_path, 2026, 1, order=["KC", "DEN", "BUF", "GB", "MIN"])
    factories.write_power_csvs(tmp_path, 2026, 0, order=["DEN", "KC", "GB", "BUF", "MIN"])
    current = tmp_path / "power_rankings_season_2026_week_01.csv"
    previous = tmp_path / "power_rankings_season_2026_week_00.csv"
    table = power.read_rankings(current, previous)
    rows = {row["team_abbr"]: row for row in table.rows}
    assert (
        rows["KC"]["rank"] == 1
        and rows["KC"]["previous_rank"] == 2
        and rows["KC"]["rank_change"] == 1
    )
    assert rows["DEN"]["rank_change"] == -1
    assert rows["MIN"]["rank_change"] == 0
    assert rows["KC"]["record"] == "9-1"
    no_prev = power.read_rankings(current, None)
    assert no_prev.rows[0]["rank_change"] is None
    standings = power.read_standings(tmp_path / "projected_standings_season_2026_week_01.csv")
    pcts = [row["projected_win_pct"] for row in standings.rows]
    assert pcts == sorted(pcts, reverse=True)
    division = power.read_standings(
        tmp_path / "projected_division_standings_season_2026_week_01.csv"
    )
    assert "projected_division_rank" in division.visible_columns


def test_power_record_with_ties() -> None:
    """Ties appear in the record string only when non-zero."""
    df = pl.DataFrame(
        {"team_abbr": ["A", "B"], "wins": [3, 2], "losses": [1, 2], "ties": [1, 0], "rank": [1, 2]}
    )
    out = power.with_movement(df, None)
    assert power.read_standings.__name__ == "read_standings"
    records = power._with_record(out)["record"].to_list()  # noqa: SLF001 - helper under test
    assert records == ["3-1-1", "2-2"]
    assert power._with_record(pl.DataFrame({"x": [1]})).columns == ["x"]  # noqa: SLF001


def test_model_reader(tmp_path: Path) -> None:
    """Metadata, metrics, importance, comparison, and calibration are assembled per run kind."""
    weekly = factories.make_run_dir(tmp_path, "weekly")
    files = resolve_run_files(weekly)
    payload = model.model_payload(files)
    assert payload["metadata"]["feature_count"] == 3
    assert payload["metadata"]["config"]["model_kind"] == "margin_total"
    assert payload["metrics"]["holdout"]["brier"] == pytest.approx(0.2192)
    assert payload["metrics"]["pool"]["weeks"] == 18
    assert [row["feature"] for row in payload["feature_importance"]] == [
        "away_elo_pre",
        "home_rest",
        "away_rest",
    ]
    assert payload["feature_importance"][0]["margin_gain"] == 3.0
    assert payload["wf_compare"] is not None
    assert payload["wf_compare"].rows[0]["wf_rank"] == 1
    assert payload["wf_best"]["candidate_key"] == "a"
    assert payload["calibration"] is None
    (weekly / "wf_compare").mkdir()
    (weekly / "wf_compare" / "wf_candidate_other.json").write_text(
        json.dumps({"candidate_key": "zzz"}), encoding="utf-8"
    )
    (weekly / "wf_compare" / "wf_candidate_bad.json").write_text("nope", encoding="utf-8")
    (weekly / "wf_compare" / "wf_candidate_a.json").write_text(
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
    cache.clear()
    payload = model.model_payload(files)
    assert payload["calibration"]["bin_count"] == 1
    assert payload["calibration"]["source"] == "wf_candidate_a.json"

    wf = factories.make_run_dir(tmp_path, "wf", kind="walk_forward")
    wf_payload = model.model_payload(resolve_run_files(wf))
    assert wf_payload["metrics"]["overall"]["brier"] == pytest.approx(0.2277)
    assert wf_payload["calibration"]["bin_count"] == 2
    assert wf_payload["feature_importance"] == []
    assert wf_payload["wf_compare"] is None

    training = factories.make_run_dir(tmp_path, "train", kind="training")
    assert model.model_payload(resolve_run_files(training))["calibration"] is None


def test_model_reader_tolerates_odd_shapes(tmp_path: Path) -> None:
    """Non-dict JSON and missing blocks produce empty results rather than errors."""
    (tmp_path / "meta.json").write_text("[1, 2]", encoding="utf-8")
    assert model.metadata_summary(tmp_path / "meta.json") == {}
    assert model.metrics_summary(tmp_path / "meta.json") == {}
    (tmp_path / "fi.json").write_text(
        json.dumps({"base_features": {"feature_names": "x"}}), encoding="utf-8"
    )
    assert model.feature_importance(tmp_path / "fi.json") == []
    (tmp_path / "wf.csv").write_text(
        "label,brier,log_loss\nb,0.3,0.8\na,0.2,0.7\n", encoding="utf-8"
    )
    table = model.wf_compare(tmp_path / "wf.csv")
    assert [row["label"] for row in table.rows] == ["a", "b"]
    assert table.rows[0]["wf_rank"] == 1
    run = factories.make_run_dir(tmp_path, "nobest")
    (run / "wf_best.json").write_text("{}", encoding="utf-8")
    cache.clear()
    assert model.best_candidate_calibration(resolve_run_files(run)) is None
    (run / "wf_best.json").write_text(json.dumps({"candidate_key": "a"}), encoding="utf-8")
    cache.clear()
    assert model.best_candidate_calibration(resolve_run_files(run)) is None


def test_data_status_helpers(project_root: Path, db: Database) -> None:
    """File status, cache coverage, leakage audits, and unattached files are discovered."""
    data = project_root / "data"
    (data / "all_data_ml.csv").write_text("season,week,x\n2024,1,1\n2025,2,2\n", encoding="utf-8")
    (data / "qb_elos.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    status = data_status.file_status(data, "all_data_ml.csv", "d")
    assert status.exists and status.rows == 2 and status.seasons == (2024, 2025)
    assert data_status.file_status(data, "qb_elos.csv", "d").seasons is None
    assert not data_status.file_status(data, "missing.csv", "d").exists
    cache_dir = data / "cache" / "nflreadpy"
    cache_dir.mkdir(parents=True)
    for name in (
        "pbp_2024_reg.parquet",
        "schedule_2025.parquet",
        "schedule_2024.parquet",
        "junk.txt",
    ):
        (cache_dir / name).write_bytes(b"")
    assert data_status.cache_coverage(cache_dir) == {"schedule": [2024, 2025], "pbp": [2024]}
    assert data_status.cache_coverage(data / "nope") == {"schedule": [], "pbp": []}
    assert data_status.latest_leakage_audit(project_root / "models") is None
    (project_root / "models" / "old_leakage_audit.json").write_text(
        json.dumps({"ok": False}), encoding="utf-8"
    )
    nested = project_root / "models" / "run"
    nested.mkdir()
    (nested / "leakage_audit.json").write_text(
        json.dumps({"ok": True, "findings": []}), encoding="utf-8"
    )
    import os

    os.utime(nested / "leakage_audit.json", (2_000_000_000, 2_000_000_000))
    audit = data_status.latest_leakage_audit(project_root / "models", project_root / "reports")
    assert (
        audit is not None
        and audit["ok"] is True
        and audit["path"].endswith("run/leakage_audit.json")
    )
    (nested / "leakage_audit.json").write_text("broken", encoding="utf-8")
    assert data_status.latest_leakage_audit(project_root / "models") is None
    (nested / "leakage_audit.json").write_text("[1]", encoding="utf-8")
    assert data_status.latest_leakage_audit(project_root / "models") is None

    predict = data / "predict"
    factories.write_predictions_csv(predict / "week_03_predictions.csv", 2026, 3)
    (predict / "week_04_games_to_predict.csv").write_text("game_id\nx\n", encoding="utf-8")
    (predict / "other.csv").write_text("a\n1\n", encoding="utf-8")
    (project_root / "reports" / "week_01_2026_betting.xlsx").write_bytes(b"PK")
    items = data_status.unattached_files(data, project_root / "reports")
    assert [(i["name"], i["season"], i["week"]) for i in items] == [
        ("week_03_predictions.csv", 2026, 3),
        ("week_04_games_to_predict.csv", None, 4),
        ("week_01_2026_betting.xlsx", None, None),
    ]
    season, week = data_status.current_season_week()
    assert season >= 2026 and 0 <= week <= 22

    fingerprints = data_status.FingerprintCache(db)
    target = data / "all_data_ml.csv"
    assert fingerprints.get(data / "missing.csv") is None
    assert fingerprints.get(target) is None
    import time

    for _ in range(50):
        if fingerprints.get(target) is not None:
            break
        time.sleep(0.05)
    payload = fingerprints.get(target)
    assert payload is not None and len(payload["sha256"]) == 64
    db.set_value(f"{data_status.FINGERPRINT_KEY_PREFIX}{target}", "not json")
    assert fingerprints.get(target) is None
