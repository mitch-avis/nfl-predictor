"""Tests for the paired walk-forward comparison and the ``compare`` command.

The fixtures are hand-built fold checkpoints: a few games with known probabilities and results,
so every metric can be checked by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from tests import snapshots

from nfl_predictor.cli import compare as command
from nfl_predictor.cli import main as front_door
from nfl_predictor.reporting import run_comparison

# (game_id, season, week, p, actual margin); home wins when the margin is positive.
GAMES = [
    ("g1", 2024, 1, 0.80, 7),
    ("g2", 2024, 1, 0.30, -3),
    ("g3", 2024, 1, 0.60, 0),
    ("g4", 2024, 2, 0.45, 3),
    ("g5", 2024, 3, 0.70, 10),
    ("g6", 2024, 19, 0.55, -6),
]


def _predictions(probs: dict[str, float] | None = None, margin_shift: float = 0.0) -> pd.DataFrame:
    """Return prediction rows for ``GAMES``, optionally overriding probabilities."""
    rows = []
    for game_id, season, week, p, margin in GAMES:
        prob = (probs or {}).get(game_id, p)
        rows.append(
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "deterministic_home_win_prob": prob,
                "market_home_win_prob": 0.5,
                "actual_home_win": int(margin > 0),
                "actual_margin": float(margin),
                "actual_total": 40.0,
                "predicted_margin": float(margin) + 2.0 + margin_shift,
                "predicted_total": 44.0,
                "calibration_method": "none",
            }
        )
    return pd.DataFrame(rows)


def _write_checkpoints(directory: Path, predictions: pd.DataFrame) -> Path:
    """Write one fold file per (season, week), as a walk-forward run does."""
    directory.mkdir(parents=True)
    for (season, week), frame in predictions.groupby(["season", "week"]):
        payload = {
            "metrics": {
                "margin_model.best_iteration": 199,
                "total_model.best_iteration": 199,
                "margin_model.early_stopped": False,
                "total_model.early_stopped": False,
            },
            "predictions": frame.reset_index(drop=True),
        }
        joblib.dump(payload, directory / f"fold_{season}_w{week:02d}.joblib")
    return directory


def _write_run(root: Path, name: str, predictions: pd.DataFrame, seed: int) -> Path:
    """Write a run directory whose metadata names its checkpoint directory."""
    checkpoints = _write_checkpoints(root / "wf_checkpoints" / name, predictions)
    run_dir = root / name
    run_dir.mkdir()
    metadata = {
        "dataset_hash": "abc",
        "git_commit_hash": "0123456",
        "config": {
            "random_seed": seed,
            "recency_half_life_seasons": None,
            "checkpoint": {"dir": str(checkpoints)},
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir


def _loaded(tmp_path: Path, name: str, predictions: pd.DataFrame) -> run_comparison.LoadedRun:
    """Load a checkpoint directory written from ``predictions``."""
    directory = _write_checkpoints(tmp_path / name, predictions)
    return run_comparison.load_run(run_comparison.resolve_run(directory))


def test_per_game_scores_follow_the_documented_definitions() -> None:
    """Brier, pick correctness (ties incorrect) and pool ranks match a hand calculation."""
    games = run_comparison.per_game_scores(_predictions())
    by_id = games.set_index("game_id")

    assert by_id.loc["g1", "brier"] == pytest.approx(0.04)
    assert by_id.loc["g2", "brier"] == pytest.approx(0.09)
    assert by_id.loc["g3", "brier"] == pytest.approx(0.36)
    assert by_id["correct"].to_dict() == {
        "g1": 1.0,
        "g2": 1.0,
        "g3": 0.0,
        "g4": 0.0,
        "g5": 1.0,
        "g6": 0.0,
    }
    # Week 1 confidence |p - 0.5|: g1 0.3, g2 0.2, g3 0.1, so ranks 3, 2, 1.
    assert by_id.loc[["g1", "g2", "g3"], "rank"].tolist() == [3, 2, 1]
    assert by_id.loc[["g1", "g2", "g3"], "pool"].sum() == 5
    assert by_id.loc["g1", "log_loss"] == pytest.approx(-np.log(0.8))


def test_identical_runs_compare_to_zero(tmp_path: Path) -> None:
    """A run against itself differs by exactly zero in every window and column."""
    candidate = _loaded(tmp_path, "a", _predictions())
    reference = _loaded(tmp_path, "b", _predictions())

    report = run_comparison.compare_runs([candidate], [reference], resamples=50)

    for window in report["windows"].values():
        for values in window["contrast"].values():
            assert values == (0.0, 0.0, 0.0)
    assert report["market_view_identical"] is True


def test_windows_split_the_weeks_and_leave_out_the_postseason_from_weeks_3_18(
    tmp_path: Path,
) -> None:
    """Week 1, week 2, weeks 3-18 and all weeks count the fixture's games correctly."""
    run = _loaded(tmp_path, "a", _predictions())
    report = run_comparison.compare_runs([run], [run], resamples=10)

    games = {label: window["games"] for label, window in report["windows"].items()}
    assert games == {"week 1": 3, "week 2": 1, "weeks 3-18": 1, "all weeks": 6}


def test_a_better_candidate_shows_the_hand_computed_difference(tmp_path: Path) -> None:
    """Moving g3 from 0.6 to 0.4 lowers its Brier by 0.2 and makes it the only change."""
    candidate = _loaded(tmp_path, "a", _predictions({"g3": 0.4}))
    reference = _loaded(tmp_path, "b", _predictions())

    report = run_comparison.compare_runs([candidate], [reference], resamples=100)

    week1 = report["windows"]["week 1"]["contrast"]
    assert week1["brier"][0] == pytest.approx((0.16 - 0.36) / 3)
    assert week1["correct"][0] == 0.0
    assert week1["margin_ae"] == (0.0, 0.0, 0.0)


def test_two_seed_pairs_average_the_per_game_differences(tmp_path: Path) -> None:
    """With two pairs, the estimate is the mean of the two single-pair estimates."""
    reference = _loaded(tmp_path, "ref", _predictions())
    seed_a = _loaded(tmp_path, "a", _predictions(margin_shift=1.0))
    seed_b = _loaded(tmp_path, "b", _predictions(margin_shift=-3.0))

    both = run_comparison.compare_runs([seed_a, seed_b], [reference, reference], resamples=20)
    one_a = run_comparison.compare_runs([seed_a], [reference], resamples=20)
    one_b = run_comparison.compare_runs([seed_b], [reference], resamples=20)

    estimate = both["windows"]["all weeks"]["contrast"]["margin_ae"][0]
    expected = (
        one_a["windows"]["all weeks"]["contrast"]["margin_ae"][0]
        + one_b["windows"]["all weeks"]["contrast"]["margin_ae"][0]
    ) / 2
    assert estimate == pytest.approx(expected)


def test_runs_over_different_games_are_rejected(tmp_path: Path) -> None:
    """A paired comparison needs the same games on both sides."""
    candidate = _loaded(tmp_path, "a", _predictions())
    reference = _loaded(tmp_path, "b", _predictions().iloc[:-1])

    with pytest.raises(ValueError, match="different games"):
        run_comparison.compare_runs([candidate], [reference])
    with pytest.raises(ValueError, match="one reference run per candidate"):
        run_comparison.compare_runs([candidate, candidate], [reference])


def test_checkpoints_without_the_probability_views_are_rejected(tmp_path: Path) -> None:
    """Old checkpoints that predate the market view cannot be compared, and say why."""
    predictions = _predictions().drop(columns=["market_home_win_prob"])
    directory = _write_checkpoints(tmp_path / "old", predictions)

    with pytest.raises(ValueError, match="market_home_win_prob"):
        run_comparison.load_run(run_comparison.resolve_run(directory))


def test_a_path_that_is_not_a_run_is_rejected(tmp_path: Path) -> None:
    """Neither a run directory nor a checkpoint directory is an error, not an empty run."""
    with pytest.raises(ValueError, match="neither"):
        run_comparison.resolve_run(tmp_path)


def test_run_directories_bring_provenance_and_config_differences(tmp_path: Path) -> None:
    """A run directory resolves to its checkpoints and reports the settings that differ."""
    candidate = run_comparison.load_run(
        run_comparison.resolve_run(_write_run(tmp_path, "cand", _predictions(), seed=7))
    )
    reference = run_comparison.load_run(
        run_comparison.resolve_run(_write_run(tmp_path, "ref", _predictions(), seed=42))
    )

    report = run_comparison.compare_runs([candidate], [reference], resamples=10)

    assert report["config_differences"] == {"cand vs ref": {"random_seed": [7, 42]}}
    provenance = report["provenance"]["cand"]
    assert provenance["folds"] == 4
    assert provenance["dataset_hash"] == "abc"
    assert provenance["early_stopped"] is False
    assert provenance["best_iteration"] == ["margin_model=199", "total_model=199"]


def test_the_command_writes_its_reports_and_exits_zero(tmp_path: Path) -> None:
    """``nfl-predictor compare`` prints the tables and writes JSON and Markdown copies."""
    candidate = _write_run(tmp_path, "cand", _predictions({"g3": 0.4}), seed=42)
    reference = _write_run(tmp_path, "ref", _predictions(), seed=42)
    out_json = tmp_path / "out" / "compare.json"
    out_md = tmp_path / "out" / "compare.md"

    code = front_door.main(
        [
            "compare",
            "--candidate",
            str(candidate),
            "--reference",
            str(reference),
            "--resamples",
            "25",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert code == 0
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["resamples"] == 25
    assert set(report["windows"]) == {"week 1", "week 2", "weeks 3-18", "all weeks"}
    markdown = out_md.read_text(encoding="utf-8")
    assert "## weeks 3-18 (1 games)" in markdown
    assert "| candidate - reference |" in markdown


def test_the_command_exits_two_when_the_runs_cannot_be_compared(tmp_path: Path) -> None:
    """An unusable input is reported and the exit code is 2."""
    reference = _write_run(tmp_path, "ref", _predictions(), seed=42)

    assert command.main(["--candidate", str(tmp_path), "--reference", str(reference)]) == 2


# Two seasons with several games per week, so every season and week bucket has games and the
# per-week pool bootstrap has more than one week to resample in the bigger windows.
# (game_id, season, week, p, actual margin)
SEASON_GAMES = [
    ("2023_01_a", 2023, 1, 0.70, 7),
    ("2023_01_b", 2023, 1, 0.40, 3),
    ("2023_01_c", 2023, 1, 0.55, -2),
    ("2023_02_a", 2023, 2, 0.65, 10),
    ("2023_02_b", 2023, 2, 0.35, -4),
    ("2023_03_a", 2023, 3, 0.80, 14),
    ("2023_03_b", 2023, 3, 0.52, -1),
    ("2023_04_a", 2023, 4, 0.30, -6),
    ("2023_04_b", 2023, 4, 0.60, 0),
    ("2024_01_a", 2024, 1, 0.62, 3),
    ("2024_01_b", 2024, 1, 0.45, -7),
    ("2024_02_a", 2024, 2, 0.75, -3),
    ("2024_02_b", 2024, 2, 0.58, 6),
    ("2024_03_a", 2024, 3, 0.40, 2),
    ("2024_03_b", 2024, 3, 0.66, 9),
    ("2024_19_a", 2024, 19, 0.57, -10),
]
SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "run_comparison"
TMP_PLACEHOLDER = "<tmp>"


def _season_predictions(shift: float = 0.0, market: float = 0.5) -> pd.DataFrame:
    """Return prediction rows for ``SEASON_GAMES``; ``shift`` moves every probability."""
    rows = []
    for index, (game_id, season, week, p, margin) in enumerate(SEASON_GAMES):
        rows.append(
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "deterministic_home_win_prob": min(max(p + shift, 0.01), 0.99),
                "market_home_win_prob": market,
                "actual_home_win": int(margin > 0),
                "actual_margin": float(margin),
                "actual_total": 40.0 + index,
                "predicted_margin": float(margin) + (index % 5) - 2.0 + 3 * shift,
                "predicted_total": 44.0 - shift * 10,
                "calibration_method": "none",
            }
        )
    return pd.DataFrame(rows)


def _season_comparison(tmp_path: Path) -> dict[str, Any]:
    """Compare a shifted candidate with a reference on the two-season fixture."""
    candidate = _loaded(tmp_path, "cand", _season_predictions(shift=0.05, market=0.55))
    reference = _loaded(tmp_path, "ref", _season_predictions(market=0.55))
    return run_comparison.compare_runs([candidate], [reference], resamples=200)


def _without_tmp(text: str, tmp_path: Path) -> str:
    """Replace the test's temporary directory in ``text`` with a fixed placeholder."""
    return text.replace(str(tmp_path), TMP_PLACEHOLDER)


def _assert_contains(where: str, actual: object, expected: object) -> None:
    """Assert every key in ``expected`` is in ``actual`` with an exactly equal value."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), where
        for key, value in expected.items():
            assert key in actual, f"{where}: missing {key}"
            _assert_contains(f"{where}.{key}", actual[key], value)
    else:
        assert actual == expected, f"{where}: {actual!r} != {expected!r}"


def test_the_existing_comparison_report_is_unchanged(tmp_path: Path) -> None:
    """The window tables, provenance and JSON keys that ``compare`` wrote before stay exact.

    The snapshot was written from the report before the per-season view was added; the report
    may add sections after it and keys beside it, but every existing line and value is kept
    byte for byte, so the command still reproduces earlier reviewed rescores.
    """
    report = _season_comparison(tmp_path)
    lines = [_without_tmp(line, tmp_path) for line in run_comparison.format_report(report)]
    payload = json.loads(_without_tmp(json.dumps(report, indent=1, default=str), tmp_path))
    markdown_snapshot = SNAPSHOT_DIR / "compare_report.txt"
    json_snapshot = SNAPSHOT_DIR / "compare_report.json"
    if snapshots.updating():
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        markdown_snapshot.write_text("\n".join(lines) + "\n", encoding="utf-8")
        json_snapshot.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        return

    expected_lines = markdown_snapshot.read_text(encoding="utf-8").splitlines()
    assert lines[: len(expected_lines)] == expected_lines
    _assert_contains("report", payload, json.loads(json_snapshot.read_text(encoding="utf-8")))


def test_the_stability_view_splits_every_window_by_season() -> None:
    """Each window has an all-seasons row and one row per season with games in it."""
    stability = run_comparison.stability_report(_season_predictions(), resamples=50)

    games = {
        label: {season: row["games"] for season, row in rows.items()}
        for label, rows in stability["windows"].items()
    }
    assert games == {
        "week 1": {"all seasons": 5, "2023": 3, "2024": 2},
        "week 2": {"all seasons": 4, "2023": 2, "2024": 2},
        "weeks 3-18": {"all seasons": 6, "2023": 4, "2024": 2},
        "all weeks": {"all seasons": 16, "2023": 9, "2024": 7},
    }
    assert stability["resamples"] == 50
    assert stability["bootstrap_seed"] == run_comparison.DEFAULT_BOOTSTRAP_SEED


def test_a_season_row_matches_a_hand_calculation() -> None:
    """Season 2023, week 1: Brier, pick accuracy, pool points and the market contrast."""
    stability = run_comparison.stability_report(_season_predictions(), resamples=50)

    row = stability["windows"]["week 1"]["2023"]
    # p 0.70/0.40/0.55 against home results 1/1/0.
    assert row["brier"] == pytest.approx((0.09 + 0.36 + 0.3025) / 3)
    assert row["correct"] == pytest.approx(1 / 3)
    # Confidence ranks 3, 2, 1; only the most confident pick (rank 3) is right.
    assert row["pool"] == 3.0
    assert row["market_brier"] == pytest.approx(0.25)
    estimate, low, high = row["det_minus_market_brier"]
    assert estimate == pytest.approx((0.09 + 0.36 + 0.3025) / 3 - 0.25)
    assert low is not None
    assert high is not None
    assert low <= estimate <= high
    for column in ("log_loss", "margin_ae", "total_ae"):
        assert column in row


def test_the_all_seasons_row_is_the_comparison_window_row(tmp_path: Path) -> None:
    """One definition: a run's stability rows equal what ``compare`` reports for that run."""
    predictions = _season_predictions(market=0.55)
    stability = run_comparison.stability_report(predictions, resamples=200)
    report = _season_comparison(tmp_path)

    for label, window in report["windows"].items():
        all_seasons = dict(stability["windows"][label]["all seasons"])
        assert all_seasons.pop("games") == window["games"]
        assert all_seasons == window["runs"]["ref"]
        for season, row in window["seasons"].items():
            season_row = dict(stability["windows"][label][season])
            assert season_row.pop("games") == row["games"]
            assert row["runs"]["ref"] == season_row


def test_the_comparison_adds_per_season_rows_for_runs_and_the_contrast(tmp_path: Path) -> None:
    """Every window gets per-season run rows and a per-season paired contrast."""
    report = _season_comparison(tmp_path)

    week1 = report["windows"]["week 1"]["seasons"]
    assert sorted(week1) == ["2023", "2024"]
    assert week1["2023"]["games"] == 3
    assert set(week1["2023"]["runs"]) == {"cand", "ref"}
    # The candidate adds 0.05 to every probability; in 2023 week 1 the results are 1, 1, 0.
    reference = (0.09 + 0.36 + 0.3025) / 3
    candidate = (0.25**2 + 0.55**2 + 0.60**2) / 3
    assert week1["2023"]["contrast"]["brier"][0] == pytest.approx(candidate - reference)
    all_weeks = report["windows"]["all weeks"]["seasons"]
    assert sum(row["games"] for row in all_weeks.values()) == 16


def test_a_single_week_has_no_pool_interval(tmp_path: Path) -> None:
    """Pool points resample weeks, so one week alone gets an estimate and no interval."""
    report = _season_comparison(tmp_path)

    season = report["windows"]["week 1"]["seasons"]["2023"]
    estimate, low, high = season["contrast"]["pool"]
    assert (low, high) == (None, None)
    assert estimate == season["runs"]["cand"]["pool"] - season["runs"]["ref"]["pool"]
    several = report["windows"]["weeks 3-18"]["seasons"]["2023"]["contrast"]["pool"]
    assert None not in several


def _one_game_season(predictions: pd.DataFrame) -> pd.DataFrame:
    """Drop a game so season 2024's week 1 holds a single game."""
    return predictions[predictions["game_id"] != "2024_01_b"].reset_index(drop=True)


def test_a_single_game_row_has_no_market_interval() -> None:
    """A run's season row with one game gets the det - market estimate and no interval."""
    stability = run_comparison.stability_report(
        _one_game_season(_season_predictions()), resamples=50
    )

    row = stability["windows"]["week 1"]["2024"]
    assert row["games"] == 1
    estimate, low, high = row["det_minus_market_brier"]
    assert (low, high) == (None, None)
    assert estimate == pytest.approx(0.38**2 - 0.5**2)
    assert None not in stability["windows"]["week 1"]["2023"]["det_minus_market_brier"]


def test_a_single_game_contrast_has_no_intervals(tmp_path: Path) -> None:
    """A paired season row with one game keeps each estimate and prints ``[n/a]``."""
    candidate = _loaded(
        tmp_path, "cand", _one_game_season(_season_predictions(shift=0.05, market=0.55))
    )
    reference = _loaded(tmp_path, "ref", _one_game_season(_season_predictions(market=0.55)))
    report = run_comparison.compare_runs([candidate], [reference], resamples=50)

    contrast = report["windows"]["week 1"]["seasons"]["2024"]["contrast"]
    for column in run_comparison.LOSS_COLUMNS:
        assert contrast[column][1:] == (None, None), column
    assert contrast["brier"][0] == pytest.approx(0.33**2 - 0.38**2)
    row = next(
        line
        for line in run_comparison.format_report(report)
        if line.startswith("| candidate - reference | 2024 | 1 | ")
    )
    brier_cell = row.split(" | ")[3]
    assert brier_cell == f"{0.33**2 - 0.38**2:+.5f} [n/a]"


def test_the_comparison_report_ends_with_the_season_tables(tmp_path: Path) -> None:
    """The Markdown gains a stability section with run rows and contrast rows per season."""
    lines = run_comparison.format_report(_season_comparison(tmp_path))

    start = lines.index("## Stability by season")
    section = lines[start:]
    assert "### week 1" in section
    assert any(line.startswith("| cand | 2023 | 3 | ") for line in section)
    assert any(line.startswith("| ref | 2024 | 2 | ") for line in section)
    assert any(line.startswith("| candidate - reference | 2023 | 3 | ") for line in section)
    assert any("[n/a]" in line for line in section)


def test_the_stability_table_formats_one_row_per_season() -> None:
    """A single run's view prints an all-seasons row, then each season, per window."""
    stability = run_comparison.stability_report(_season_predictions(), resamples=50)

    lines = run_comparison.format_stability(stability)

    assert lines[0] == "## Stability by season"
    week1 = lines.index("### week 1")
    assert lines[week1 + 2].startswith("| season | games | det Brier |")
    assert lines[week1 + 4].startswith("| all seasons | 5 | ")
    assert lines[week1 + 5].startswith("| 2023 | 3 | 0.25083 | ")
    assert lines[week1 + 6].startswith("| 2024 | 2 | ")


def _command_outputs(tmp_path: Path, *extra: str) -> tuple[int, list[str], dict[str, Any]]:
    """Run ``compare`` on the two-season fixture's run directories; return its written reports.

    The candidate shifts every probability by 0.05; both runs carry a 0.55 market. ``extra``
    adds options. The Markdown lines and the JSON have the temporary directory replaced.
    """
    candidate = _write_run(tmp_path, "cand", _season_predictions(shift=0.05, market=0.55), 42)
    reference = _write_run(tmp_path, "ref", _season_predictions(market=0.55), 42)
    out_json = tmp_path / "out" / "compare.json"
    out_md = tmp_path / "out" / "compare.md"
    code = command.main(
        [
            "--candidate",
            str(candidate),
            "--reference",
            str(reference),
            "--resamples",
            "200",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            *extra,
        ]
    )
    if code != 0:
        return code, [], {}
    lines = _without_tmp(out_md.read_text(encoding="utf-8"), tmp_path).splitlines()
    payload = json.loads(_without_tmp(out_json.read_text(encoding="utf-8"), tmp_path))
    return code, lines, payload


def test_the_command_output_without_a_market_run_is_unchanged(tmp_path: Path) -> None:
    """Without ``--market-from`` the written Markdown and JSON stay exactly as before.

    The snapshot holds the command's complete output from before the option existed: every
    Markdown line byte for byte, and the JSON with exactly the same keys (floats to the
    snapshot tolerance), so no key is added when each run's own market rows are used.
    """
    code, lines, payload = _command_outputs(tmp_path)
    markdown_snapshot = SNAPSHOT_DIR / "compare_command.txt"
    json_snapshot = SNAPSHOT_DIR / "compare_command.json"
    if snapshots.updating():
        markdown_snapshot.write_text("\n".join(lines) + "\n", encoding="utf-8")
        json_snapshot.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        return

    assert code == 0
    assert lines == markdown_snapshot.read_text(encoding="utf-8").splitlines()
    snapshots.assert_json_match(
        "report", payload, json.loads(json_snapshot.read_text(encoding="utf-8"))
    )
