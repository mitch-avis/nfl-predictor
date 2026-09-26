"""Tests for the paired walk-forward comparison and the ``compare`` command.

The fixtures are hand-built fold checkpoints: a few games with known probabilities and results,
so every metric can be checked by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

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
