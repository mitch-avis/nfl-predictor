"""Tests for the betting-label test (``.agents/sos_port/betting_labels.py``)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from tests.sos_port_scripts import load_sos_port_script

from nfl_predictor.reporting import betting_report

if TYPE_CHECKING:
    from pathlib import Path

labels = load_sos_port_script("betting_labels")

_SIGMA = 13.5


def _claims(
    p_model: list[float], p_spread: list[float], p_ml: list[float], outcome: list[int]
) -> pd.DataFrame:
    """Return scored games with the given probabilities, one season, ids in order."""
    count = len(p_model)
    frame = pd.DataFrame(
        {
            "game_id": [f"2025_01_G{index}" for index in range(count)],
            "season": [2025] * count,
            "p_model": p_model,
            "p_spread": p_spread,
            "p_ml": p_ml,
            "actual_home_win": outcome,
        }
    )
    frame["edge"] = frame["p_model"] - frame["p_ml"]
    frame["gap"] = frame["p_spread"] - frame["p_ml"]
    frame["beyond"] = frame["p_model"] - frame["p_spread"]
    return frame


def test_add_claims_q_beyond_is_moneyline_plus_model_minus_spread() -> None:
    # Arrange
    games = _claims([0.70, 0.30], [0.60, 0.45], [0.55, 0.40], [1, 0])

    # Act
    claimed = labels.add_claims(games)

    # Assert
    np.testing.assert_allclose(
        claimed["q_beyond"], [0.55 + 0.70 - 0.60, 0.40 + 0.30 - 0.45], rtol=0, atol=1e-15
    )


def test_brier_difference_is_beyond_claim_minus_model_claim_per_game() -> None:
    # Arrange
    q_beyond = np.array([0.6, 0.2])
    p_model = np.array([0.7, 0.4])
    outcome = np.array([1.0, 0.0])

    # Act
    difference = labels.brier_difference(q_beyond, p_model, outcome)

    # Assert
    np.testing.assert_allclose(difference, [0.16 - 0.09, 0.04 - 0.16], rtol=0, atol=1e-15)


def test_brier_difference_out_of_range_claim_is_not_clipped() -> None:
    # Arrange
    q_beyond = np.array([1.2])
    p_model = np.array([0.9])
    outcome = np.array([1.0])

    # Act
    difference = labels.brier_difference(q_beyond, p_model, outcome)

    # Assert
    assert difference[0] == pytest.approx(0.04 - 0.01, abs=1e-15)


def test_out_of_range_count_counts_claims_outside_zero_to_one() -> None:
    # Arrange
    q_beyond = np.array([1.2, -0.1, 0.5, 0.0, 1.0])

    # Act
    count = labels.out_of_range_count(q_beyond)

    # Assert
    assert count == 2


def test_log_loss_difference_clips_each_probability() -> None:
    # Arrange
    q_beyond = np.array([1.2, -0.1])
    p_model = np.array([0.5, 0.5])
    outcome = np.array([1.0, 1.0])

    # Act
    difference = labels.log_loss_difference(q_beyond, p_model, outcome)

    # Assert
    expected = [-np.log(1 - 1e-6) - np.log(2), -np.log(1e-6) - np.log(2)]
    np.testing.assert_allclose(difference, expected, rtol=1e-12, atol=0)


def _seed_frame(q_beyond: list[float], p_model: list[float], ids: list[str]) -> pd.DataFrame:
    """Return one seed's governing games with the claims given, all home wins."""
    count = len(ids)
    p_ml = np.full(count, 0.5)
    p_model_array = np.array(p_model)
    p_spread = p_ml + p_model_array - np.array(q_beyond)
    frame = pd.DataFrame(
        {
            "game_id": ids,
            "season": [2025] * count,
            "p_model": p_model_array,
            "p_spread": p_spread,
            "p_ml": p_ml,
            "actual_home_win": [1] * count,
        }
    )
    frame["edge"] = frame["p_model"] - frame["p_ml"]
    frame["gap"] = frame["p_spread"] - frame["p_ml"]
    frame["beyond"] = frame["p_model"] - frame["p_spread"]
    return labels.add_claims(frame)


def test_pair_seeds_averages_each_games_difference_over_the_seeds() -> None:
    # Arrange
    ids = ["2025_01_A_B", "2025_01_C_D"]
    first = _seed_frame([0.6, 0.8], [0.7, 0.7], ids)
    second = _seed_frame([0.5, 0.9], [0.6, 0.8], ids)

    # Act
    paired = labels.pair_seeds({42: first, 7: second})

    # Assert
    seed_42 = [(0.4**2 - 0.3**2), (0.2**2 - 0.3**2)]
    seed_7 = [(0.5**2 - 0.4**2), (0.1**2 - 0.2**2)]
    np.testing.assert_allclose(paired["brier_d_42"], seed_42, rtol=0, atol=1e-15)
    np.testing.assert_allclose(paired["brier_d_7"], seed_7, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        paired["brier_d"], (np.array(seed_42) + np.array(seed_7)) / 2, rtol=0, atol=1e-15
    )


def test_pair_seeds_matches_games_by_id_not_by_row_order() -> None:
    # Arrange
    first = _seed_frame([0.6, 0.8], [0.7, 0.7], ["2025_01_A_B", "2025_01_C_D"])
    second = _seed_frame([0.9, 0.5], [0.8, 0.6], ["2025_01_C_D", "2025_01_A_B"])

    # Act
    paired = labels.pair_seeds({42: first, 7: second}).set_index("game_id")

    # Assert
    assert paired.loc["2025_01_A_B", "brier_d_7"] == pytest.approx(0.5**2 - 0.4**2, abs=1e-15)


def test_pair_seeds_mismatched_games_raise() -> None:
    # Arrange
    first = _seed_frame([0.6, 0.8], [0.7, 0.7], ["2025_01_A_B", "2025_01_C_D"])
    second = _seed_frame([0.6, 0.8], [0.7, 0.7], ["2025_01_A_B", "2025_01_E_F"])

    # Act & Assert
    with pytest.raises(ValueError, match="same games"):
        labels.pair_seeds({42: first, 7: second})


def test_summarize_difference_bootstraps_the_seed_averaged_differences() -> None:
    # Arrange
    rng = np.random.default_rng(12)
    differences = rng.normal(0.0, 0.05, size=40)

    # Act
    summary = labels.summarize_difference(differences)

    # Assert
    draws = np.random.default_rng(0)
    means = [differences[draws.integers(0, 40, size=40)].mean() for _ in range(5_000)]
    low, high = np.percentile(means, [2.5, 97.5])
    assert summary.games == 40
    assert summary.mean == pytest.approx(differences.mean(), abs=1e-15)
    assert (summary.low, summary.high) == pytest.approx((low, high), abs=1e-15)


def test_summarize_difference_two_calls_give_the_same_interval() -> None:
    # Arrange
    differences = np.random.default_rng(3).normal(0.0, 0.05, size=30)

    # Act
    first, second = (
        labels.summarize_difference(differences),
        labels.summarize_difference(differences),
    )

    # Assert
    assert first == second


@pytest.mark.parametrize(
    ("low", "high", "branch", "recommend_beyond"),
    [
        (-0.02, -0.001, "below zero", True),
        (-0.01, 0.01, "contains zero", False),
        (0.001, 0.02, "above zero", False),
        (-0.01, 0.0, "contains zero", False),
    ],
)
def test_apply_rule_reads_where_the_interval_lies(
    low: float, high: float, branch: str, *, recommend_beyond: bool
) -> None:
    # Arrange
    difference = labels.Difference(games=10, mean=(low + high) / 2, low=low, high=high)

    # Act
    outcome = labels.apply_rule(difference)

    # Assert
    assert (outcome.branch, outcome.recommend_beyond) == (branch, recommend_beyond)


def test_label_changes_use_the_reports_mapping() -> None:
    # Arrange
    edge = [0.05, 0.015, 0.01, -0.12, 0.0]
    beyond = [0.03, 0.009, -0.01, -0.11, 0.0]
    games = pd.DataFrame({"edge": edge, "beyond": beyond})

    # Act
    changes = labels.label_changes(games)

    # Assert
    action = sum(
        betting_report._edge_to_action(e) != betting_report._edge_to_action(b)
        for e, b in zip(edge, beyond, strict=True)
    )
    confidence = sum(
        betting_report._edge_to_confidence_1_to_10(e)
        != betting_report._edge_to_confidence_1_to_10(b)
        for e, b in zip(edge, beyond, strict=True)
    )
    assert (changes.games, changes.action, changes.confidence, changes.side) == (
        5,
        action,
        confidence,
        1,
    )
    assert (action, confidence) == (1, 2)


def test_label_changes_zero_edge_points_home_like_the_report() -> None:
    # Arrange
    games = pd.DataFrame({"edge": [0.0, -0.01], "beyond": [-0.001, 0.0]})

    # Act
    changes = labels.label_changes(games)

    # Assert
    assert changes.side == 2


def test_top_decile_takes_the_ceiling_of_a_tenth_by_the_seed_mean_magnitude() -> None:
    # Arrange
    paired = pd.DataFrame(
        {
            "edge_42": [0.01, -0.30, 0.02, 0.03, 0.20, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
            "edge_7": [0.01, -0.10, 0.02, 0.03, 0.10, 0.04, 0.05, 0.06, 0.07, 0.08, 0.30],
        }
    )

    # Act
    top = labels.top_decile(paired, "edge", seeds=(42, 7))

    # Assert
    assert sorted(top.index.tolist()) == [1, 10]


# The whole run, on synthetic fold checkpoints.


def _fold_games(season: int, week: int, sigma: float) -> pd.DataFrame:
    """Return three fold rows whose floor matches their predicted margins."""
    margins = np.array([3.0, -6.5, 10.0])
    return pd.DataFrame(
        {
            "game_id": [f"{season}_{week:02d}_{name}" for name in ("A_B", "C_D", "E_F")],
            "season": [season] * 3,
            "week": [week] * 3,
            "game_type": ["REG"] * 3,
            "home_moneyline": [-150.0, 220.0, -400.0],
            "away_moneyline": [130.0, -260.0, 320.0],
            "home_spread": [-3.0, 6.5, -9.5],
            "predicted_margin": margins,
            "deterministic_home_win_prob": norm.cdf(margins / sigma),
            "actual_home_win": [1, 0, 0],
        }
    )


def _write_run(
    directory: Path, seed: int, folds: list[tuple[int, int]], stored_sigma: float = _SIGMA
) -> Path:
    """Write a walk-forward's checkpoints: a manifest with its seed and one file per fold.

    ``stored_sigma`` is the sigma behind each fold's stored probability; any other value than
    ``_SIGMA`` makes the floor fail to reproduce.
    """
    directory.mkdir()
    manifest = {"config": {"random_seed": seed}, "fingerprint": directory.name}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for season, week in folds:
        payload = {
            "fingerprint": directory.name,
            "metrics": {"floor_sigma": _SIGMA},
            "predictions": _fold_games(season, week, stored_sigma),
        }
        joblib.dump(payload, directory / f"fold_{season}_w{week:02d}.joblib")
    return directory


def _write_metadata(path: Path, derived: dict[int, int]) -> Path:
    """Write market-lines metadata giving each season's count of map-derived moneylines."""
    seasons = {
        str(season): {
            "games": 3,
            "moneylines_from_map": by_map,
            "moneylines_from_fixed_conversion": 0,
        }
        for season, by_map in derived.items()
    }
    path.write_text(json.dumps({"seasons": seasons}), encoding="utf-8")
    return path


def _argv(tmp_path: Path, seed_7_sigma: float = _SIGMA) -> list[str]:
    """Return the arguments of a run on two synthetic seeds of two folds each."""
    folds = [(2021, 1), (2025, 1)]
    first = _write_run(tmp_path / "run42", 42, folds)
    second = _write_run(tmp_path / "run7", 7, folds, stored_sigma=seed_7_sigma)
    metadata = _write_metadata(tmp_path / "metadata.json", {2021: 3, 2025: 0})
    earlier = _write_metadata(tmp_path / "earlier.json", {2021: 3, 2025: 0})
    return [
        "--checkpoint-dir",
        str(first),
        "--checkpoint-dir",
        str(second),
        "--expected-folds",
        "2",
        "--market-lines-metadata",
        str(metadata),
        "--earlier-market-lines-metadata",
        str(earlier),
        "--out",
        str(tmp_path / "BETTING_LABELS.md"),
    ]


def test_main_floor_that_does_not_reproduce_stops_the_run(tmp_path: Path) -> None:
    # Arrange
    argv = _argv(tmp_path, seed_7_sigma=_SIGMA + 0.5)

    # Act & Assert
    with pytest.raises(ValueError, match="does not reproduce"):
        labels.main(argv)


def test_main_floor_that_does_not_reproduce_writes_no_report(tmp_path: Path) -> None:
    # Arrange
    argv = _argv(tmp_path, seed_7_sigma=_SIGMA + 0.5)

    # Act
    with pytest.raises(ValueError, match="does not reproduce"):
        labels.main(argv)

    # Assert
    assert not (tmp_path / "BETTING_LABELS.md").exists()


def test_main_checkpoints_of_another_seed_raise(tmp_path: Path) -> None:
    # Arrange
    argv = _argv(tmp_path)
    argv[1], argv[3] = argv[3], argv[1]

    # Act & Assert
    with pytest.raises(ValueError, match="seed"):
        labels.main(argv)


def test_main_writes_the_governing_games_and_the_rules_outcome(tmp_path: Path) -> None:
    # Arrange
    argv = _argv(tmp_path)

    # Act
    labels.main(argv)

    # Assert
    record = json.loads((tmp_path / "betting_labels.json").read_text(encoding="utf-8"))
    assert record["governing"]["seasons"] == [2025]
    assert record["governing"]["games"] == 3
    assert record["rule"]["branch"] in {"below zero", "contains zero", "above zero"}


def test_main_count_other_than_the_earlier_builds_is_explained_in_the_report(
    tmp_path: Path,
) -> None:
    # Arrange
    argv = _argv(tmp_path)

    # Act
    labels.main(argv)

    # Assert
    report = (tmp_path / "BETTING_LABELS.md").read_text(encoding="utf-8")
    assert "differs from the 815" in report


def test_main_records_the_hash_of_the_code_and_every_input(tmp_path: Path) -> None:
    # Arrange
    argv = _argv(tmp_path)

    # Act
    labels.main(argv)

    # Assert
    provenance = json.loads((tmp_path / "betting_labels.json").read_text(encoding="utf-8"))[
        "provenance"
    ]
    assert set(provenance["code"]) >= {
        "betting_labels.py",
        "betting_edges.py",
        "betting_report.py",
    }
    assert set(provenance["inputs"]) == {
        str(tmp_path / "run42"),
        str(tmp_path / "run7"),
        str(tmp_path / "metadata.json"),
        str(tmp_path / "earlier.json"),
    }
    assert {"git_commit", "command", "seconds"} <= set(provenance)


def test_main_two_runs_write_the_same_decisions_and_scores(tmp_path: Path) -> None:
    # Arrange
    argv = _argv(tmp_path)
    labels.main(argv)
    first = json.loads((tmp_path / "betting_labels.json").read_text(encoding="utf-8"))

    # Act
    labels.main(argv)

    # Assert
    second = json.loads((tmp_path / "betting_labels.json").read_text(encoding="utf-8"))
    first.pop("provenance")
    second.pop("provenance")
    assert first == second
