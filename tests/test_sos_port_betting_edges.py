"""Tests for the betting-edge decomposition (``.agents/sos_port/betting_edges.py``)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from tests.sos_port_scripts import load_sos_port_script

if TYPE_CHECKING:
    from pathlib import Path

edges = load_sos_port_script("betting_edges")

_SIGMA = 13.5


def _games(**overrides: list[object]) -> pd.DataFrame:
    """Return three regular-season fold rows whose floor matches their predicted margins."""
    margins = np.array([3.0, -6.5, 10.0])
    columns: dict[str, list[object]] = {
        "game_id": ["2010_01_A_B", "2010_01_C_D", "2010_01_E_F"],
        "season": [2010, 2010, 2010],
        "week": [1, 1, 1],
        "game_type": ["REG", "REG", "REG"],
        "home_moneyline": [-150.0, 220.0, -400.0],
        "away_moneyline": [130.0, -260.0, 320.0],
        "home_spread": [-3.0, 6.5, -9.5],
        "predicted_margin": margins.tolist(),
        "floor_sigma": [_SIGMA] * 3,
        "deterministic_home_win_prob": norm.cdf(margins / _SIGMA).tolist(),
        "actual_home_win": [1, 0, 1],
    }
    columns.update(overrides)
    return pd.DataFrame(columns)


def _decomposed(edge: list[float], gap: list[float]) -> pd.DataFrame:
    """Return decomposed games with the given edges and gaps, the rest of the edge beyond."""
    edge_array, gap_array = np.array(edge), np.array(gap)
    p_ml = np.full(len(edge), 0.5)
    return pd.DataFrame(
        {
            "p_ml": p_ml,
            "p_spread": p_ml + gap_array,
            "p_model": p_ml + edge_array,
            "edge": edge_array,
            "gap": gap_array,
            "beyond": edge_array - gap_array,
            "actual_home_win": np.arange(len(edge)) % 2,
        }
    )


def test_decompose_edge_is_the_sum_of_beyond_and_gap() -> None:
    # Arrange
    games = _games()

    # Act
    decomposed = edges.decompose(games)

    # Assert
    np.testing.assert_allclose(
        decomposed["edge"], decomposed["beyond"] + decomposed["gap"], rtol=0, atol=1e-15
    )


def test_decompose_spread_probability_is_the_floor_at_the_market_margin() -> None:
    # Arrange
    games = _games()

    # Act
    decomposed = edges.decompose(games)

    # Assert
    expected = norm.cdf(np.array([3.0, -6.5, 9.5]) / _SIGMA)
    np.testing.assert_allclose(decomposed["p_spread"], expected, rtol=0, atol=1e-15)


def test_decompose_moneyline_probability_removes_the_vig() -> None:
    # Arrange
    games = _games()

    # Act
    decomposed = edges.decompose(games)

    # Assert
    home_raw, away_raw = 150.0 / 250.0, 100.0 / 230.0
    assert decomposed["p_ml"].iloc[0] == pytest.approx(home_raw / (home_raw + away_raw), abs=1e-15)


def test_decompose_drops_games_without_both_moneylines_or_a_spread() -> None:
    # Arrange
    games = _games(
        away_moneyline=[float("nan"), -260.0, 320.0], home_spread=[-3.0, float("nan"), -9.5]
    )

    # Act
    decomposed = edges.decompose(games)

    # Assert
    assert decomposed["game_id"].tolist() == ["2010_01_E_F"]


def test_decompose_keeps_only_regular_season_games_from_2007_to_2025() -> None:
    # Arrange
    games = _games(game_type=["REG", "WC", "REG"], season=[2010, 2010, 2026])

    # Act
    decomposed = edges.decompose(games)

    # Assert
    assert decomposed["game_id"].tolist() == ["2010_01_A_B"]


def test_check_floor_reproduction_matching_floor_returns_the_largest_difference() -> None:
    # Arrange
    games = _games()

    # Act
    difference = edges.check_floor_reproduction(games)

    # Assert
    assert difference <= 1e-12


def test_check_floor_reproduction_mismatch_raises() -> None:
    # Arrange
    games = _games(floor_sigma=[_SIGMA, _SIGMA, _SIGMA + 0.5])

    # Act & Assert
    with pytest.raises(ValueError, match="does not reproduce"):
        edges.check_floor_reproduction(games)


def test_top_decile_takes_the_ceiling_of_a_tenth_by_absolute_edge() -> None:
    # Arrange
    decomposed = _decomposed(
        edge=[0.01, -0.30, 0.02, 0.03, 0.20, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
        gap=[0.0] * 11,
    )

    # Act
    top = edges.top_decile(decomposed)

    # Assert
    assert sorted(top["edge"].tolist()) == [-0.30, 0.20]


def test_summarize_gap_dominated_top_decile_recommends_beyond() -> None:
    # Arrange
    decomposed = _decomposed(edge=[0.10, -0.08, 0.01, 0.01], gap=[0.09, -0.07, 0.0, 0.0])

    # Act
    summary = edges.summarize(decomposed)

    # Assert
    assert summary.share_gap_dominates == 1.0
    assert summary.recommend_beyond


def test_summarize_half_gap_dominated_top_decile_keeps_the_edge() -> None:
    # Arrange
    edge = [0.10, -0.09] + [0.01] * 18
    gap = [0.09, -0.01] + [0.0] * 18

    # Act
    summary = edges.summarize(_decomposed(edge=edge, gap=gap))

    # Assert
    assert summary.share_gap_dominates == 0.5
    assert not summary.recommend_beyond


def test_paired_bootstrap_interval_constant_difference_collapses_to_it() -> None:
    # Arrange
    differences = np.full(50, 0.01)

    # Act
    low, high = edges.paired_bootstrap_interval(differences, resamples=200, seed=0)

    # Assert
    assert (low, high) == pytest.approx((0.01, 0.01), abs=1e-15)


def _write_fold(
    directory: Path, season: int, week: int, sigma: float, fingerprint: str | None = None
) -> None:
    """Write one fold checkpoint whose metrics carry ``sigma``.

    Its fingerprint is the directory's name, as in a walk-forward's checkpoint directory, unless
    ``fingerprint`` says otherwise.
    """
    payload = {
        "fingerprint": directory.name if fingerprint is None else fingerprint,
        "metrics": {"floor_sigma": sigma},
        "predictions": _games(week=[week] * 3, season=[season] * 3),
    }
    joblib.dump(payload, directory / f"fold_{season}_w{week:02d}.joblib")


def _write_metadata(path: Path, derived: dict[int, tuple[int, int]]) -> Path:
    """Write market-lines metadata giving each season's (map, fixed) derived moneyline counts."""
    seasons = {
        str(season): {"moneylines_from_map": by_map, "moneylines_from_fixed_conversion": fixed}
        for season, (by_map, fixed) in derived.items()
    }
    path.write_text(json.dumps({"seasons": seasons}), encoding="utf-8")
    return path


def test_load_fold_games_takes_each_weeks_sigma_from_its_metrics(tmp_path: Path) -> None:
    # Arrange
    _write_fold(tmp_path, 2010, 1, 12.0)
    _write_fold(tmp_path, 2010, 2, 14.0)

    # Act
    games = edges.load_fold_games(tmp_path, expected_folds=2)

    # Assert
    assert games.groupby("week")["floor_sigma"].unique().map(list).to_dict() == {
        1: [12.0],
        2: [14.0],
    }


def test_load_fold_games_unexpected_fold_count_raises(tmp_path: Path) -> None:
    # Arrange
    _write_fold(tmp_path, 2010, 1, 12.0)

    # Act & Assert
    with pytest.raises(ValueError, match="Expected 2 fold checkpoints"):
        edges.load_fold_games(tmp_path, expected_folds=2)


def test_render_report_gap_dominated_recommends_beyond_and_names_the_command(
    tmp_path: Path,
) -> None:
    # Arrange
    summary = edges.summarize(
        _decomposed(edge=[0.10, -0.08, 0.01, 0.01], gap=[0.09, -0.07, 0.0, 0.0])
    )
    inputs = edges.ReportInputs(
        checkpoint_dir=tmp_path,
        folds=1,
        fold_rows=4,
        floor_difference=0.0,
        metadata_path=tmp_path / "market_lines_metadata.json",
    )

    # Act
    report = edges.render_report([(edges.ALL_GAMES, summary)], inputs)

    # Assert
    assert ".agents/sos_port/betting_edges.py" in report
    assert edges.RECOMMEND_BEYOND in report


def test_main_writes_every_group_of_the_scored_games(tmp_path: Path) -> None:
    # Arrange
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    _write_fold(checkpoints, 2025, 1, _SIGMA)
    metadata = _write_metadata(tmp_path / "metadata.json", {2025: (0, 0)})
    out = tmp_path / "BETTING_EDGES.md"
    argv = ["--checkpoint-dir", str(checkpoints), "--expected-folds", "1", "--out", str(out)]

    # Act
    edges.main([*argv, "--market-lines-metadata", str(metadata)])

    # Assert
    assert "| Scored games | 3 | 3 | 3 |" in out.read_text(encoding="utf-8")


def test_load_fold_games_accepts_the_full_fingerprint_the_directory_abbreviates(
    tmp_path: Path,
) -> None:
    # Arrange
    checkpoints = tmp_path / "df20b775abf7f2657d13"
    checkpoints.mkdir()
    _write_fold(checkpoints, 2010, 1, 12.0, fingerprint="df20b775abf7f2657d13d05694d016dd")

    # Act
    games = edges.load_fold_games(checkpoints, expected_folds=1)

    # Assert
    assert len(games) == 3


def test_load_fold_games_foreign_fingerprint_raises(tmp_path: Path) -> None:
    # Arrange
    _write_fold(tmp_path, 2010, 1, 12.0, fingerprint="another_run")

    # Act & Assert
    with pytest.raises(ValueError, match="fingerprint"):
        edges.load_fold_games(tmp_path, expected_folds=1)


def test_real_moneyline_seasons_skip_seasons_with_derived_moneylines(tmp_path: Path) -> None:
    # Arrange
    metadata = _write_metadata(
        tmp_path / "metadata.json",
        {2006: (0, 0), 2021: (285, 0), 2022: (0, 0), 2023: (0, 3), 2025: (0, 0)},
    )

    # Act
    seasons = edges.real_moneyline_seasons(metadata)

    # Assert
    assert seasons == [2022, 2025]


def test_summary_groups_add_the_real_moneyline_seasons_and_the_last_season() -> None:
    # Arrange
    decomposed = _decomposed(
        edge=[0.10, -0.08, 0.05, 0.02, 0.03, -0.04], gap=[0.01, 0.02, -0.01, 0.0, 0.01, 0.02]
    ).assign(season=[2021, 2021, 2024, 2024, 2025, 2025])

    # Act
    groups = edges.summary_groups(decomposed, real_seasons=[2022, 2024, 2025])

    # Assert
    assert [(name, summary.games) for name, summary in groups] == [
        ("All scored games", 6),
        ("Real moneyline pairs (2022, 2024, 2025)", 4),
        ("2025 alone", 2),
    ]


def test_summary_groups_skip_a_group_without_games() -> None:
    # Arrange
    decomposed = _decomposed(edge=[0.10, -0.08, 0.05], gap=[0.01, 0.02, -0.01]).assign(
        season=[2021, 2021, 2021]
    )

    # Act
    groups = edges.summary_groups(decomposed, real_seasons=[2022])

    # Assert
    assert [name for name, _ in groups] == ["All scored games"]
