"""Tests for the shared pieces of the rating-definition tests (``.agents/sos_port``)."""

import json
import re
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from scipy import stats

from tests.sos_port_scripts import load_sos_port_script

if TYPE_CHECKING:
    from pathlib import Path

common = load_sos_port_script("rating_tests_common")


def _constant_by_season(values: dict[int, tuple[float, int]]) -> pl.DataFrame:
    """Return one row per game: each season's difference repeated over its game count."""
    rows = [
        {"season": season, "difference": difference}
        for season, (difference, games) in values.items()
        for _ in range(games)
    ]
    return pl.DataFrame(rows, schema={"season": pl.Int64, "difference": pl.Float64})


def test_comparison_confidence_three_comparisons_is_bonferroni_level() -> None:
    # Act
    level = common.comparison_confidence(3)

    # Assert
    assert level == pytest.approx(1.0 - 0.05 / 3.0)


def test_season_bootstrap_constant_within_seasons_matches_weighted_resampled_means() -> None:
    # Arrange
    values = {2001: (0.5, 3), 2002: (-1.0, 5), 2003: (2.0, 2), 2004: (0.25, 4)}
    differences = _constant_by_season(values)
    seasons = sorted(values)
    draws = common.season_draws(len(seasons), resamples=400, seed=3)
    expected = np.array(
        [
            sum(values[seasons[i]][0] * values[seasons[i]][1] for i in row)
            / sum(values[seasons[i]][1] for i in row)
            for row in draws
        ]
    )
    lower, upper = np.quantile(expected, [0.025, 0.975])

    # Act
    interval = common.season_bootstrap(differences, seasons, draws, 0.95)

    # Assert
    assert interval.lower == pytest.approx(lower, abs=1e-12)
    assert interval.upper == pytest.approx(upper, abs=1e-12)


def test_season_bootstrap_estimate_is_mean_over_every_game() -> None:
    # Arrange
    differences = _constant_by_season({2001: (1.0, 1), 2002: (-2.0, 3)})
    draws = common.season_draws(2, resamples=50, seed=0)

    # Act
    interval = common.season_bootstrap(differences, [2001, 2002], draws, 0.95)

    # Assert
    assert interval.estimate == pytest.approx(-5.0 / 4.0)


def test_season_bootstrap_season_without_games_raises() -> None:
    # Arrange
    differences = _constant_by_season({2001: (1.0, 2)})
    draws = common.season_draws(2, resamples=10, seed=0)

    # Act & Assert
    with pytest.raises(ValueError, match="2002"):
        common.season_bootstrap(differences, [2001, 2002], draws, 0.95)


def test_season_draws_same_seed_gives_same_matrix() -> None:
    # Arrange
    first = common.season_draws(19, resamples=20, seed=0)

    # Act
    second = common.season_draws(19, resamples=20, seed=0)

    # Assert
    assert np.array_equal(first, second)
    assert first.shape == (20, 19)


def test_season_t_interval_uses_season_means_and_student_t() -> None:
    # Arrange
    differences = _constant_by_season({2001: (1.0, 2), 2002: (3.0, 5), 2003: (-1.0, 1)})
    means = np.array([1.0, 3.0, -1.0])
    half = stats.t.ppf(0.975, df=2) * means.std(ddof=1) / np.sqrt(3.0)

    # Act
    interval = common.season_t_interval(differences, [2001, 2002, 2003], 0.95)

    # Assert
    assert interval.estimate == pytest.approx(1.0)
    assert interval.lower == pytest.approx(1.0 - half)
    assert interval.upper == pytest.approx(1.0 + half)


def test_season_t_interval_season_without_rows_raises_a_value_error() -> None:
    # Arrange
    differences = pl.DataFrame({"season": [2000, 2000], "difference": [0.1, -0.1]})

    # Act & Assert
    with pytest.raises(ValueError, match="2001"):
        common.season_t_interval(differences, [2000, 2001], 0.95)


def test_game_bootstrap_same_seed_is_reproducible() -> None:
    # Arrange
    values = np.linspace(-1.0, 2.0, 37)
    first = common.game_bootstrap(values, confidence=0.9, resamples=1_200, seed=5)

    # Act
    second = common.game_bootstrap(values, confidence=0.9, resamples=1_200, seed=5)

    # Assert
    assert first == second
    assert first.estimate == pytest.approx(values.mean())


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [(-2.0, -0.5, (True, False)), (0.1, 0.4, (False, True)), (-0.1, 0.2, (False, False))],
)
def test_interval_sides_of_zero(lower: float, upper: float, expected: tuple[bool, bool]) -> None:
    # Arrange
    interval = common.Interval(estimate=(lower + upper) / 2.0, lower=lower, upper=upper)

    # Act
    sides = (interval.below_zero, interval.above_zero)

    # Assert
    assert sides == expected


def test_interval_flipped_negates_and_swaps_bounds() -> None:
    # Arrange
    interval = common.Interval(estimate=0.5, lower=-0.25, upper=1.5)

    # Act
    flipped = interval.flipped()

    # Assert
    assert flipped == common.Interval(estimate=-0.5, lower=-1.5, upper=0.25)


def test_wrap_markdown_fills_prose_and_bullets_to_the_width() -> None:
    # Arrange
    long = " ".join(["word"] * 40)
    lines = ["# Heading", "", long, f"- {long}", f"12. {long}", "| a | b |", "```text", long, "```"]

    # Act
    wrapped = common.wrap_markdown(lines, width=60)

    # Assert
    code_line = wrapped.index("```text") + 1
    assert all(
        len(line) <= 60
        for index, line in enumerate(wrapped)
        if index != code_line and not line.startswith("|")
    )
    assert wrapped[0] == "# Heading"
    assert wrapped[code_line] == long
    assert any(line.startswith("  word") for line in wrapped)
    assert any(line.startswith("    word") for line in wrapped)


def test_wrap_markdown_keeps_hyphenated_spans_whole() -> None:
    # Arrange
    lines = ["seasons " * 9 + "1999-2025 and weeks 5-7"]

    # Act
    wrapped = common.wrap_markdown(lines, width=40)

    # Assert
    assert any("1999-2025" in line for line in wrapped)


@pytest.mark.parametrize(
    ("value", "text"),
    [(0.00047, "+0.0005"), (-0.0004, "-0.0004"), (0.0123, "+0.012"), (-1.5, "-1.500")],
)
def test_bound_text_keeps_a_bound_near_zero_visible(value: float, text: str) -> None:
    # Act
    formatted = common.bound_text(value)

    # Assert
    assert formatted == text


def test_perturbed_numbers_changes_every_numeric_value_of_the_rows_only() -> None:
    # Arrange
    frame = pl.DataFrame(
        {
            "season": [2010, 2010],
            "week": [1, 2],
            "team": ["AAA", "AAA"],
            "plays": [10, 10],
            "epa": [1.0, 1.0],
        }
    )

    # Act
    changed = common.perturbed_numbers(frame, pl.col("week") >= 2)

    # Assert
    assert changed.row(0) == (2010, 1, "AAA", 10, 1.0)
    assert changed.row(1) == (2010, 2, "AAA", 15, 4.0)
    assert changed.schema == frame.schema


def test_run_provenance_records_the_git_commit_and_the_run(tmp_path: Path) -> None:
    # Arrange
    code = tmp_path / "script.py"
    code.write_text("print()\n")
    ledger = common.InputLedger()
    ledger.record(code)

    # Act
    provenance = common.run_provenance([code], ledger, "python script.py", 12.5)

    # Assert
    assert re.fullmatch(r"[0-9a-f]{40}", provenance["git_commit"])
    assert provenance["code"] == {"script.py": common.sha256_file(code)}
    assert (provenance["inputs"], provenance["fingerprint"]) == (1, ledger.fingerprint())
    assert (provenance["command"], provenance["seconds"]) == ("python script.py", 12.5)


def test_write_json_record_ends_with_a_newline(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "record.json"

    # Act
    common.write_json_record(path, {"value": 1.5, "rows": [1, 2]})

    # Assert
    text = path.read_text()
    assert text.endswith("}\n")
    assert json.loads(text) == {"value": 1.5, "rows": [1, 2]}


def test_write_json_record_passes_unwritable_values_to_the_default(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "record.json"

    # Act
    common.write_json_record(path, {"pair": ("A", "B"), "path": tmp_path}, default=str)

    # Assert
    assert json.loads(path.read_text()) == {"pair": ["A", "B"], "path": str(tmp_path)}
