"""The weekly fixture with columns from four feature groups, for the feature-group tests.

``build_group_fixture`` writes the synthetic seasons of ``tests/weekly_fixture.py`` and adds, to
every games file (the completed games, both all-games files and the week to predict), columns
that belong to the ``elo``, ``strength``, ``own_elo`` and ``own_elo_wt`` groups of
``constants.FEATURE_GROUP_COLUMN_MARKERS``. They sit inside the feature range, right after
``power_diff``, so every fit reads them unless something leaves them out. Their values are drawn
around each side's ``power`` from one seeded generator, so every call writes the same bytes.

``GROUP_COLUMNS`` names the columns each group owns in this fixture, written out by hand so a
marker that drifts onto another column, or misses one, fails the tests that use it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml
from tests.weekly_fixture import build_fixture, write_weekly_config

if TYPE_CHECKING:
    from pathlib import Path

# The fixture column the group columns follow, inside the feature range.
INSERT_AFTER = "power_diff"
# The games files the weekly fixture writes; the strength snapshots and the reference run are
# not games files.
GAMES_FILES = ("completed", "all_ml", "all", "predict")
GROUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "elo": (
        "away_elo_pre",
        "home_elo_pre",
        "elo_pre_diff",
        "away_qb_value_pre",
        "home_qb_value_pre",
        "qb_value_pre_diff",
    ),
    "strength": (
        "away_adj_srs",
        "home_adj_srs",
        "adj_srs_diff",
        "away_sos_played_adj",
        "home_sos_played_adj",
        "sos_played_adj_diff",
    ),
    "own_elo": ("away_own_elo", "home_own_elo", "own_elo_diff", "own_elo_home_win_prob"),
    "own_elo_wt": (
        "away_own_wt_elo",
        "home_own_wt_elo",
        "own_wt_elo_diff",
        "own_wt_elo_home_win_prob",
    ),
}
# Elo points of home field in the fixture's own-Elo win probabilities.
ELO_HOME_FIELD = 48.0


def _elo_home_win_prob(away: pd.Series, home: pd.Series) -> pd.Series:
    """Return the Elo win probability of the home side, home field included."""
    return (1.0 / (1.0 + 10.0 ** (-(home - away + ELO_HOME_FIELD) / 400.0))).round(5)


def _sides(
    rng: np.random.Generator, games: pd.DataFrame, center: float, scale: float, noise: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return away and home values drawn around each side's power, and their difference."""
    away = (center + scale * games["away_power"] + rng.normal(0, noise, len(games))).round(3)
    home = (center + scale * games["home_power"] + rng.normal(0, noise, len(games))).round(3)
    return away, home, (away - home).round(3)


def group_columns_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Return every group column for ``games``, keyed by ``game_id``, in fixture order."""
    rng = np.random.default_rng(20261010)
    columns: dict[str, pd.Series] = {"game_id": games["game_id"]}
    for names, (center, scale, noise) in (
        (GROUP_COLUMNS["elo"][:3], (1505.0, 25.0, 30.0)),
        (GROUP_COLUMNS["elo"][3:], (150.0, 4.0, 40.0)),
        (GROUP_COLUMNS["strength"][:3], (0.0, 1.0, 1.5)),
        (GROUP_COLUMNS["strength"][3:], (0.0, 0.0, 1.0)),
        (GROUP_COLUMNS["own_elo"][:3], (1505.0, 24.0, 25.0)),
        (GROUP_COLUMNS["own_elo_wt"][:3], (1505.0, 26.0, 20.0)),
    ):
        columns.update(zip(names, _sides(rng, games, center, scale, noise), strict=True))
    for group in ("own_elo", "own_elo_wt"):
        away, home, _, probability = GROUP_COLUMNS[group]
        columns[probability] = _elo_home_win_prob(columns[away], columns[home])
    return pd.DataFrame(columns)


def _with_group_columns(frame: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with the group columns inserted after ``INSERT_AFTER``, rows in place."""
    added = frame[["game_id"]].merge(groups, on="game_id", how="left", validate="one_to_one")
    position = list(frame.columns).index(INSERT_AFTER) + 1
    names = [name for name in groups.columns if name != "game_id"]
    return pd.concat(
        [frame.iloc[:, :position], added[names], frame.iloc[:, position:]], axis="columns"
    )


def build_group_fixture(root: Path) -> dict[str, Path]:
    """Write the weekly fixture with the group columns added; return its paths."""
    paths = build_fixture(root)
    groups = group_columns_frame(pd.read_csv(paths["all_ml"]))
    for name in GAMES_FILES:
        frame = pd.read_csv(paths[name])
        _with_group_columns(frame, groups).to_csv(paths[name], index=False)
    return paths


def write_group_weekly_config(
    root: Path, paths: dict[str, Path], overrides: dict[str, Any] | None = None
) -> Path:
    """Write the fixture's weekly config with ``overrides`` on top; return its path.

    It is ``tests/weekly_fixture.py``'s adaptation of the shipped config (the run under
    ``root/run``, outputs under ``root/out``), with the power rankings skipped: the feature-group
    tests read only the walk-forward, the final fit and the predictions.
    """
    config_path = write_weekly_config(root, paths, root / "out")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.update({"skip_power_rankings": True, **(overrides or {})})
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return config_path
