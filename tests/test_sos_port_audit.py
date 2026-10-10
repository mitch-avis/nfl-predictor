"""The port audit scripts beside ``.agents/sos_port/README.md``: inventory and parity check.

The scripts run from their own directory and import each other by file name, so each test puts
that directory on ``sys.path`` and imports the module it needs. Every input here is built in a
temporary directory; nothing reads the checkout's ``data/`` or the network.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils.polars import loaders
from nfl_predictor.utils.polars.season_stats_schema import SEASON_STATS_SCHEMAS

if TYPE_CHECKING:
    from types import ModuleType

AUDIT_DIR = Path(__file__).resolve().parents[1] / ".agents" / "sos_port"


@pytest.fixture
def port_audit(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Return the audit scripts' shared module, imported from their directory."""
    monkeypatch.syspath_prepend(str(AUDIT_DIR))
    return importlib.import_module("port_audit")


@pytest.fixture
def inventory(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Return the inventory script, imported from its directory."""
    monkeypatch.syspath_prepend(str(AUDIT_DIR))
    return importlib.import_module("inventory")


@pytest.fixture
def parity(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Return the parity check, imported from its directory."""
    monkeypatch.syspath_prepend(str(AUDIT_DIR))
    return importlib.import_module("parity")


# Two plays of one game as nflverse publishes them, in the columns the parity check needs.
_RAW_PLAYS = {
    "season": [2025, 2025],
    "season_type": ["REG", "REG"],
    "week": [1, 1],
    "game_id": ["2025_01_BUF_MIA", "2025_01_BUF_MIA"],
    "play_id": [40.0, 61.0],
    "posteam": ["BUF", "MIA"],
    "defteam": ["MIA", "BUF"],
    "home_team": ["MIA", "MIA"],
    "away_team": ["BUF", "BUF"],
    "qb_scramble": [0.0, 1.0],
    "desc": ["pass", "scramble"],
    "epa": [0.25, -0.5],
}


def _snapshot_and_cache(tmp_path: Path, *, cache_epa: float = 0.25, stale: bool = False) -> Path:
    """Write the raw plays as a snapshot file and, prepared, as the season's cache file.

    ``cache_epa`` is the first play's EPA in the cache; ``stale`` writes the cache without
    ``desc`` and as if the code had not asked for it then.
    """
    raw = pl.DataFrame(_RAW_PLAYS)
    raw.write_parquet(tmp_path / "snapshot.parquet")
    prepared = loaders._prepare_pbp(raw, regular_season_only=True).with_columns(
        pl.Series("epa", [cache_epa, -0.5])
    )
    requested = tuple(constants.PBP_COLUMNS)
    if stale:
        prepared = prepared.drop("desc")
        requested = tuple(column for column in requested if column != "desc")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    path = loaders._pbp_cache_path(cache_dir, 2025, regular_season_only=True)
    loaders._write_cached_frame(prepared, path, requested)
    return cache_dir


def _column(inventory: ModuleType, kind: str, column: str, **fields: str) -> Any:
    """Return an inventory column of ``kind`` resolved to itself, with ``fields`` overridden."""
    values = {"base": column, "prefix": "", "suffix": "", "registry_source": "", "entity": "team"}
    return inventory.Column(kind=kind, column=column, seasons=[2025], **(values | fields))


def _write_ratings_tables(data_dir: Path, inventory: ModuleType, *, skip: str = "") -> None:
    """Write one small table per ratings kind where the ETL writes it, except ``skip``."""
    folder = data_dir / "ratings"
    folder.mkdir(parents=True)
    last = inventory.SEASONS[-1]
    kinds = [(kind, last) for kind in inventory.PUBLISHED_KINDS]
    kinds += [(kind, last + 1) for kind in inventory.IN_PROGRESS_KINDS]
    for kind, season in kinds:
        if kind != skip:
            pl.DataFrame({"team": ["BUF"], f"{kind}_value": [1.0]}).write_parquet(
                folder / f"{season}_{kind}.parquet"
            )


def test_parse_args_without_data_dir_reads_the_main_checkout_data(port_audit: ModuleType) -> None:
    # Act
    args = port_audit.parse_args("audit", [])

    # Assert
    assert args.data_dir == port_audit.MAIN / "data"


def test_parse_args_data_dir_returns_the_resolved_directory(
    port_audit: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    (tmp_path / "build" / "data").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    # Act
    args = port_audit.parse_args("audit", ["--data-dir", "build/data"])

    # Assert
    assert args.data_dir == (tmp_path / "build" / "data").resolve()


def test_parse_args_data_dir_that_does_not_exist_exits(
    port_audit: ModuleType, tmp_path: Path
) -> None:
    # Act & Assert
    with pytest.raises(SystemExit):
        port_audit.parse_args("audit", ["--data-dir", str(tmp_path / "missing")])


def test_port_catalog_reads_ratings_tables_from_the_data_directory(
    inventory: ModuleType, tmp_path: Path
) -> None:
    # Arrange
    _write_ratings_tables(tmp_path, inventory)
    audit = inventory.Audit()

    # Act
    port = inventory.port_catalog(tmp_path, audit)

    # Assert
    assert port["rating_pairs"] == ["team", "rating_pairs_value"]
    assert port["qb_rating_ranges_by_week"] == ["team", "qb_rating_ranges_by_week_value"]
    assert len(audit.files) == len(inventory.PUBLISHED_KINDS) + len(inventory.IN_PROGRESS_KINDS)
    assert audit.failures == []


def test_port_catalog_takes_season_statistics_columns_from_their_schema(
    inventory: ModuleType, tmp_path: Path
) -> None:
    # Arrange
    _write_ratings_tables(tmp_path, inventory)

    # Act
    port = inventory.port_catalog(tmp_path, inventory.Audit())

    # Assert
    assert port["combined"] == list(SEASON_STATS_SCHEMAS["combined"])


def test_port_catalog_missing_ratings_table_fails_the_audit(
    inventory: ModuleType, tmp_path: Path
) -> None:
    # Arrange
    _write_ratings_tables(tmp_path, inventory, skip="team_prior")
    audit = inventory.Audit()

    # Act
    port = inventory.port_catalog(tmp_path, audit)

    # Assert
    assert "team_prior" not in port
    assert [failure.what for failure in audit.failures] == ["port catalog"]


@pytest.mark.parametrize(("kind", "expected"), [("combined", True), ("ratings", False)])
def test_check_counterpart_ported_reference_reads_the_columns_own_kind(
    inventory: ModuleType, kind: str, *, expected: bool
) -> None:
    # Arrange
    port = {"combined": ["passing_epa"], "ratings": ["team_rating"]}

    # Act
    found = inventory.check_counterpart("ratings:passing_epa", {}, port, kind)

    # Assert
    assert found is expected


def test_classify_names_the_ported_column_before_the_rules_counterparts(
    inventory: ModuleType,
) -> None:
    # Arrange
    item = _column(inventory, "combined", "passing_epa")
    spec = {"id": "epa", "kinds": ["combined"], "columns": ["passing_epa"], "class": "different"}
    audit = inventory.Audit()
    rules = inventory.validate_rules([spec | {"here": "dataset:passing_epa"}], audit, "rules")

    # Act
    inventory.classify(
        [item], rules, {"dataset": ["passing_epa"]}, {"combined": ["passing_epa"]}, audit
    )

    # Assert
    assert item.here == ["ratings:passing_epa", "dataset:passing_epa"]
    assert audit.failures == []


def test_classify_mirror_names_its_own_ported_column(inventory: ModuleType) -> None:
    # Arrange
    game = _column(inventory, "qb_game_logs", "qb_attempts")
    season = _column(inventory, "qb_per_game_stats", "qb_attempts_total", base="qb_attempts")
    specs = [
        {"id": "game", "kinds": ["qb_game_logs"], "columns": ["qb_attempts"]}
        | {"class": "different", "here": "qb_game:attempts"},
        {"id": "season", "kinds": ["qb_per_game_stats"], "columns": ["qb_attempts_total"]}
        | {"class": "different", "mirror_kind": "qb_game_logs", "mirror_base": True},
    ]
    port = {"qb_game_logs": ["qb_attempts"], "qb_per_game_stats": ["qb_attempts_total"]}
    audit = inventory.Audit()
    rules = inventory.validate_rules(specs, audit, "rules")

    # Act
    inventory.classify([game, season], rules, {"qb_game": ["attempts"]}, port, audit)

    # Assert
    assert season.here == ["ratings:qb_attempts_total", "qb_game:attempts"]
    assert audit.failures == []


def test_classify_mirror_of_a_column_only_the_port_writes_fails(inventory: ModuleType) -> None:
    # Arrange
    game = _column(inventory, "team_game_logs", "turnovers_on_downs")
    season = _column(inventory, "combined", "turnovers_on_downs")
    specs = [
        {"id": "game", "kinds": ["team_game_logs"], "columns": ["turnovers_on_downs"]}
        | {"class": "matched", "here": "ratings:{column}"},
        {"id": "season", "kinds": ["combined"], "columns": ["turnovers_on_downs"]}
        | {"class": "different", "mirror_kind": "team_game_logs"},
    ]
    port = {"team_game_logs": ["turnovers_on_downs"], "combined": ["turnovers_on_downs"]}
    audit = inventory.Audit()
    rules = inventory.validate_rules(specs, audit, "rules")

    # Act
    inventory.classify([game, season], rules, {}, port, audit)

    # Assert
    assert [failure.what for failure in audit.failures] == ["column rules"]


def test_guard_port_fails_a_missing_column_the_port_writes(inventory: ModuleType) -> None:
    # Arrange
    item = _column(inventory, "opponent_profiles", "passing_epa", prefix="opp_")
    item.classification = "missing"
    audit = inventory.Audit()

    # Act
    checked = inventory.guard_port([item], {"opponent_profiles": ["passing_epa"]}, audit)

    # Assert
    assert checked == 1
    assert [failure.what for failure in audit.failures] == ["port guard"]


def test_json_body_writes_columns_by_kind_and_catalogs_as_counts(inventory: ModuleType) -> None:
    # Arrange
    every = _column(inventory, "combined", "passing_epa")
    every.seasons = [2024, 2025]
    late = _column(inventory, "combined", "rushing_epa")
    payload = {
        "kinds": {"combined": [2024, 2025]},
        "columns": [every.__dict__, late.__dict__],
        "here_catalog": {"constants": ["a", "b"], "dataset": ["passing_epa"]},
        "port_catalog": {"combined": ["passing_epa", "rushing_epa"]},
        "registry": {"metrics": [{"name": "passing_epa", "entity": "team"}]},
    }

    # Act
    body = inventory.json_body(payload)

    # Assert
    assert body["column_fields"][0] == "column"
    assert [row[0] for row in body["columns"]["combined"]] == ["passing_epa", "rushing_epa"]
    assert body["partial_seasons"] == {"combined.rushing_epa": "2025"}
    assert body["here_catalog"] == {"constants": "2 names", "dataset": ["passing_epa"]}
    assert body["port_catalog"] == {"combined": "2 columns"}
    assert body["registry"]["metric_fields"] == ["name", "entity"]
    assert body["registry"]["metrics"] == [["passing_epa", "team"]]


def test_guard_simple_missing_item_this_repository_has_fails(inventory: ModuleType) -> None:
    # Arrange
    rows = [{"key": "nflreadpy.load_players", "class": "missing"}]
    audit = inventory.Audit()

    # Act
    inventory.guard_simple(
        rows, {"load:nflreadpy.load_players"}, audit, "source rules", ref_prefix="load:"
    )

    # Assert
    assert [failure.what for failure in audit.failures] == ["source rules"]


def test_classify_simple_fills_the_items_key_into_its_counterpart(inventory: ModuleType) -> None:
    # Arrange
    spec = {"id": "players", "match": "nflreadpy.load_play.*", "class": "matched"}
    audit = inventory.Audit()

    # Act
    rows = inventory.classify_simple(
        [("nflreadpy.load_players", None)],
        [spec | {"here": "load:{key}"}],
        {"load:nflreadpy.load_players"},
        audit,
        "source rules",
    )

    # Assert
    assert rows[0]["here"] == ["load:nflreadpy.load_players"]
    assert audit.failures == []


def test_season_pbp_returns_the_cache_the_snapshot_equals(
    parity: ModuleType, tmp_path: Path
) -> None:
    # Arrange
    cache_dir = _snapshot_and_cache(tmp_path)
    audit = parity.Audit()

    # Act
    found = parity.season_pbp(2025, tmp_path / "snapshot.parquet", cache_dir, audit)

    # Assert
    assert found is not None
    pbp, report = found
    assert pbp["desc"].to_list() == ["pass", "scramble"]
    assert report["identical_to_cache"] is True
    assert audit.failures == []


def test_season_pbp_snapshot_that_differs_from_the_cache_fails(
    parity: ModuleType, tmp_path: Path
) -> None:
    # Arrange
    cache_dir = _snapshot_and_cache(tmp_path, cache_epa=0.75)
    audit = parity.Audit()

    # Act
    parity.season_pbp(2025, tmp_path / "snapshot.parquet", cache_dir, audit)

    # Assert
    assert [failure.what for failure in audit.failures] == ["snapshot"]


def test_season_pbp_cache_that_predates_the_code_fails(parity: ModuleType, tmp_path: Path) -> None:
    # Arrange
    cache_dir = _snapshot_and_cache(tmp_path, stale=True)
    audit = parity.Audit()

    # Act
    found = parity.season_pbp(2025, tmp_path / "snapshot.parquet", cache_dir, audit)

    # Assert
    assert found is None
    assert [failure.what for failure in audit.failures] == ["cache"]


def test_season_pbp_missing_cache_fails(parity: ModuleType, tmp_path: Path) -> None:
    # Arrange
    pl.DataFrame(_RAW_PLAYS).write_parquet(tmp_path / "snapshot.parquet")
    audit = parity.Audit()

    # Act
    found = parity.season_pbp(2025, tmp_path / "snapshot.parquet", tmp_path, audit)

    # Assert
    assert found is None
    assert [failure.what for failure in audit.failures] == ["cache"]
