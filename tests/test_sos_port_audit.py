"""The port audit scripts beside ``.agents/sos_port/README.md``: inventory and parity check.

The scripts run from their own directory and import each other by file name, so each test imports
the module it needs through the shared loader (``tests/sos_port_scripts.py``). Every input here is
built in a temporary directory; nothing reads the checkout's ``data/`` or the network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
import yaml

from nfl_predictor import constants
from nfl_predictor.utils.polars import loaders, pbp
from nfl_predictor.utils.polars.season_stats_schema import SEASON_STATS_SCHEMAS
from tests.sos_port_scripts import load_sos_port_script

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


@pytest.fixture
def port_audit() -> ModuleType:
    """Return the audit scripts' shared module, imported from their directory."""
    return load_sos_port_script("port_audit")


@pytest.fixture
def inventory() -> ModuleType:
    """Return the inventory script, imported from its directory."""
    return load_sos_port_script("inventory")


@pytest.fixture
def parity() -> ModuleType:
    """Return the parity check, imported from its directory."""
    return load_sos_port_script("parity")


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


@pytest.fixture
def parity_rules() -> ModuleType:
    """Return the parity check's cause adjustments, imported from their directory."""
    return load_sos_port_script("parity_rules")


def _plays(**flags: list[float]) -> pl.DataFrame:
    """Return Buffalo's plays at Miami, one per list entry, with ``flags`` set and 0 elsewhere."""
    count = len(next(iter(flags.values())))
    names = ("rush", "rush_attempt", "qb_dropback", "qb_kneel", "qb_spike", "qb_scramble")
    columns: dict[str, list[Any]] = {
        "season": [2025] * count,
        "season_type": ["REG"] * count,
        "week": [1] * count,
        "game_id": ["2025_01_BUF_MIA"] * count,
        "posteam": ["BUF"] * count,
        "defteam": ["MIA"] * count,
        "home_team": ["MIA"] * count,
        "away_team": ["BUF"] * count,
        "play_type": ["run"] * count,
        "two_point_attempt": [0.0] * count,
        "down": [1.0] * count,
        "epa": [float(n + 1) for n in range(count)],
    }
    columns |= {name: flags.get(name, [0.0] * count) for name in names}
    return pl.DataFrame(columns)


def _rush_epa(plays: pl.DataFrame) -> float:
    """Return Buffalo's ``rush_epa_sum`` from this repository's team-game aggregate."""
    games = pbp.aggregate_pbp_team_game_stats(plays)
    return float(games.filter(pl.col("team_abbr") == "BUF")["rush_epa_sum"][0])


def _wiped_out_passes(epa: list[float]) -> pl.DataFrame:
    """Return Buffalo's passes a penalty wiped out (``no_play``), one per EPA value."""
    count = len(epa)
    return pl.DataFrame(
        {
            "posteam": ["BUF"] * count,
            "defteam": ["MIA"] * count,
            "play_type": ["no_play"] * count,
            "penalty": [1.0] * count,
            "rush": [0.0] * count,
            "pass": [1.0] * count,
            "qb_dropback": [1.0] * count,
            "epa": epa,
        }
    )


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


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("box_score[opponent]:rush_attempts", True),
        ("box_score[opponent]:carries_faced", False),
        ("box_score[home]:rush_attempts", False),
    ],
)
def test_check_counterpart_opponent_row_marker_names_a_column_of_its_catalog(
    inventory: ModuleType, ref: str, *, expected: bool
) -> None:
    # Arrange
    catalog = {"box_score": ["rush_attempts"]}

    # Act
    found = inventory.check_counterpart(ref, catalog, {}, "team_game_logs")

    # Assert
    assert found is expected


@pytest.mark.parametrize("catalog", ["team_game", "box_score", "team_stats"])
def test_check_counterpart_opponent_row_marker_on_a_team_game_catalog_is_accepted(
    inventory: ModuleType, catalog: str
) -> None:
    # Arrange
    catalogs = {catalog: ["rating"]}

    # Act
    found = inventory.check_counterpart(f"{catalog}[opponent]:rating", catalogs, {}, "x")

    # Assert
    assert found is True


@pytest.mark.parametrize("catalog", ["ratings", "strength", "dataset", "schedule"])
def test_check_counterpart_opponent_row_marker_on_a_catalog_without_team_game_rows_fails(
    inventory: ModuleType, catalog: str
) -> None:
    # Arrange
    catalogs = {catalog: ["rating"]}
    port = {"x": ["rating"]}

    # Act
    found = inventory.check_counterpart(f"{catalog}[opponent]:rating", catalogs, port, "x")

    # Assert
    assert found is False


def test_mapping_rules_on_the_opponents_row_mark_every_counterpart(inventory: ModuleType) -> None:
    # Arrange
    rules = yaml.safe_load(inventory.MAPPING.read_text(encoding="utf-8"))["columns"]
    opponent_rules = [r for r in rules if r.get("note", "").startswith("The opponent's row")]

    # Act
    unmarked = [
        f"{rule['id']}: {ref}"
        for rule in opponent_rules
        for ref in ([rule["here"]] if "here" in rule else rule["here_map"].values())
        if "[opponent]:" not in ref
    ]

    # Assert
    assert (len(opponent_rules), unmarked) == (5, [])


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


def test_wiped_out_run_epa_puts_a_penalty_wiped_run_into_rush_epa(
    parity_rules: ModuleType,
) -> None:
    # Arrange: a designed run (EPA 1) and a run a penalty wiped out (EPA 2).
    plays = _plays(rush=[1.0, 1.0], rush_attempt=[1.0, 0.0]).with_columns(
        play_type=pl.Series(["run", "no_play"])
    )

    # Act
    adjusted = parity_rules.CAUSES["wiped_out_run_epa"].ours(plays)

    # Assert
    assert (_rush_epa(plays), _rush_epa(adjusted)) == (1.0, 3.0)


def test_unflagged_rush_attempt_epa_takes_an_unflagged_carry_out_of_rush_epa(
    parity_rules: ModuleType,
) -> None:
    # Arrange: a designed run (EPA 1), a rush attempt without the run flag (EPA 2), a kneel (3).
    plays = _plays(rush=[1.0, 0.0, 0.0], rush_attempt=[1.0, 1.0, 1.0], qb_kneel=[0.0, 0.0, 1.0])

    # Act
    adjusted = parity_rules.CAUSES["unflagged_rush_attempt_epa"].ours(plays)

    # Assert
    assert (_rush_epa(plays), _rush_epa(adjusted)) == (3.0, 1.0)


def test_unflagged_rush_attempt_epa_keeps_a_spike_filed_as_a_rush_attempt(
    parity_rules: ModuleType,
) -> None:
    # Arrange: a designed run (EPA 1) and a spike filed as a rush attempt (EPA 2), a scrimmage
    # snap there.
    plays = _plays(rush=[1.0, 0.0], rush_attempt=[1.0, 1.0], qb_spike=[0.0, 1.0])

    # Act
    adjusted = parity_rules.CAUSES["unflagged_rush_attempt_epa"].ours(plays)

    # Assert
    assert _rush_epa(adjusted) == _rush_epa(plays) == 3.0


def test_adjust_takes_an_unflagged_two_point_run_out_after_two_point_tries(
    parity_rules: ModuleType,
) -> None:
    # Arrange: a designed run (EPA 1) and a two-point try nflverse files as a rush attempt without
    # the run flag (EPA 2), no scrimmage snap there.
    plays = _plays(rush=[1.0, 0.0], rush_attempt=[1.0, 1.0]).with_columns(
        two_point_attempt=pl.Series([0.0, 1.0])
    )
    causes = frozenset({"two_point_tries", "unflagged_rush_attempt_epa"})

    # Act
    adjusted = parity_rules.adjust(plays, causes)

    # Assert
    assert _rush_epa(adjusted) == 1.0


def test_kneels_and_spikes_puts_every_kneel_down_into_rush_epa(parity_rules: ModuleType) -> None:
    # Arrange: a designed run (EPA 1), a kneel-down filed as a rush attempt (EPA 2) and one that
    # is not (EPA 3).
    plays = _plays(rush_attempt=[1.0, 1.0, 0.0], rush=[1.0, 0.0, 0.0], qb_kneel=[0.0, 1.0, 1.0])

    # Act
    adjusted = parity_rules.CAUSES["kneels_and_spikes"].ours(plays)

    # Assert
    assert (_rush_epa(plays), _rush_epa(adjusted)) == (1.0, 6.0)


def test_season_pbp_missing_cache_fails(parity: ModuleType, tmp_path: Path) -> None:
    # Arrange
    pl.DataFrame(_RAW_PLAYS).write_parquet(tmp_path / "snapshot.parquet")
    audit = parity.Audit()

    # Act
    found = parity.season_pbp(2025, tmp_path / "snapshot.parquet", tmp_path, audit)

    # Assert
    assert found is None
    assert [failure.what for failure in audit.failures] == ["cache"]


def test_wiped_out_plays_writes_the_epa_sum_rounded_to_nine_decimals(
    parity_rules: ModuleType,
) -> None:
    # Arrange
    plays = _wiped_out_passes([0.25, 1e-10])

    # Act
    found = parity_rules.wiped_out_plays(plays)

    # Assert
    assert found["pass_epa"] == round(0.25 + 1e-10, 9)


def test_wiped_out_plays_epa_sums_in_either_summation_order_serialize_identically(
    parity_rules: ModuleType,
) -> None:
    # Arrange
    report_text = load_sos_port_script("report_text")
    forward, backward = _wiped_out_passes([0.1, 0.2, 0.3]), _wiped_out_passes([0.3, 0.2, 0.1])

    # Act
    written = [report_text.dump_json(parity_rules.wiped_out_plays(p)) for p in (forward, backward)]

    # Assert
    assert written[0] == written[1]


@pytest.mark.parametrize(
    "epa", [[0.1000000005, 1.0, -1.0], [-1.0, 1.0, 0.1000000005]], ids=["forward", "backward"]
)
def test_wiped_out_plays_epa_sum_straddling_a_ninth_decimal_boundary_rounds_the_exact_sum(
    parity_rules: ModuleType, epa: list[float]
) -> None:
    # Arrange
    plays = _wiped_out_passes(epa)

    # Act
    found = parity_rules.wiped_out_plays(plays)

    # Assert
    assert found["pass_epa"] == round(0.1000000005, 9)


def _carries_faced_pair(parity_rules: ModuleType) -> Any:
    """Return a pair judging its ``carries_faced`` against this repository's opponent row."""
    return parity_rules.Pair(
        id="carries_faced",
        ours=parity_rules.expression("rush_attempts"),
        theirs=parity_rules.expression("carries_faced"),
        kind="count",
        causes=(),
        null_as_zero=False,
        row="opponent",
    )


def _one_game(value_column: str, buffalo: float, miami: float) -> pl.DataFrame:
    """Return Buffalo at Miami as two team-game rows, with ``value_column`` set per team."""
    return pl.DataFrame(
        {
            "week": [1, 1],
            "team_abbr": ["BUF", "MIA"],
            "opponent_abbr": ["MIA", "BUF"],
            "game_id": ["2025_01_BUF_MIA", "2025_01_BUF_MIA"],
            value_column: [buffalo, miami],
        }
    )


def test_compare_opponent_row_pair_matches_the_value_on_the_opponents_row(
    parity: ModuleType, parity_rules: ModuleType
) -> None:
    # Arrange
    ours = _one_game("rush_attempts", 20.0, 30.0)
    theirs = _one_game("carries_faced", 30.0, 20.0)

    # Act
    compared = parity.compare(_carries_faced_pair(parity_rules), ours, theirs, 1e-6)

    # Assert
    assert compared.sort("team_abbr")["match"].to_list() == [True, True]


def test_compare_opponent_row_pair_mismatches_the_value_on_the_teams_own_row(
    parity: ModuleType, parity_rules: ModuleType
) -> None:
    # Arrange
    ours = _one_game("rush_attempts", 20.0, 30.0)
    theirs = _one_game("carries_faced", 20.0, 30.0)

    # Act
    compared = parity.compare(_carries_faced_pair(parity_rules), ours, theirs, 1e-6)

    # Assert
    assert compared.sort("team_abbr")["match"].to_list() == [False, False]


def test_check_opponent_rows_team_game_without_its_opponents_row_fails_the_audit(
    parity: ModuleType,
) -> None:
    # Arrange
    ours = _one_game("rush_attempts", 20.0, 30.0).head(1)
    theirs = _one_game("carries_faced", 30.0, 20.0).head(1)
    audit = parity.Audit()

    # Act
    missing = parity.check_opponent_rows(2025, ours, theirs, audit)

    # Assert
    assert (missing, [f.what for f in audit.failures]) == (1, ["team-game keys"])


def test_check_opponent_rows_complete_game_passes(parity: ModuleType) -> None:
    # Arrange
    ours = _one_game("rush_attempts", 20.0, 30.0)
    theirs = _one_game("carries_faced", 30.0, 20.0)
    audit = parity.Audit()

    # Act
    missing = parity.check_opponent_rows(2025, ours, theirs, audit)

    # Assert
    assert (missing, audit.failures) == (0, [])


def test_load_pairs_unknown_row_fails_the_audit(parity: ModuleType) -> None:
    # Arrange
    entry = {"id": "x", "ours": "a", "theirs": "b", "kind": "count", "causes": [], "row": "home"}
    audit = parity.Audit()

    # Act
    parity.load_pairs({"pairs": [entry], "causes": {}}, audit)

    # Assert
    assert ("pair", "x: unknown row home") in [(f.what, f.detail) for f in audit.failures]


_RULE_10_PAIRS = {
    "carries_faced": "opponent",
    "attempts_faced": "opponent",
    "completions_allowed": "opponent",
    "rush_success_rate_allowed": "own",
    "third_down_pct_allowed": "own",
    "fourth_down_pct_allowed": "own",
    "two_pt_conversion_rate_allowed": "own",
}


def test_load_pairs_reads_the_allowed_rate_and_carries_faced_pairs_from_parity_yaml(
    parity: ModuleType,
) -> None:
    # Arrange
    rules = yaml.safe_load(parity.RULES.read_text(encoding="utf-8"))
    audit = parity.Audit()

    # Act
    pairs = parity.load_pairs(rules, audit)

    # Assert
    rows = {pair.id: pair.row for pair in pairs if pair.id in _RULE_10_PAIRS}
    assert rows == _RULE_10_PAIRS
    assert audit.failures == []


def test_pair_table_lists_the_parity_yaml_pairs_with_their_row(parity: ModuleType) -> None:
    # Arrange
    rules = yaml.safe_load(parity.RULES.read_text(encoding="utf-8"))
    pairs = parity.load_pairs(rules, parity.Audit())
    payload = {"tolerance": 1e-6, "pairs": parity.pair_entries(pairs, {})}
    lines: list[str] = []

    # Act
    load_sos_port_script("parity_report")._pairs(payload, lines.append)

    # Assert
    listed = {line.split(" | ")[0].strip("| `"): line for line in lines if line.startswith("| `")}
    assert {
        pair: f" {row} |" in listed[pair] for pair, row in _RULE_10_PAIRS.items()
    } == dict.fromkeys(_RULE_10_PAIRS, True)


def test_check_coverage_empty_schedule_fails_the_audit_instead_of_raising(
    parity: ModuleType,
) -> None:
    # Arrange
    schedule = pl.DataFrame(
        schema={
            "season": pl.Int64,
            "game_id": pl.String,
            "game_type": pl.String,
            "home_score": pl.Int64,
            "away_score": pl.Int64,
            "home_abbr": pl.String,
            "away_abbr": pl.String,
            "week": pl.Int64,
        }
    )
    team_games = pl.DataFrame(
        schema={
            "game_id": pl.String,
            "week": pl.Int64,
            "team_abbr": pl.String,
            "opponent_abbr": pl.String,
        }
    )
    audit = parity.Audit()

    # Act
    parity.check_coverage(
        schedule, team_games, team_games, pl.DataFrame({"game_id": []}), set(), audit
    )

    # Assert
    assert [failure.what for failure in audit.failures] == ["coverage"]
