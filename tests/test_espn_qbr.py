"""Tests for the cached, validation-only ESPN QBR loader.

Downloads are replaced by a fake ``requests.get`` that serves a small Parquet asset shaped like
nflverse's (every season in one file) or fails the way ``requests`` fails.
"""

from __future__ import annotations

import ast
import io
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import polars as pl
import pytest
import requests

import nfl_predictor
from nfl_predictor import constants
from nfl_predictor.cli import main as front_door
from nfl_predictor.utils.polars import espn_qbr

if TYPE_CHECKING:
    from collections.abc import Callable

CURRENT = 2024
PACKAGE_DIR = Path(nfl_predictor.__file__).parent
# The first HTTP status ``requests`` treats as an error.
HTTP_ERROR_STATUS = 400


class _Response:
    """A downloaded file: its bytes and its HTTP status."""

    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content = content
        self.status = status

    def raise_for_status(self) -> None:
        """Raise as ``requests`` does for an error status."""
        if self.status >= HTTP_ERROR_STATUS:
            msg = f"{self.status} Client Error"
            raise requests.HTTPError(msg)


def _season_level(rows: list[dict[str, object]]) -> pl.DataFrame:
    """Return a season-level QBR asset: one row per entry, with nflverse's column types."""
    base: dict[str, object] = {
        "season": 2020,
        "season_type": "Regular",
        "game_week": "Season Total",
        "team_abb": "KC",
        "player_id": "3139477",
        "name_short": "P. Mahomes",
        "name_display": "Patrick Mahomes",
        "qb_plays": 640.0,
        "qbr_total": 82.9,
        "qbr_raw": 80.1,
        "pts_added": 40.0,
        "qualified": True,
    }
    frame = pl.DataFrame([{**base, **row} for row in rows])
    return frame.with_columns(pl.col("season").cast(pl.Int32))


def _parquet(frame: pl.DataFrame) -> bytes:
    """Return ``frame`` written as Parquet bytes."""
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def _serve(content: bytes, urls: list[str], status: int = 200) -> Callable[..., _Response]:
    """Return a fake ``requests.get`` serving ``content`` and recording each URL."""

    def get(url: str, **_kwargs: object) -> _Response:
        urls.append(url)
        return _Response(content, status)

    return get


def _unreachable(url: str, **_kwargs: object) -> _Response:
    """Simulate a network failure."""
    msg = f"cannot reach {url}"
    raise requests.ConnectionError(msg)


def test_one_download_serves_every_requested_season_from_its_own_cache_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    urls: list[str] = []
    asset = _season_level([{"season": 2019}, {"season": 2020}])
    monkeypatch.setattr(espn_qbr.requests, "get", _serve(_parquet(asset), urls))

    # Act
    qbr = espn_qbr.load_espn_qbr([2019, 2020], cache_dir=tmp_path, current_season=CURRENT)

    # Assert
    assert urls == [constants.ESPN_QBR_RELEASE_URLS["season"]]
    assert qbr.get_column("season").to_list() == [2019, 2020]
    assert qbr.columns == constants.ESPN_QBR_SEASON_COLUMNS
    assert (tmp_path / "espn_qbr_season_2019_reg.parquet").is_file()
    assert (tmp_path / "espn_qbr_season_2020_reg.parquet").is_file()


def test_a_cached_historical_season_needs_no_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    asset = _season_level([{"season": 2020, "qbr_total": 70.0}])
    monkeypatch.setattr(espn_qbr.requests, "get", _serve(_parquet(asset), []))
    espn_qbr.load_espn_qbr([2020], cache_dir=tmp_path, current_season=CURRENT)
    monkeypatch.setattr(espn_qbr.requests, "get", _unreachable)

    # Act
    qbr = espn_qbr.load_espn_qbr([2020], cache_dir=tmp_path, current_season=CURRENT)

    # Assert
    assert qbr.get_column("qbr_total").to_list() == [70.0]


def test_the_current_season_is_downloaded_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    first = _season_level([{"season": CURRENT, "qbr_total": 50.0}])
    monkeypatch.setattr(espn_qbr.requests, "get", _serve(_parquet(first), []))
    espn_qbr.load_espn_qbr([CURRENT], cache_dir=tmp_path, current_season=CURRENT)
    second = _season_level([{"season": CURRENT, "qbr_total": 65.0}])
    monkeypatch.setattr(espn_qbr.requests, "get", _serve(_parquet(second), []))

    # Act
    qbr = espn_qbr.load_espn_qbr([CURRENT], cache_dir=tmp_path, current_season=CURRENT)

    # Assert
    assert qbr.get_column("qbr_total").to_list() == [65.0]


def test_only_the_regular_season_is_kept_with_canonical_team_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    asset = _season_level(
        [
            {"team_abb": "OAK"},
            {"team_abb": "STL"},
            {"team_abb": "WSH"},
            {"team_abb": "SD", "season_type": "Playoffs"},
        ]
    )
    monkeypatch.setattr(espn_qbr.requests, "get", _serve(_parquet(asset), []))

    # Act
    qbr = espn_qbr.load_espn_qbr([2020], cache_dir=tmp_path, current_season=CURRENT)

    # Assert
    assert qbr.get_column("team_abb").to_list() == ["LV", "LAR", "WSH"]


def test_the_week_level_reads_its_own_asset_and_normalizes_opponents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    urls: list[str] = []
    asset = _season_level([{"opp_abb": "SD", "game_id": "401220225"}]).with_columns(
        pl.lit(3, dtype=pl.Int32).alias("game_week")
    )
    monkeypatch.setattr(espn_qbr.requests, "get", _serve(_parquet(asset), urls))

    # Act
    qbr = espn_qbr.load_espn_qbr([2020], level="week", cache_dir=tmp_path, current_season=CURRENT)

    # Assert
    assert urls == [constants.ESPN_QBR_RELEASE_URLS["week"]]
    assert qbr.columns == constants.ESPN_QBR_WEEK_COLUMNS
    assert qbr.select("game_week", "opp_abb").rows() == [(3, "LAC")]


def test_an_unknown_level_is_rejected_before_any_download(tmp_path: Path) -> None:
    # Arrange
    level = cast("Literal['season', 'week']", "quarter")

    # Act & Assert
    with pytest.raises(ValueError, match="season, week"):
        espn_qbr.load_espn_qbr([2020], level=level, cache_dir=tmp_path)


def test_a_failed_download_of_a_historical_season_is_raised_as_a_connection_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(espn_qbr.requests, "get", _unreachable)

    # Act & Assert
    with pytest.raises(ConnectionError, match="cannot reach"):
        espn_qbr.load_espn_qbr([2020], cache_dir=tmp_path, current_season=CURRENT)


def test_a_current_season_the_asset_cannot_serve_falls_back_to_its_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    asset = _season_level([{"season": CURRENT, "qbr_total": 58.0}])
    monkeypatch.setattr(espn_qbr.requests, "get", _serve(_parquet(asset), []))
    espn_qbr.load_espn_qbr([CURRENT], cache_dir=tmp_path, current_season=CURRENT)
    monkeypatch.setattr(espn_qbr.requests, "get", _serve(b"", [], status=404))

    # Act
    qbr = espn_qbr.load_espn_qbr([CURRENT], cache_dir=tmp_path, current_season=CURRENT)

    # Assert
    assert qbr.get_column("qbr_total").to_list() == [58.0]


@pytest.mark.parametrize(
    ("content", "status"), [(b"", 404), (b"not parquet", 200)], ids=["missing", "unreadable"]
)
def test_a_current_season_the_asset_cannot_serve_without_a_cache_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: bytes, status: int
) -> None:
    # Arrange
    monkeypatch.setattr(espn_qbr.requests, "get", _serve(content, [], status=status))

    # Act
    qbr = espn_qbr.load_espn_qbr([CURRENT], cache_dir=tmp_path, current_season=CURRENT)

    # Assert
    assert qbr.is_empty()
    assert qbr.columns == constants.ESPN_QBR_SEASON_COLUMNS


def test_an_asset_without_seasons_is_an_unreadable_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    asset = _season_level([{}]).drop("season")
    monkeypatch.setattr(espn_qbr.requests, "get", _serve(_parquet(asset), []))

    # Act & Assert
    with pytest.raises(ValueError, match="no season column"):
        espn_qbr.load_espn_qbr([2020], cache_dir=tmp_path, current_season=CURRENT)


def test_a_season_before_espn_qbr_exists_needs_no_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.setattr(espn_qbr.requests, "get", _unreachable)

    # Act
    qbr = espn_qbr.load_espn_qbr(
        [constants.ESPN_QBR_MIN_SEASON - 1], cache_dir=tmp_path, current_season=CURRENT
    )

    # Assert
    assert qbr.is_empty()


# The package's modules allowed to import the QBR loader: validation code only. A module added
# here must never feed the game dataset or the model.
QBR_IMPORTERS_ALLOWED = frozenset({espn_qbr.__name__})
# Entry points beside the command table: the web app, which builds every router at import.
EXTRA_ENTRY_MODULES = frozenset({"nfl_predictor.api"})


def _module_name(path: Path) -> str:
    """Return the dotted name of a source file of the package."""
    parts = list(path.relative_to(PACKAGE_DIR.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imported_modules(source: str, package: str) -> set[str]:
    """Return every module ``source`` imports anywhere, function bodies included.

    ``package`` resolves relative imports. A ``from`` import adds its module and each name it
    imports from it, since the name may be a submodule. A string constant is added as it is, so
    a module named for ``importlib.import_module`` counts too.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                module = f"{base}.{module}" if module else base
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
    return found


def _package_of(path: Path) -> str:
    """Return the package a source file's relative imports start from."""
    name = _module_name(path)
    return name if path.name == "__init__.py" else name.rpartition(".")[0]


@pytest.mark.parametrize(
    "source",
    [
        "import nfl_predictor.utils.polars.espn_qbr\n",
        "from nfl_predictor.utils.polars import espn_qbr\n",
        "from nfl_predictor.utils.polars.espn_qbr import load_espn_qbr\n",
        "def load():\n    from . import espn_qbr\n",
        "def load():\n    from .espn_qbr import load_espn_qbr\n",
        "import importlib\n\nimportlib.import_module('nfl_predictor.utils.polars.espn_qbr')\n",
    ],
    ids=[
        "import",
        "from-package",
        "from-module",
        "relative-in-function",
        "relative-module",
        "importlib",
    ],
)
def test_the_import_scan_finds_every_way_to_import_espn_qbr(source: str) -> None:
    # Act
    imported = _imported_modules(source, "nfl_predictor.utils.polars")

    # Assert
    assert espn_qbr.__name__ in imported


def test_no_module_outside_the_allow_list_imports_espn_qbr() -> None:
    """ESPN QBR is a validation reference: no other module of the package imports it anywhere."""
    # Arrange
    sources = sorted(PACKAGE_DIR.rglob("*.py"))

    # Act
    importers = sorted(
        _module_name(path)
        for path in sources
        if espn_qbr.__name__
        in _imported_modules(path.read_text(encoding="utf-8"), _package_of(path))
    )

    # Assert
    assert len(sources) > len(QBR_IMPORTERS_ALLOWED)
    assert set(importers) <= QBR_IMPORTERS_ALLOWED


def test_no_command_the_weekly_run_or_the_web_app_loads_espn_qbr() -> None:
    """Importing every command module and the web app leaves the QBR loader unloaded."""
    # Arrange
    modules = sorted({command.module for command in front_door.COMMANDS} | EXTRA_ENTRY_MODULES)
    script = (
        "import importlib\nimport sys\n"
        f"for module in {modules!r}:\n    importlib.import_module(module)\n"
        f"print({espn_qbr.__name__!r} in sys.modules)\n"
    )

    # Act
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )

    # Assert
    assert "nfl_predictor.lines_refresh" in modules
    assert "nfl_predictor.weekly_run.pipeline" in modules
    assert result.stdout.strip() == "False"
