"""Tests for the stats catalogs generated from the metric registry."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from nfl_predictor.season_stats.metrics import CategoryDef, MetricDef, catalog_docs
from nfl_predictor.season_stats.metrics.catalog_docs import (
    CATALOG_PATHS,
    REGENERATE_COMMAND,
    render_catalog,
    write_catalogs,
)
from nfl_predictor.season_stats.metrics.registry import MetricRegistry

if TYPE_CHECKING:
    from nfl_predictor.season_stats.metrics.schema import Entity

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_render_catalog_lists_the_headline_team_rating() -> None:
    # Act
    text = render_catalog("team")

    # Assert
    assert "| `team_rating` | Team Rating |" in text


def test_render_catalog_lists_qb_metrics_under_their_category() -> None:
    # Act
    text = render_catalog("qb")

    # Assert
    assert text.index("## Rushing") < text.index("| `qb_designed_carries` |")


def test_render_catalog_marks_itself_as_generated() -> None:
    # Act
    text = render_catalog("team")

    # Assert
    assert REGENERATE_COMMAND in "\n".join(text.splitlines()[2:4])


@pytest.mark.parametrize(("entity", "relative_path"), CATALOG_PATHS.items())
def test_committed_catalog_matches_the_registry(entity: Entity, relative_path: str) -> None:
    # Arrange
    committed = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    # Act
    generated = render_catalog(entity)

    # Assert
    assert committed == generated, f"run `{REGENERATE_COMMAND}` to refresh {relative_path}"


def test_render_catalog_skips_categories_without_metrics() -> None:
    # Arrange
    metric = MetricDef(
        name="alpha",
        label="Alpha",
        full_name="Alpha",
        description="A synthetic metric used only for catalog tests.",
        entity="team",
        category="Filled",
        shape="count",
        polarity="higher",
        source="D",
    )
    registry = MetricRegistry(
        [metric],
        [
            CategoryDef(name="Empty", entity="team", description="No metrics here."),
            CategoryDef(name="Filled", entity="team", description="One metric."),
        ],
    )

    # Act
    text = render_catalog("team", registry)

    # Assert
    assert "## Empty" not in text
    assert "## Filled" in text


def test_write_catalogs_writes_both_rendered_catalogs(tmp_path: Path) -> None:
    # Act
    written = write_catalogs(tmp_path)

    # Assert
    assert [path.relative_to(tmp_path).as_posix() for path in written] == list(
        CATALOG_PATHS.values()
    )
    assert written[0].read_text(encoding="utf-8") == render_catalog("team")


def test_main_reports_each_written_catalog(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    def write_catalogs_stub() -> list[Path]:
        return [Path("docs/x.md")]

    monkeypatch.setattr(catalog_docs, "write_catalogs", write_catalogs_stub)
    caplog.set_level(logging.INFO)

    # Act
    catalog_docs.main()

    # Assert
    assert [record.getMessage() for record in caplog.records] == ["Wrote docs/x.md"]
