"""Payloads for the team and quarterback pages: ratings, statistics, ranges and schedules."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from nfl_predictor.api.schemas.common import ColumnMetaOut, Polarity

EntityKind = Literal["teams", "qbs"]
"""Which page family a route serves: the team pages or the quarterback pages."""
MetricShape = Literal["count", "rate", "avg", "max", "flag", "id", "score"]
"""How a registry metric behaves across views (``season_stats.metrics.schema.Shape``)."""
RegistryEntity = Literal["team", "qb"]


class MetricColumnOut(ColumnMetaOut):
    """One column in the app's column format, plus what the metric registry says about it.

    ``full_name`` is the long name, ``category`` and ``subcategory`` the column's place in the
    registry's taxonomy, ``contextual`` marks context (the opponents faced, schedule strength)
    rather than a grade, ``shape`` how the stat behaves across views, and ``percent`` a proportion
    shown as a percentage. ``decimals`` stays null: the page picks them from the values it shows.
    """

    full_name: str
    category: str | None = None
    subcategory: str | None = None
    contextual: bool = False
    shape: MetricShape | None = None
    percent: bool = False


class MetricTableOut(BaseModel):
    """Rows plus registry metadata, in the shape of the app's ``TablePayload``."""

    rows: list[dict[str, Any]]
    visible_columns: list[str]
    column_groups: dict[str, list[str]]
    column_metadata: dict[str, MetricColumnOut]


class CategoryOut(BaseModel):
    """A display category of one entity's metrics, with its ordered subcategories."""

    name: str
    description: str
    subcategories: list[str]


class RatingsSeasonsOut(BaseModel):
    """The seasons with team and quarterback tables, newest first, and the season under way."""

    seasons: list[int]
    current_season: int


class EntityTableOut(BaseModel):
    """One season's team or quarterback table, for the index and detail pages.

    ``in_progress`` is true only for the season being played while a team still has
    regular-season games left; ``games_played`` is the most games any team has played and
    ``games_per_team`` a full regular season. ``categories`` lists the entity's metric categories
    in display order, the ratings category first, and ``team_names`` the franchise name of every
    team code in the table.
    """

    kind: EntityKind
    season: int
    in_progress: bool
    games_played: int | None
    games_per_team: int
    table: MetricTableOut
    categories: list[CategoryOut]
    team_names: dict[str, str]


class SeasonTableOut(BaseModel):
    """One season table for a page section: a league-wide one, or one team's or quarterback's."""

    kind: EntityKind
    season: int
    entity_id: str | None
    table: MetricTableOut


class ScheduleRanksOut(BaseModel):
    """Every completed team-season ranked on schedule strength (``sos``), softest first.

    ``seasons`` are the seasons ranked; ``skipped_seasons`` have ratings but are still in progress.
    """

    count: int
    seasons: list[int]
    skipped_seasons: list[int]
    table: MetricTableOut


class RegistryCategoryOut(BaseModel):
    """A metric category as the registry defines it."""

    name: str
    description: str
    subcategories: list[str]


class RegistryEntityOut(BaseModel):
    """One entity's metric categories, in display order."""

    categories: list[RegistryCategoryOut]


class RegistryMetricOut(BaseModel):
    """One metric of the registry, with every field the glossary shows."""

    label: str
    full_name: str
    description: str
    entity: RegistryEntity
    category: str
    subcategory: str | None
    shape: MetricShape
    polarity: Polarity
    source: str
    denominator: str | None
    since: int | None
    duplicate_of: str | None
    contextual: bool
    formula: str | None
    note: str | None
    percent: bool


class PrefixRuleOut(BaseModel):
    """How a column prefix (``opp_``, ``qopp_``, ...) changes a metric's name and meaning."""

    prefix: str
    label_template: str
    full_name_template: str
    description_note: str
    contextual: bool
    invert_polarity_for_qb: bool


class MetricRegistryOut(BaseModel):
    """The metric registry the glossary is built from."""

    entities: dict[RegistryEntity, RegistryEntityOut]
    metrics: dict[str, RegistryMetricOut]
    prefix_rules: list[PrefixRuleOut]
