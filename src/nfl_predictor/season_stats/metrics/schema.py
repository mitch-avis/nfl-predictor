"""Typed building blocks for the metric registry single source of truth.

Every stat, rating, and metric the project publishes is defined once as a
:class:`MetricDef`. Concrete data columns are either exact metric names or
affix products (a prefix such as ``opp_`` and/or a suffix such as
``_per_game`` around a base metric), resolved by the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, Protocol, TypedDict, Unpack

Entity = Literal["team", "qb"]
"""Which page family a metric belongs to: the Teams pages or the QBs pages."""

Shape = Literal["count", "rate", "avg", "max", "flag", "id", "score"]
"""How a metric behaves across views.

- ``count``: a summable total (yards, touchdowns). Valid in every view.
- ``rate``: an intrinsic ratio with its own denominator. Never divided again.
- ``avg``: a per-event mean re-averaged over events, not over weeks.
- ``max``: the largest single value, such as the longest play. The season row keeps the largest
  game value, so it is never summed, multiplied by games, or divided by plays.
- ``flag``: a boolean marker (eligibility, comeback credit).
- ``id``: identity text (team codes, player names, game ids).
- ``score``: a model output on its own scale (ratings and SRS in points per game).
"""

Polarity = Literal["higher", "lower", "neutral"]
"""Which end of the scale is good for the subject of the row."""


@dataclass(frozen=True, slots=True)
class MetricDef:
    """One stat/rating/metric definition — the single source of truth entry.

    ``percent`` marks a proportion, where 0.653 means 65.3%: a share of its denominator (completion
    percentage), events per play (havoc rate), a chance, or the difference of two shares. The
    analyst app shows it as a percentage; data files, the API, and CSV exports keep the proportion.
    A value already in percentage points (completion percentage above expectation) is not one.
    """

    name: str
    label: str
    full_name: str
    description: str
    entity: Entity
    category: str
    shape: Shape
    polarity: Polarity
    source: str
    subcategory: str | None = None
    denominator: str | None = None
    since: int | None = None
    duplicate_of: str | None = None
    contextual: bool = False
    formula: str | None = None
    note: str | None = None
    percent: bool = False


@dataclass(frozen=True, slots=True)
class CategoryDef:
    """A display category for one entity, with ordered subcategories."""

    name: str
    entity: Entity
    description: str
    subcategories: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PrefixRule:
    """How a column prefix transforms the base metric's presentation."""

    prefix: str
    label_template: str
    full_name_template: str
    description_note: str
    contextual: bool = False
    invert_polarity_for_qb: bool = False
    category_override: str | None = None


@dataclass(frozen=True, slots=True)
class SuffixRule:
    """How a column suffix transforms the base metric's presentation.

    ``polarity``, when set, replaces the base metric's polarity: a suffix that turns a grade into
    something else (a change, for example) says which end of the new scale is good, if either.
    """

    suffix: str
    label_template: str
    full_name_template: str
    description_note: str
    polarity: Polarity | None = None


@dataclass(frozen=True, slots=True)
class ResolvedColumn:
    """A concrete data column resolved to its base metric plus affix context."""

    column: str
    base: MetricDef
    label: str
    full_name: str
    description: str
    polarity: Polarity
    contextual: bool
    category: str
    subcategory: str | None


class MetricFields(TypedDict):
    """Keyword fields accepted by section builders (entity/category preset)."""

    name: str
    label: str
    full_name: str
    description: str
    shape: Shape
    polarity: Polarity
    source: str
    denominator: NotRequired[str | None]
    since: NotRequired[int | None]
    duplicate_of: NotRequired[str | None]
    contextual: NotRequired[bool]
    formula: NotRequired[str | None]
    note: NotRequired[str | None]
    percent: NotRequired[bool]


class MetricBuilder(Protocol):
    """A callable that builds metrics with entity/category/subcategory preset."""

    def __call__(self, **fields: Unpack[MetricFields]) -> MetricDef:
        """Build one metric definition from the remaining fields."""
        ...


def section(entity: Entity, category: str, subcategory: str | None = None) -> MetricBuilder:
    """Return a builder that stamps entity, category, and subcategory."""

    def build(**fields: Unpack[MetricFields]) -> MetricDef:
        """Build one metric definition inside the preset section."""
        return MetricDef(entity=entity, category=category, subcategory=subcategory, **fields)

    return build
