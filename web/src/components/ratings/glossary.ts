/**
 * The team and quarterback half of the glossary, built from the metric registry: a "Start here"
 * section with the ideas the pages rely on and the headline ratings, then every other metric by
 * entity and category in the registry's order, with a search over all of it.
 */
import type { MetricRegistry, Polarity, RegistryMetric } from '@/api/types'

export interface GlossaryEntry {
  id: string
  title: string
  /** The table label, when it differs from the full name. */
  shownAs: string | null
  description: string
  direction: string | null
  formula: string | null
  note: string | null
  /** The first season with data, when that is after the first play-by-play season. */
  since: number | null
  /** The full name of the metric this one repeats. */
  sameAs: string | null
}

export interface GlossarySection {
  id: string
  title: string
  description: string | null
  entries: GlossaryEntry[]
}

const FIRST_SEASON = 1999
const ENTITY_TITLES = { team: 'Teams', qb: 'Quarterbacks' } as const
/** The ratings each page ranks by and their schedule context, shown once, in Start here. */
const HEADLINE_METRICS = [
  'team_rating',
  'offense_rating',
  'defense_rating',
  'special_teams_rating',
  'sos',
  'SRS',
  'adj_qb_epa_per_dropback',
  'qb_epa_per_dropback',
  'qb_faced_pass_defense',
]

function concept(id: string, title: string, description: string): GlossaryEntry {
  return { id, title, shownAs: null, description, direction: null, formula: null, note: null, since: null, sameAs: null }
}

const CONCEPTS: GlossaryEntry[] = [
  concept(
    'epa',
    'Expected points added (EPA)',
    'How much a play changed the offense’s expected points, from nflverse’s model of down, distance, field position and time left. The ratings are built from EPA per play, so a 20-yard gain on 3rd and 25 counts for less than a 5-yard gain on 3rd and 2.',
  ),
  concept(
    'schedule-adjustment',
    'Schedule adjustment',
    'Every team or quarterback is rated against the opponents it actually faced, and each of those opponents against everyone it faced, in one fit. Early in a season a quarterback’s rating leans toward the league average, and a team’s (from 2003 on) also leans on its rating the season before, a little less after each game until its 9th.',
  ),
  concept(
    'schedule-strength',
    'Schedule strength',
    'SoS is the average Team Rating of the opponents a team played, each rated without its games against that team. Positive means a harder schedule than average. Pass Defense Faced is the same idea for a quarterback, weighted by his dropbacks.',
  ),
  concept(
    'opponent-profile',
    'Opponents faced',
    'The opponent columns (Opp ...) average what the opponents did in their games against everyone else, so a team’s own games against them do not count twice.',
  ),
  concept(
    'rank-range',
    'Rank range',
    'Where a team or quarterback ranks when the season’s games are redrawn at random, with repeats, and the ratings are refit each time. The middle 50% of those ranks shows how much a rank depends on which games happened to be played, not whether the method is right.',
  ),
  concept(
    'head-to-head',
    'Head-to-head chance',
    'How often one team or quarterback is rated above another across those redrawn seasons: 79% means it came out ahead in 79 of every 100 redraws.',
  ),
  concept(
    'shading',
    'Shading',
    'Green cells mark better within the season, deeper for better. Slate cells mark context, such as schedule strength or the opponents faced, deeper for tougher.',
  ),
]

/** Which way is better for a metric, for hints and the glossary; null when neither is. */
export function directionFor(polarity: Polarity, contextual: boolean): string | null {
  if (contextual) return 'Context, not a grade: it describes the opposition, not this team or quarterback.'
  if (polarity === 'higher') return 'Higher is better.'
  if (polarity === 'lower') return 'Lower is better.'
  return null
}

function metricEntry(name: string, metric: RegistryMetric, metrics: MetricRegistry['metrics']): GlossaryEntry {
  const duplicate = metric.duplicate_of ? metrics[metric.duplicate_of] : undefined
  return {
    id: name,
    title: metric.full_name,
    shownAs: metric.label !== metric.full_name ? metric.label : null,
    description: metric.description,
    direction: directionFor(metric.polarity, metric.contextual),
    formula: metric.formula,
    note: metric.note,
    since: metric.since !== null && metric.since > FIRST_SEASON ? metric.since : null,
    sameAs: duplicate ? duplicate.full_name : null,
  }
}

function slug(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
}

/** Start here, then one section per entity and category, in the registry's category order. */
export function buildGlossary(registry: MetricRegistry): GlossarySection[] {
  const { metrics } = registry
  const headline = HEADLINE_METRICS.filter((name) => name in metrics).map((name) => metricEntry(name, metrics[name], metrics))
  const sections: GlossarySection[] = [
    {
      id: 'start-here',
      title: 'Start here',
      description: 'The ideas the team and quarterback pages rely on, then the headline ratings and their schedule context.',
      entries: [...CONCEPTS, ...headline],
    },
  ]
  for (const entity of ['team', 'qb'] as const) {
    const named = registry.entities[entity].categories
    const byCategory = new Map<string, GlossaryEntry[]>()
    for (const [name, metric] of Object.entries(metrics)) {
      if (metric.entity !== entity || HEADLINE_METRICS.includes(name)) continue
      byCategory.set(metric.category, [...(byCategory.get(metric.category) ?? []), metricEntry(name, metric, metrics)])
    }
    const order = named.map((category) => category.name)
    const unnamed = [...byCategory.keys()].filter((category) => !order.includes(category)).sort()
    for (const category of [...order, ...unnamed]) {
      const entries = byCategory.get(category)
      if (!entries) continue
      sections.push({
        id: `${entity}-${slug(category)}`,
        title: `${ENTITY_TITLES[entity]}: ${category}`,
        description: named.find((candidate) => candidate.name === category)?.description ?? null,
        entries,
      })
    }
  }
  return sections
}

/** The sections narrowed to entries whose name, label or description holds `query`, in any case. */
export function filterGlossary(sections: GlossarySection[], query: string): GlossarySection[] {
  const needle = query.trim().toLowerCase()
  if (needle === '') return sections
  return sections
    .map((section) => ({
      ...section,
      entries: section.entries.filter((entry) =>
        [entry.id, entry.title, entry.shownAs ?? '', entry.description].some((text) => text.toLowerCase().includes(needle)),
      ),
    }))
    .filter((section) => section.entries.length > 0)
}
