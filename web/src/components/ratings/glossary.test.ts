import { describe, expect, it } from 'vitest'

import type { MetricRegistry, RegistryMetric } from '@/api/types'

import { buildGlossary, filterGlossary } from './glossary'

function metric(overrides: Partial<RegistryMetric>): RegistryMetric {
  return {
    label: 'Label',
    full_name: 'Full Name',
    description: 'What it measures, in plain words.',
    entity: 'team',
    category: 'Offense',
    subcategory: null,
    shape: 'count',
    polarity: 'higher',
    source: 'play-by-play',
    denominator: null,
    since: null,
    duplicate_of: null,
    contextual: false,
    formula: null,
    note: null,
    percent: false,
    ...overrides,
  }
}

const REGISTRY: MetricRegistry = {
  entities: {
    team: {
      categories: [
        { name: 'Schedule-Adjusted Ratings', description: 'The ratings.', subcategories: [] },
        { name: 'Offense', description: 'What the offense did.', subcategories: ['Passing'] },
      ],
    },
    qb: { categories: [{ name: 'Passing Efficiency', description: 'Quality per play.', subcategories: [] }] },
  },
  metrics: {
    team_rating: metric({ label: 'Team Rating', full_name: 'Team Rating', category: 'Schedule-Adjusted Ratings', shape: 'score' }),
    sos: metric({ label: 'SoS', full_name: 'Strength of Schedule', category: 'Schedule-Adjusted Ratings', contextual: true }),
    passing_yards: metric({ label: 'Pass Yds', full_name: 'Passing Yards', formula: 'Sum of passing yards.', since: 2006 }),
    net_passing_yards: metric({ label: 'Net Pass Yds', full_name: 'Net Passing Yards', duplicate_of: 'passing_yards', since: 1999 }),
    qb_passer_rating: metric({ entity: 'qb', label: 'Rating', full_name: 'Passer Rating', category: 'Passing Efficiency', polarity: 'higher' }),
    turnovers: metric({ label: 'TO', full_name: 'Turnovers', polarity: 'lower', category: 'Offense' }),
  },
  prefix_rules: [],
}

describe('buildGlossary', () => {
  it('starts with the ideas and the headline ratings, then each entity’s categories', () => {
    // Act
    const sections = buildGlossary(REGISTRY)

    // Assert
    expect(sections.map((section) => section.title)).toEqual(['Start here', 'Teams: Offense', 'Quarterbacks: Passing Efficiency'])
    const start = sections[0].entries.map((entry) => entry.id)
    expect(start).toContain('epa')
    expect(start.slice(-2)).toEqual(['team_rating', 'sos'])
  })

  it('says which way is better, or that a column is context', () => {
    // Arrange
    const sections = buildGlossary(REGISTRY)
    const entries = sections.flatMap((section) => section.entries)

    // Act
    const directions = Object.fromEntries(entries.map((entry) => [entry.id, entry.direction]))

    // Assert
    expect(directions.turnovers).toBe('Lower is better.')
    expect(directions.qb_passer_rating).toBe('Higher is better.')
    expect(directions.sos).toBe('Context, not a grade: it describes the opposition, not this team or quarterback.')
  })

  it('carries the table label, the formula, the first season and the metric repeated', () => {
    // Arrange
    const sections = buildGlossary(REGISTRY)
    const entries = Object.fromEntries(sections.flatMap((section) => section.entries).map((entry) => [entry.id, entry]))

    // Act
    const passing = entries.passing_yards
    const net = entries.net_passing_yards

    // Assert
    expect([passing.shownAs, passing.formula, passing.since]).toEqual(['Pass Yds', 'Sum of passing yards.', 2006])
    expect([net.sameAs, net.since]).toEqual(['Passing Yards', null])
  })

  it('leaves out the garbage-time filter, which this app does not show', () => {
    // Act
    const ids = buildGlossary(REGISTRY).flatMap((section) => section.entries.map((entry) => entry.id))

    // Assert
    expect(ids).not.toContain('garbage-time')
  })
})

describe('filterGlossary', () => {
  it('keeps the entries whose name, label or description holds the query, in any case', () => {
    // Arrange
    const sections = buildGlossary(REGISTRY)

    // Act
    const shown = filterGlossary(sections, '  PASSER ')

    // Assert
    expect(shown.map((section) => [section.title, section.entries.map((entry) => entry.id)])).toEqual([
      ['Quarterbacks: Passing Efficiency', ['qb_passer_rating']],
    ])
  })

  it('keeps everything for an empty query', () => {
    // Arrange
    const sections = buildGlossary(REGISTRY)

    // Act
    const shown = filterGlossary(sections, ' ')

    // Assert
    expect(shown).toBe(sections)
  })
})
