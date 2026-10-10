import { describe, expect, it } from 'vitest'

import type { MetricCategory, MetricColumnMeta, MetricTable, Row } from '@/api/types'

import { gameLogColumns, resolveViewState, seasonView, VIEWS } from './views'

function column(key: string, overrides: Partial<MetricColumnMeta>): MetricColumnMeta {
  return {
    key,
    label: key,
    description: `${key} described.`,
    group: 'per_game_rates',
    kind: 'float',
    polarity: 'higher',
    heatmap: true,
    decimals: null,
    actionable: true,
    sticky: false,
    full_name: key,
    category: 'Offense',
    subcategory: 'Passing',
    contextual: false,
    shape: 'count',
    percent: false,
    ...overrides,
  }
}

const TEAM_CATEGORIES: MetricCategory[] = [
  { name: 'Schedule-Adjusted Ratings', description: 'Ratings.', subcategories: [] },
  { name: 'Overall', description: 'Overall.', subcategories: [] },
  { name: 'Offense', description: 'Offense.', subcategories: ['Total', 'Passing', 'Rushing'] },
  { name: 'Defense', description: 'Defense.', subcategories: ['Passing', 'Rushing'] },
  { name: 'Special Teams', description: 'Kicks.', subcategories: [] },
]

const QB_CATEGORIES: MetricCategory[] = [
  { name: 'Schedule-Adjusted Ratings', description: 'Ratings.', subcategories: [] },
  { name: 'Passing Volume', description: 'Volume.', subcategories: [] },
  { name: 'Passing Efficiency', description: 'Efficiency.', subcategories: [] },
]

const META: Record<string, MetricColumnMeta> = {
  team: column('team', { kind: 'text', shape: 'id', category: 'Overall', subcategory: null, group: 'identity', polarity: 'neutral' }),
  team_rating: column('team_rating', { shape: 'score', category: 'Schedule-Adjusted Ratings', subcategory: null, group: 'ratings' }),
  sos: column('sos', { shape: 'score', category: 'Schedule-Adjusted Ratings', subcategory: null, group: 'ratings', contextual: true }),
  games_played: column('games_played', { kind: 'int', category: 'Overall', subcategory: null, polarity: 'neutral' }),
  points_for: column('points_for', { category: 'Overall', subcategory: null }),
  passing_yards: column('passing_yards', {}),
  completion_pct: column('completion_pct', { shape: 'rate', percent: true, kind: 'prob' }),
  rushing_yards: column('rushing_yards', { subcategory: 'Rushing' }),
  longest_pass: column('longest_pass', { shape: 'max' }),
  passing_yards_per_offensive_snap: column('passing_yards_per_offensive_snap', { group: 'per_snap_rates' }),
  passing_yards_allowed: column('passing_yards_allowed', { category: 'Defense', polarity: 'lower' }),
  opp_passing_yards: column('opp_passing_yards', { contextual: true, group: 'opponent_context' }),
  opp_completion_pct: column('opp_completion_pct', { contextual: true, shape: 'rate', percent: true, group: 'opponent_context' }),
}

const TEAM_TABLE: MetricTable = {
  rows: [
    { team: 'BUF', team_rating: 6.5, sos: 0.5, games_played: 4, points_for: 27.5, passing_yards: 250.25, completion_pct: 0.68, rushing_yards: 120, longest_pass: 61, passing_yards_per_offensive_snap: 3.9, passing_yards_allowed: 201, opp_passing_yards: 220, opp_completion_pct: 0.64 },
    { team: 'KC', team_rating: 4, sos: -0.2, games_played: 3, points_for: 24, passing_yards: 270, completion_pct: 0.7, rushing_yards: 101, longest_pass: 55, passing_yards_per_offensive_snap: 4.1, passing_yards_allowed: 230, opp_passing_yards: 210, opp_completion_pct: 0.62 },
  ],
  visible_columns: Object.keys(META),
  column_groups: {
    identity: ['team'],
    ratings: ['team_rating', 'sos'],
    rating_companions: [],
    per_game_rates: ['games_played', 'points_for', 'passing_yards', 'completion_pct', 'rushing_yards', 'longest_pass', 'passing_yards_allowed'],
    per_snap_rates: ['passing_yards_per_offensive_snap'],
    opponent_context: ['opp_passing_yards', 'opp_completion_pct'],
  },
  column_metadata: META,
}

describe('VIEWS', () => {
  it('offers the ratings, the stat views and the opponent views', () => {
    // Act
    const ids = VIEWS.map((view) => view.id)

    // Assert
    expect(ids).toEqual(['ratings', 'per_game', 'per_play', 'totals', 'opponent_per_game', 'opponent_per_play'])
  })
})

describe('resolveViewState', () => {
  it('opens a team page on the ratings and the offense', () => {
    // Act
    const state = resolveViewState('teams', TEAM_CATEGORIES, {})

    // Assert
    expect(state.view).toBe('ratings')
    expect(state.category).toBe('Offense')
    expect(state.categoryOptions).toEqual(['Overall', 'Offense', 'Defense', 'Special Teams'])
    expect(state.subcategoryOptions).toEqual(['Total', 'Passing', 'Rushing'])
  })

  it('falls back to the offense when the asked category does not exist', () => {
    // Act
    const state = resolveViewState('teams', TEAM_CATEGORIES, { view: 'per_game', category: 'Coaching' })

    // Assert
    expect(state.category).toBe('Offense')
  })

  it('offers a quarterback page its categories as the subcategory filter', () => {
    // Act
    const state = resolveViewState('qbs', QB_CATEGORIES, { view: 'per_game' })

    // Assert
    expect(state.categoryOptions).toEqual([])
    expect(state.subcategoryOptions).toEqual(['Passing Volume', 'Passing Efficiency'])
  })
})

describe('seasonView', () => {
  it('shows the identity and ratings columns in the ratings view', () => {
    // Arrange
    const state = resolveViewState('teams', TEAM_CATEGORIES, { view: 'ratings' })

    // Act
    const view = seasonView('teams', TEAM_TABLE, state)

    // Assert
    expect(view.columns).toEqual(['team', 'team_rating', 'sos'])
  })

  it('shows the per-game counts and rates of the chosen category and subcategories', () => {
    // Arrange
    const state = resolveViewState('teams', TEAM_CATEGORIES, { view: 'per_game', category: 'Offense', hidden: ['Rushing'] })

    // Act
    const view = seasonView('teams', TEAM_TABLE, state)

    // Assert
    expect(view.metricColumns).toEqual(['passing_yards', 'completion_pct', 'longest_pass'])
  })

  it('shows the per-snap stats and rates in the per-play view', () => {
    // Arrange
    const state = resolveViewState('teams', TEAM_CATEGORIES, { view: 'per_play', category: 'Offense' })

    // Act
    const view = seasonView('teams', TEAM_TABLE, state)

    // Assert
    expect(view.metricColumns).toEqual(['completion_pct', 'passing_yards_per_offensive_snap'])
  })

  it('turns per-game counts into season totals and keeps rates and maxima', () => {
    // Arrange
    const state = resolveViewState('teams', TEAM_CATEGORIES, { view: 'totals', category: 'Offense' })

    // Act
    const view = seasonView('teams', TEAM_TABLE, state)

    // Assert
    const buffalo = view.table.rows[0] as Row
    expect(buffalo.passing_yards).toBe(1001)
    expect(buffalo.completion_pct).toBe(0.68)
    expect(buffalo.longest_pass).toBe(61)
  })

  it('shows only the opponent profiles in the opponent views', () => {
    // Arrange
    const state = resolveViewState('teams', TEAM_CATEGORIES, { view: 'opponent_per_game', category: 'Offense' })

    // Act
    const view = seasonView('teams', TEAM_TABLE, state)

    // Assert
    expect(view.metricColumns).toEqual(['opp_passing_yards', 'opp_completion_pct'])
  })

  it('narrows a quarterback page by category', () => {
    // Arrange
    const qbMeta = {
      qb_name: column('qb_name', { kind: 'text', shape: 'id', category: 'Identity & Availability', group: 'identity', sticky: true }),
      qb_attempts_per_game: column('qb_attempts_per_game', { category: 'Passing Volume', subcategory: null }),
      qb_passer_rating: column('qb_passer_rating', { shape: 'rate', category: 'Passing Efficiency', subcategory: null }),
    }
    const table: MetricTable = {
      rows: [{ qb_name: 'J.Allen', qb_attempts_per_game: 33, qb_passer_rating: 101.4 }],
      visible_columns: Object.keys(qbMeta),
      column_groups: { identity: ['qb_name'], ratings: [], per_game_rates: ['qb_attempts_per_game', 'qb_passer_rating'] },
      column_metadata: qbMeta,
    }
    const state = resolveViewState('qbs', QB_CATEGORIES, { view: 'per_game', hidden: ['Passing Volume'] })

    // Act
    const view = seasonView('qbs', table, state)

    // Assert
    expect(view.columns).toEqual(['qb_name', 'qb_passer_rating'])
  })
})

describe('gameLogColumns', () => {
  const logMeta: Record<string, MetricColumnMeta> = {
    game_id: column('game_id', { kind: 'text', shape: 'id', category: 'Overall', group: 'identity' }),
    week: column('week', { kind: 'text', shape: 'id', category: 'Overall', group: 'identity' }),
    opponent_team: column('opponent_team', { kind: 'text', shape: 'id', category: 'Overall', group: 'identity' }),
    points_for: column('points_for', { category: 'Overall', subcategory: null, group: 'results' }),
    point_margin: column('point_margin', { category: 'Overall', subcategory: null, group: 'results' }),
    passing_yards: column('passing_yards', { group: 'raw_totals' }),
    opp_defense_rating: column('opp_defense_rating', { shape: 'score', contextual: true, category: 'Schedule-Adjusted Ratings', group: 'opponent_ratings' }),
    opp_team_rating: column('opp_team_rating', { shape: 'score', contextual: true, category: 'Schedule-Adjusted Ratings', group: 'opponent_ratings' }),
  }
  const logs: MetricTable = {
    rows: [],
    visible_columns: Object.keys(logMeta),
    column_groups: {},
    column_metadata: logMeta,
  }

  it('leads with the game, the result and the opponent defense on the offense view', () => {
    // Arrange
    const state = resolveViewState('teams', TEAM_CATEGORIES, { view: 'per_game', category: 'Offense' })

    // Act
    const selection = gameLogColumns('teams', logs, state)

    // Assert
    expect(selection.columns).toEqual(['week', 'opponent_team', 'points_for', 'point_margin', 'opp_defense_rating', 'opp_team_rating', 'passing_yards'])
    expect(selection.metricColumns).toEqual(['passing_yards'])
  })

  it('reads the ratings view as per game', () => {
    // Arrange
    const state = resolveViewState('teams', TEAM_CATEGORIES, { view: 'ratings', category: 'Offense' })

    // Act
    const selection = gameLogColumns('teams', logs, state)

    // Assert
    expect(selection.metricColumns).toEqual(['passing_yards'])
  })
})
