/**
 * Which columns the team and quarterback pages show: one of six views, then (outside the ratings)
 * a category and its subcategories. Categories come from the metric registry with each table.
 */
import type { EntityKind, MetricCategory, MetricColumnMeta, MetricTable, Row } from '@/api/types'

import { entityConfig } from './entities'

export type StatView = 'ratings' | 'per_game' | 'per_play' | 'totals' | 'opponent_per_game' | 'opponent_per_play'

export interface ViewOption {
  id: StatView
  label: string
  description: (kind: EntityKind) => string
}

export const VIEWS: ViewOption[] = [
  {
    id: 'ratings',
    label: 'Ratings',
    description: (kind) =>
      kind === 'teams'
        ? 'The schedule-adjusted ratings, schedule strength and SRS.'
        : 'The schedule-adjusted rating, the pass defense faced, and the unadjusted rate and sample behind them.',
  },
  {
    id: 'per_game',
    label: 'Per game',
    description: (kind) =>
      kind === 'teams'
        ? 'Counting stats as per-game averages over the games played. Rates such as completion percentage are shown as they are.'
        : 'Counting stats divided by the games he played. Rates such as passer rating are shown as they are.',
  },
  {
    id: 'per_play',
    label: 'Per play',
    description: (kind) =>
      kind === 'teams'
        ? 'Stats divided by the plays behind them (per snap, dropback, pass attempt, carry or drive), so busy and slow teams compare fairly.'
        : 'Stats divided by the plays behind them (per dropback, pass attempt or carry), plus rates such as passer rating.',
  },
  {
    id: 'totals',
    label: 'Season totals',
    description: () => 'Season totals for counting stats such as yards, touchdowns and sacks. Rates and averages are shown as they are.',
  },
  {
    id: 'opponent_per_game',
    label: 'Opponents per game',
    description: (kind) =>
      kind === 'teams'
        ? 'What this team’s opponents did per game in their other games, each opponent counted once: the schedule, not this team.'
        : 'What the defenses he faced did per game in their other games: points allowed, sacks, interceptions and what they allowed other passers.',
  },
  {
    id: 'opponent_per_play',
    label: 'Opponents per play',
    description: (kind) =>
      kind === 'teams'
        ? 'The same opponent profile per play (per snap, dropback, pass attempt, carry or drive).'
        : 'The same profile of the defenses he faced, per dropback, pass attempt or carry.',
  },
]

/** The view, category and hidden subcategories asked for, from the URL or the page's state. */
export interface ViewRequest {
  view?: StatView | null
  category?: string | null
  hidden?: string[]
}

/** A view request checked against the entity's categories. */
export interface ViewState {
  view: StatView
  /** The team category shown outside the ratings view; empty for quarterbacks. */
  category: string
  categoryOptions: string[]
  /** The subcategories (teams) or categories (quarterbacks) that can be switched off. */
  subcategoryOptions: string[]
  hidden: string[]
}

export interface SeasonViewTable {
  /** The identity columns shown, then `metricColumns`. */
  columns: string[]
  metricColumns: string[]
  /** The table with its rows as the view shows them (season totals multiplied out). */
  table: MetricTable
}

const RATINGS_CATEGORY = 'Schedule-Adjusted Ratings'
const DEFAULT_TEAM_CATEGORY = 'Offense'
const PLAY_SUFFIXES = [
  '_per_offensive_snap',
  '_per_defensive_snap',
  '_per_dropback',
  '_per_attempt',
  '_per_carry',
  '_per_drive',
  '_per_series',
  '_per_target',
  '_per_reception',
  '_per_return',
  '_per_punt',
  '_per_kickoff',
  '_per_fg_att',
]
const ALREADY_TOTALS = new Set(['games_played', 'games', 'wins', 'losses', 'ties'])
const LOG_IDENTITY = ['week', 'opponent_team']
const LOG_RESULTS = ['points_for', 'points_allowed', 'point_margin', 'win_value', 'turnover_margin']

export function isStatView(value: string | null | undefined): value is StatView {
  return VIEWS.some((view) => view.id === value)
}

/** Check a view request against the entity's categories: unknown views and categories fall back. */
export function resolveViewState(kind: EntityKind, categories: MetricCategory[], request: ViewRequest): ViewState {
  const statCategories = categories.filter((category) => category.name !== RATINGS_CATEGORY)
  const view = request.view && isStatView(request.view) ? request.view : 'ratings'
  if (kind === 'qbs') {
    return {
      view,
      category: '',
      categoryOptions: [],
      subcategoryOptions: statCategories.map((category) => category.name),
      hidden: request.hidden ?? [],
    }
  }
  const names = statCategories.map((category) => category.name)
  const category = request.category && names.includes(request.category) ? request.category : DEFAULT_TEAM_CATEGORY
  return {
    view,
    category,
    categoryOptions: names,
    subcategoryOptions: statCategories.find((candidate) => candidate.name === category)?.subcategories ?? [],
    hidden: request.hidden ?? [],
  }
}

function isPlayColumn(column: string): boolean {
  return PLAY_SUFFIXES.some((suffix) => column.endsWith(suffix))
}

function isTotalColumn(column: string): boolean {
  return column.endsWith('_total')
}

function isPerGameColumn(column: string): boolean {
  return column.endsWith('_per_game')
}

/** Whether a column belongs to the view's category and the subcategories switched on. */
function matchesTaxonomy(kind: EntityKind, meta: MetricColumnMeta | undefined, state: ViewState): boolean {
  if (!meta?.category || meta.category === RATINGS_CATEGORY) return false
  if (kind === 'qbs') return !state.hidden.includes(meta.category)
  if (meta.category !== state.category) return false
  if (state.subcategoryOptions.length === 0 || !meta.subcategory) return true
  return !state.hidden.includes(meta.subcategory)
}

/** Whether a column fits the view: its scale (total, per game, per play) and context flag. */
function matchesView(column: string, meta: MetricColumnMeta | undefined, view: StatView): boolean {
  if (!meta || view === 'ratings') return false
  if (view.startsWith('opponent_') !== meta.contextual) return false
  const shape = meta.shape
  switch (view) {
    case 'totals':
      if (isTotalColumn(column)) return true
      if (isPerGameColumn(column) || isPlayColumn(column)) return false
      return shape !== 'score' && shape !== 'id'
    case 'per_game':
    case 'opponent_per_game':
      if (isTotalColumn(column) || isPlayColumn(column)) return false
      return shape !== 'score' && shape !== 'id'
    case 'per_play':
    case 'opponent_per_play':
      if (isPlayColumn(column)) return true
      if (isTotalColumn(column) || isPerGameColumn(column)) return false
      return shape === 'rate' || shape === 'avg' || shape === 'flag'
  }
}

/**
 * In a team's season-totals view, a per-game count becomes its season total (times games played).
 * Rates and averages keep their own denominator, and a maximum (the longest play) is already the
 * season's largest game.
 */
function toSeasonTotals(row: Row, columns: string[], metadata: Record<string, MetricColumnMeta>): Row {
  const games = typeof row.games_played === 'number' ? row.games_played : null
  if (games === null) return row
  const totals: Row = { ...row }
  for (const column of columns) {
    const value = row[column]
    if (
      metadata[column]?.shape === 'count' &&
      typeof value === 'number' &&
      Number.isFinite(value) &&
      !ALREADY_TOTALS.has(column) &&
      !isTotalColumn(column) &&
      !isPerGameColumn(column) &&
      !isPlayColumn(column)
    ) {
      totals[column] = Number((value * games).toFixed(6))
    }
  }
  return totals
}

/** The season table as one view shows it: its columns, and its rows with totals multiplied out. */
export function seasonView(kind: EntityKind, table: MetricTable, state: ViewState): SeasonViewTable {
  const groups = table.column_groups
  const identity = groups.identity ?? []
  const ratings = groups.ratings ?? []
  const shownIdentity = identity.filter((column) => entityConfig(kind).identityColumns.includes(column))
  const metricColumns =
    state.view === 'ratings'
      ? [...ratings, ...(groups.rating_companions ?? [])]
      : table.visible_columns.filter(
          (column) =>
            !identity.includes(column) &&
            !ratings.includes(column) &&
            matchesTaxonomy(kind, table.column_metadata[column], state) &&
            matchesView(column, table.column_metadata[column], state.view),
        )
  const rows =
    kind === 'teams' && state.view === 'totals'
      ? table.rows.map((row) => toSeasonTotals(row, metricColumns, table.column_metadata))
      : table.rows
  return { columns: [...shownIdentity, ...metricColumns], metricColumns, table: { ...table, rows } }
}

/** The opponent ratings each game shows beside the stats a view judges against them. */
function opponentContext(kind: EntityKind, available: string[], state: ViewState): string[] {
  const candidates =
    kind === 'qbs' || state.category === 'Offense'
      ? ['opp_defense_rating', 'opp_team_rating']
      : state.category === 'Defense'
        ? ['opp_offense_rating', 'opp_team_rating']
        : ['opp_team_rating', 'opp_SRS']
  return candidates.filter((column) => available.includes(column))
}

/**
 * The game log's columns for a view: the week and opponent, the score and result, the opponent's
 * season ratings that matter for the view, then the view's stats (the ratings view reads as per
 * game). In an opponent view the stats are the opponent ratings themselves.
 */
export function gameLogColumns(kind: EntityKind, logs: MetricTable, state: ViewState): { columns: string[]; metricColumns: string[] } {
  const view: StatView = state.view === 'ratings' ? 'per_game' : state.view
  const visible = logs.visible_columns
  const identity = LOG_IDENTITY.filter((column) => visible.includes(column))
  const results = LOG_RESULTS.filter((column) => visible.includes(column))
  const context = opponentContext(kind, visible, state)
  const opponentRatings = new Set(logs.column_groups.opponent_ratings ?? context)
  const metricColumns = view.startsWith('opponent_')
    ? context
    : visible.filter(
        (column) =>
          !identity.includes(column) &&
          !results.includes(column) &&
          !opponentRatings.has(column) &&
          matchesTaxonomy(kind, logs.column_metadata[column], state) &&
          matchesView(column, logs.column_metadata[column], view),
      )
  const shownContext = view.startsWith('opponent_') ? [] : context
  return { columns: [...new Set([...identity, ...results, ...shownContext, ...metricColumns])], metricColumns }
}
