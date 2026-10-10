/**
 * One row per opponent from a team's or quarterback's game log: how often they met, when, the
 * results, the opponent's season ratings and league tier, and the average score of those games.
 * An opponent met twice (a division rival) gets one row.
 */
import type { MetricColumnMeta, MetricTable, Row } from '@/api/types'

const OPPONENT_RATINGS = ['opp_team_rating', 'opp_team_rank', 'opp_offense_rating', 'opp_defense_rating', 'opp_special_teams_rating', 'opp_SRS']
const RANK_COLUMN = 'opp_team_rank'
const TIER_COLUMN = 'opp_tier'
/** Opponents ranked this high (or this low) on Team Rating read as Tougher (or Softer). */
const TIER_SIZE = 10
const AVERAGED = ['points_for', 'points_allowed', 'point_margin']
const RESULT_LETTERS: Record<number, string> = { 1: 'W', 0: 'L', 0.5: 'T' }
const AVERAGE_NOTE = 'Averaged over the games against this opponent.'

function local(key: string, label: string, description: string, kind: MetricColumnMeta['kind']): MetricColumnMeta {
  return {
    key,
    label,
    description,
    group: 'opponents',
    kind,
    polarity: 'neutral',
    heatmap: false,
    decimals: kind === 'int' ? 0 : null,
    actionable: true,
    sticky: false,
    full_name: label,
    category: null,
    subcategory: null,
    contextual: false,
    shape: kind === 'int' ? 'count' : 'id',
    percent: false,
  }
}

const LOCAL_COLUMNS: Record<string, MetricColumnMeta> = {
  games: local('games', 'Games', 'Games against this opponent this season.', 'int'),
  weeks: local('weeks', 'Weeks', 'The weeks of the games against this opponent.', 'text'),
  results: local('results', 'Results', 'Each game’s result in week order: W won, L lost, T tied.', 'text'),
}

const TIER_META = local(
  TIER_COLUMN,
  'Opp Tier',
  `Where this opponent ranked in the league on Team Rating over its whole season: Tougher is the top ${TIER_SIZE}, Softer the bottom ${TIER_SIZE}, and Middle the rest.`,
  'text',
)

function average(values: number[]): number {
  return Number((values.reduce((total, value) => total + value, 0) / values.length).toFixed(6))
}

function numbers(rows: Row[], column: string): number[] {
  return rows.map((row) => row[column]).filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
}

/** Tougher for a top-10 team, Softer for a bottom-10 team, Middle otherwise. */
function tier(rank: Row[string], leagueSize: number): string | null {
  if (typeof rank !== 'number') return null
  if (rank <= TIER_SIZE) return 'Tougher'
  if (rank > leagueSize - TIER_SIZE) return 'Softer'
  return 'Middle'
}

/**
 * The ledger of one game log, one row per opponent, in opponent order. With `leagueSize` (the
 * teams the opponents' ranks run to) each opponent also gets its tier.
 */
export function opponentLedger(logs: MetricTable, leagueSize: number | null): MetricTable {
  const byOpponent = new Map<string, Row[]>()
  const ordered = [...logs.rows].sort((left, right) => Number(left.week ?? 0) - Number(right.week ?? 0))
  for (const row of ordered) {
    const opponent = String(row.opponent_team ?? '')
    byOpponent.set(opponent, [...(byOpponent.get(opponent) ?? []), row])
  }
  const ratings = OPPONENT_RATINGS.filter((column) => logs.visible_columns.includes(column))
  const tiered = leagueSize !== null && ratings.includes(RANK_COLUMN)
  const averaged = AVERAGED.filter((column) => logs.visible_columns.includes(column))
  const rows: Row[] = [...byOpponent.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([opponent, games]) => {
      const row: Row = {
        opponent_team: opponent,
        games: games.length,
        weeks: games.map((game) => game.week).join(', '),
        results: numbers(games, 'win_value')
          .map((value) => RESULT_LETTERS[value] ?? '?')
          .join(', '),
      }
      for (const column of ratings) row[column] = games.find((game) => game[column] !== null && game[column] !== undefined)?.[column] ?? null
      if (tiered) row[TIER_COLUMN] = tier(row[RANK_COLUMN] ?? null, leagueSize)
      for (const column of averaged) {
        const values = numbers(games, column)
        row[column] = values.length > 0 ? average(values) : null
      }
      return row
    })
  const metadata: Record<string, MetricColumnMeta> = { ...LOCAL_COLUMNS }
  const opponentMeta = logs.column_metadata.opponent_team
  if (opponentMeta) metadata.opponent_team = { ...opponentMeta, sticky: true, group: 'opponents' }
  for (const column of ratings) {
    const meta = logs.column_metadata[column]
    if (meta) metadata[column] = { ...meta, group: 'opponents' }
  }
  if (tiered) metadata[TIER_COLUMN] = TIER_META
  for (const column of averaged) {
    const meta = logs.column_metadata[column]
    if (meta) metadata[column] = { ...meta, kind: 'float', group: 'opponents', description: `${meta.description} ${AVERAGE_NOTE}` }
  }
  const ratingColumns = ratings.flatMap((column) => (column === RANK_COLUMN && tiered ? [column, TIER_COLUMN] : [column]))
  const visible = ['opponent_team', 'games', 'weeks', 'results', ...ratingColumns, ...averaged].filter((column) => column in metadata)
  return { rows, visible_columns: visible, column_groups: { opponents: visible }, column_metadata: metadata }
}
