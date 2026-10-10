/** Small, real-shaped ratings payloads for the ratings page and component tests. */
import type { EntityTable, GameLogsTable, MetricCategory, MetricColumnMeta, MetricTable, RankRangeTable, Row, SeasonTable } from '@/api/types'

export function metricMeta(key: string, overrides: Partial<MetricColumnMeta> = {}): MetricColumnMeta {
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

export const TEAM_CATEGORIES: MetricCategory[] = [
  { name: 'Schedule-Adjusted Ratings', description: 'The ratings.', subcategories: [] },
  { name: 'Overall', description: 'Results.', subcategories: [] },
  { name: 'Offense', description: 'Offense.', subcategories: ['Passing', 'Rushing'] },
  { name: 'Defense', description: 'Defense.', subcategories: ['Passing'] },
]

export const QB_CATEGORIES: MetricCategory[] = [
  { name: 'Schedule-Adjusted Ratings', description: 'The ratings.', subcategories: [] },
  { name: 'Identity & Availability', description: 'Who.', subcategories: [] },
  { name: 'Passing Volume', description: 'Volume.', subcategories: [] },
]

const TEAM_META: Record<string, MetricColumnMeta> = {
  team: metricMeta('team', { label: 'Team', kind: 'text', shape: 'id', polarity: 'neutral', category: 'Overall', subcategory: null, group: 'identity', heatmap: false, sticky: true }),
  team_rating: metricMeta('team_rating', { label: 'Team Rating', shape: 'score', category: 'Schedule-Adjusted Ratings', subcategory: null, group: 'ratings' }),
  offense_rating: metricMeta('offense_rating', { label: 'Off Rating', shape: 'score', category: 'Schedule-Adjusted Ratings', subcategory: null, group: 'ratings' }),
  sos: metricMeta('sos', { label: 'SoS', shape: 'score', category: 'Schedule-Adjusted Ratings', subcategory: null, group: 'ratings', contextual: true }),
  games_played: metricMeta('games_played', { label: 'Games Played', kind: 'int', polarity: 'neutral', category: 'Overall', subcategory: null, heatmap: false }),
  epa_margin_per_play: metricMeta('epa_margin_per_play', { label: 'EPA Margin/Play', shape: 'rate', category: 'Overall', subcategory: null }),
  passing_yards: metricMeta('passing_yards', { label: 'Pass Yds' }),
  rushing_yards: metricMeta('rushing_yards', { label: 'Rush Yds', subcategory: 'Rushing' }),
  opp_passing_yards: metricMeta('opp_passing_yards', { label: 'Opp Pass Yds', contextual: true, group: 'opponent_context' }),
}

export const TEAM_ROWS: Row[] = [
  { team: 'BUF', team_rating: 6.5, offense_rating: 4.25, sos: 0.5, games_played: 17, epa_margin_per_play: 0.12, passing_yards: 250.5, rushing_yards: 130.25, opp_passing_yards: 221 },
  { team: 'KC', team_rating: 4, offense_rating: 3, sos: -0.25, games_played: 17, epa_margin_per_play: 0.09, passing_yards: 270, rushing_yards: 101, opp_passing_yards: 210 },
  { team: 'WSH', team_rating: -8.5, offense_rating: -3, sos: 0.1, games_played: 17, epa_margin_per_play: -0.11, passing_yards: 190.75, rushing_yards: 95, opp_passing_yards: 230 },
]

export function teamTable(overrides: Partial<EntityTable> = {}): EntityTable {
  const table: MetricTable = {
    rows: TEAM_ROWS,
    visible_columns: Object.keys(TEAM_META),
    column_groups: {
      identity: ['team'],
      ratings: ['team_rating', 'offense_rating', 'sos'],
      rating_companions: [],
      per_game_rates: ['games_played', 'epa_margin_per_play', 'passing_yards', 'rushing_yards'],
      per_snap_rates: [],
      opponent_context: ['opp_passing_yards'],
    },
    column_metadata: TEAM_META,
  }
  return {
    kind: 'teams',
    season: 2025,
    in_progress: false,
    games_played: 17,
    games_per_team: 17,
    table,
    categories: TEAM_CATEGORIES,
    team_names: { BUF: 'Buffalo Bills', KC: 'Kansas City Chiefs', WSH: 'Washington Commanders' },
    ...overrides,
  }
}

const QB_META: Record<string, MetricColumnMeta> = {
  qb_id: metricMeta('qb_id', { label: 'QB ID', kind: 'text', shape: 'id', polarity: 'neutral', category: 'Identity & Availability', subcategory: null, group: 'identity', heatmap: false }),
  qb_name: metricMeta('qb_name', { label: 'QB', kind: 'text', shape: 'id', polarity: 'neutral', category: 'Identity & Availability', subcategory: null, group: 'identity', heatmap: false, sticky: true }),
  team: metricMeta('team', { label: 'Team', kind: 'text', shape: 'id', polarity: 'neutral', category: 'Identity & Availability', subcategory: null, group: 'identity', heatmap: false }),
  adj_qb_epa_per_dropback: metricMeta('adj_qb_epa_per_dropback', { label: 'Adj EPA/Dropback', shape: 'rate', category: 'Schedule-Adjusted Ratings', subcategory: null, group: 'ratings' }),
  qb_faced_pass_defense: metricMeta('qb_faced_pass_defense', { label: 'Pass Defense Faced', shape: 'rate', category: 'Schedule-Adjusted Ratings', subcategory: null, group: 'ratings', contextual: true }),
  qb_epa_per_dropback: metricMeta('qb_epa_per_dropback', { label: 'EPA/Dropback', shape: 'rate', category: 'Passing Volume', subcategory: null, group: 'per_dropback_rates' }),
  qb_dropbacks_total: metricMeta('qb_dropbacks_total', { label: 'Dropbacks', kind: 'int', polarity: 'neutral', category: 'Identity & Availability', subcategory: null, group: 'raw_totals', heatmap: false }),
  qb_is_eligible: metricMeta('qb_is_eligible', { label: 'Qualified', kind: 'bool', shape: 'flag', polarity: 'neutral', category: 'Identity & Availability', subcategory: null, group: 'raw_totals', heatmap: false }),
}

export function qbTable(): EntityTable {
  const table: MetricTable = {
    rows: [
      { qb_id: '00-1', qb_name: 'J.Allen', team: 'BUF', adj_qb_epa_per_dropback: 0.22, qb_faced_pass_defense: 0.01, qb_epa_per_dropback: 0.25, qb_dropbacks_total: 600, qb_is_eligible: true },
      { qb_id: '00-2', qb_name: 'P.Mahomes', team: 'KC', adj_qb_epa_per_dropback: 0.12, qb_faced_pass_defense: 0.02, qb_epa_per_dropback: 0.15, qb_dropbacks_total: 620, qb_is_eligible: true },
      { qb_id: '00-9', qb_name: 'M.Backup', team: 'KC', adj_qb_epa_per_dropback: null, qb_faced_pass_defense: null, qb_epa_per_dropback: -0.4, qb_dropbacks_total: 14, qb_is_eligible: false },
    ],
    visible_columns: Object.keys(QB_META),
    column_groups: {
      identity: ['qb_id', 'qb_name', 'team'],
      ratings: ['adj_qb_epa_per_dropback', 'qb_faced_pass_defense'],
      rating_companions: ['qb_epa_per_dropback', 'qb_dropbacks_total'],
      raw_totals: ['qb_dropbacks_total', 'qb_is_eligible'],
      per_game_rates: [],
      per_dropback_rates: ['qb_epa_per_dropback'],
      opponent_context: [],
    },
    column_metadata: QB_META,
  }
  return {
    kind: 'qbs',
    season: 2025,
    in_progress: false,
    games_played: 17,
    games_per_team: 17,
    table,
    categories: QB_CATEGORIES,
    team_names: { BUF: 'Buffalo Bills', KC: 'Kansas City Chiefs' },
  }
}

function rankQuantiles(base: string, ranks: number[]): Row {
  const keys = ['q025', 'q100', 'q250', 'q500', 'q750', 'q900', 'q975']
  return Object.fromEntries(keys.map((key, index) => [`${base}_${key}`, ranks[index]]))
}

export function teamRankRanges(): RankRangeTable {
  const row = (team: string, rank: number, ranks: number[]) => ({
    team,
    team_rank: rank,
    ...rankQuantiles('team_rank', ranks),
    team_rank_top5_probability: 0.9,
    team_rank_top10_probability: 1,
    team_rank_missing_share: 0,
    team_rank_probabilities: [0.5, 0.3, 0.2],
    offense_rank: rank,
    ...rankQuantiles('offense_rank', ranks.map((value) => Math.max(1, value - 1))),
    defense_rank: 4 - rank,
    ...rankQuantiles('defense_rank', ranks.map((value) => 4 - value)),
  })
  return {
    kind: 'teams',
    season: 2025,
    entity_id: null,
    table: {
      rows: [row('BUF', 1, [1, 1, 1, 1, 2, 2, 2]), row('KC', 2, [1, 1, 2, 2, 2, 3, 3]), row('WSH', 3, [2, 2, 3, 3, 3, 3, 3])],
      visible_columns: [],
      column_groups: {},
      column_metadata: {},
    },
    team_names: { BUF: 'Buffalo Bills', KC: 'Kansas City Chiefs', WSH: 'Washington Commanders' },
  }
}

export function teamPairs(team: string): SeasonTable {
  const others = ['BUF', 'KC', 'WSH'].filter((other) => other !== team)
  return {
    kind: 'teams',
    season: 2025,
    entity_id: team,
    table: {
      rows: others.map((other) => ({
        team,
        other_team: other,
        team_rated_above_probability: team === 'BUF' ? 0.81 : 0.3,
        team_rating_gap_q025: -1,
        team_rating_gap_q500: 2.5,
        team_rating_gap_q975: 6,
        team_pair_share: 1,
      })),
      visible_columns: [],
      column_groups: {},
      column_metadata: {},
    },
  }
}

const LOG_META: Record<string, MetricColumnMeta> = {
  game_id: metricMeta('game_id', { label: 'Game ID', kind: 'text', shape: 'id', polarity: 'neutral', category: 'Overall', subcategory: null, group: 'identity', heatmap: false }),
  week: metricMeta('week', { label: 'Week', kind: 'text', shape: 'id', polarity: 'neutral', category: 'Overall', subcategory: null, group: 'identity', heatmap: false }),
  team: metricMeta('team', { label: 'Team', kind: 'text', shape: 'id', polarity: 'neutral', category: 'Overall', subcategory: null, group: 'identity', heatmap: false, sticky: true }),
  opponent_team: metricMeta('opponent_team', { label: 'Opponent', kind: 'text', shape: 'id', polarity: 'neutral', category: 'Overall', subcategory: null, group: 'identity', heatmap: false }),
  points_for: metricMeta('points_for', { label: 'Points For', kind: 'int', category: 'Overall', subcategory: null, group: 'results' }),
  points_allowed: metricMeta('points_allowed', { label: 'Points Allowed', kind: 'int', polarity: 'lower', category: 'Overall', subcategory: null, group: 'results' }),
  win_value: metricMeta('win_value', { label: 'Win Value', shape: 'avg', category: 'Overall', subcategory: null, group: 'results' }),
  is_home: metricMeta('is_home', { label: 'Home', kind: 'bool', shape: 'flag', polarity: 'neutral', category: 'Overall', subcategory: null, group: 'raw_totals', heatmap: false }),
  passing_yards: metricMeta('passing_yards', { label: 'Pass Yds', kind: 'int', group: 'raw_totals' }),
  opp_defense_rating: metricMeta('opp_defense_rating', { label: 'Opp Def Rating', shape: 'score', contextual: true, category: 'Schedule-Adjusted Ratings', subcategory: null, group: 'opponent_ratings' }),
  opp_team_rating: metricMeta('opp_team_rating', { label: 'Opp Team Rating', shape: 'score', contextual: true, category: 'Schedule-Adjusted Ratings', subcategory: null, group: 'opponent_ratings' }),
  opp_team_rank: metricMeta('opp_team_rank', { label: 'Opp Rank', kind: 'int', shape: 'score', polarity: 'lower', contextual: true, category: 'Schedule-Adjusted Ratings', subcategory: null, group: 'opponent_ratings' }),
}

/** One team's game log in a 32-team league: Kansas City ranks 3rd, Washington 30th. */
export function teamGameLogs(team = 'BUF'): GameLogsTable {
  return {
    kind: 'teams',
    season: 2025,
    entity_id: team,
    table: {
      rows: [
        { game_id: '2025_01_BUF_KC', week: 1, team, opponent_team: 'KC', points_for: 31, points_allowed: 24, win_value: 1, is_home: false, passing_yards: 280, opp_defense_rating: 1.5, opp_team_rating: 4, opp_team_rank: 3 },
        { game_id: '2025_02_WSH_BUF', week: 2, team, opponent_team: 'WSH', points_for: 17, points_allowed: 20, win_value: 0, is_home: true, passing_yards: 210, opp_defense_rating: -2, opp_team_rating: -8.5, opp_team_rank: 30 },
      ],
      visible_columns: Object.keys(LOG_META),
      column_groups: {
        identity: ['game_id', 'week', 'team', 'opponent_team'],
        results: ['points_for', 'points_allowed', 'win_value'],
        raw_totals: ['is_home', 'passing_yards'],
        per_snap_rates: [],
        opponent_ratings: ['opp_defense_rating', 'opp_team_rating', 'opp_team_rank'],
      },
      column_metadata: LOG_META,
    },
    league_size: 32,
  }
}

export function teamRatingHistory(team = 'BUF'): SeasonTable {
  return {
    kind: 'teams',
    season: 2025,
    entity_id: team,
    table: {
      rows: [1, 2, 3].map((week) => ({ week, team, games_played: week, team_rating: 2 * week, offense_rating: week })),
      visible_columns: ['week', 'team', 'games_played', 'team_rating', 'offense_rating'],
      column_groups: { identity: ['week', 'team'], sample: ['games_played'], ratings: ['team_rating', 'offense_rating'] },
      column_metadata: {
        week: LOG_META.week,
        team: LOG_META.team,
        games_played: metricMeta('games_played', { label: 'Games Played', kind: 'int', polarity: 'neutral', category: 'Overall', subcategory: null, heatmap: false }),
        team_rating: TEAM_META.team_rating,
        offense_rating: TEAM_META.offense_rating,
      },
    },
  }
}
