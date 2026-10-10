/** What the team pages and the quarterback pages each show, and how their rows are named. */
import type { EntityKind, Row } from '@/api/types'

export interface EntityConfig {
  kind: EntityKind
  title: string
  singular: string
  /** The column that identifies a row and its detail page. */
  idColumn: string
  /** The column that names a row. */
  labelColumn: string
  /** The published rating each page ranks by, best first. */
  ratingColumn: string
  /** The identity columns the tables show (a quarterback's player id stays hidden). */
  identityColumns: string[]
  /** The detail page's lead tiles, headline first. */
  summaryColumns: string[]
  /** The headline rating's value before the schedule adjustment. */
  unadjustedColumn: string
  /** What the rating means, in one paragraph. */
  ratingDescription: string
  /** How to read the page, a sentence each. */
  notes: string[]
}

const CONFIG: Record<EntityKind, EntityConfig> = {
  teams: {
    kind: 'teams',
    title: 'Teams',
    singular: 'team',
    idColumn: 'team',
    labelColumn: 'team',
    ratingColumn: 'team_rating',
    identityColumns: ['team'],
    summaryColumns: ['team_rating', 'offense_rating', 'defense_rating', 'special_teams_rating', 'sos', 'SRS'],
    unadjustedColumn: 'epa_margin_per_play',
    ratingDescription:
      'Team Rating is points per game better (+) or worse (−) than an average team on a neutral field, after adjusting for every opponent faced; 0 is average. It is the sum of the offense, defense and special-teams ratings.',
    notes: [
      'Team Rating is built from expected points added (EPA), which scores each play by how much it changed the offense’s expected points. Each team is judged against its opponents, and those opponents against theirs, all at once.',
      'Offense, defense and special-teams ratings are on the same points-per-game scale, so they add up to Team Rating.',
      'SoS (schedule strength) is the average Team Rating of the opponents played, one entry per game, each opponent rated without its games against this team. Positive means a tougher schedule.',
      'SRS is the classic score-based rating: average point margin adjusted for the opponents played, a check built from final scores instead of plays.',
      'Green shading marks better or worse within the season; slate shading marks context, such as schedule strength or the opponents faced, deeper for tougher.',
    ],
  },
  qbs: {
    kind: 'qbs',
    title: 'Quarterbacks',
    singular: 'quarterback',
    idColumn: 'qb_id',
    labelColumn: 'qb_name',
    ratingColumn: 'adj_qb_epa_per_dropback',
    identityColumns: ['qb_name', 'team'],
    summaryColumns: ['adj_qb_epa_per_dropback', 'qb_faced_pass_defense', 'qb_dropbacks_total'],
    unadjustedColumn: 'qb_epa_per_dropback',
    ratingDescription:
      'Adjusted EPA per dropback is the quarterback’s expected points added per dropback after adjusting for the pass defenses he faced. It reads on the same scale as raw EPA per dropback, whose league average is usually a little above 0.',
    notes: [
      'Each passer is compared with the defenses he actually faced, and each defense with every passer it faced.',
      'Pass Defense Faced is the average strength of those pass defenses, weighted by his dropbacks against each, each defense rated only on its plays against other passers. Positive means tougher.',
      'Small samples are pulled toward the league average, so a backup with a few big plays does not top the table.',
      'A quarterback is ranked once he has 14 pass attempts for every game his team has played; the switch above the table lists the passers below that mark too.',
      'EPA also reflects his line, receivers and play-calling, which public play-by-play cannot separate, so the rating describes the passing offense he led.',
    ],
  },
}

export function entityConfig(kind: EntityKind): EntityConfig {
  return CONFIG[kind]
}

/** The id of a team or quarterback row: its team code or player id. */
export function entityId(kind: EntityKind, row: Row): string {
  return String(row[CONFIG[kind].idColumn] ?? '')
}

/** The name of a row: a team's franchise name when `teamNames` has it, a passer's name, else the id. */
export function entityLabel(kind: EntityKind, row: Row, teamNames: Record<string, string> = {}): string {
  const id = entityId(kind, row)
  if (kind === 'teams') return teamNames[id] ?? id
  const label = row[CONFIG[kind].labelColumn]
  return label === null || label === undefined || label === '' ? id : String(label)
}

export function findEntityRow(kind: EntityKind, rows: Row[], id: string): Row | undefined {
  return rows.find((row) => entityId(kind, row) === id)
}

/** The detail page of one team or quarterback for a season. */
export function entityPath(kind: EntityKind, id: string, season: number): string {
  return `/${kind}/${encodeURIComponent(id)}?season=${season}`
}

/** The head-to-head page of a pair of teams or quarterbacks in one season. */
export function headToHeadPath(kind: EntityKind, season: number, a: string, b: string): string {
  return `/head-to-head?${new URLSearchParams({ kind, season: String(season), a, b }).toString()}`
}

/** The index page of the teams or quarterbacks, keeping the season when there is one. */
export function ratingsPagePath(kind: EntityKind, season: number | null): string {
  return season === null ? `/${kind}` : `/${kind}?season=${season}`
}
