/** Hand-maintained API payload types for the routes the frontend consumes. */

export type Role = 'viewer' | 'admin'

export interface User {
  id: number
  username: string
  role: Role
  created_at: string
}

export interface Session {
  user: User
}

export type RunKind = 'weekly' | 'training' | 'walk_forward'

export interface RunSummary {
  run_id: string
  created_at: string
  kind: RunKind
  season: number | null
  week: number | null
  stages: Record<string, boolean>
  complete: boolean
  git_commit_hash: string | null
  dataset_hash: string | null
  model_kind: string | null
  holdout: Record<string, number> | null
  files: Record<string, boolean>
  is_active: boolean
}

export interface RunList {
  active_run_id: string | null
  pinned_run_id: string | null
  runs: RunSummary[]
}

export interface RunDetail extends RunSummary {
  config: Record<string, unknown> | null
  splits: Record<string, unknown> | null
  library_versions: Record<string, unknown> | null
  feature_count: number | null
}

export type ColumnKind =
  | 'text' | 'int' | 'float' | 'pct' | 'prob' | 'money' | 'spread' | 'datetime' | 'date' | 'team' | 'action' | 'bool'
export type Polarity = 'higher' | 'lower' | 'neutral'

export interface ColumnMeta {
  key: string
  label: string
  description: string
  group: string
  kind: ColumnKind
  polarity: Polarity
  heatmap: boolean
  decimals: number | null
  actionable: boolean
  sticky: boolean
  /** Context rather than a grade (schedule strength, the opponents faced); shaded in a muted hue. */
  contextual?: boolean
}

export type Row = Record<string, string | number | boolean | null>

export interface TablePayload {
  rows: Row[]
  visible_columns: string[]
  column_groups: Record<string, string[]>
  column_metadata: Record<string, ColumnMeta>
}

export interface Registry {
  columns: Record<string, ColumnMeta>
  groups: Record<string, string[]>
}

export interface WeekRef {
  season: number | null
  week: number | null
  source: 'active' | 'run' | 'unattached' | 'available'
  run_id: string | null
  label: string
}

export interface PredictionsSummary {
  games: number
  avg_confidence: number | null
  market_disagreements: number
  games_with_lines: number
  first_kickoff: string | null
  last_kickoff: string | null
}

export interface PredictionsOut {
  run_id: string | null
  source: string
  season: number | null
  week: number | null
  generated_at: string | null
  table: TablePayload
  summary: PredictionsSummary
  weeks: WeekRef[]
}

export interface PicksOut {
  run_id: string | null
  season: number | null
  week: number | null
  table: TablePayload
}

export interface BettingOut {
  run_id: string | null
  season: number | null
  week: number | null
  generated_at: string | null
  table: TablePayload
  ladder: { action: string; min_edge: number }[]
  notes: string[]
}

export interface PowerOut {
  run_id: string
  season: number | null
  through_week: number | null
  previous_run_id: string | null
  rankings: TablePayload
  standings: TablePayload | null
  division_standings: TablePayload | null
}

/** One game on the confidence pool sheet; `result` and `points_won` are set once it is final. */
export interface SheetRow {
  game_id: string
  points: number | null
  pick: string | null
  opponent: string | null
  pick_is_home: boolean | null
  away_abbr: string
  home_abbr: string
  kickoff: string | null
  pick_win_prob: number | null
  expected_points: number | null
  away_score: number | null
  home_score: number | null
  result: 'win' | 'loss' | 'tie' | null
  points_won: number | null
  /** On a sheet of the picks entered before the lock: the points and pick predicted since, when they differ. */
  newer_points: number | null
  newer_pick: string | null
}

/**
 * A week's confidence pool sheet: picks by points, the lock (first kickoff) and the scoring. When
 * the predictions were rewritten after the lock, `picks_file` names the pre-lock picks it shows.
 */
export interface ConfidenceSheet {
  run_id: string | null
  source: string
  season: number | null
  week: number | null
  generated_at: string | null
  rows: SheetRow[]
  games: number
  max_points: number
  expected_points: number | null
  games_final: number
  points_won: number | null
  points_possible: number | null
  lock_at: string | null
  generated_after_lock: boolean | null
  rank_problem: string | null
  picks_file: string | null
  newer_predictions_at: string | null
  changed_after_lock: number
}

/** A game's whole-point tiebreak entry and the p10 to p90 bands behind it. */
export interface TiebreakGame {
  game_id: string
  away_abbr: string
  home_abbr: string
  kickoff: string | null
  slot: string | null
  total_points: number | null
  home_points: number | null
  away_points: number | null
  total_raw: number | null
  home_raw: number | null
  away_raw: number | null
  total_p10: number | null
  total_p90: number | null
  home_p10: number | null
  home_p90: number | null
  away_p10: number | null
  away_p90: number | null
}

/** The week's highest or lowest team score, and the team projected to have it. */
export interface ScoreEstimate {
  points: number | null
  median: number | null
  p10: number | null
  p90: number | null
  projected_team: string | null
  projected_points: number | null
}

export interface Tiebreakers {
  run_id: string | null
  source: string
  season: number | null
  week: number | null
  generated_at: string | null
  games: TiebreakGame[]
  default_game_ids: string[]
  highest: ScoreEstimate
  lowest: ScoreEstimate
}

export interface SurvivorRequest {
  season: number
  start_week: number
  used_teams: string[]
  /** How many weeks from the start week the plans show as near-term picks; every week is solved. */
  horizon_weeks: number
}

/** One week's pick in a survivor plan; `later` marks a lookahead pick. */
export interface SurvivorPick {
  week: number
  team: string
  opponent: string
  is_home: boolean
  win_prob: number
  game_id: string
  kickoff: string | null
  later: boolean
  cumulative_survival: number
}

export interface SurvivorPlan {
  rank: number
  picks: SurvivorPick[]
  survival: number
  later_survival: number | null
  total_survival: number
}

export interface SurvivorOut {
  season: number
  start_week: number
  planned_weeks: number[]
  later_weeks: number[]
  missing_weeks: number[]
  used_teams: string[]
  teams: string[]
  plans: SurvivorPlan[]
  future_values: { team: string; value: number }[]
}

export interface CalibrationBin {
  bin_lower: number
  bin_upper: number
  count: number
  avg_pred: number
  avg_actual: number
}

/** Which importance measure ranks the features: SHAP for current runs, the gains for older ones. */
export type FeatureImportanceMeasure = 'mean_abs_shap' | 'total_gain' | 'summed_average_gain'

/** One base feature; `value`, `margin_value` and `total_value` are in the payload's measure. */
export interface FeatureImportanceRow {
  feature: string
  value: number
  margin_value: number | null
  total_value: number | null
  splits: number | null
}

/**
 * `rows` ranks the features over both heads (the default view); `margin_rows` ranks every feature
 * by the margin head alone, with `value` and `splits` for that head, and is empty when the run
 * recorded no per-head numbers.
 */
export interface FeatureImportance {
  measure: FeatureImportanceMeasure | null
  rows: FeatureImportanceRow[]
  margin_rows: FeatureImportanceRow[]
}

/** Which ranking the feature importance chart plots: both heads added, or the margin head alone. */
export type ImportanceHead = 'combined' | 'margin'

export interface MetricStrategyEntry {
  metric: string
  direction: 'higher' | 'lower'
}

export interface ModelOut {
  run_id: string
  kind: RunKind
  metadata: {
    created_at?: string
    run_id?: string
    git_commit_hash?: string | null
    dataset_hash?: string | null
    library_versions?: Record<string, string> | null
    params?: Record<string, unknown> | null
    tuned_params?: Record<string, unknown> | null
    early_stopping?: Record<string, unknown> | null
    optuna_summary?: Record<string, unknown> | null
    splits?: Record<string, unknown> | null
    feature_count?: number | null
    config?: Record<string, unknown> | null
  }
  metrics: {
    kind?: string | null
    holdout?: Record<string, number> | null
    pool?: Record<string, number> | null
    missing_data?: { total_rows?: number; groups?: Record<string, Record<string, number>> } | null
    overall?: Record<string, number> | null
    per_season?: Record<string, number>[] | null
    per_week?: Record<string, number | string>[] | null
    summary_table?: { metric: string; priority: string; direction: string; overall: number; fold_mean?: number; fold_variance?: number }[] | null
    metric_strategy?: Record<string, MetricStrategyEntry[]> | null
    calibration?: { bin_count: number; bins: CalibrationBin[] } | null
  }
  feature_importance: FeatureImportance
  calibration: { bin_count: number; bins: CalibrationBin[]; source?: string } | null
  wf_compare: TablePayload | null
  wf_best: Record<string, unknown> | null
  shap_available: boolean
}

export interface DataFile {
  name: string
  description: string
  exists: boolean
  size: number | null
  modified_at: string | null
  rows: number | null
  seasons: [number, number] | null
}

export interface PredictFile {
  name: string
  kind: 'games_to_predict' | 'predictions' | 'xlsx'
  season: number | null
  week: number | null
  size: number
  modified_at: string
  path: string
}

/** One column the leakage audit flagged; the statistics appear only on the statistical flags. */
export interface LeakageFlag {
  column: string
  /** `suspicious_name`, or a comma-joined subset of the statistical reasons. */
  reason: string
  corr_margin?: number | null
  corr_total?: number | null
  matches_home_score_rate?: number | null
  matches_away_score_rate?: number | null
}

export interface LeakageAudit {
  ok: boolean
  row_count?: number
  feature_count?: number
  failures?: string[]
  warnings?: string[]
  flagged_columns?: LeakageFlag[]
  path?: string
  modified_at?: string
}

export interface DataStatusOut {
  current_season: number
  current_week: number
  files: DataFile[]
  predict_files: PredictFile[]
  fingerprint: { sha256: string; size: number; mtime: number; path: string } | null
  cache: { schedule: number[]; pbp: number[] }
  leakage_audit: LeakageAudit | null
}

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled'
export type ParamKind = 'int' | 'float' | 'str' | 'bool' | 'choice'

export interface ParamSpec {
  name: string
  label: string
  kind: ParamKind
  description: string
  required: boolean
  default: string | number | boolean | null
  choices: string[]
  minimum: number | null
  maximum: number | null
}

export interface JobTemplate {
  id: string
  label: string
  description: string
  category: string
  exclusive_group: string | null
  chain_template_id: string | null
  writes_datasets: boolean
  needs_active_run: boolean
  params: ParamSpec[]
}

export interface JobCatalog {
  templates: JobTemplate[]
  busy_groups: string[]
}

export interface JobProgress {
  current: number
  total: number
  label: string
}

export interface Job {
  id: string
  template_id: string
  template_label: string
  params: Record<string, string | number | boolean>
  status: JobStatus
  created_at: string
  started_at: string | null
  finished_at: string | null
  exit_code: number | null
  created_by: string | null
  progress: JobProgress | null
  error: string | null
  parent_job_id: string | null
}

export interface JobList {
  jobs: Job[]
  busy_groups: string[]
}

export interface JobLogLine {
  seq: number
  ts: string
  level: string
  line: string
}

export interface JobLogs {
  job_id: string
  status: JobStatus
  next_seq: number
  lines: JobLogLine[]
}

export type JobParams = Record<string, string | number | boolean>

/** Which page family a ratings route serves: the team pages or the quarterback pages. */
export type EntityKind = 'teams' | 'qbs'

/** How a registry metric behaves across views: a summable count, a rate, a maximum, ... */
export type MetricShape = 'count' | 'rate' | 'avg' | 'max' | 'flag' | 'id' | 'score'

/** A ratings column: the app's column format plus what the metric registry says about it. */
export interface MetricColumnMeta extends ColumnMeta {
  full_name: string
  category: string | null
  subcategory: string | null
  contextual: boolean
  shape: MetricShape | null
  /** A proportion (0.653) shown as a percentage (65.3%). */
  percent: boolean
}

export interface MetricTable {
  rows: Row[]
  visible_columns: string[]
  column_groups: Record<string, string[]>
  column_metadata: Record<string, MetricColumnMeta>
}

/** A metric category of the team or quarterback pages, with its subcategories in display order. */
export interface MetricCategory {
  name: string
  description: string
  subcategories: string[]
}

export interface RatingsSeasons {
  seasons: number[]
  current_season: number
}

/**
 * One season's team or quarterback table. `in_progress` holds only for the season under way while
 * a team has games left; `games_played` is the most any team has played.
 */
export interface EntityTable {
  kind: EntityKind
  season: number
  in_progress: boolean
  games_played: number | null
  games_per_team: number
  table: MetricTable
  categories: MetricCategory[]
  team_names: Record<string, string>
}

/** A league-wide section (rank ranges) or one team's or quarterback's (game logs, histories, pairs). */
export interface SeasonTable {
  kind: EntityKind
  season: number
  entity_id: string | null
  table: MetricTable
}

/** One team's or quarterback's games; `league_size` is how many teams the opponents' ranks run to. */
export interface GameLogsTable extends SeasonTable {
  league_size: number | null
}

/** A rank-range row: the scalar columns plus `<rank>_probabilities`, the chance of each rank. */
export type RankRangeRow = Record<string, Row[string] | number[]>

/** A season's rank ranges, with the franchise name of every team code in them. */
export interface RankRangeTable extends Omit<SeasonTable, 'table'> {
  table: Omit<MetricTable, 'rows'> & { rows: RankRangeRow[] }
  team_names: Record<string, string>
}

/** Every completed team-season ranked on schedule strength, softest first. */
export interface ScheduleRanks {
  count: number
  seasons: number[]
  skipped_seasons: number[]
  table: MetricTable
}

/** One metric of the registry, as the glossary shows it. */
export interface RegistryMetric {
  label: string
  full_name: string
  description: string
  entity: 'team' | 'qb'
  category: string
  subcategory: string | null
  shape: MetricShape
  polarity: Polarity
  source: string
  denominator: string | null
  since: number | null
  duplicate_of: string | null
  contextual: boolean
  formula: string | null
  note: string | null
  percent: boolean
}

/** How a column prefix (`opp_`, `qopp_`, ...) changes a metric's name and meaning. */
export interface PrefixRule {
  prefix: string
  label_template: string
  full_name_template: string
  description_note: string
  contextual: boolean
  invert_polarity_for_qb: boolean
}

export interface MetricRegistry {
  entities: Record<'team' | 'qb', { categories: MetricCategory[] }>
  metrics: Record<string, RegistryMetric>
  prefix_rules: PrefixRule[]
}
