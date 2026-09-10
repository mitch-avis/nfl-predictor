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
  source: 'active' | 'run' | 'unattached'
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
  xlsx_available: boolean
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

export interface CalibrationBin {
  bin_lower: number
  bin_upper: number
  count: number
  avg_pred: number
  avg_actual: number
}

export interface FeatureImportanceRow {
  feature: string
  gain: number
  weight: number | null
  margin_gain: number | null
  total_gain: number | null
}

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
  feature_importance: FeatureImportanceRow[]
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

export interface LeakageAudit {
  ok: boolean
  row_count?: number
  feature_count?: number
  failures?: unknown[]
  warnings?: unknown[]
  flagged_columns?: string[]
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
