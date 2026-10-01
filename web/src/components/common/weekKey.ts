/** Stable identity for a week entry: the run id only matters for non-active runs. */
export function weekKey(ref: { season: number | null; week: number | null; source?: string | null; run_id?: string | null }): string {
  const owner = ref.source === 'run' ? (ref.run_id ?? '') : ''
  return `${ref.season ?? 'x'}-${ref.week ?? 'x'}-${ref.source ?? ''}-${owner}`
}
