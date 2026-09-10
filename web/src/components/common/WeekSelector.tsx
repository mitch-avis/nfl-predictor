import type { WeekRef } from '@/api/types'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

/** Stable identity for a week entry: the run id only matters for non-active runs. */
export function weekKey(ref: { season: number | null; week: number | null; source?: string; run_id?: string | null }): string {
  const owner = ref.source === 'run' ? (ref.run_id ?? '') : ''
  return `${ref.season ?? 'x'}-${ref.week ?? 'x'}-${ref.source ?? ''}-${owner}`
}

/** Pick which week's predictions to view. */
export function WeekSelector({ weeks, value, onChange }: { weeks: WeekRef[]; value: string | null; onChange: (ref: WeekRef | null) => void }) {
  if (weeks.length === 0) return null
  return (
    <Select value={value ?? weekKey(weeks[0])} onValueChange={(key) => onChange(weeks.find((w) => weekKey(w) === key) ?? null)}>
      <SelectTrigger className="w-[240px]" aria-label="Week">
        <SelectValue placeholder="Week" />
      </SelectTrigger>
      <SelectContent>
        {weeks.map((ref) => (
          <SelectItem key={weekKey(ref)} value={weekKey(ref)}>
            {ref.season ? `${ref.season} · ` : ''}
            {ref.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
