import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

/** Pick the season the ratings pages show, newest first. */
export function SeasonPicker({ seasons, season, onChange }: { seasons: number[]; season: number | null; onChange: (season: number) => void }) {
  if (seasons.length === 0 || season === null) return null
  return (
    <Select value={String(season)} onValueChange={(value) => onChange(Number(value))}>
      <SelectTrigger className="w-[120px]" aria-label="Season">
        <SelectValue placeholder="Season" />
      </SelectTrigger>
      <SelectContent>
        {seasons.map((option) => (
          <SelectItem key={option} value={String(option)}>
            {option}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
