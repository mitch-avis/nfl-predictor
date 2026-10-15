import { Button } from '@/components/ui/button'
import { cn } from '@/utils/cn'

/** Toggle buttons for the teams a survivor entry has already used this season. */
export function TeamPicker({ teams, selected, onChange }: { teams: string[]; selected: string[]; onChange: (next: string[]) => void }) {
  const chosen = new Set(selected)
  const toggle = (team: string) => {
    const next = new Set(chosen)
    if (next.has(team)) next.delete(team)
    else next.add(team)
    onChange([...next].sort())
  }
  return (
    <div className="space-y-2">
      <div role="group" aria-label="Teams already used" className="grid grid-cols-4 gap-1.5 sm:grid-cols-8">
        {teams.map((team) => {
          const used = chosen.has(team)
          return (
            <button
              key={team}
              type="button"
              aria-pressed={used}
              onClick={() => toggle(team)}
              className={cn(
                'h-11 rounded-md border text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-ring sm:h-9',
                used ? 'border-transparent bg-muted text-muted-foreground line-through' : 'bg-card hover:bg-accent',
              )}
            >
              {team}
            </button>
          )
        })}
      </div>
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{selected.length === 0 ? 'No team used yet.' : `${selected.length} used: ${selected.join(', ')}`}</span>
        {selected.length > 0 ? (
          <Button variant="ghost" size="xs" onClick={() => onChange([])}>
            Clear
          </Button>
        ) : null}
      </div>
    </div>
  )
}
