import { useId, useState } from 'react'

import { useSurvivor } from '@/api/queries'
import { ErrorState } from '@/components/common/ErrorState'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { useLocalStorage } from '@/hooks/useLocalStorage'

import { SurvivorPlans } from './SurvivorPlans'
import { TeamPicker } from './TeamPicker'

const PLANNED_WEEKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
const BEAM_WIDTHS = [16, 32, 64, 128, 256]

function NumberChoice({ label, value, choices, onChange }: { label: string; value: number; choices: number[]; onChange: (value: number) => void }) {
  const id = useId()
  return (
    <div className="flex items-center gap-2 text-sm">
      <label htmlFor={id} className="text-muted-foreground">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-9 rounded-md border bg-background px-2 text-sm focus-visible:outline-2 focus-visible:outline-ring"
      >
        {choices.map((choice) => (
          <option key={choice} value={choice}>
            {choice}
          </option>
        ))}
      </select>
    </div>
  )
}

/** The survivor planner: the teams already used (kept in this browser per season) and the plans. */
export function SurvivorPanel({ season, week }: { season: number | null; week: number | null }) {
  const [used, setUsed] = useLocalStorage<string[]>(`nflp.survivor.used.${season ?? 'none'}`, [])
  const [horizon, setHorizon] = useState(4)
  const [beam, setBeam] = useState(64)
  const request = season !== null && week !== null ? { season, start_week: week, used_teams: used, horizon_weeks: horizon, beam_width: beam } : null
  const query = useSurvivor(request)
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Survivor plan from week {week ?? '?'}</CardTitle>
          <CardDescription>
            One team a week, each team once; a loss or a tie ends the entry. Plans multiply the model’s win probabilities and treat the games as independent.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {query.data ? <TeamPicker teams={query.data.teams} selected={used} onChange={setUsed} /> : null}
          <div className="flex flex-wrap items-center gap-4">
            <NumberChoice label="Weeks planned pick by pick" value={horizon} choices={PLANNED_WEEKS} onChange={setHorizon} />
            <NumberChoice label="Search width" value={beam} choices={BEAM_WIDTHS} onChange={setBeam} />
            {query.isFetching && query.isPlaceholderData ? (
              <span role="status" className="text-xs text-muted-foreground">
                Updating…
              </span>
            ) : null}
          </div>
        </CardContent>
      </Card>
      {request === null ? <p className="text-sm text-muted-foreground">Choose a week with predictions to plan from.</p> : null}
      {query.isLoading ? <Skeleton className="h-48 w-full" /> : null}
      {query.isError ? <ErrorState error={query.error} title="No survivor plan" /> : null}
      {query.data ? <SurvivorPlans result={query.data} /> : null}
    </div>
  )
}
