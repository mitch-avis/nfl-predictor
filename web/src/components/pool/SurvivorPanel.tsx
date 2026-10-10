import { useId, useState } from 'react'

import { useSurvivor } from '@/api/queries'
import { ErrorState } from '@/components/common/ErrorState'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { useLocalStorage } from '@/hooks/useLocalStorage'

import { SurvivorPlans } from './SurvivorPlans'
import { TeamPicker } from './TeamPicker'

/** How many weeks a plan can show as near-term picks: up to a full regular season. */
const NEAR_TERM_WEEKS = Array.from({ length: 18 }, (_, index) => index + 1)

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
  const request = season !== null && week !== null ? { season, start_week: week, used_teams: used, horizon_weeks: horizon } : null
  const query = useSurvivor(request)
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Survivor plan from week {week ?? '?'}</CardTitle>
          <CardDescription>
            One team a week, each team once; a loss or a tie ends the entry. Each plan is the best one for its pick this week over every week with predictions,
            multiplying the model’s win probabilities as independent games.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {query.data ? <TeamPicker teams={query.data.teams} selected={used} onChange={setUsed} /> : null}
          <div className="flex flex-wrap items-center gap-4">
            <NumberChoice label="Near-term weeks" value={horizon} choices={NEAR_TERM_WEEKS} onChange={setHorizon} />
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
