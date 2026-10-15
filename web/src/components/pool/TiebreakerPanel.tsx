import { useId, useState } from 'react'

import type { ScoreEstimate, TiebreakGame, Tiebreakers } from '@/api/types'
import { EmptyState } from '@/components/common/EmptyState'
import { InfoTooltip } from '@/components/common/InfoTooltip'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { formatNumber } from '@/utils/format'

import { formatKickoff } from './poolFormat'

function band(low: number | null, high: number | null): string {
  if (low === null || high === null) return '—'
  return `${Math.round(low)}–${Math.round(high)}`
}

function Entry({ label, value, low, high }: { label: string; value: number | null; low: number | null; high: number | null }) {
  return (
    <div className="min-w-0">
      <div className="truncate text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</div>
      <output aria-label={`${label} points`} className="tabular block text-3xl font-semibold leading-tight">
        {value ?? '—'}
      </output>
      <div className="tabular text-xs text-muted-foreground" title="10th to 90th percentile">
        {band(low, high)}
      </div>
    </div>
  )
}

function GameCard({ title, games, selected, onSelect }: { title: string; games: TiebreakGame[]; selected: string; onSelect: (gameId: string) => void }) {
  const selectId = useId()
  const game = games.find((g) => g.game_id === selected) ?? games[0]
  return (
    <section aria-label={title}>
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="text-base">{title}</CardTitle>
          <CardDescription>
            <label htmlFor={selectId} className="sr-only">
              Choose the {title.toLowerCase()}
            </label>
            <select
              id={selectId}
              value={game.game_id}
              onChange={(event) => onSelect(event.target.value)}
              className="mt-1 h-9 w-full rounded-md border bg-background px-2 text-sm text-foreground focus-visible:outline-2 focus-visible:outline-ring"
            >
              {games.map((g) => (
                <option key={g.game_id} value={g.game_id}>
                  {g.away_abbr} @ {g.home_abbr}
                  {g.slot ? ` · ${g.slot}` : ''}
                </option>
              ))}
            </select>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <div className="font-medium">
              {game.away_abbr} @ {game.home_abbr}
            </div>
            <div className="text-xs text-muted-foreground">
              {game.slot ? `${game.slot} · ` : ''}
              {formatKickoff(game.kickoff)}
            </div>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <Entry label="Total" value={game.total_points} low={game.total_p10} high={game.total_p90} />
            <Entry label={game.home_abbr} value={game.home_points} low={game.home_p10} high={game.home_p90} />
            <Entry label={game.away_abbr} value={game.away_points} low={game.away_p10} high={game.away_p90} />
          </div>
        </CardContent>
      </Card>
    </section>
  )
}

function Extreme({ label, estimate, word }: { label: string; estimate: ScoreEstimate; word: string }) {
  return (
    <div className="min-w-0">
      <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</div>
      <output aria-label={label} className="tabular block text-3xl font-semibold leading-tight">
        {estimate.points ?? '—'}
      </output>
      <div className="tabular text-xs text-muted-foreground">likely {band(estimate.p10, estimate.p90)}</div>
      {estimate.projected_team ? (
        <div className="text-xs text-muted-foreground">
          {estimate.projected_team} projected {word} at {formatNumber(estimate.projected_points, 1)}
        </div>
      ) : null}
    </div>
  )
}

/**
 * The tiebreaker card: whole-point entries for the pool's two tiebreak games and the week's highest
 * and lowest team scores. The pool names its games each week, so both are selectable.
 */
export function TiebreakerPanel({ card }: { card: Tiebreakers }) {
  const [first, setFirst] = useState(card.default_game_ids[0] ?? card.games[0]?.game_id ?? '')
  const [second, setSecond] = useState(card.default_game_ids[1] ?? card.games[1]?.game_id ?? card.games[0]?.game_id ?? '')
  if (card.games.length === 0) return <EmptyState title="No games to break ties with this week." />
  return (
    <div className="space-y-4">
      <p className="max-w-prose text-sm text-muted-foreground">
        Yahoo breaks a tie on the first tiebreak game’s total points, then its home and road points, then the same for the second game, then the week’s highest and lowest
        team scores. Yahoo names the two games each week; the Sunday and Monday night games are preselected, so change them to match.
      </p>
      <div className="grid gap-4 md:grid-cols-2">
        <GameCard title="First tiebreak game" games={card.games} selected={first} onSelect={setFirst} />
        <GameCard title="Second tiebreak game" games={card.games} selected={second} onSelect={setSecond} />
      </div>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-1 text-base">
            Highest and lowest team score
            <InfoTooltip content="The week's top and bottom team scores, not the projections: with every team's score spread around its prediction, someone almost always runs past the best projection. Entries are the median of each extreme; the range is its 10th to 90th percentile." />
          </CardTitle>
          <CardDescription>Over every team playing this week.</CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-4">
          <Extreme label="Highest team score" estimate={card.highest} word="highest" />
          <Extreme label="Lowest team score" estimate={card.lowest} word="lowest" />
        </CardContent>
      </Card>
    </div>
  )
}
