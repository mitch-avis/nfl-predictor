import { ArrowLeft, Swords } from 'lucide-react'
import { Link } from 'react-router'

import { isMissing, useEntitySection, useEntityTable, useRankRanges } from '@/api/queries'
import type { EntityKind, EntityTable, Row } from '@/api/types'
import { EmptyState } from '@/components/common/EmptyState'
import { ErrorState } from '@/components/common/ErrorState'
import { PageHeader } from '@/components/common/PageHeader'
import { GameLogTable } from '@/components/ratings/GameLogTable'
import { HeadToHeadCard } from '@/components/ratings/HeadToHeadCard'
import { MetricSections } from '@/components/ratings/MetricSections'
import { RankHistoryCard } from '@/components/ratings/RankHistoryCard'
import { RankRangeCard } from '@/components/ratings/RankRangeCard'
import { RatingSummary } from '@/components/ratings/RatingSummary'
import { RatingsSeasonState, UnavailableSeason } from '@/components/ratings/RatingsStates'
import { SeasonPicker } from '@/components/ratings/SeasonPicker'
import { SectionCard } from '@/components/ratings/SectionCard'
import { TrendChart } from '@/components/ratings/TrendChart'
import { ViewControls, type ViewChange } from '@/components/ratings/ViewControls'
import { entityConfig, entityLabel, findEntityRow, headToHeadPath, ratingsPagePath } from '@/components/ratings/entities'
import { opponentLedger } from '@/components/ratings/opponents'
import { useRatingsParams, useRatingsSeason, viewPatch } from '@/components/ratings/useRatingsParams'
import { gameLogColumns, resolveViewState, seasonView, VIEWS, type ViewState } from '@/components/ratings/views'
import { withDecimals } from '@/components/ratings/decimals'
import { DataTable } from '@/components/table/DataTable'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'

/** The stats views of the detail page: its summary stands in for the ratings view. */
const STAT_VIEWS = VIEWS.filter((view) => view.id !== 'ratings')
/** The game-by-game chart opens on the per-game form of what the rating measures. */
const TREND_PREFERRED: Record<EntityKind, string[]> = { teams: ['epa_margin_per_play', 'point_margin'], qbs: ['qb_epa_per_dropback'] }
/** The first season whose early team ratings lean on the season before (the prior needs three earlier season pairs). */
const FIRST_PRIOR_SEASON = 2003

function RatingHistory({ kind, season, entityId }: { kind: EntityKind; season: number; entityId: string }) {
  const query = useEntitySection('rating-history', kind, season, entityId)
  if (query.isError) return isMissing(query.error) ? null : <ErrorState error={query.error} title="Could not load the rating by week" />
  if (!query.data) return null
  const { table } = query.data
  const ratings = table.column_groups.ratings ?? []
  const early =
    kind === 'teams' && season >= FIRST_PRIOR_SEASON
      ? 'Early points also lean on the team’s rating last season, a little less after each game until the 9th, because a few games are thin evidence.'
      : `Early points sit near ${kind === 'teams' ? '0 (an average team)' : 'the league average'}, because a few games are thin evidence.`
  return (
    <SectionCard title="Rating by week" description={`Each point is the rating using only the ${season} games through that week. ${early} The last point is the season rating above.`}>
      <TrendChart rows={table.rows} columns={ratings} metadata={table.column_metadata} reference={kind === 'teams' ? { value: 0, caption: 'Dashed line: an average team (0).' } : null} />
    </SectionCard>
  )
}

function SeasonStats({ kind, data, entityId, state, onChange }: { kind: EntityKind; data: EntityTable; entityId: string; state: ViewState; onChange: (change: ViewChange) => void }) {
  const statState: ViewState = { ...state, view: state.view === 'ratings' ? 'per_game' : state.view }
  const view = seasonView(kind, data.table, statState)
  const viewRow: Row = findEntityRow(kind, view.table.rows, entityId) ?? {}
  return (
    <SectionCard title="Season stats" description={VIEWS.find((option) => option.id === statState.view)?.description(kind)}>
      <ViewControls kind={kind} state={statState} onChange={onChange} views={STAT_VIEWS} />
      <MetricSections row={viewRow} rows={view.table.rows} columns={view.metricColumns} metadata={data.table.column_metadata} categories={data.categories} />
    </SectionCard>
  )
}

function GameByGame({ kind, season, entityId, state }: { kind: EntityKind; season: number; entityId: string; state: ViewState }) {
  const query = useEntitySection('game-logs', kind, season, entityId)
  if (query.isLoading) return <Skeleton className="h-64 w-full" />
  if (query.isError) return isMissing(query.error) ? null : <ErrorState error={query.error} title="Could not load the game log" />
  if (!query.data) return null
  const logs = query.data.table
  const selection = gameLogColumns(kind, logs, state)
  const ledger = opponentLedger(logs)
  return (
    <>
      <SectionCard
        title="Game by game"
        description={`Every ${season} game${kind === 'qbs' ? ' he played' : ''}: score and result first, then the opponent’s season ratings and the stats for the view picked. The opponent ratings describe that opponent’s whole season, not this game.`}
      >
        <TrendChart rows={logs.rows} columns={selection.metricColumns} metadata={logs.column_metadata} preferred={TREND_PREFERRED[kind]} />
        <GameLogTable kind={kind} season={season} logs={logs} columns={selection.columns} />
      </SectionCard>
      <SectionCard
        title="Opponents faced"
        description="One row per opponent: an opponent played twice gets one row averaging both games. Its ratings are its whole season’s. What these opponents did against everyone else is in the Opponents per game and per play views of Season stats."
      >
        <DataTable table={withDecimals(ledger, ledger.visible_columns)} defaultSort={{ key: 'opponent_team' }} rowKey={(row) => String(row.opponent_team)} dense />
      </SectionCard>
    </>
  )
}

/** One team's or quarterback's season: ratings, rank range, weekly history, stats, games, opponents. */
export function RatingsDetailPage({ kind, entityId }: { kind: EntityKind; entityId: string }) {
  const config = entityConfig(kind)
  const { query: seasonsQuery, season, seasons, unavailable, setSeason } = useRatingsSeason()
  const { update, request } = useRatingsParams()
  const tableQuery = useEntityTable(kind, season)
  const rankRanges = useRankRanges(kind, season)
  const data = tableQuery.data
  const row = data ? findEntityRow(kind, data.table.rows, entityId) : undefined
  const back = ratingsPagePath(kind, season)

  if (season === null) {
    return (
      <>
        <PageHeader title={config.title} />
        <RatingsSeasonState isLoading={seasonsQuery.isLoading} error={seasonsQuery.error} />
      </>
    )
  }
  const allLink = (
    <Button asChild variant="outline" size="sm">
      <Link to={back}>
        <ArrowLeft /> All {config.title.toLowerCase()}
      </Link>
    </Button>
  )
  if (tableQuery.isLoading) return <Skeleton className="h-96 w-full" />
  if (tableQuery.isError) return <ErrorState error={tableQuery.error} title={`Could not load the ${season} ${config.title.toLowerCase()}`} />
  if (!data) return null
  if (!row) {
    return (
      <>
        <PageHeader title={config.title} actions={<SeasonPicker seasons={seasons} season={season} onChange={setSeason} />} />
        <EmptyState title={`No ${config.singular} ${entityId} in ${season}`} description="Pick another season, or go back to the full list." action={allLink} />
      </>
    )
  }

  const label = entityLabel(kind, row, data.team_names)
  const team = String(row.team ?? '')
  const subtitle = kind === 'teams' ? `Team · ${season} regular season` : `Quarterback · ${data.team_names[team] ?? team} · ${season} regular season`
  const state = resolveViewState(kind, data.categories, request)
  const onChange = (change: ViewChange) => update(viewPatch(change))
  const rankCount = rankRanges.data?.table.rows.length

  return (
    <>
      <PageHeader
        title={label}
        description={subtitle}
        actions={
          <>
            <SeasonPicker seasons={seasons} season={season} onChange={setSeason} />
            {allLink}
            <Button asChild variant="outline" size="sm">
              <Link to={headToHeadPath(kind, season, entityId)}>
                <Swords /> Head to head
              </Link>
            </Button>
          </>
        }
      />
      <div className="flex flex-col gap-5">
        {unavailable !== null ? <UnavailableSeason requested={unavailable} shown={season} /> : null}
        <Card className="px-4 py-3">
          <RatingSummary kind={kind} row={row} rows={data.table.rows} metadata={data.table.column_metadata} />
        </Card>
        <div className="grid gap-5 lg:grid-cols-2">
          {rankRanges.isError && !isMissing(rankRanges.error) ? <ErrorState error={rankRanges.error} title="Could not load the rank ranges" /> : null}
          <RankRangeCard kind={kind} season={season} entityId={entityId} ranges={rankRanges.data} row={row} />
          <HeadToHeadCard kind={kind} season={season} entityId={entityId} rows={data.table.rows} names={data.team_names} />
        </div>
        <RatingHistory kind={kind} season={season} entityId={entityId} />
        <RankHistoryCard kind={kind} season={season} entityId={entityId} count={rankCount} />
        <SeasonStats kind={kind} data={data} entityId={entityId} state={state} onChange={onChange} />
        <GameByGame kind={kind} season={season} entityId={entityId} state={state} />
      </div>
    </>
  )
}
