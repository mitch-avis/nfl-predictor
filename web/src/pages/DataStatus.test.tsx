import { screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DataStatusOut, LeakageAudit } from '@/api/types'
import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'

import { DataStatusPage } from './DataStatus'

function status(leakageAudit: LeakageAudit | null, overrides: Partial<DataStatusOut> = {}): DataStatusOut {
  return {
    current_season: 2026,
    current_week: 6,
    files: [],
    predict_files: [],
    fingerprint: null,
    cache: { schedule: [], pbp: [] },
    cache_files: [],
    other_seasons: { team_rankings: [], etl_seasons: [] },
    leakage_audit: leakageAudit,
    ...overrides,
  }
}

async function cardTitled(title: string): Promise<HTMLElement> {
  const heading = await screen.findByText(title)
  const card = heading.closest('[data-slot="card"]')
  if (!(card instanceof HTMLElement)) throw new Error(`${title} card not found`)
  return card
}

describe('DataStatusPage cache card', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('shows a season strip with a plain label for every cached source', async () => {
    // Arrange
    stubApi({
      '/api/data/status': status(null, {
        cache: {
          schedule: [2025, 2026],
          pbp: [2026],
          official_team_stats: [2024],
          snap_counts: [2012, 2026],
          rosters_weekly: [2003],
          depth_charts: [2020],
        },
        other_seasons: { team_rankings: [2003, 2004], etl_seasons: [1999] },
      }),
    })

    // Act
    renderWithProviders(<DataStatusPage />)

    // Assert
    const card = await cardTitled('Source caches')
    for (const label of [
      'Schedules',
      'Play-by-play',
      'Official team stats',
      'Snap counts',
      'Weekly rosters',
      'Depth charts',
      'TeamRankings scrapes',
      'ETL season builds',
    ]) {
      expect(within(card).getByText(label)).toBeInTheDocument()
    }
    expect(card).toHaveTextContent('2012–2026')
  })

  it('lists the cache entries that are not season files with their file counts', async () => {
    // Arrange
    stubApi({
      '/api/data/status': status(null, {
        cache_files: [
          { cache: 'nflreadpy', name: 'players.parquet', file_count: 1, size: 439522, modified_at: '2026-10-16T07:55:07Z' },
          { cache: 'nfelo', name: 'snapshots/', file_count: 5, size: 5954537, modified_at: '2026-10-16T07:55:11Z' },
        ],
      }),
    })

    // Act
    renderWithProviders(<DataStatusPage />)

    // Assert
    const card = await cardTitled('Source caches')
    const items = within(card).getAllByRole('listitem')
    expect(items).toHaveLength(2)
    expect(items[0]).toHaveTextContent('nflreadpy/players.parquet')
    expect(items[1]).toHaveTextContent('nfelo/snapshots/')
    expect(items[1]).toHaveTextContent('5 files')
  })

  it('shows a single cached season once, not as a range', async () => {
    // Arrange
    stubApi({ '/api/data/status': status(null, { cache: { schedule: [], pbp: [2026] } }) })

    // Act
    renderWithProviders(<DataStatusPage />)

    // Assert
    const card = await cardTitled('Source caches')
    expect(within(card).getByText('2026')).toBeInTheDocument()
    expect(card).not.toHaveTextContent('2026–2026')
  })

  it('names the seasons missing inside a strip in words', async () => {
    // Arrange
    stubApi({ '/api/data/status': status(null, { cache: { schedule: [2003, 2005, 2006, 2008], pbp: [] } }) })

    // Act
    renderWithProviders(<DataStatusPage />)

    // Assert
    const card = await cardTitled('Source caches')
    expect(within(card).getByText('Missing: 2004, 2007')).toBeInTheDocument()
  })

  it('names no missing seasons when a strip has no gaps', async () => {
    // Arrange
    stubApi({ '/api/data/status': status(null, { cache: { schedule: [2003, 2004, 2005], pbp: [2026] } }) })

    // Act
    renderWithProviders(<DataStatusPage />)

    // Assert
    const card = await cardTitled('Source caches')
    expect(card).not.toHaveTextContent('Missing:')
  })

  it('still renders when nothing is cached', async () => {
    // Arrange
    stubApi({ '/api/data/status': status(null) })

    // Act
    renderWithProviders(<DataStatusPage />)

    // Assert
    const card = await cardTitled('Source caches')
    expect(within(card).getByText('Schedules: none cached')).toBeInTheDocument()
    expect(within(card).getByText('Play-by-play: none cached')).toBeInTheDocument()
    expect(within(card).getByText('TeamRankings scrapes: none cached')).toBeInTheDocument()
    expect(within(card).getByText('No other cache files.')).toBeInTheDocument()
  })
})

describe('DataStatusPage files card', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('shows a directory entry with its file count', async () => {
    // Arrange
    stubApi({
      '/api/data/status': status(null, {
        files: [
          {
            name: 'ratings/',
            description: 'The ported ratings tables.',
            exists: true,
            size: 44891284,
            modified_at: '2026-10-16T08:00:00Z',
            rows: null,
            seasons: [1999, 2026],
            file_count: 534,
          },
        ],
      }),
    })

    // Act
    renderWithProviders(<DataStatusPage />)

    // Assert
    const card = await cardTitled('Dataset files')
    expect(within(card).getByText('ratings/')).toBeInTheDocument()
    expect(card).toHaveTextContent('534 files')
    expect(card).toHaveTextContent('1999–2026')
  })
})

function audit(overrides: Partial<LeakageAudit>): LeakageAudit {
  return {
    ok: true,
    row_count: 1234,
    feature_count: 87,
    failures: [],
    warnings: [],
    flagged_columns: [],
    modified_at: '2026-10-15T12:00:00Z',
    ...overrides,
  }
}

async function leakageCard(): Promise<HTMLElement> {
  const title = await screen.findByText('Leakage audit')
  const card = title.closest('[data-slot="card"]')
  if (!(card instanceof HTMLElement)) throw new Error('leakage audit card not found')
  return card
}

describe('DataStatusPage leakage-audit card', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('lists each flagged column by name with its readable reasons', async () => {
    // Arrange
    stubApi({
      '/api/data/status': status(
        audit({
          flagged_columns: [
            { column: 'elo_post', reason: 'suspicious_name' },
            {
              column: 'home_points_proxy',
              reason: 'high_corr_margin,matches_home_score',
              corr_margin: 0.91,
              corr_total: null,
              matches_home_score_rate: 0.42,
              matches_away_score_rate: null,
            },
          ],
        }),
      ),
    })

    // Act
    renderWithProviders(<DataStatusPage />)

    // Assert
    const card = await leakageCard()
    const items = within(card).getAllByRole('listitem')
    expect(items).toHaveLength(2)
    expect(items[0]).toHaveTextContent('elo_post')
    expect(items[0]).toHaveTextContent('suspicious name')
    expect(items[1]).toHaveTextContent('home_points_proxy')
    expect(items[1]).toHaveTextContent('high corr margin, matches home score')
    expect(card).toHaveTextContent('87 features · 1,234 rows · 2 flagged')
    expect(card).not.toHaveTextContent('[object Object]')
  })

  it('lists a column flagged by name and by statistics twice without a duplicate-key warning', async () => {
    // Arrange
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
    stubApi({
      '/api/data/status': status(
        audit({
          flagged_columns: [
            { column: 'final_margin_proxy', reason: 'suspicious_name' },
            { column: 'final_margin_proxy', reason: 'high_corr_margin', corr_margin: 0.996 },
          ],
        }),
      ),
    })

    // Act
    renderWithProviders(<DataStatusPage />)

    // Assert
    const card = await leakageCard()
    expect(within(card).getAllByRole('listitem')).toHaveLength(2)
    expect(consoleError.mock.calls.filter((args) => String(args[0]).includes('same key'))).toEqual([])
  })

  it('shows a flag with no reason without a trailing separator', async () => {
    // Arrange
    stubApi({ '/api/data/status': status(audit({ flagged_columns: [{ column: 'x_col', reason: '' }] })) })

    // Act
    renderWithProviders(<DataStatusPage />)

    // Assert
    const card = await leakageCard()
    expect(within(card).getByRole('listitem').textContent).toBe('x_col')
  })

  it('lists the failure messages when the audit failed', async () => {
    // Arrange
    stubApi({
      '/api/data/status': status(
        audit({
          ok: false,
          failures: ["Target columns present in features: ['home_score']", 'Column x matches home_score in 99% of rows'],
        }),
      ),
    })

    // Act
    renderWithProviders(<DataStatusPage />)

    // Assert
    const card = await leakageCard()
    expect(within(card).getByText('Findings')).toBeInTheDocument()
    expect(within(card).getByText("Target columns present in features: ['home_score']")).toBeInTheDocument()
    expect(within(card).getByText('Column x matches home_score in 99% of rows')).toBeInTheDocument()
  })

  it('shows zero flagged and no list for a clean audit', async () => {
    // Arrange
    stubApi({ '/api/data/status': status(audit({})) })

    // Act
    renderWithProviders(<DataStatusPage />)

    // Assert
    const card = await leakageCard()
    expect(within(card).getByText('Clean')).toBeInTheDocument()
    expect(card).toHaveTextContent('0 flagged')
    expect(within(card).queryByRole('list')).not.toBeInTheDocument()
  })

  it('says no report was found when there is no audit', async () => {
    // Arrange
    stubApi({ '/api/data/status': status(null) })

    // Act
    renderWithProviders(<DataStatusPage />)

    // Assert
    const card = await leakageCard()
    expect(within(card).getByText('No audit report found under models/ or reports/.')).toBeInTheDocument()
  })
})
