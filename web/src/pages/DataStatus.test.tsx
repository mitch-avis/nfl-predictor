import { screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DataStatusOut, LeakageAudit } from '@/api/types'
import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'

import { DataStatusPage } from './DataStatus'

function status(leakageAudit: LeakageAudit | null): DataStatusOut {
  return {
    current_season: 2026,
    current_week: 6,
    files: [],
    predict_files: [],
    fingerprint: null,
    cache: { schedule: [], pbp: [] },
    leakage_audit: leakageAudit,
  }
}

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
