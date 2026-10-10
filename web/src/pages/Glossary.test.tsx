import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { MetricRegistry, Registry } from '@/api/types'
import { stubApi } from '@/test/api'
import { renderWithProviders } from '@/test/render'

import { GlossaryPage } from './Glossary'

const MODEL_REGISTRY: Registry = {
  columns: {
    home_win_prob: {
      key: 'home_win_prob',
      label: 'Home win %',
      description: 'The model’s chance that the home team wins.',
      group: 'Predictions',
      kind: 'prob',
      polarity: 'neutral',
      heatmap: false,
      decimals: 1,
      actionable: true,
      sticky: false,
    },
  },
  groups: { Predictions: ['home_win_prob'] },
}

const METRICS: MetricRegistry = {
  entities: {
    team: { categories: [{ name: 'Schedule-Adjusted Ratings', description: 'Ratings.', subcategories: [] }, { name: 'Offense', description: 'What the offense did.', subcategories: [] }] },
    qb: { categories: [{ name: 'Passing Efficiency', description: 'Quality per play.', subcategories: [] }] },
  },
  metrics: {
    team_rating: {
      label: 'Team Rating',
      full_name: 'Team Rating',
      description: 'Points per game better or worse than an average team.',
      entity: 'team',
      category: 'Schedule-Adjusted Ratings',
      subcategory: null,
      shape: 'score',
      polarity: 'higher',
      source: 'ratings',
      denominator: null,
      since: null,
      duplicate_of: null,
      contextual: false,
      formula: null,
      note: null,
      percent: false,
    },
    qb_passer_rating: {
      label: 'Rating',
      full_name: 'Passer Rating',
      description: 'The NFL passer rating.',
      entity: 'qb',
      category: 'Passing Efficiency',
      subcategory: null,
      shape: 'rate',
      polarity: 'higher',
      source: 'play-by-play',
      denominator: 'attempts',
      since: null,
      duplicate_of: null,
      contextual: false,
      formula: 'The NFL formula.',
      note: null,
      percent: false,
    },
  },
  prefix_rules: [],
}

describe('GlossaryPage', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('opens on the model’s columns, grouped as before', async () => {
    // Arrange
    stubApi({ '/api/registry': MODEL_REGISTRY, '/api/ratings/metrics': METRICS })

    // Act
    renderWithProviders(<GlossaryPage />, { route: '/glossary' })

    // Assert
    const group = await screen.findByRole('region', { name: 'Predictions' })
    expect(within(group).getByText('The model’s chance that the home team wins.')).toBeInTheDocument()
  })

  it('describes the team and quarterback stats from the metric registry', async () => {
    // Arrange
    stubApi({ '/api/registry': MODEL_REGISTRY, '/api/ratings/metrics': METRICS })

    // Act
    renderWithProviders(<GlossaryPage />, { route: '/glossary?tab=ratings' })

    // Assert
    const start = await screen.findByRole('region', { name: 'Start here' })
    expect(within(start).getByText('Team Rating')).toBeInTheDocument()
    const efficiency = screen.getByRole('region', { name: 'Quarterbacks: Passing Efficiency' })
    expect(within(efficiency).getByText('How it’s computed: The NFL formula.')).toBeInTheDocument()
  })

  it('narrows the entries to the search', async () => {
    // Arrange
    stubApi({ '/api/registry': MODEL_REGISTRY, '/api/ratings/metrics': METRICS })
    const user = userEvent.setup()
    renderWithProviders(<GlossaryPage />, { route: '/glossary?tab=ratings' })
    await screen.findByRole('region', { name: 'Start here' })

    // Act
    await user.type(screen.getByRole('searchbox', { name: 'Search the glossary' }), 'passer')

    // Assert
    expect(screen.queryByRole('region', { name: 'Start here' })).not.toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Quarterbacks: Passing Efficiency' })).toBeInTheDocument()
  })
})
