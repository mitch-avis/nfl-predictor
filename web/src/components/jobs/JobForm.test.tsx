import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { JobTemplate, ParamSpec } from '@/api/types'

import { JobForm, initialValues, submittableValues } from './JobForm'

function spec(overrides: Partial<ParamSpec> & { name: string }): ParamSpec {
  return {
    label: overrides.name,
    kind: 'str',
    description: '',
    required: false,
    default: null,
    choices: [],
    minimum: null,
    maximum: null,
    ...overrides,
  }
}

function template(params: ParamSpec[]): JobTemplate {
  return {
    id: 'lines_refresh',
    label: 'Refresh market lines',
    description: 'Update the spread, total, and moneyline columns.',
    category: 'Data',
    exclusive_group: 'datasets',
    chain_template_id: 'predict',
    writes_datasets: true,
    needs_active_run: false,
    params,
  }
}

describe('JobForm values', () => {
  it('prefers the caller preset over the declared default', () => {
    const values = initialValues(
      template([spec({ name: 'season', kind: 'int', default: 2025 }), spec({ name: 'week', kind: 'int' })]),
      { season: 2026 },
    )
    expect(values).toEqual({ season: 2026, week: '' })
  })

  it('falls back to defaults, and to false for checkboxes', () => {
    const values = initialValues(
      template([spec({ name: 'rounding', kind: 'choice', default: 'nfl', choices: ['nfl', 'none'] }), spec({ name: 'refresh', kind: 'bool' })]),
    )
    expect(values).toEqual({ rounding: 'nfl', refresh: false })
  })

  it('drops blank fields so the backend applies its own defaults', () => {
    expect(submittableValues({ season: 2026, week: '', refresh: false })).toEqual({ season: 2026, refresh: false })
  })
})

describe('JobForm', () => {
  it('renders a field per parameter and submits what was typed', async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(
      <JobForm
        template={template([spec({ name: 'season', label: 'Season', kind: 'int', required: true, default: 2026 }), spec({ name: 'week', label: 'Week', kind: 'int' })])}
        onSubmit={onSubmit}
      />,
    )

    await user.type(screen.getByLabelText(/Week/), '3')
    await user.click(screen.getByRole('button', { name: 'Run job' }))

    expect(onSubmit).toHaveBeenCalledWith({ season: 2026, week: '3' })
  })

  it('says so when a job takes no options', () => {
    render(<JobForm template={template([])} onSubmit={() => {}} />)
    expect(screen.getByText('This job takes no options.')).toBeInTheDocument()
  })
})
