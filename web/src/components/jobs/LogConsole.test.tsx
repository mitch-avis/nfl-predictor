import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import type { JobLogLine } from '@/api/types'

import { LogConsole, filterLines } from './LogConsole'
import { JobStatusBadge } from './JobStatusBadge'

const lines: JobLogLine[] = [
  { seq: 1, ts: '2026-09-10 12:00:00', level: 'DEBUG', line: 'noisy detail' },
  { seq: 2, ts: '2026-09-10 12:00:01', level: 'INFO', line: 'loading schedule' },
  { seq: 3, ts: '2026-09-10 12:00:02', level: 'WARNING', line: 'no lines yet' },
  { seq: 4, ts: '2026-09-10 12:00:03', level: 'ERROR', line: 'it broke' },
]

describe('filterLines', () => {
  it('keeps the chosen level and everything more severe', () => {
    expect(filterLines(lines, 'WARNING').map((line) => line.seq)).toEqual([3, 4])
    expect(filterLines(lines, 'DEBUG')).toHaveLength(4)
  })

  it('treats an unrecognized level as INFO', () => {
    const odd: JobLogLine[] = [{ seq: 9, ts: '', level: 'NOTICE', line: 'hello' }]
    expect(filterLines(odd, 'INFO')).toHaveLength(1)
    expect(filterLines(odd, 'WARNING')).toHaveLength(0)
  })
})

describe('LogConsole', () => {
  it('hides debug output until the filter asks for it', async () => {
    const user = userEvent.setup()
    render(<LogConsole lines={lines} />)

    expect(screen.queryByText('noisy detail')).not.toBeInTheDocument()
    expect(screen.getByText('loading schedule')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'DEBUG' }))

    expect(screen.getByText('noisy detail')).toBeInTheDocument()
  })

  it('shows a placeholder when there is nothing to display', () => {
    render(<LogConsole lines={[]} empty="This job produced no output." />)
    expect(screen.getByText('This job produced no output.')).toBeInTheDocument()
  })
})

describe('JobStatusBadge', () => {
  it('names each state', () => {
    render(<JobStatusBadge status="running" />)
    expect(screen.getByText('Running')).toBeInTheDocument()
    render(<JobStatusBadge status="canceled" />)
    expect(screen.getByText('Canceled')).toBeInTheDocument()
  })
})
