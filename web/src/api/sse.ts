/**
 * Follow one job's output over server-sent events.
 *
 * The backend replays everything after a cursor, so a dropped connection reconnects without
 * losing or repeating lines. Incoming lines are buffered and flushed on a short interval: a
 * chatty job would otherwise re-render the console once per line.
 */
import { useEffect, useRef, useState } from 'react'

import { fetchJobLogs } from './queries'
import type { JobLogLine, JobStatus } from './types'

const FLUSH_MS = 120

interface StatusEvent {
  status: JobStatus
  progress: { current: number; total: number; label: string } | null
  error: string | null
}

interface StreamState {
  jobId: string | null
  lines: JobLogLine[]
  status: JobStatus | null
  connected: boolean
}

export type JobStreamState = Omit<StreamState, 'jobId'>

function empty(): Omit<StreamState, 'jobId'> {
  return { lines: [], status: null, connected: false }
}

/**
 * Stream `jobId`'s logs. Returns every line received so far, the latest status the stream
 * reported, and whether the connection is currently open.
 */
export function useJobStream(jobId: string | null): JobStreamState {
  const [state, setState] = useState<StreamState>({ jobId, ...empty() })
  const buffer = useRef<JobLogLine[]>([])
  const cursor = useRef(0)

  // Watching a different job starts from a clean slate; resetting during render rather than in
  // an effect keeps the console from flashing the previous job's output.
  if (state.jobId !== jobId) setState({ jobId, ...empty() })

  useEffect(() => {
    if (jobId === null) return
    let ended = false
    buffer.current = []
    cursor.current = 0

    const flush = () => {
      if (buffer.current.length === 0) return
      const batch = buffer.current
      buffer.current = []
      setState((previous) => ({ ...previous, lines: [...previous.lines, ...batch] }))
    }
    const timer = window.setInterval(flush, FLUSH_MS)

    const source = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/stream`)
    source.onopen = () => setState((previous) => ({ ...previous, connected: true }))
    source.addEventListener('log', (event) => {
      const row = JSON.parse((event as MessageEvent<string>).data) as JobLogLine
      cursor.current = row.seq
      buffer.current.push(row)
    })
    source.addEventListener('status', (event) => {
      const payload = JSON.parse((event as MessageEvent<string>).data) as StatusEvent
      setState((previous) => ({ ...previous, status: payload.status }))
    })
    source.addEventListener('end', () => {
      ended = true
      source.close()
      setState((previous) => ({ ...previous, connected: false }))
    })
    source.onerror = () => {
      setState((previous) => ({ ...previous, connected: false }))
      if (ended) return
      // The browser retries on its own; fetch anything missed so a long outage still catches up.
      void fetchJobLogs(jobId, cursor.current).then((page) => {
        if (page.lines.length > 0) {
          cursor.current = page.next_seq
          buffer.current.push(...page.lines)
        }
        setState((previous) => ({ ...previous, status: page.status }))
      })
    }

    return () => {
      ended = true
      window.clearInterval(timer)
      source.close()
    }
  }, [jobId])

  return { lines: state.lines, status: state.status, connected: state.connected }
}
