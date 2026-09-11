import { useEffect, useMemo, useRef, useState } from 'react'

import type { JobLogLine } from '@/api/types'
import { Button } from '@/components/ui/button'
import { cn } from '@/utils/cn'

/** Levels in severity order; picking one shows it and everything above it. */
export const LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR'] as const
export type Level = (typeof LEVELS)[number]

const RANK: Record<string, number> = { DEBUG: 0, INFO: 1, WARNING: 2, ERROR: 3, CRITICAL: 4 }
const LINE_STYLE: Record<string, string> = {
  DEBUG: 'text-muted-foreground',
  INFO: 'text-foreground',
  WARNING: 'text-warning',
  ERROR: 'text-destructive',
  CRITICAL: 'text-destructive font-semibold',
}
/** Rendering every line of a long walk-forward run would stall the page; keep the newest ones. */
export const MAX_RENDERED = 2000

/** Filter log lines to those at or above `level`. */
export function filterLines(lines: JobLogLine[], level: Level): JobLogLine[] {
  const floor = RANK[level]
  return lines.filter((line) => (RANK[line.level] ?? RANK.INFO) >= floor)
}

/** Scrolling console with a level filter and follow-the-tail behavior. */
export function LogConsole({ lines, empty = 'Waiting for output…' }: { lines: JobLogLine[]; empty?: string }) {
  const [level, setLevel] = useState<Level>('INFO')
  const [follow, setFollow] = useState(true)
  const boxRef = useRef<HTMLDivElement>(null)

  const visible = useMemo(() => {
    const filtered = filterLines(lines, level)
    return filtered.length > MAX_RENDERED ? filtered.slice(-MAX_RENDERED) : filtered
  }, [lines, level])
  const hidden = filterLines(lines, level).length - visible.length

  useEffect(() => {
    if (!follow || boxRef.current === null) return
    boxRef.current.scrollTop = boxRef.current.scrollHeight
  }, [visible, follow])

  return (
    <div className="rounded-xl border">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b px-3 py-2">
        <div className="flex flex-wrap items-center gap-1">
          {LEVELS.map((option) => (
            <Button
              key={option}
              type="button"
              size="sm"
              variant={option === level ? 'secondary' : 'ghost'}
              onClick={() => setLevel(option)}
            >
              {option}
            </Button>
          ))}
        </div>
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <input
            type="checkbox"
            checked={follow}
            onChange={(event) => setFollow(event.target.checked)}
            className="size-3.5 accent-current"
          />
          Follow output
        </label>
      </div>
      <div
        ref={boxRef}
        onScroll={(event) => {
          const box = event.currentTarget
          const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 24
          if (atBottom !== follow) setFollow(atBottom)
        }}
        className="h-[60vh] overflow-auto bg-muted/30 p-3 font-mono text-xs leading-relaxed"
      >
        {hidden > 0 ? (
          <div className="mb-2 text-muted-foreground">… {hidden} earlier lines not shown</div>
        ) : null}
        {visible.length === 0 ? <div className="text-muted-foreground">{empty}</div> : null}
        {visible.map((line) => (
          <div key={line.seq} className={cn('whitespace-pre-wrap break-words', LINE_STYLE[line.level])}>
            {line.line}
          </div>
        ))}
      </div>
    </div>
  )
}
