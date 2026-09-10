import { cn } from '@/utils/cn'

const STYLES: Record<string, string> = {
  PASS: 'bg-muted text-muted-foreground',
  LEAN: 'bg-secondary text-secondary-foreground',
  SMALL: 'bg-success/20 text-foreground',
  MEDIUM: 'bg-success/45 text-foreground',
  STRONG: 'bg-success text-success-foreground',
}

/** Colored chip for PASS / LEAN / SMALL / MEDIUM / STRONG. */
export function ActionBadge({ action, muted = false }: { action: string | null | undefined; muted?: boolean }) {
  const label = action ?? '—'
  return (
    <span
      className={cn(
        'inline-flex min-w-16 justify-center rounded-md px-2 py-0.5 text-[11px] font-semibold tracking-wide',
        STYLES[label] ?? 'bg-muted text-muted-foreground',
        muted && 'opacity-50 grayscale',
      )}
    >
      {label}
    </span>
  )
}
