import { AlertTriangle } from 'lucide-react'

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'

/** Render a query or mutation error in a consistent way. */
export function ErrorState({ error, title = 'Something went wrong' }: { error: unknown; title?: string }) {
  const message = error instanceof Error ? error.message : String(error)
  return (
    <Alert variant="destructive">
      <AlertTriangle className="size-4" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>{message}</AlertDescription>
    </Alert>
  )
}
