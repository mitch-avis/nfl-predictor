import { Navigate, useLocation } from 'react-router'

import { useSession } from '@/api/queries'
import { Skeleton } from '@/components/ui/skeleton'

import { AppShell } from './AppShell'

/** Gate the app behind a session; unauthenticated users go to the login page. */
export function RequireAuth() {
  const session = useSession()
  const location = useLocation()
  if (session.isLoading) {
    return (
      <div className="mx-auto max-w-3xl space-y-4 p-6">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-40 w-full" />
      </div>
    )
  }
  if (session.isError) {
    return (
      <div className="p-6 text-sm text-destructive">
        Could not reach the API: {(session.error as Error).message}
      </div>
    )
  }
  if (!session.data) {
    return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />
  }
  return <AppShell user={session.data.user} />
}
