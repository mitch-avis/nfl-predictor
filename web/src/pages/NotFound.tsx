import { Link, useLocation } from 'react-router'

import { EmptyState } from '@/components/common/EmptyState'
import { PageHeader } from '@/components/common/PageHeader'
import { Button } from '@/components/ui/button'

/** Shown for any path the app has no page for. */
export function NotFoundPage() {
  const location = useLocation()
  return (
    <>
      <PageHeader title="Page not found" />
      <EmptyState
        title="This page doesn’t exist."
        description={
          <>
            Nothing lives at <code>{location.pathname}</code>. Pick a page from the sidebar.
          </>
        }
        action={
          <Button variant="outline" asChild>
            <Link to="/">Go to the overview</Link>
          </Button>
        }
      />
    </>
  )
}
