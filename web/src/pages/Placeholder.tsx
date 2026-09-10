import { Construction } from 'lucide-react'
import { useLocation } from 'react-router'

import { NAV_ITEMS } from '@/app/nav'
import { PageHeader } from '@/components/common/PageHeader'
import { EmptyState } from '@/components/common/EmptyState'

/** Shown for routes whose page has not been built yet. */
export function PlaceholderPage() {
  const location = useLocation()
  const item = NAV_ITEMS.find((nav) => nav.to !== '/' && location.pathname.startsWith(nav.to))
  return (
    <>
      <PageHeader title={item?.label ?? 'Coming soon'} description={item?.description} />
      <EmptyState
        title="This page is on the roadmap"
        description={
          <span className="inline-flex items-center gap-1">
            <Construction className="size-4" /> It lands in phase {item?.phase ?? '?'} of the web UI plan.
          </span>
        }
      />
    </>
  )
}
