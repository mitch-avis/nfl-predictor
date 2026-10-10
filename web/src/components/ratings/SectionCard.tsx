import { useId, type ReactNode } from 'react'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

/** A titled card that is also a named region, so each section of a long page can be found. */
export function SectionCard({ title, description, children }: { title: string; description?: ReactNode; children: ReactNode }) {
  const titleId = useId()
  return (
    <Card role="region" aria-labelledby={titleId} className="gap-4">
      <CardHeader>
        <CardTitle id={titleId} className="text-base">
          {title}
        </CardTitle>
        {description ? <CardDescription>{description}</CardDescription> : null}
      </CardHeader>
      <CardContent className="flex flex-col gap-4">{children}</CardContent>
    </Card>
  )
}
