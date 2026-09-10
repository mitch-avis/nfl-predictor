import { Check, Copy } from 'lucide-react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'

/** Copy `text` to the clipboard with a brief confirmation. */
export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false)
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text)
          setDone(true)
          setTimeout(() => setDone(false), 1500)
        } catch {
          /* clipboard unavailable */
        }
      }}
    >
      {done ? <Check className="size-4" /> : <Copy className="size-4" />} {done ? 'Copied' : label}
    </Button>
  )
}
