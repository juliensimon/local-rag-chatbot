/**
 * Display for rewritten query badge
 */

import { Badge } from '@/components/ui/badge'
import { RefreshCw } from 'lucide-react'

interface RewrittenQueryProps {
  query: string
}

export function RewrittenQuery({ query }: RewrittenQueryProps) {
  return (
    <div className="flex items-start gap-2 rounded-md bg-muted/50 p-2">
      <RefreshCw className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
      <div className="space-y-1">
        <Badge variant="secondary" className="text-xs">
          Rewritten Query
        </Badge>
        <p className="text-sm text-muted-foreground">{query}</p>
      </div>
    </div>
  )
}
