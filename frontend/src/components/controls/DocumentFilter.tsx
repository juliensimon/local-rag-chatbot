/**
 * Document filter dropdown
 */

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { FileText } from 'lucide-react'
import { useSettings } from '@/context/SettingsContext'
import { useSources } from '@/hooks/useSources'

export function DocumentFilter() {
  const { docFilter, setDocFilter, ragEnabled } = useSettings()
  const { data: sources, isLoading } = useSources()

  if (!ragEnabled) return null

  if (isLoading) {
    return <Skeleton className="h-9 w-48" />
  }

  return (
    <div className="space-y-2">
      <label className="flex items-center gap-1 text-sm font-medium">
        <FileText className="h-4 w-4" />
        Document
      </label>
      <Select
        value={docFilter || 'all'}
        onValueChange={(value) => setDocFilter(value === 'all' ? null : value)}
      >
        <SelectTrigger className="w-48" aria-label="Filter by document">
          <SelectValue placeholder="All Documents" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All Documents</SelectItem>
          {sources?.map((source) => (
            <SelectItem key={source} value={source}>
              {source}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
