/**
 * Main statistics dashboard component
 */

import { useQuery } from '@tanstack/react-query'
import { explorerAPI } from '@/api/client'
import { CollectionStats } from './CollectionStats'
import { SourceStats } from './SourceStats'
import { QualityMetrics } from './QualityMetrics'
import { EmbeddingQuality } from './EmbeddingQuality'
import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { AlertCircle } from 'lucide-react'

export function StatisticsDashboard() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['stats'],
    queryFn: () => explorerAPI.getStats(),
  })

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-destructive" />
            Error Loading Statistics
          </CardTitle>
          <CardDescription>
            {error instanceof Error ? error.message : 'Failed to load statistics'}
          </CardDescription>
        </CardHeader>
      </Card>
    )
  }

  if (!data) {
    return null
  }

  return (
    <div className="space-y-6">
      <CollectionStats stats={data.collection} />
      <SourceStats stats={data.sources} />
      <QualityMetrics metrics={data.quality} />
      {data.embedding && <EmbeddingQuality quality={data.embedding} />}
    </div>
  )
}
