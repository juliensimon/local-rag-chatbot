/**
 * Recommendations panel component
 */

import { useQuery } from '@tanstack/react-query'
import { explorerAPI } from '@/api/client'
import { RecommendationCard } from './RecommendationCard'
import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { AlertCircle } from 'lucide-react'

export function RecommendationsPanel() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['recommendations'],
    queryFn: () => explorerAPI.getRecommendations(),
  })

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    )
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-destructive" />
            Error Loading Recommendations
          </CardTitle>
          <CardDescription>
            {error instanceof Error ? error.message : 'Failed to load recommendations'}
          </CardDescription>
        </CardHeader>
      </Card>
    )
  }

  if (!data || data.recommendations.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Optimization Recommendations</CardTitle>
          <CardDescription>No recommendations at this time</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Your vector store appears to be well-optimized!
          </p>
        </CardContent>
      </Card>
    )
  }

  // Group by category
  const byCategory = data.recommendations.reduce(
    (acc, rec) => {
      if (!acc[rec.category]) {
        acc[rec.category] = []
      }
      acc[rec.category].push(rec)
      return acc
    },
    {} as Record<string, typeof data.recommendations>
  )

  return (
    <div className="space-y-6">
      {Object.entries(byCategory).map(([category, recommendations]) => (
        <div key={category}>
          <h2 className="text-lg font-semibold mb-4 capitalize">{category}</h2>
          <div className="space-y-4">
            {recommendations.map((rec, idx) => (
              <RecommendationCard key={idx} recommendation={rec} />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
