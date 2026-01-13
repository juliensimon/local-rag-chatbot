/**
 * Individual recommendation card component
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { Recommendation } from '@/types/explorer'
import { AlertTriangle, Info, AlertCircle } from 'lucide-react'

interface RecommendationCardProps {
  recommendation: Recommendation
}

function getSeverityIcon(severity: string) {
  switch (severity) {
    case 'high':
      return <AlertTriangle className="h-5 w-5 text-destructive" />
    case 'medium':
      return <AlertCircle className="h-5 w-5 text-yellow-500" />
    default:
      return <Info className="h-5 w-5 text-blue-500" />
  }
}

function getSeverityVariant(severity: string): 'default' | 'secondary' | 'destructive' {
  switch (severity) {
    case 'high':
      return 'destructive'
    case 'medium':
      return 'secondary'
    default:
      return 'default'
  }
}

export function RecommendationCard({ recommendation }: RecommendationCardProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3 flex-1">
            {getSeverityIcon(recommendation.severity)}
            <div className="flex-1">
              <CardTitle className="text-base">{recommendation.title}</CardTitle>
              <CardDescription className="mt-1">{recommendation.description}</CardDescription>
            </div>
          </div>
          <Badge variant={getSeverityVariant(recommendation.severity)}>
            {recommendation.severity}
          </Badge>
        </div>
      </CardHeader>
      {recommendation.action_items.length > 0 && (
        <CardContent>
          <div className="space-y-2">
            <h4 className="text-sm font-semibold">Recommended Actions:</h4>
            <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
              {recommendation.action_items.map((item, idx) => (
                <li key={idx}>{item}</li>
              ))}
            </ul>
          </div>
        </CardContent>
      )}
    </Card>
  )
}
