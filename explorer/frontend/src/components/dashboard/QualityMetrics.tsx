/**
 * Quality metrics component
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { QualityMetrics as QualityMetricsType } from '@/types/explorer'

interface QualityMetricsProps {
  metrics: QualityMetricsType
}

export function QualityMetrics({ metrics }: QualityMetricsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Quality Metrics</CardTitle>
        <CardDescription>Analysis of chunk quality and metadata completeness</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div>
            <h3 className="text-sm font-semibold mb-3">Chunk Size Issues</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-lg border p-4">
                <div className="text-sm text-muted-foreground">Too Short (&lt;50 chars)</div>
                <div className="text-2xl font-bold">
                  {metrics.outliers_short}
                  {metrics.outliers_short > 0 && (
                    <Badge variant="destructive" className="ml-2">Issue</Badge>
                  )}
                </div>
              </div>
              <div className="rounded-lg border p-4">
                <div className="text-sm text-muted-foreground">Too Long (&gt;2000 chars)</div>
                <div className="text-2xl font-bold">
                  {metrics.outliers_long}
                  {metrics.outliers_long > 0 && (
                    <Badge variant="destructive" className="ml-2">Issue</Badge>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold mb-3">Metadata Completeness</h3>
            <div className="space-y-2">
              {Object.entries(metrics.metadata_completeness).map(([field, pct]) => (
                <div key={field} className="flex items-center justify-between">
                  <span className="text-sm capitalize">{field}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-32 bg-secondary rounded-full h-2">
                      <div
                        className="bg-primary h-2 rounded-full"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium w-12 text-right">{pct.toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold mb-3">Content Quality</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-lg border p-4">
                <div className="text-sm text-muted-foreground">Duplicate Chunks</div>
                <div className="text-2xl font-bold">
                  {metrics.duplicate_chunks}
                  {metrics.duplicate_chunks > 0 && (
                    <Badge variant="destructive" className="ml-2">Issue</Badge>
                  )}
                </div>
              </div>
              <div className="rounded-lg border p-4">
                <div className="text-sm text-muted-foreground">Low Information Chunks</div>
                <div className="text-2xl font-bold">
                  {metrics.low_information_chunks}
                  {metrics.low_information_chunks > 0 && (
                    <Badge variant="secondary" className="ml-2">Warning</Badge>
                  )}
                </div>
              </div>
            </div>
          </div>

          {metrics.overlap_analysis && (
            <div>
              <h3 className="text-sm font-semibold mb-3">Overlap Analysis</h3>
              <div className="rounded-lg border p-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-muted-foreground">Sources with Overlap</div>
                    <div className="text-xl font-bold">
                      {metrics.overlap_analysis.sources_with_overlap || 0}
                    </div>
                  </div>
                  {metrics.overlap_analysis.avg_overlap_ratio !== undefined && (
                    <div>
                      <div className="text-sm text-muted-foreground">Avg Overlap Ratio</div>
                      <div className="text-xl font-bold">
                        {(metrics.overlap_analysis.avg_overlap_ratio * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
