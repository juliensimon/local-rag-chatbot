/**
 * Source-level statistics component
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { SourceStatsResponse } from '@/types/explorer'
import { Badge } from '@/components/ui/badge'

interface SourceStatsProps {
  stats: SourceStatsResponse
}

export function SourceStats({ stats }: SourceStatsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Source Statistics</CardTitle>
        <CardDescription>
          Statistics for each source document in the vector store
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[600px]">
          <div className="space-y-4">
            {stats.sources.map((source, idx) => (
              <div
                key={idx}
                className="rounded-lg border p-4 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold">{source.filename}</h3>
                  <Badge variant="secondary">{source.chunk_count} chunks</Badge>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Pages</div>
                    <div className="font-medium">
                      {source.pages_with_chunks} / {source.total_pages}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Coverage</div>
                    <div className="font-medium">{source.page_coverage.toFixed(1)}%</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Avg Size</div>
                    <div className="font-medium">{source.avg_chunk_size.toFixed(0)} chars</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Size Range</div>
                    <div className="font-medium">
                      {source.min_chunk_size} - {source.max_chunk_size}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
