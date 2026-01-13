/**
 * Collection-level statistics component
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { CollectionStats as CollectionStatsType } from '@/types/explorer'
import { ChunkSizeHistogram } from '@/components/visualization/ChunkSizeHistogram'

interface CollectionStatsProps {
  stats: CollectionStatsType
}

function formatBytes(bytes: number | null): string {
  if (bytes === null) return 'N/A'
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`
}

export function CollectionStats({ stats }: CollectionStatsProps) {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Collection Overview</CardTitle>
          <CardDescription>High-level statistics about the vector store</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <div>
              <div className="text-2xl font-bold">{stats.total_chunks.toLocaleString()}</div>
              <div className="text-sm text-muted-foreground">Total Chunks</div>
            </div>
            <div>
              <div className="text-2xl font-bold">{stats.unique_sources}</div>
              <div className="text-sm text-muted-foreground">Unique Sources</div>
            </div>
            <div>
              <div className="text-2xl font-bold">{stats.avg_chunk_size.toFixed(0)}</div>
              <div className="text-sm text-muted-foreground">Avg Chunk Size (chars)</div>
            </div>
            <div>
              <div className="text-2xl font-bold">{stats.embedding_dimensions}</div>
              <div className="text-sm text-muted-foreground">Embedding Dimensions</div>
            </div>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-muted-foreground">Chunk Size Range</div>
              <div className="text-lg font-semibold">
                {stats.min_chunk_size} - {stats.max_chunk_size} chars
              </div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground">Database Size</div>
              <div className="text-lg font-semibold">{formatBytes(stats.db_size_bytes)}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {Object.keys(stats.chunk_size_distribution).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Chunk Size Distribution</CardTitle>
            <CardDescription>Distribution of chunk sizes across the collection</CardDescription>
          </CardHeader>
          <CardContent>
            <ChunkSizeHistogram distribution={stats.chunk_size_distribution} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}
