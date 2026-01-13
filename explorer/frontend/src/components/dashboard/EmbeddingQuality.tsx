/**
 * Embedding quality metrics component
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import type { EmbeddingQuality as EmbeddingQualityType } from '@/types/explorer'
import { SimilarityDistribution } from '@/components/visualization/SimilarityDistribution'

interface EmbeddingQualityProps {
  quality: EmbeddingQualityType
}

export function EmbeddingQuality({ quality }: EmbeddingQualityProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Embedding Quality</CardTitle>
        <CardDescription>Analysis of embedding similarity distribution</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {quality.avg_similarity !== null && (
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-sm text-muted-foreground">Average Similarity</div>
                <div className="text-2xl font-bold">{quality.avg_similarity.toFixed(3)}</div>
              </div>
              {quality.min_similarity !== null && (
                <div>
                  <div className="text-sm text-muted-foreground">Min Similarity</div>
                  <div className="text-2xl font-bold">{quality.min_similarity.toFixed(3)}</div>
                </div>
              )}
              {quality.max_similarity !== null && (
                <div>
                  <div className="text-sm text-muted-foreground">Max Similarity</div>
                  <div className="text-2xl font-bold">{quality.max_similarity.toFixed(3)}</div>
                </div>
              )}
            </div>
          )}

          {Object.keys(quality.similarity_distribution).length > 0 && (
            <div>
              <SimilarityDistribution distribution={quality.similarity_distribution} />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
