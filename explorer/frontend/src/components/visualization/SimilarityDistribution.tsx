/**
 * Similarity distribution chart
 */

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface SimilarityDistributionProps {
  distribution: Record<string, number>
}

export function SimilarityDistribution({ distribution }: SimilarityDistributionProps) {
  const data = Object.entries(distribution)
    .map(([similarity, count]) => ({
      similarity: parseFloat(similarity),
      count,
    }))
    .sort((a, b) => a.similarity - b.similarity)

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="similarity" />
        <YAxis />
        <Tooltip />
        <Bar dataKey="count" fill="hsl(var(--primary))" />
      </BarChart>
    </ResponsiveContainer>
  )
}
