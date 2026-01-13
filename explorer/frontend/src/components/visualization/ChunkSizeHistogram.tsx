/**
 * Chunk size distribution histogram
 */

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface ChunkSizeHistogramProps {
  distribution: Record<string, number>
}

export function ChunkSizeHistogram({ distribution }: ChunkSizeHistogramProps) {
  const data = Object.entries(distribution)
    .map(([bin, count]) => ({
      bin,
      count,
    }))
    .sort((a, b) => {
      // Sort by bin start value
      const aStart = parseInt(a.bin.split('-')[0]) || 0
      const bStart = parseInt(b.bin.split('-')[0]) || 0
      return aStart - bStart
    })

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="bin" />
        <YAxis />
        <Tooltip />
        <Bar dataKey="count" fill="hsl(var(--primary))" />
      </BarChart>
    </ResponsiveContainer>
  )
}
