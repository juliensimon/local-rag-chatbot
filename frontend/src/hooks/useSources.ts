/**
 * Hook for fetching available document sources
 */

import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'

export function useSources() {
  return useQuery({
    queryKey: ['sources'],
    queryFn: () => api.sources(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
    select: (data) => data.sources,
  })
}
