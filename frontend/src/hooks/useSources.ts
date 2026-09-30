/**
 * Hook for fetching available document sources
 */

import { useQuery } from '@tanstack/react-query'
import { api, ApiError } from '@/api/client'

export function useSources(userId: string | null = null) {
  return useQuery({
    queryKey: ['sources', userId],
    queryFn: () => api.sources(userId),
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
    // 404 means the user has no documents; retrying won't change that
    retry: (failureCount, error) =>
      !(error instanceof ApiError && error.status === 404) && failureCount < 3,
    select: (data) => data.sources,
  })
}
