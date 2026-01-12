/**
 * Hook for API health monitoring
 */

import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'

export function useHealthCheck() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => api.health(),
    refetchInterval: 30000, // Check every 30 seconds
    refetchOnWindowFocus: true,
    retry: 1,
  })
}
