/**
 * API client for explorer backend
 */

import type {
  DocumentsResponse,
  HealthResponse,
  RecommendationsResponse,
  SearchRequest,
  SearchResponse,
  SimilarityResponse,
  StatsResponse,
} from '@/types/explorer'

const API_BASE = '/api/explorer'

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(error.detail || `HTTP error! status: ${response.status}`)
  }

  return response.json()
}

export const explorerAPI = {
  health: (): Promise<HealthResponse> => fetchAPI('/health'),

  getStats: (): Promise<StatsResponse> => fetchAPI('/stats'),

  getCollectionStats: () => fetchAPI('/stats/collection'),

  getSourceStats: () => fetchAPI('/stats/sources'),

  getQualityMetrics: () => fetchAPI('/stats/quality'),

  getRecommendations: (): Promise<RecommendationsResponse> =>
    fetchAPI('/recommendations'),

  search: (request: SearchRequest): Promise<SearchResponse> =>
    fetchAPI('/search', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  getDocuments: (params: {
    page?: number
    page_size?: number
    source?: string
    min_page?: number
    max_page?: number
    min_chunk_size?: number
    max_chunk_size?: number
  }): Promise<DocumentsResponse> => {
    const searchParams = new URLSearchParams()
    if (params.page) searchParams.set('page', params.page.toString())
    if (params.page_size) searchParams.set('page_size', params.page_size.toString())
    if (params.source) searchParams.set('source', params.source)
    if (params.min_page !== undefined)
      searchParams.set('min_page', params.min_page.toString())
    if (params.max_page !== undefined)
      searchParams.set('max_page', params.max_page.toString())
    if (params.min_chunk_size !== undefined)
      searchParams.set('min_chunk_size', params.min_chunk_size.toString())
    if (params.max_chunk_size !== undefined)
      searchParams.set('max_chunk_size', params.max_chunk_size.toString())

    const query = searchParams.toString()
    return fetchAPI(`/documents${query ? `?${query}` : ''}`)
  },

  getDocument: (docId: string) => fetchAPI(`/documents/${docId}`),

  getSimilarDocuments: (docId: string, k?: number): Promise<SimilarityResponse> => {
    const params = k ? `?k=${k}` : ''
    return fetchAPI(`/similarity/${docId}${params}`)
  },
}
