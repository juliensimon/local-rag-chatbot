/**
 * API client for RAG chatbot backend
 */

import type {
  ChatRequest,
  ChatResponse,
  HealthResponse,
  SourcesResponse
} from '@/types/api'

const API_BASE = import.meta.env.VITE_API_BASE || ''

class ApiError extends Error {
  status: number

  constructor(response: Response, message?: string) {
    super(message || `API error: ${response.status} ${response.statusText}`)
    this.name = 'ApiError'
    this.status = response.status
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new ApiError(response)
  }
  return response.json()
}

/**
 * API client methods
 */
export const api = {
  /**
   * Check API health status
   */
  async health(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE}/api/health`)
    return handleResponse<HealthResponse>(response)
  },

  /**
   * Get list of available document sources
   */
  async sources(): Promise<SourcesResponse> {
    const response = await fetch(`${API_BASE}/api/sources`)
    return handleResponse<SourcesResponse>(response)
  },

  /**
   * Non-streaming chat endpoint
   */
  async chat(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
    return handleResponse<ChatResponse>(response)
  },

  /**
   * Start streaming chat request
   * Returns Response object for SSE processing
   */
  async chatStream(request: ChatRequest, signal?: AbortSignal): Promise<Response> {
    const response = await fetch(`${API_BASE}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    })

    if (!response.ok) {
      throw new ApiError(response)
    }

    return response
  },
}

export { ApiError }
