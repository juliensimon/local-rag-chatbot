/**
 * API types matching backend schemas (api/schemas.py)
 */

/** A single chat message */
export interface Message {
  role: 'user' | 'assistant'
  content: string
}

/** Request body for chat endpoints */
export interface ChatRequest {
  message: string
  history?: Message[]
  rag_enabled?: boolean
  search_type?: 'mmr' | 'similarity' | 'hybrid'
  doc_filter?: string | null
  use_query_rewriting?: boolean
  use_reranking?: boolean
  hybrid_alpha?: number // 0-100
}

/** A retrieved source document */
export interface SourceDocument {
  content: string
  source: string
  page: number
  score?: number | null
  semantic_score?: number | null
  keyword_score?: number | null
  is_top: boolean
}

/** Context information from RAG retrieval */
export interface ContextResponse {
  sources: SourceDocument[]
  rewritten_query?: string | null
}

/** Response for non-streaming chat */
export interface ChatResponse {
  response: string
  context?: ContextResponse | null
}

/** Health check response */
export interface HealthResponse {
  status: string
  vectorstore_ready: boolean
  llm_ready: boolean
}

/** Available document sources */
export interface SourcesResponse {
  sources: string[]
}

/** SSE event types from backend */
export type SSEEventType = 'token' | 'context' | 'done' | 'error'

/** SSE token event data */
export interface SSETokenData {
  content: string
}

/** SSE error event data */
export interface SSEErrorData {
  message: string
}

/** Union type for all SSE event data */
export type SSEEventData = SSETokenData | ContextResponse | SSEErrorData | Record<string, never>

/** Parsed SSE event */
export interface SSEEvent {
  type: SSEEventType
  data: SSEEventData
}

/** Search type options */
export type SearchType = 'mmr' | 'similarity' | 'hybrid'

/** Default values for chat settings */
export const DEFAULT_CHAT_SETTINGS = {
  rag_enabled: false,
  search_type: 'mmr' as SearchType,
  doc_filter: null as string | null,
  use_query_rewriting: false,
  use_reranking: false,
  hybrid_alpha: 70,
} as const
