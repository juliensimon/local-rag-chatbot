/** TypeScript types for explorer API responses */

export interface CollectionStats {
  total_chunks: number
  unique_sources: number
  avg_chunk_size: number
  min_chunk_size: number
  max_chunk_size: number
  embedding_dimensions: number
  db_size_bytes: number | null
  chunk_size_distribution: Record<string, number>
}

export interface SourceStat {
  source: string
  filename: string
  chunk_count: number
  total_pages: number
  pages_with_chunks: number
  page_coverage: number
  avg_chunk_size: number
  min_chunk_size: number
  max_chunk_size: number
}

export interface SourceStatsResponse {
  sources: SourceStat[]
}

export interface QualityMetrics {
  chunk_length_distribution: Record<string, number>
  outliers_short: number
  outliers_long: number
  overlap_analysis: {
    sources_with_overlap?: number
    avg_overlap_ratio?: number
  }
  metadata_completeness: Record<string, number>
  duplicate_chunks: number
  low_information_chunks: number
}

export interface EmbeddingQuality {
  similarity_distribution: Record<string, number>
  avg_similarity: number | null
  min_similarity: number | null
  max_similarity: number | null
}

export interface StatsResponse {
  collection: CollectionStats
  sources: SourceStatsResponse
  quality: QualityMetrics
  embedding: EmbeddingQuality | null
}

export interface Recommendation {
  category: "chunking" | "retrieval" | "metadata" | "embedding" | "content"
  severity: "low" | "medium" | "high"
  title: string
  description: string
  action_items: string[]
  metrics: Record<string, any>
}

export interface RecommendationsResponse {
  recommendations: Recommendation[]
}

export interface DocumentScore {
  content: string
  source: string
  page: number
  score: number | null
  semantic_score: number | null
  keyword_score: number | null
  doc_id: string
}

export interface SearchRequest {
  query: string
  search_type: "mmr" | "similarity" | "hybrid"
  k: number
  hybrid_alpha: number
  source_filter?: string | null
  page_filter?: { min?: number; max?: number } | null
}

export interface SearchResponse {
  results: DocumentScore[]
  query: string
  search_type: string
  total_results: number
}

export interface DocumentResponse {
  doc_id: string
  content: string
  source: string
  page: number
  chunk_size: number
  metadata: Record<string, any>
}

export interface DocumentsResponse {
  documents: DocumentResponse[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface SimilarityResponse {
  doc_id: string
  similar_docs: DocumentScore[]
}

export interface HealthResponse {
  status: string
  vectorstore_ready: boolean
}
