"""Pydantic models for explorer API requests and responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CollectionStats(BaseModel):
    """Collection-level statistics."""

    total_chunks: int
    unique_sources: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    embedding_dimensions: int
    db_size_bytes: Optional[int] = None
    chunk_size_distribution: Dict[str, int] = Field(default_factory=dict)


class SourceStat(BaseModel):
    """Statistics for a single source file."""

    source: str
    filename: str
    chunk_count: int
    total_pages: int
    pages_with_chunks: int
    page_coverage: float
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int


class SourceStatsResponse(BaseModel):
    """Source-level statistics response."""

    sources: List[SourceStat]


class QualityMetrics(BaseModel):
    """Quality metrics for chunks."""

    chunk_length_distribution: Dict[str, int] = Field(default_factory=dict)
    outliers_short: int = 0  # Chunks < 50 chars
    outliers_long: int = 0  # Chunks > 2000 chars
    overlap_analysis: Dict[str, Any] = Field(default_factory=dict)
    metadata_completeness: Dict[str, float] = Field(default_factory=dict)
    duplicate_chunks: int = 0
    low_information_chunks: int = 0


class EmbeddingQuality(BaseModel):
    """Embedding quality metrics."""

    similarity_distribution: Dict[str, int] = Field(default_factory=dict)
    avg_similarity: Optional[float] = None
    min_similarity: Optional[float] = None
    max_similarity: Optional[float] = None


class StatsResponse(BaseModel):
    """Complete statistics response."""

    collection: CollectionStats
    sources: SourceStatsResponse
    quality: QualityMetrics
    embedding: Optional[EmbeddingQuality] = None


class Recommendation(BaseModel):
    """A single optimization recommendation."""

    category: str  # "chunking", "retrieval", "metadata", "embedding", "content"
    severity: str  # "low", "medium", "high"
    title: str
    description: str
    action_items: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class RecommendationsResponse(BaseModel):
    """Optimization recommendations response."""

    recommendations: List[Recommendation]


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., min_length=1, max_length=10000)
    search_type: str = Field(default="mmr", pattern="^(mmr|similarity|hybrid)$")
    k: int = Field(default=5, ge=1, le=50)
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    source_filter: Optional[str] = None
    page_filter: Optional[Dict[str, int]] = None  # {"min": 1, "max": 10}


class DocumentScore(BaseModel):
    """Document with score information."""

    content: str
    source: str
    page: int
    score: Optional[float] = None
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    doc_id: str


class SearchResponse(BaseModel):
    """Search results response."""

    results: List[DocumentScore]
    query: str
    search_type: str
    total_results: int


class DocumentResponse(BaseModel):
    """Single document chunk response."""

    doc_id: str
    content: str
    source: str
    page: int
    chunk_size: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SimilarityResponse(BaseModel):
    """Similar documents response."""

    doc_id: str
    similar_docs: List[DocumentScore]


class DocumentsResponse(BaseModel):
    """Paginated documents response."""

    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    vectorstore_ready: bool
