"""Pydantic models for API request/response validation."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single chat message."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for chat endpoints."""

    message: str = Field(..., min_length=1, max_length=10000)
    history: list[Message] = Field(default_factory=list)
    rag_enabled: bool = False
    search_type: Literal["mmr", "similarity", "hybrid"] = "mmr"
    doc_filter: Optional[str] = None
    use_query_rewriting: bool = False
    use_reranking: bool = False
    hybrid_alpha: int = Field(default=70, ge=0, le=100)


class SourceDocument(BaseModel):
    """A retrieved source document."""

    content: str
    source: str
    page: int
    score: Optional[float] = None
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    is_top: bool = False


class ContextResponse(BaseModel):
    """Context information from RAG retrieval."""

    sources: list[SourceDocument]
    rewritten_query: Optional[str] = None


class ChatResponse(BaseModel):
    """Response for non-streaming chat."""

    response: str
    context: Optional[ContextResponse] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    vectorstore_ready: bool
    llm_ready: bool


class SourcesResponse(BaseModel):
    """Available document sources."""

    sources: list[str]
