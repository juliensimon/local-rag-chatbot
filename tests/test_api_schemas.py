"""Tests for API schema validation."""

import pytest
from pydantic import ValidationError

from api.schemas import (
    ChatRequest,
    ChatResponse,
    ContextResponse,
    HealthResponse,
    Message,
    SourceDocument,
    SourcesResponse,
)


class TestMessage:
    """Tests for Message schema."""

    def test_valid_user_message(self):
        """Test creating a valid user message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_valid_assistant_message(self):
        """Test creating a valid assistant message."""
        msg = Message(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_invalid_role(self):
        """Test that invalid roles are rejected."""
        with pytest.raises(ValidationError):
            Message(role="system", content="Test")


class TestChatRequest:
    """Tests for ChatRequest schema."""

    def test_minimal_request(self):
        """Test creating request with only required fields."""
        request = ChatRequest(message="What is RAG?")
        assert request.message == "What is RAG?"
        assert request.history == []
        assert request.rag_enabled is False
        assert request.search_type == "mmr"
        assert request.doc_filter is None
        assert request.use_query_rewriting is False
        assert request.use_reranking is False
        assert request.hybrid_alpha == 70

    def test_full_request(self):
        """Test creating request with all fields."""
        request = ChatRequest(
            message="Test question",
            history=[
                Message(role="user", content="Previous"),
                Message(role="assistant", content="Response"),
            ],
            rag_enabled=True,
            search_type="hybrid",
            doc_filter="test.pdf",
            use_query_rewriting=True,
            use_reranking=True,
            hybrid_alpha=50,
        )
        assert request.rag_enabled is True
        assert request.search_type == "hybrid"
        assert len(request.history) == 2

    def test_empty_message_rejected(self):
        """Test that empty messages are rejected."""
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_message_max_length(self):
        """Test that overly long messages are rejected."""
        with pytest.raises(ValidationError):
            ChatRequest(message="x" * 10001)

    def test_invalid_search_type(self):
        """Test that invalid search types are rejected."""
        with pytest.raises(ValidationError):
            ChatRequest(message="Test", search_type="invalid")

    def test_hybrid_alpha_bounds(self):
        """Test hybrid alpha must be 0-100."""
        # Valid bounds
        ChatRequest(message="Test", hybrid_alpha=0)
        ChatRequest(message="Test", hybrid_alpha=100)

        # Invalid bounds
        with pytest.raises(ValidationError):
            ChatRequest(message="Test", hybrid_alpha=-1)
        with pytest.raises(ValidationError):
            ChatRequest(message="Test", hybrid_alpha=101)


class TestSourceDocument:
    """Tests for SourceDocument schema."""

    def test_minimal_source(self):
        """Test creating source with required fields."""
        source = SourceDocument(
            content="Test content",
            source="test.pdf",
            page=1,
        )
        assert source.content == "Test content"
        assert source.source == "test.pdf"
        assert source.page == 1
        assert source.score is None
        assert source.is_top is False

    def test_full_source(self):
        """Test creating source with all fields."""
        source = SourceDocument(
            content="Test content",
            source="test.pdf",
            page=5,
            score=0.95,
            semantic_score=0.9,
            keyword_score=0.85,
            is_top=True,
        )
        assert source.score == 0.95
        assert source.semantic_score == 0.9
        assert source.keyword_score == 0.85
        assert source.is_top is True


class TestContextResponse:
    """Tests for ContextResponse schema."""

    def test_empty_context(self):
        """Test creating empty context."""
        ctx = ContextResponse(sources=[])
        assert ctx.sources == []
        assert ctx.rewritten_query is None

    def test_context_with_sources(self):
        """Test creating context with sources."""
        sources = [
            SourceDocument(content="Test", source="test.pdf", page=1),
        ]
        ctx = ContextResponse(sources=sources, rewritten_query="rewritten query")
        assert len(ctx.sources) == 1
        assert ctx.rewritten_query == "rewritten query"


class TestChatResponse:
    """Tests for ChatResponse schema."""

    def test_vanilla_response(self):
        """Test vanilla LLM response without context."""
        resp = ChatResponse(response="Hello!", context=None)
        assert resp.response == "Hello!"
        assert resp.context is None

    def test_rag_response(self):
        """Test RAG response with context."""
        ctx = ContextResponse(sources=[])
        resp = ChatResponse(response="Based on the documents...", context=ctx)
        assert resp.context is not None


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_healthy_status(self):
        """Test healthy status."""
        resp = HealthResponse(
            status="healthy",
            vectorstore_ready=True,
            llm_ready=True,
        )
        assert resp.status == "healthy"

    def test_degraded_status(self):
        """Test degraded status."""
        resp = HealthResponse(
            status="degraded",
            vectorstore_ready=False,
            llm_ready=True,
        )
        assert resp.status == "degraded"


class TestSourcesResponse:
    """Tests for SourcesResponse schema."""

    def test_empty_sources(self):
        """Test empty sources list."""
        resp = SourcesResponse(sources=[])
        assert resp.sources == []

    def test_sources_list(self):
        """Test sources list."""
        resp = SourcesResponse(sources=["doc1.pdf", "doc2.pdf"])
        assert len(resp.sources) == 2
