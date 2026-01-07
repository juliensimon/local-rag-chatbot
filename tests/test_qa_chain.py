"""Tests for qa_chain module."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.documents import Document

from qa_chain import QAChainWrapper, create_qa_chain


@pytest.fixture
def mock_prompt():
    """Create a mock prompt template."""
    return MagicMock()


@pytest.fixture
def qa_chain_wrapper(mock_vectorstore, mock_prompt):
    """Create a QAChainWrapper instance."""
    return QAChainWrapper(mock_vectorstore, mock_prompt)


def test_qa_chain_wrapper_init(mock_vectorstore, mock_prompt):
    """Test QAChainWrapper initialization."""
    wrapper = QAChainWrapper(mock_vectorstore, mock_prompt)
    assert wrapper._vectorstore == mock_vectorstore
    assert wrapper._prompt == mock_prompt
    assert wrapper._hybrid_retriever is not None
    assert wrapper._reranker is None


def test_qa_chain_wrapper_retriever_property(qa_chain_wrapper):
    """Test retriever property."""
    retriever = qa_chain_wrapper.retriever
    assert retriever is not None


@patch("qa_chain.CrossEncoder")
def test_get_reranker_success(mock_cross_encoder, qa_chain_wrapper):
    """Test successful reranker loading."""
    mock_reranker = MagicMock()
    mock_cross_encoder.return_value = mock_reranker

    result = qa_chain_wrapper._get_reranker()
    assert result == mock_reranker
    assert qa_chain_wrapper._reranker == mock_reranker


@patch("qa_chain.CrossEncoder")
def test_get_reranker_failure(mock_cross_encoder, qa_chain_wrapper):
    """Test reranker loading failure."""
    mock_cross_encoder.side_effect = Exception("Load error")

    result = qa_chain_wrapper._get_reranker()
    assert result is None
    assert qa_chain_wrapper._reranker is None


@patch("qa_chain.create_llm")
def test_rewrite_query_success(mock_create_llm, qa_chain_wrapper):
    """Test successful query rewriting."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="rewritten query")
    mock_create_llm.return_value = mock_llm

    result = qa_chain_wrapper.rewrite_query("original question")
    assert result == "rewritten query"


@patch("qa_chain.create_llm")
def test_rewrite_query_too_short(mock_create_llm, qa_chain_wrapper):
    """Test query rewriting that returns too short result."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="x")  # Too short
    mock_create_llm.return_value = mock_llm

    result = qa_chain_wrapper.rewrite_query("original question that is long")
    assert result == "original question that is long"  # Should return original


@patch("qa_chain.create_llm")
def test_rewrite_query_failure(mock_create_llm, qa_chain_wrapper):
    """Test query rewriting failure."""
    mock_create_llm.side_effect = Exception("Error")

    result = qa_chain_wrapper.rewrite_query("original question")
    assert result == "original question"  # Should return original on error


@patch("qa_chain.create_llm")
def test_rewrite_query_same(mock_create_llm, qa_chain_wrapper):
    """Test query rewriting that returns same as original."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="original question")
    mock_create_llm.return_value = mock_llm

    result = qa_chain_wrapper.rewrite_query("original question")
    assert result == "original question"


def test_rerank_documents_empty(qa_chain_wrapper):
    """Test reranking empty document list."""
    result = qa_chain_wrapper.rerank_documents("query", [])
    assert result == []


@patch("qa_chain.CrossEncoder")
def test_rerank_documents_success(mock_cross_encoder, qa_chain_wrapper, sample_documents):
    """Test successful document reranking."""
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.9, 0.8, 0.7]
    mock_cross_encoder.return_value = mock_reranker

    result = qa_chain_wrapper.rerank_documents("query", sample_documents[:3], top_k=2)
    assert len(result) == 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in result)


@patch("qa_chain.CrossEncoder")
def test_rerank_documents_failure(mock_cross_encoder, qa_chain_wrapper, sample_documents):
    """Test reranking failure."""
    mock_reranker = MagicMock()
    mock_reranker.predict.side_effect = Exception("Error")
    mock_cross_encoder.return_value = mock_reranker

    result = qa_chain_wrapper.rerank_documents("query", sample_documents[:2], top_k=2)
    assert len(result) == 2
    assert all(r[1] is None for r in result)  # Scores should be None


def test_get_retriever_with_filter_mmr(qa_chain_wrapper):
    """Test getting retriever with MMR search type."""
    retriever = qa_chain_wrapper.get_retriever_with_filter(search_type="mmr")
    assert retriever is not None


def test_get_retriever_with_filter_similarity(qa_chain_wrapper):
    """Test getting retriever with similarity search type."""
    retriever = qa_chain_wrapper.get_retriever_with_filter(search_type="similarity")
    assert retriever is not None


def test_get_retriever_with_filter_metadata(qa_chain_wrapper):
    """Test getting retriever with metadata filter."""
    metadata_filter = {"source": {"$eq": "test.pdf"}}
    retriever = qa_chain_wrapper.get_retriever_with_filter(
        metadata_filter=metadata_filter
    )
    assert retriever is not None


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_mmr(mock_format_history, mock_create_llm, qa_chain_wrapper, mock_prompt):
    """Test streaming with MMR search."""
    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_llm.stream.return_value = [
        MagicMock(content="Chunk "),
        MagicMock(content="1"),
    ]
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="Test", metadata={"source": "test.pdf", "page": 1})
    ]
    qa_chain_wrapper._retriever = mock_retriever

    # Mock the chain operator
    mock_chain = MagicMock()
    mock_chain.stream.return_value = [
        MagicMock(content="Chunk "),
        MagicMock(content="1"),
    ]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question",
        "chat_history": [],
        "search_type": "mmr",
    }

    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0
    assert all("chunk" in r for r in results)


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_hybrid(mock_format_history, mock_create_llm, qa_chain_wrapper, mock_hybrid_results, mock_prompt):
    """Test streaming with hybrid search."""
    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_llm.stream.return_value = [MagicMock(content="Response")]
    mock_create_llm.return_value = mock_llm

    qa_chain_wrapper._hybrid_retriever.hybrid_search = MagicMock(
        return_value=mock_hybrid_results
    )

    # Mock the chain operator
    mock_chain = MagicMock()
    mock_chain.stream.return_value = [MagicMock(content="Response")]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question",
        "chat_history": [],
        "search_type": "hybrid",
    }

    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_error(mock_format_history, mock_create_llm, qa_chain_wrapper, mock_prompt):
    """Test streaming error handling."""
    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="Test", metadata={"source": "test.pdf", "page": 1})
    ]
    qa_chain_wrapper._retriever = mock_retriever

    # Mock the chain operator to raise an error
    mock_chain = MagicMock()
    mock_chain.stream.side_effect = Exception("Stream error")
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question",
        "chat_history": [],
    }

    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0
    assert "Error" in results[0]["chunk"]


def test_create_qa_chain(mock_vectorstore):
    """Test creating QA chain."""
    chain = create_qa_chain(mock_vectorstore)
    assert isinstance(chain, QAChainWrapper)

