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
    # _reranker is set to sentinel _RERANKER_LOAD_FAILED, not None
    assert qa_chain_wrapper._reranker is not None


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
    assert "error occurred" in results[0]["chunk"]


def test_create_qa_chain(mock_vectorstore):
    """Test creating QA chain."""
    chain = create_qa_chain(mock_vectorstore)
    assert isinstance(chain, QAChainWrapper)


# Tests merged from test_qa_chain_edge_cases.py


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_similarity_search(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test streaming with similarity search type."""
    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    # Create documents with matching content
    doc_content = "Test content for matching"
    mock_doc = Document(page_content=doc_content, metadata={"page": 1, "source": "test.pdf"})
    mock_scored_doc = Document(page_content=doc_content, metadata={"page": 1, "source": "test.pdf"})

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]
    qa_chain_wrapper._retriever = mock_retriever

    mock_vectorstore = qa_chain_wrapper._vectorstore
    mock_vectorstore.similarity_search_with_score.return_value = [
        (mock_scored_doc, 0.5)
    ]

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question",
        "chat_history": [],
        "search_type": "similarity",
    }

    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_with_reranking(mock_format_history, mock_create_llm, qa_chain_wrapper, sample_documents):
    """Test streaming with reranking enabled."""
    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = sample_documents[:3]
    qa_chain_wrapper._retriever = mock_retriever

    # Mock reranker
    with patch("qa_chain.CrossEncoder") as mock_cross_encoder:
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.8, 0.7]
        mock_cross_encoder.return_value = mock_reranker
        qa_chain_wrapper._reranker = mock_reranker

        mock_chain = MagicMock()
        mock_chain.stream.return_value = [mock_chunk]
        qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

        inputs = {
            "question": "test question",
            "chat_history": [],
            "search_type": "mmr",
            "use_reranking": True,
        }

        results = list(qa_chain_wrapper.stream(inputs))
        assert len(results) > 0


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_with_query_rewriting(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test streaming with query rewriting enabled."""
    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_rewrite_response = MagicMock()
    mock_rewrite_response.content = "rewritten query"
    mock_llm.invoke.return_value = mock_rewrite_response
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="Test", metadata={"source": "test.pdf", "page": 1})
    ]
    qa_chain_wrapper._retriever = mock_retriever

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question that is long enough",
        "chat_history": [],
        "use_query_rewriting": True,
    }

    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0
    assert results[0].get("rewritten_query") == "rewritten query"


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_similarity_search_error(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test streaming with similarity search error."""
    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    mock_doc = Document(page_content="Test", metadata={"page": 1, "source": "test.pdf"})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]
    qa_chain_wrapper._retriever = mock_retriever

    mock_vectorstore = qa_chain_wrapper._vectorstore
    mock_vectorstore.similarity_search_with_score.side_effect = Exception("Error")

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question",
        "chat_history": [],
        "search_type": "similarity",
    }

    results = list(qa_chain_wrapper.stream(inputs))
    # Should still work even if similarity_search_with_score fails
    assert len(results) > 0


# Tests merged from test_qa_chain_remaining.py


def test_rerank_documents_no_reranker(qa_chain_wrapper, sample_documents):
    """Test rerank_documents when reranker is None (line 115)."""
    # Set reranker to None and ensure _get_reranker returns None
    qa_chain_wrapper._reranker = None
    with patch.object(qa_chain_wrapper, '_get_reranker', return_value=None):
        result = qa_chain_wrapper.rerank_documents("query", sample_documents[:3], top_k=2)
        assert len(result) == 2
        assert all(r[1] is None for r in result)  # All scores should be None


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_reranking_similarity(mock_format_history, mock_create_llm, qa_chain_wrapper, sample_documents):
    """Test streaming with reranking and similarity search (line 247)."""
    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    # Mock retriever for similarity search with reranking
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = sample_documents[:3]
    qa_chain_wrapper._vectorstore.as_retriever.return_value = mock_retriever

    # Mock reranker
    with patch("qa_chain.CrossEncoder") as mock_cross_encoder:
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.8, 0.7]
        mock_cross_encoder.return_value = mock_reranker
        qa_chain_wrapper._reranker = mock_reranker

        mock_chain = MagicMock()
        mock_chain.stream.return_value = [mock_chunk]
        qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

        inputs = {
            "question": "test question",
            "chat_history": [],
            "search_type": "similarity",
            "use_reranking": True,
        }

        results = list(qa_chain_wrapper.stream(inputs))
        assert len(results) > 0


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_similarity_doc_matching(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test similarity search with document matching (lines 268-276)."""
    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    # Create documents with matching content (first 100 chars must match)
    doc_content = "A" * 50 + "B" * 50 + "C" * 50  # 150 chars total
    doc1 = Document(page_content=doc_content, metadata={"page": 1, "source": "test.pdf"})
    doc2 = Document(page_content=doc_content, metadata={"page": 1, "source": "test.pdf"})

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc1]
    qa_chain_wrapper._retriever = mock_retriever

    mock_vectorstore = qa_chain_wrapper._vectorstore
    mock_vectorstore.similarity_search_with_score.return_value = [
        (doc2, 0.5)  # Same content, should match (lines 269-274)
    ]

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question",
        "chat_history": [],
        "search_type": "similarity",
    }

    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0
    # Verify that scores were matched
    assert results[0].get("docs_with_scores") is not None

    # Test case where doc doesn't match (else clause line 276)
    doc3 = Document(
        page_content="X" * 150,  # Different content
        metadata={"page": 2, "source": "other.pdf"}
    )
    mock_retriever.invoke.return_value = [doc3]
    mock_vectorstore.similarity_search_with_score.return_value = [
        (doc2, 0.5)  # Different content, shouldn't match
    ]

    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0


# Tests merged from test_qa_chain_final.py


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_similarity_exact_match(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test similarity search with exact document matching (lines 269-274)."""
    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    # Create documents where first 100 chars match exactly
    matching_prefix = "A" * 100
    doc1 = Document(
        page_content=matching_prefix + " rest of content 1",
        metadata={"page": 1, "source": "test.pdf"},
    )
    doc2 = Document(
        page_content=matching_prefix + " rest of content 2",  # Same first 100 chars
        metadata={"page": 1, "source": "test.pdf"},
    )

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc1]
    qa_chain_wrapper._retriever = mock_retriever

    mock_vectorstore = qa_chain_wrapper._vectorstore
    # Return doc2 with score - should match doc1 based on first 100 chars and page
    mock_vectorstore.similarity_search_with_score.return_value = [
        (doc2, 0.5)
    ]

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question",
        "chat_history": [],
        "search_type": "similarity",
    }

    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0
    # Verify that the matching logic was executed (lines 269-274)
    assert results[0].get("docs_with_scores") is not None

