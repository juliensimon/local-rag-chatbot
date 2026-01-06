"""Tests for retrievers module."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from retrievers import HybridRetriever


def test_hybrid_retriever_init(mock_vectorstore):
    """Test HybridRetriever initialization."""
    retriever = HybridRetriever(mock_vectorstore)
    assert retriever._vectorstore == mock_vectorstore
    assert retriever._bm25 is None
    assert retriever._documents is None


def test_hybrid_retriever_tokenize():
    """Test tokenization."""
    retriever = HybridRetriever(MagicMock())
    tokens = retriever._tokenize("Hello World Test")
    assert tokens == ["hello", "world", "test"]


def test_hybrid_retriever_tokenize_empty():
    """Test tokenization of empty string."""
    retriever = HybridRetriever(MagicMock())
    tokens = retriever._tokenize("")
    assert tokens == []


def test_hybrid_retriever_build_bm25_index(mock_vectorstore):
    """Test BM25 index building."""
    retriever = HybridRetriever(mock_vectorstore)
    retriever._build_bm25_index()
    assert retriever._bm25 is not None
    assert len(retriever._documents) == 2


def test_hybrid_retriever_build_bm25_index_empty(mock_vectorstore):
    """Test BM25 index building with empty collection."""
    mock_vectorstore.get.return_value = {"documents": [], "metadatas": []}
    retriever = HybridRetriever(mock_vectorstore)
    retriever._build_bm25_index()
    assert retriever._bm25 is None
    assert retriever._documents == []


def test_hybrid_retriever_build_bm25_index_no_documents(mock_vectorstore):
    """Test BM25 index building when collection is None."""
    mock_vectorstore.get.return_value = None
    retriever = HybridRetriever(mock_vectorstore)
    retriever._build_bm25_index()
    assert retriever._bm25 is None


def test_hybrid_retriever_matches_filter_no_filter():
    """Test filter matching with no filter."""
    retriever = HybridRetriever(MagicMock())
    doc = Mock(metadata={"source": "test.pdf"})
    assert retriever._matches_filter(doc, None) is True


def test_hybrid_retriever_matches_filter_eq():
    """Test filter matching with $eq operator."""
    retriever = HybridRetriever(MagicMock())
    doc = Mock(metadata={"source": "test.pdf"})
    filter_dict = {"source": {"$eq": "test.pdf"}}
    assert retriever._matches_filter(doc, filter_dict) is True

    doc2 = Mock(metadata={"source": "other.pdf"})
    assert retriever._matches_filter(doc2, filter_dict) is False


def test_hybrid_retriever_matches_filter_in():
    """Test filter matching with $in operator."""
    retriever = HybridRetriever(MagicMock())
    doc = Mock(metadata={"source": "test.pdf"})
    filter_dict = {"source": {"$in": ["test.pdf", "other.pdf"]}}
    assert retriever._matches_filter(doc, filter_dict) is True

    doc2 = Mock(metadata={"source": "notin.pdf"})
    assert retriever._matches_filter(doc2, filter_dict) is False


def test_hybrid_retriever_matches_filter_missing_key():
    """Test filter matching with missing metadata key."""
    retriever = HybridRetriever(MagicMock())
    doc = Mock(metadata={})
    filter_dict = {"source": {"$eq": "test.pdf"}}
    assert retriever._matches_filter(doc, filter_dict) is False


@patch("retrievers.BM25Okapi")
def test_hybrid_search(mock_bm25, mock_vectorstore, sample_documents):
    """Test hybrid search."""
    # Setup mocks
    mock_vectorstore.get.return_value = {
        "documents": [doc.page_content for doc in sample_documents[:2]],
        "metadatas": [doc.metadata for doc in sample_documents[:2]],
    }
    mock_vectorstore.similarity_search_with_score.return_value = [
        (sample_documents[0], 0.1),
        (sample_documents[1], 0.2),
    ]

    mock_bm25_instance = MagicMock()
    mock_bm25_instance.get_scores.return_value = [0.5, 0.3]
    mock_bm25.return_value = mock_bm25_instance

    retriever = HybridRetriever(mock_vectorstore)
    results = retriever.hybrid_search("test query", k=2)

    assert len(results) <= 2
    assert all("doc" in r for r in results)
    assert all("fused_score" in r for r in results)


def test_hybrid_search_empty_documents(mock_vectorstore):
    """Test hybrid search with no documents."""
    mock_vectorstore.get.return_value = {"documents": [], "metadatas": []}
    retriever = HybridRetriever(mock_vectorstore)
    results = retriever.hybrid_search("test query")
    assert results == []


def test_hybrid_search_semantic_error(mock_vectorstore, sample_documents):
    """Test hybrid search when semantic search fails."""
    mock_vectorstore.get.return_value = {
        "documents": [doc.page_content for doc in sample_documents[:2]],
        "metadatas": [doc.metadata for doc in sample_documents[:2]],
    }
    mock_vectorstore.similarity_search_with_score.side_effect = Exception("Error")

    retriever = HybridRetriever(mock_vectorstore)
    results = retriever.hybrid_search("test query")
    # Should still return results from BM25
    assert isinstance(results, list)

