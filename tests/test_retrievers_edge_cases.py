"""Additional edge case tests for retrievers module."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from retrievers import HybridRetriever


def test_hybrid_retriever_build_bm25_index_no_metadata(mock_vectorstore):
    """Test BM25 index building with missing metadata."""
    mock_vectorstore.get.return_value = {
        "documents": ["Doc 1", "Doc 2"],
        "metadatas": [{"source": "test.pdf"}, None],  # One None metadata
    }

    retriever = HybridRetriever(mock_vectorstore)
    retriever._build_bm25_index()

    assert retriever._bm25 is not None
    assert len(retriever._documents) == 2


def test_hybrid_search_normalize_scores_edge_cases(mock_vectorstore):
    """Test hybrid search score normalization edge cases."""
    from langchain_core.documents import Document

    mock_vectorstore.get.return_value = {
        "documents": ["Doc 1", "Doc 2"],
        "metadatas": [
            {"source": "test1.pdf", "page": 1},
            {"source": "test2.pdf", "page": 2},
        ],
    }

    mock_vectorstore.similarity_search_with_score.return_value = [
        (Document(page_content="Doc 1", metadata={"source": "test1.pdf", "page": 1}), 0.1),
    ]

    with patch("retrievers.BM25Okapi") as mock_bm25:
        mock_bm25_instance = MagicMock()
        # Test case: all scores are the same
        mock_bm25_instance.get_scores.return_value = [0.5, 0.5]
        mock_bm25.return_value = mock_bm25_instance

        retriever = HybridRetriever(mock_vectorstore)
        results = retriever.hybrid_search("test query", k=2)

        assert isinstance(results, list)

        # Test case: all scores are zero
        mock_bm25_instance.get_scores.return_value = [0.0, 0.0]
        results = retriever.hybrid_search("test query", k=2)
        assert isinstance(results, list)


def test_hybrid_search_negative_distance(mock_vectorstore):
    """Test hybrid search with negative distance (edge case)."""
    from langchain_core.documents import Document

    mock_vectorstore.get.return_value = {
        "documents": ["Doc 1"],
        "metadatas": [{"source": "test.pdf", "page": 1}],
    }

    doc = Document(page_content="Doc 1", metadata={"source": "test.pdf", "page": 1})
    # Negative distance should be handled
    mock_vectorstore.similarity_search_with_score.return_value = [(doc, -0.1)]

    with patch("retrievers.BM25Okapi") as mock_bm25:
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.get_scores.return_value = [0.5]
        mock_bm25.return_value = mock_bm25_instance

        retriever = HybridRetriever(mock_vectorstore)
        results = retriever.hybrid_search("test query", k=1)

        assert len(results) > 0

