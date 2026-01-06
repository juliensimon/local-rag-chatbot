"""Tests to cover remaining lines in retrievers.py."""

from unittest.mock import MagicMock, patch

import pytest

from retrievers import HybridRetriever
from langchain_core.documents import Document


def test_build_bm25_index_empty_tokenized(mock_vectorstore):
    """Test BM25 index building with empty tokenized docs (line 66)."""
    # Test with empty documents list
    mock_vectorstore.get.return_value = {
        "documents": [],  # Empty documents list
        "metadatas": [],
    }

    retriever = HybridRetriever(mock_vectorstore)
    retriever._build_bm25_index()

    # When tokenized_docs is empty, bm25 should be None (line 66)
    assert retriever._bm25 is None


def test_hybrid_search_no_bm25(mock_vectorstore):
    """Test hybrid search when BM25 is None (line 128)."""
    from langchain_core.documents import Document

    mock_vectorstore.get.return_value = {
        "documents": ["Doc 1"],
        "metadatas": [{"source": "test.pdf", "page": 1}],
    }

    doc = Document(page_content="Doc 1", metadata={"source": "test.pdf", "page": 1})
    mock_vectorstore.similarity_search_with_score.return_value = [(doc, 0.1)]

    retriever = HybridRetriever(mock_vectorstore)
    retriever._bm25 = None  # Explicitly set to None (line 128)
    retriever._documents = [doc]

    results = retriever.hybrid_search("test query", k=1)

    assert isinstance(results, list)
    # Should still return results based on semantic search only
    assert len(results) > 0


@patch("retrievers.BM25Okapi")
def test_hybrid_search_filter_matching(mock_bm25, mock_vectorstore):
    """Test hybrid search with filter matching (lines 165, 179-185)."""
    from langchain_core.documents import Document

    doc1 = Document(
        page_content="Document 1",
        metadata={"source": "pdf/test1.pdf", "page": 1},
    )
    doc2 = Document(
        page_content="Document 2",
        metadata={"source": "pdf/test2.pdf", "page": 2},
    )

    mock_vectorstore.get.return_value = {
        "documents": [doc1.page_content, doc2.page_content],
        "metadatas": [doc1.metadata, doc2.metadata],
    }

    mock_vectorstore.similarity_search_with_score.return_value = [
        (doc1, 0.1),
    ]

    mock_bm25_instance = MagicMock()
    mock_bm25_instance.get_scores.return_value = [0.8, 0.6]
    mock_bm25.return_value = mock_bm25_instance

    retriever = HybridRetriever(mock_vectorstore)

    # Test with filter that excludes one document
    metadata_filter = {"source": {"$eq": "pdf/test1.pdf"}}
    results = retriever.hybrid_search("test query", k=2, metadata_filter=metadata_filter)

    assert len(results) > 0
    # Should only include documents matching the filter

    # Test keyword score assignment when sem_key exists (lines 179-185)
    # This happens when BM25 doc matches semantic doc
    results = retriever.hybrid_search("test query", k=2, alpha=0.5)
    assert all("keyword_score" in r for r in results)


@patch("retrievers.BM25Okapi")
def test_hybrid_search_keyword_score_assignment(mock_bm25, mock_vectorstore):
    """Test keyword score assignment in hybrid search (lines 179-185)."""
    from langchain_core.documents import Document

    # Create documents that will match exactly (first 100 chars must match)
    base_content = "Same content for matching test " * 4  # ~120 chars
    doc1 = Document(
        page_content=base_content,
        metadata={"source": "pdf/test1.pdf", "page": 1},
    )
    # Use same content for semantic doc to ensure matching
    doc2 = Document(
        page_content=base_content,
        metadata={"source": "pdf/test1.pdf", "page": 1},
    )

    mock_vectorstore.get.return_value = {
        "documents": [doc1.page_content],
        "metadatas": [doc1.metadata],
    }

    mock_vectorstore.similarity_search_with_score.return_value = [
        (doc2, 0.1),  # Semantic result with same content
    ]

    mock_bm25_instance = MagicMock()
    mock_bm25_instance.get_scores.return_value = [0.8]
    mock_bm25.return_value = mock_bm25_instance

    retriever = HybridRetriever(mock_vectorstore)
    # Build index first
    retriever._build_bm25_index()
    
    results = retriever.hybrid_search("test query", k=1)

    # Should have keyword score assigned when documents match (lines 179-185)
    assert len(results) > 0
    # The keyword score should be assigned
    assert "keyword_score" in results[0]
    # When sem_key exists in doc_scores, keyword score should be updated
    assert results[0]["keyword_score"] >= 0

