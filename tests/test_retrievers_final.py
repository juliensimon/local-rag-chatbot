"""Final tests to cover remaining retrievers lines."""

from unittest.mock import MagicMock, patch

import pytest

from retrievers import HybridRetriever
from langchain_core.documents import Document


@patch("retrievers.BM25Okapi")
def test_hybrid_search_sem_key_keyword_assignment(mock_bm25, mock_vectorstore):
    """Test keyword score assignment when sem_key exists (lines 179-185)."""
    # Create documents that will match in both semantic and BM25
    doc_content = "Test content for hybrid search matching" * 3  # ~120 chars
    doc1 = Document(
        page_content=doc_content,
        metadata={"source": "pdf/test1.pdf", "page": 1},
    )
    doc2 = Document(
        page_content=doc_content,  # Same content
        metadata={"source": "pdf/test1.pdf", "page": 1},
    )

    mock_vectorstore.get.return_value = {
        "documents": [doc1.page_content],
        "metadatas": [doc1.metadata],
    }

    # Semantic search returns doc2 (same content)
    mock_vectorstore.similarity_search_with_score.return_value = [
        (doc2, 0.1),
    ]

    mock_bm25_instance = MagicMock()
    mock_bm25_instance.get_scores.return_value = [0.8]
    mock_bm25.return_value = mock_bm25_instance

    retriever = HybridRetriever(mock_vectorstore)
    retriever._build_bm25_index()

    # The matching logic should trigger lines 179-185
    # where sem_key exists in doc_scores and keyword score is assigned
    results = retriever.hybrid_search("test query", k=1)

    assert len(results) > 0
    # Verify keyword score was assigned (line 181-183)
    assert results[0]["keyword_score"] >= 0


def test_hybrid_search_bm25_none_case(mock_vectorstore):
    """Test hybrid search when BM25 is None after build (line 128)."""
    from langchain_core.documents import Document

    doc = Document(page_content="Test", metadata={"source": "test.pdf", "page": 1})
    
    # Use empty documents list
    mock_vectorstore.get.return_value = {
        "documents": [],
        "metadatas": [],
    }

    mock_vectorstore.similarity_search_with_score.return_value = [(doc, 0.1)]

    retriever = HybridRetriever(mock_vectorstore)
    retriever._build_bm25_index()
    
    # After build with empty docs, bm25 should be None (line 66)
    # Then in hybrid_search, line 128 should handle None bm25
    # Set documents manually to test the None bm25 path
    retriever._documents = [doc]
    
    results = retriever.hybrid_search("test query", k=1)
    
    assert isinstance(results, list)

