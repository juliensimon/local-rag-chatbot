"""Tests for retrievers module."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.documents import Document

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


# Tests merged from test_retrievers_edge_cases.py


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
    doc = Document(page_content="Doc 1", metadata={"source": "test.pdf", "page": 1})
    mock_vectorstore.get.return_value = {
        "documents": ["Doc 1"],
        "metadatas": [{"source": "test.pdf", "page": 1}],
    }

    # Negative distance should be handled
    mock_vectorstore.similarity_search_with_score.return_value = [(doc, -0.1)]

    with patch("retrievers.BM25Okapi") as mock_bm25:
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.get_scores.return_value = [0.5]
        mock_bm25.return_value = mock_bm25_instance

        retriever = HybridRetriever(mock_vectorstore)
        results = retriever.hybrid_search("test query", k=1)

        assert len(results) > 0


# Tests merged from test_retrievers_remaining.py


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
    doc = Document(page_content="Doc 1", metadata={"source": "test.pdf", "page": 1})
    mock_vectorstore.get.return_value = {
        "documents": ["Doc 1"],
        "metadatas": [{"source": "test.pdf", "page": 1}],
    }

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


# Tests merged from test_retrievers_final.py


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



