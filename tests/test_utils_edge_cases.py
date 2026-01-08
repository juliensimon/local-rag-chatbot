"""Additional edge case tests for utils module."""

from langchain_core.documents import Document

from utils import format_context_with_highlight


def test_format_context_score_edge_cases():
    """Test format_context_with_highlight with edge case scores."""
    docs = [
        Document(
            page_content="Test content",
            metadata={"source": "pdf/test.pdf", "page": 1},
        )
    ]

    # Test with score > 1.0 (distance score)
    docs_with_scores = [(docs[0], 1.5)]
    result = format_context_with_highlight(docs, docs_with_scores=docs_with_scores)
    assert "dist:" in result

    # Test with score exactly 1.0
    docs_with_scores = [(docs[0], 1.0)]
    result = format_context_with_highlight(docs, docs_with_scores=docs_with_scores)
    assert "rel:" in result

    # Test with None score
    docs_with_scores = [(docs[0], None)]
    result = format_context_with_highlight(docs, docs_with_scores=docs_with_scores)
    assert "⭐" in result  # Should still highlight first chunk


def test_format_context_no_scores():
    """Test format_context_with_highlight with no scores."""
    docs = [
        Document(
            page_content="Test content",
            metadata={"source": "pdf/test.pdf", "page": 1},
        )
    ]

    result = format_context_with_highlight(docs, docs_with_scores=None)
    assert "Test content" in result
    assert "⭐" in result  # First chunk should be highlighted by default



