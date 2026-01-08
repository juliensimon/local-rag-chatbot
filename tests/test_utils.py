"""Tests for utils module."""

from langchain_core.messages import AIMessage, HumanMessage

from utils import (
    format_chat_history,
    format_context_with_highlight,
    messages_to_tuples,
)


def test_format_chat_history_empty():
    """Test formatting empty chat history."""
    result = format_chat_history([])
    assert result == ""


def test_format_chat_history_tuples(sample_chat_history_tuples):
    """Test formatting chat history as tuples."""
    result = format_chat_history(sample_chat_history_tuples)
    assert "Human:" in result
    assert "Assistant:" in result
    assert "What is RAG?" in result


def test_format_chat_history_dicts(sample_chat_history):
    """Test formatting chat history as dicts."""
    result = format_chat_history(sample_chat_history)
    assert "Human:" in result
    assert "Assistant:" in result
    assert "What is RAG?" in result


def test_format_chat_history_messages():
    """Test formatting chat history as Message objects."""
    history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]
    result = format_chat_history(history)
    assert "Human:" in result
    assert "Assistant:" in result
    assert "Hello" in result
    assert "Hi there" in result


def test_format_chat_history_limit():
    """Test that chat history respects limit."""
    history = [
        {"role": "user", "content": f"Message {i}"}
        for i in range(10)
    ]
    result = format_chat_history(history, limit=5)
    # Should only include last 5 messages
    assert "Message 5" in result
    assert "Message 9" in result
    assert "Message 0" not in result


def test_messages_to_tuples_empty():
    """Test converting empty messages to tuples."""
    result = messages_to_tuples([])
    assert result == []


def test_messages_to_tuples(sample_chat_history):
    """Test converting messages to tuples."""
    result = messages_to_tuples(sample_chat_history)
    assert len(result) == 1  # Only one complete pair
    assert result[0][0] == "What is RAG?"
    assert result[0][1] == "RAG is Retrieval-Augmented Generation."


def test_messages_to_tuples_incomplete():
    """Test converting messages with incomplete pairs."""
    messages = [
        {"role": "user", "content": "Question 1"},
        {"role": "user", "content": "Question 2"},
    ]
    result = messages_to_tuples(messages)
    assert len(result) == 0  # No complete pairs


def test_format_context_with_highlight_empty():
    """Test formatting empty context."""
    result = format_context_with_highlight([])
    assert result == ""


def test_format_context_with_highlight(sample_documents):
    """Test formatting context with documents."""
    result = format_context_with_highlight(sample_documents[:2])
    assert "Sources:" in result
    assert "test1.pdf" in result
    assert "test2.pdf" in result
    assert "machine learning" in result


def test_format_context_with_scores(sample_documents):
    """Test formatting context with scores."""
    docs_with_scores = [
        (sample_documents[0], 0.9),
        (sample_documents[1], 0.8),
    ]
    result = format_context_with_highlight(
        sample_documents[:2], docs_with_scores=docs_with_scores
    )
    assert "⭐" in result  # Top chunk should be highlighted
    assert "rel:" in result or "dist:" in result  # Score info


def test_format_context_with_rewritten_query(sample_documents):
    """Test formatting context with rewritten query."""
    result = format_context_with_highlight(
        sample_documents[:2], rewritten_query="rewritten query"
    )
    assert "🔄 Rewritten:" in result
    assert "rewritten query" in result


def test_format_context_with_hybrid_scores(sample_documents):
    """Test formatting context with hybrid scores."""
    hybrid_scores = [
        (sample_documents[0], 0.9, 0.85, 0.95),
        (sample_documents[1], 0.8, 0.75, 0.85),
    ]
    result = format_context_with_highlight(
        sample_documents[:2], hybrid_scores=hybrid_scores
    )
    assert "f:" in result  # Fused score
    assert "s:" in result  # Semantic score
    assert "k:" in result  # Keyword score



