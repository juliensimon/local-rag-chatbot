"""Tests for input validation and edge cases."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from qa_chain import QAChainWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from ui.handlers import create_stream_chat_response, create_respond_handler


@pytest.fixture
def qa_chain_wrapper(mock_vectorstore):
    """Create a QAChainWrapper instance."""
    prompt = ChatPromptTemplate.from_template("Test: {question}")
    return QAChainWrapper(mock_vectorstore, prompt)


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_very_long_query(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test handling of extremely long queries (>10k characters)."""
    very_long_query = "What is " + "RAG? " * 2000  # ~10k+ characters

    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="Test", metadata={})]
    qa_chain_wrapper._retriever = mock_retriever

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": very_long_query,
        "chat_history": [],
    }

    # Should handle long query without crashing
    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0


def test_empty_whitespace_query():
    """Test handling of whitespace-only queries."""
    from ui.handlers import create_respond_handler

    stream_fn = MagicMock()
    respond_fn = create_respond_handler(stream_fn)

    # Empty string
    results = list(respond_fn("", [], False, "mmr", "All Documents", False, False, 70))
    assert results[0][0] == ""  # Should return empty immediately

    # Whitespace only - now also returns empty (treated as empty after strip)
    results = list(respond_fn("   ", [], False, "mmr", "All Documents", False, False, 70))
    assert results[0][0] == ""  # Should return empty immediately

    # Newlines only - also returns empty
    results = list(respond_fn("\n\n\n", [], False, "mmr", "All Documents", False, False, 70))
    assert results[0][0] == ""  # Should return empty immediately


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_special_characters_in_query(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test handling of special characters in queries."""
    special_chars_query = "What is RAG? @#$%^&*()[]{}|\\/<>?~`"

    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="Test", metadata={})]
    qa_chain_wrapper._retriever = mock_retriever

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": special_chars_query,
        "chat_history": [],
    }

    # Should handle special characters without crashing
    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_unicode_emoji_in_query(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test handling of Unicode and emoji in queries."""
    unicode_query = "What is RAG? 🚀 你好 مرحبا"

    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="Test", metadata={})]
    qa_chain_wrapper._retriever = mock_retriever

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": unicode_query,
        "chat_history": [],
    }

    # Should handle Unicode/emoji without crashing
    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0


def test_invalid_document_filter():
    """Test handling of invalid document filter selection."""
    from ui.handlers import create_stream_chat_response

    mock_qa_chain = MagicMock()
    mock_qa_chain.stream.return_value = [
        {
            "chunk": "Response",
            "source_documents": [],
            "docs_with_scores": None,
            "rewritten_query": None,
            "hybrid_scores": None,
        }
    ]

    # Pass valid sources - nonexistent.pdf is NOT in the list
    stream_fn = create_stream_chat_response(mock_qa_chain, available_sources=["valid.pdf"])

    # Invalid filter (document that doesn't exist in available_sources)
    results = list(
        stream_fn(
            "test question",
            [],
            "RAG",
            doc_filter="nonexistent.pdf",  # Invalid filter - not in available_sources
            search_type="mmr",
        )
    )

    # Should handle invalid filter gracefully
    assert len(results) > 0
    # Invalid filter should be ignored (no filter passed to chain)
    call_args = mock_qa_chain.stream.call_args[0][0]
    assert "filter" not in call_args  # Filter is NOT passed for invalid sources


def test_malformed_chat_history():
    """Test handling of malformed chat history."""
    from ui.handlers import messages_to_tuples

    # Missing role - should handle gracefully
    try:
        malformed_history1 = [
            {"content": "Message without role"},
        ]
        tuples1 = messages_to_tuples(malformed_history1)
        assert isinstance(tuples1, list)
        # Should return empty (no valid user/assistant pairs)
        assert tuples1 == []
    except (KeyError, TypeError):
        # Exception is acceptable for malformed input
        pass

    # Missing content - should handle gracefully
    try:
        malformed_history2 = [
            {"role": "user"},
        ]
        tuples2 = messages_to_tuples(malformed_history2)
        assert isinstance(tuples2, list)
    except (KeyError, TypeError):
        # Exception is acceptable for malformed input
        pass

    # Invalid role - should handle gracefully
    malformed_history3 = [
        {"role": "invalid", "content": "Message"},
    ]
    tuples3 = messages_to_tuples(malformed_history3)
    # Should return empty (only processes user/assistant pairs)
    assert isinstance(tuples3, list)
    assert tuples3 == []


@patch("qa_chain.create_llm")
def test_query_rewriting_with_very_short_query(mock_create_llm, qa_chain_wrapper):
    """Test query rewriting with very short query."""
    very_short_query = "RAG?"

    mock_llm = MagicMock()
    mock_response = MagicMock()
    # Make rewritten query shorter than 30% of original
    # "RAG?" is 4 chars, 30% = 1.2, so "R" (1 char) should trigger fallback
    mock_response.content = "R"
    mock_llm.invoke.return_value = mock_response
    mock_create_llm.return_value = mock_llm

    result = qa_chain_wrapper.rewrite_query(very_short_query)

    # Should return original if rewritten is too short (< 30% of original length)
    assert result == very_short_query


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_hybrid_search_with_extreme_alpha(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test hybrid search with extreme alpha values."""
    from langchain_core.documents import Document

    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="Test", metadata={})]
    qa_chain_wrapper._retriever = mock_retriever

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    # Test with alpha = 0.0 (pure keyword)
    inputs = {
        "question": "test",
        "chat_history": [],
        "search_type": "hybrid",
        "hybrid_alpha": 0.0,
    }
    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0

    # Test with alpha = 1.0 (pure semantic)
    inputs["hybrid_alpha"] = 1.0
    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0


def test_rerank_with_very_few_documents(qa_chain_wrapper, sample_documents):
    """Test re-ranking with very few documents."""
    # Re-rank with only 1 document
    result = qa_chain_wrapper.rerank_documents("query", sample_documents[:1], top_k=5)
    
    # Should return the single document
    assert len(result) == 1

    # Re-rank with more documents than top_k
    result = qa_chain_wrapper.rerank_documents("query", sample_documents[:10], top_k=3)
    
    # Should return only top_k documents
    assert len(result) == 3

