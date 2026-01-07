"""Tests for UI handlers module."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ui.handlers import (
    create_respond_handler,
    create_stream_chat_response,
    update_hybrid_alpha_visibility,
    update_rag_controls,
)


def test_create_stream_chat_response_rag(mock_vectorstore):
    """Test stream chat response for RAG mode."""
    mock_qa_chain = MagicMock()
    mock_qa_chain.stream.return_value = [
        {
            "chunk": "Response ",
            "source_documents": [Mock()],
            "docs_with_scores": [(Mock(), 0.9)],
            "rewritten_query": None,
            "hybrid_scores": None,
        },
        {
            "chunk": "chunk",
            "source_documents": [Mock()],
            "docs_with_scores": [(Mock(), 0.9)],
            "rewritten_query": None,
            "hybrid_scores": None,
        },
    ]

    stream_fn = create_stream_chat_response(mock_qa_chain)
    results = list(
        stream_fn(
            "test question",
            [],
            "RAG",
            doc_filter=None,
            search_type="mmr",
        )
    )

    assert len(results) > 0
    assert "Response chunk" in results[-1][0]


@patch("ui.handlers.create_llm")
def test_create_stream_chat_response_vanilla(mock_create_llm):
    """Test stream chat response for vanilla LLM mode."""
    mock_llm = MagicMock()
    mock_llm.stream.return_value = [
        MagicMock(content="Hello "),
        MagicMock(content="World"),
    ]
    mock_create_llm.return_value = mock_llm

    stream_fn = create_stream_chat_response(MagicMock())
    results = list(
        stream_fn(
            "test question",
            [],
            "Vanilla LLM",
        )
    )

    assert len(results) > 0
    assert "Hello World" in results[-1][0]


@patch("ui.handlers.create_llm")
def test_create_stream_chat_response_vanilla_error(mock_create_llm):
    """Test stream chat response error handling for vanilla mode."""
    mock_llm = MagicMock()
    mock_llm.stream.side_effect = Exception("Error")
    mock_create_llm.return_value = mock_llm

    stream_fn = create_stream_chat_response(MagicMock())
    results = list(
        stream_fn(
            "test question",
            [],
            "Vanilla LLM",
        )
    )

    assert len(results) > 0
    assert "Error" in results[-1][0]


def test_create_stream_chat_response_with_filter(mock_vectorstore):
    """Test stream chat response with document filter."""
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

    # Pass available_sources to allow the filter to be validated
    stream_fn = create_stream_chat_response(mock_qa_chain, available_sources=["test.pdf"])
    list(
        stream_fn(
            "test question",
            [],
            "RAG",
            doc_filter="test.pdf",
            search_type="mmr",
        )
    )

    # Check that filter was passed (only works with valid source)
    call_args = mock_qa_chain.stream.call_args[0][0]
    assert "filter" in call_args


def test_create_respond_handler_empty_message():
    """Test respond handler with empty message."""
    stream_fn = MagicMock()
    respond_fn = create_respond_handler(stream_fn)

    results = list(respond_fn("", [], False, "mmr", "All Documents", False, False, 70))
    assert len(results) > 0
    assert results[0][0] == ""  # Empty message cleared


def test_create_respond_handler_rag(mock_vectorstore):
    """Test respond handler for RAG mode."""
    mock_qa_chain = MagicMock()
    mock_qa_chain.stream.return_value = [
        {
            "chunk": "Response",
            "source_documents": [Mock(metadata={"source": "test.pdf", "page": 1})],
            "docs_with_scores": [(Mock(), 0.9)],
            "rewritten_query": None,
            "hybrid_scores": None,
        }
    ]

    stream_fn = create_stream_chat_response(mock_qa_chain)
    respond_fn = create_respond_handler(stream_fn)

    results = list(
        respond_fn(
            "test question",
            [],
            True,  # RAG enabled
            "mmr",
            "All Documents",
            False,
            False,
            70,
        )
    )

    assert len(results) > 0
    # Check that history was updated
    assert results[-1][1][-1]["role"] == "assistant"


def test_update_rag_controls_enabled():
    """Test RAG controls update when enabled."""
    result = update_rag_controls(True)
    assert len(result) == 4
    assert all(r.get("visible") for r in result)


def test_update_rag_controls_disabled():
    """Test RAG controls update when disabled."""
    result = update_rag_controls(False)
    assert len(result) == 4
    assert all(not r.get("visible") for r in result)


def test_update_hybrid_alpha_visibility_hybrid():
    """Test hybrid alpha visibility for hybrid search."""
    result = update_hybrid_alpha_visibility("hybrid")
    assert result.get("visible") is True
    assert result.get("interactive") is True


def test_update_hybrid_alpha_visibility_other():
    """Test hybrid alpha visibility for non-hybrid search."""
    result = update_hybrid_alpha_visibility("mmr")
    assert result.get("visible") is False
    assert result.get("interactive") is False

