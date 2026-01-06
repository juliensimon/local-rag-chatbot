"""Tests for switching between RAG and Vanilla LLM modes."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ui.handlers import create_stream_chat_response, create_respond_handler


@patch("ui.handlers.create_llm")
def test_switch_from_rag_to_vanilla(mock_create_llm):
    """Test switching from RAG mode to Vanilla LLM mode."""
    mock_qa_chain = MagicMock()
    mock_llm = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.content = "Vanilla "
    mock_chunk2 = MagicMock()
    mock_chunk2.content = "response"
    mock_llm.stream.return_value = [mock_chunk1, mock_chunk2]
    mock_create_llm.return_value = mock_llm

    stream_fn = create_stream_chat_response(mock_qa_chain)
    respond_fn = create_respond_handler(stream_fn)

    # First: RAG query
    history_rag = []
    results_rag = list(
        respond_fn(
            "What is RAG?",
            history_rag,
            True,  # RAG enabled
            "mmr",
            "All Documents",
            False,
            False,
            70,
        )
    )

    # Second: Switch to Vanilla (RAG disabled)
    history_after_rag = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG response"},
    ]

    results_vanilla = list(
        respond_fn(
            "Tell me a joke",
            history_after_rag,
            False,  # RAG disabled - switched to Vanilla
            "mmr",
            "All Documents",
            False,
            False,
            70,
        )
    )

    # Verify Vanilla LLM was used (not RAG chain)
    assert len(results_vanilla) > 0
    # Note: RAG chain might be called during first turn, so we check that vanilla was also called
    assert mock_create_llm.called  # Vanilla LLM should be called
    # Verify history was preserved
    assert len(results_vanilla[-1][1]) >= len(history_after_rag)


@patch("ui.handlers.create_llm")
def test_switch_from_vanilla_to_rag(mock_create_llm):
    """Test switching from Vanilla LLM mode to RAG mode."""
    mock_qa_chain = MagicMock()
    mock_qa_chain.stream.return_value = [
        {
            "chunk": "RAG response",
            "source_documents": [Mock(metadata={"source": "test.pdf", "page": 1})],
            "docs_with_scores": [(Mock(), 0.9)],
            "rewritten_query": None,
            "hybrid_scores": None,
        }
    ]

    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Vanilla response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    stream_fn = create_stream_chat_response(mock_qa_chain)
    respond_fn = create_respond_handler(stream_fn)

    # First: Vanilla query
    history_vanilla = []
    results_vanilla = list(
        respond_fn(
            "Tell me a joke",
            history_vanilla,
            False,  # RAG disabled
            "mmr",
            "All Documents",
            False,
            False,
            70,
        )
    )

    # Second: Switch to RAG
    history_after_vanilla = [
        {"role": "user", "content": "Tell me a joke"},
        {"role": "assistant", "content": "Vanilla response"},
    ]

    results_rag = list(
        respond_fn(
            "What is RAG?",
            history_after_vanilla,
            True,  # RAG enabled - switched to RAG
            "mmr",
            "All Documents",
            False,
            False,
            70,
        )
    )

    # Verify RAG chain was used
    assert len(results_rag) > 0
    assert mock_qa_chain.stream.called  # RAG chain should be called
    # Verify history was preserved
    assert len(results_rag[-1][1]) > len(history_after_vanilla)


@patch("ui.handlers.create_llm")
def test_chat_history_preserved_across_mode_switches(mock_create_llm):
    """Test that chat history is preserved when switching modes multiple times."""
    mock_qa_chain = MagicMock()
    mock_qa_chain.stream.return_value = [
        {
            "chunk": "RAG answer",
            "source_documents": [],
            "docs_with_scores": None,
            "rewritten_query": None,
            "hybrid_scores": None,
        }
    ]

    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Vanilla answer"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    stream_fn = create_stream_chat_response(mock_qa_chain)
    respond_fn = create_respond_handler(stream_fn)

    history = []

    # Turn 1: RAG
    results1 = list(
        respond_fn("Q1", history, True, "mmr", "All Documents", False, False, 70)
    )
    history = results1[-1][1]

    # Turn 2: Vanilla
    results2 = list(
        respond_fn("Q2", history, False, "mmr", "All Documents", False, False, 70)
    )
    history = results2[-1][1]

    # Turn 3: RAG again
    results3 = list(
        respond_fn("Q3", history, True, "mmr", "All Documents", False, False, 70)
    )
    history = results3[-1][1]

    # Verify all messages are preserved
    assert len(history) == 6  # 3 user + 3 assistant messages
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Q1"
    assert history[1]["role"] == "assistant"
    assert history[2]["role"] == "user"
    assert history[2]["content"] == "Q2"
    assert history[3]["role"] == "assistant"
    assert history[4]["role"] == "user"
    assert history[4]["content"] == "Q3"
    assert history[5]["role"] == "assistant"


@patch("ui.handlers.create_llm")
def test_ui_state_consistency_during_mode_switch(mock_create_llm):
    """Test that UI state remains consistent when switching modes."""
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

    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    stream_fn = create_stream_chat_response(mock_qa_chain)
    respond_fn = create_respond_handler(stream_fn)

    # Start with RAG enabled
    results1 = list(
        respond_fn(
            "Question",
            [],
            True,  # RAG enabled
            "mmr",
            "test.pdf",
            False,
            False,
            70,
        )
    )

    # Switch to Vanilla
    results2 = list(
        respond_fn(
            "Question 2",
            results1[-1][1],
            False,  # RAG disabled
            "mmr",  # Search type should be ignored in Vanilla mode
            "test.pdf",  # Filter should be ignored in Vanilla mode
            False,
            False,
            70,
        )
    )

    # Verify state is consistent
    # RAG state should be False
    assert results2[-1][3] is False  # rag_state
    # Context should be empty for Vanilla mode
    assert results2[-1][2] == ""  # context_box
    # Document filter should be preserved
    assert results2[-1][4] == "test.pdf"  # doc_filter

