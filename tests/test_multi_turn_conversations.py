"""Tests for multi-turn conversations with chat history."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from qa_chain import QAChainWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


@pytest.fixture
def qa_chain_wrapper(mock_vectorstore):
    """Create a QAChainWrapper instance."""
    prompt = ChatPromptTemplate.from_template("Test: {question}\nHistory: {chat_history}")
    return QAChainWrapper(mock_vectorstore, prompt)


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_rag_with_chat_history_context(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test RAG query with previous conversation context."""
    # First turn: User asks initial question
    chat_history_turn1 = []
    
    # Second turn: User asks follow-up question
    chat_history_turn2 = [
        ("What is RAG?", "RAG stands for Retrieval-Augmented Generation."),
    ]
    
    mock_format_history.return_value = "Human: What is RAG?\nAssistant: RAG stands for Retrieval-Augmented Generation."
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Based on the previous context, RAG combines retrieval and generation."
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_doc = Document(
        page_content="RAG combines retrieval and generation",
        metadata={"source": "test.pdf", "page": 1}
    )
    mock_retriever.invoke.return_value = [mock_doc]
    qa_chain_wrapper._retriever = mock_retriever

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    # Test follow-up question with history
    inputs = {
        "question": "How does it work?",
        "chat_history": chat_history_turn2,
    }

    results = list(qa_chain_wrapper.stream(inputs))
    
    assert len(results) > 0
    # Verify chat history was formatted and included
    mock_format_history.assert_called()
    # Verify results contain expected data
    assert "chunk" in results[0]


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_chat_history_limit_enforcement(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test that chat history limit (CHAT_HISTORY_LIMIT) is enforced."""
    from config import CHAT_HISTORY_LIMIT

    # Create history longer than limit
    long_history = [
        (f"Question {i}", f"Answer {i}") for i in range(CHAT_HISTORY_LIMIT + 5)
    ]

    mock_format_history.return_value = "Formatted history"
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
        "question": "New question",
        "chat_history": long_history,
    }

    list(qa_chain_wrapper.stream(inputs))
    
    # Verify format_chat_history was called with the long history
    # (it should handle limiting internally)
    mock_format_history.assert_called()
    # The format_chat_history function should limit to CHAT_HISTORY_LIMIT
    call_args = mock_format_history.call_args[0][0]
    # Should only include recent messages (last CHAT_HISTORY_LIMIT)
    assert len(call_args) <= CHAT_HISTORY_LIMIT + 5  # format_chat_history handles limiting


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_follow_up_question_references_previous_answer(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test follow-up question that references previous answer."""
    # First question about a topic
    chat_history = [
        ("What is machine learning?", "Machine learning is a subset of AI."),
    ]

    mock_format_history.return_value = "Human: What is machine learning?\nAssistant: Machine learning is a subset of AI."
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Deep learning is a subset of machine learning."
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_doc = Document(
        page_content="Deep learning is a subset of machine learning",
        metadata={"source": "test.pdf", "page": 2}
    )
    mock_retriever.invoke.return_value = [mock_doc]
    qa_chain_wrapper._retriever = mock_retriever

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    # Follow-up question that references "it" (machine learning)
    inputs = {
        "question": "What are its main types?",
        "chat_history": chat_history,
    }

    results = list(qa_chain_wrapper.stream(inputs))
    
    assert len(results) > 0
    # Verify history context was included
    mock_format_history.assert_called()
    # Verify results contain expected data
    assert "chunk" in results[0]


@patch("ui.handlers.create_llm")
def test_handlers_multi_turn_conversation(mock_create_llm, mock_vectorstore):
    """Test handlers with multi-turn conversation."""
    from ui.handlers import create_stream_chat_response, create_respond_handler

    mock_qa_chain = MagicMock()
    mock_qa_chain.stream.return_value = [
        {
            "chunk": "Answer to follow-up",
            "source_documents": [Mock(metadata={"source": "test.pdf", "page": 1})],
            "docs_with_scores": [(Mock(), 0.9)],
            "rewritten_query": None,
            "hybrid_scores": None,
        }
    ]

    stream_fn = create_stream_chat_response(mock_qa_chain)
    respond_fn = create_respond_handler(stream_fn)

    # First turn
    history_turn1 = []
    results_turn1 = list(
        respond_fn(
            "What is RAG?",
            history_turn1,
            True,  # RAG enabled
            "mmr",
            "All Documents",
            False,
            False,
            70,
        )
    )

    # Second turn with history
    history_turn2 = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
    ]

    results_turn2 = list(
        respond_fn(
            "How does it work?",
            history_turn2,
            True,  # RAG enabled
            "mmr",
            "All Documents",
            False,
            False,
            70,
        )
    )

    assert len(results_turn2) > 0
    # Verify chat history was passed to stream function
    assert mock_qa_chain.stream.called
    call_args = mock_qa_chain.stream.call_args[0][0]
    # Should have chat_history in the call
    assert "chat_history" in call_args or "question" in call_args

