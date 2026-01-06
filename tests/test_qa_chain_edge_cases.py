"""Additional edge case tests for qa_chain module."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from qa_chain import QAChainWrapper
from langchain_core.prompts import ChatPromptTemplate


@pytest.fixture
def mock_prompt():
    """Create a mock prompt template."""
    from langchain_core.prompts import ChatPromptTemplate
    return ChatPromptTemplate.from_template("Test: {question}")


@pytest.fixture
def qa_chain_wrapper(mock_vectorstore, mock_prompt):
    """Create a QAChainWrapper instance."""
    return QAChainWrapper(mock_vectorstore, mock_prompt)


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_similarity_search(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test streaming with similarity search type."""
    from langchain_core.documents import Document

    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    # Create documents with matching content
    doc_content = "Test content for matching"
    mock_doc = Document(page_content=doc_content, metadata={"page": 1, "source": "test.pdf"})
    mock_scored_doc = Document(page_content=doc_content, metadata={"page": 1, "source": "test.pdf"})

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]
    qa_chain_wrapper._retriever = mock_retriever

    mock_vectorstore = qa_chain_wrapper._vectorstore
    mock_vectorstore.similarity_search_with_score.return_value = [
        (mock_scored_doc, 0.5)
    ]

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question",
        "chat_history": [],
        "search_type": "similarity",
    }

    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_with_reranking(mock_format_history, mock_create_llm, qa_chain_wrapper, sample_documents):
    """Test streaming with reranking enabled."""
    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = sample_documents[:3]
    qa_chain_wrapper._retriever = mock_retriever

    # Mock reranker
    with patch("qa_chain.CrossEncoder") as mock_cross_encoder:
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.8, 0.7]
        mock_cross_encoder.return_value = mock_reranker
        qa_chain_wrapper._reranker = mock_reranker

        mock_chain = MagicMock()
        mock_chain.stream.return_value = [mock_chunk]
        qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

        inputs = {
            "question": "test question",
            "chat_history": [],
            "search_type": "mmr",
            "use_reranking": True,
        }

        results = list(qa_chain_wrapper.stream(inputs))
        assert len(results) > 0


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_with_query_rewriting(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test streaming with query rewriting enabled."""
    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_rewrite_response = MagicMock()
    mock_rewrite_response.content = "rewritten query"
    mock_llm.invoke.return_value = mock_rewrite_response
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Mock(page_content="Test")]
    qa_chain_wrapper._retriever = mock_retriever

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question that is long enough",
        "chat_history": [],
        "use_query_rewriting": True,
    }

    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0
    assert results[0].get("rewritten_query") == "rewritten query"


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_similarity_search_error(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test streaming with similarity search error."""
    from langchain_core.documents import Document

    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    mock_doc = Document(page_content="Test", metadata={"page": 1, "source": "test.pdf"})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]
    qa_chain_wrapper._retriever = mock_retriever

    mock_vectorstore = qa_chain_wrapper._vectorstore
    mock_vectorstore.similarity_search_with_score.side_effect = Exception("Error")

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question",
        "chat_history": [],
        "search_type": "similarity",
    }

    results = list(qa_chain_wrapper.stream(inputs))
    # Should still work even if similarity_search_with_score fails
    assert len(results) > 0

