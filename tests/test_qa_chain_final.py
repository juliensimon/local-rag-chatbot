"""Final tests to cover remaining qa_chain lines."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from qa_chain import QAChainWrapper
from langchain_core.prompts import ChatPromptTemplate


@pytest.fixture
def qa_chain_wrapper(mock_vectorstore):
    """Create a QAChainWrapper instance."""
    prompt = ChatPromptTemplate.from_template("Test: {question}")
    return QAChainWrapper(mock_vectorstore, prompt)


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_similarity_exact_match(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test similarity search with exact document matching (lines 269-274)."""
    from langchain_core.documents import Document

    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    # Create documents where first 100 chars match exactly
    matching_prefix = "A" * 100
    doc1 = Document(
        page_content=matching_prefix + " rest of content 1",
        metadata={"page": 1, "source": "test.pdf"},
    )
    doc2 = Document(
        page_content=matching_prefix + " rest of content 2",  # Same first 100 chars
        metadata={"page": 1, "source": "test.pdf"},
    )

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc1]
    qa_chain_wrapper._retriever = mock_retriever

    mock_vectorstore = qa_chain_wrapper._vectorstore
    # Return doc2 with score - should match doc1 based on first 100 chars and page
    mock_vectorstore.similarity_search_with_score.return_value = [
        (doc2, 0.5)
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
    # Verify that the matching logic was executed (lines 269-274)
    assert results[0].get("docs_with_scores") is not None

