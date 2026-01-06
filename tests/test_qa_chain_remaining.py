"""Tests to cover remaining lines in qa_chain.py."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from qa_chain import QAChainWrapper
from langchain_core.prompts import ChatPromptTemplate


@pytest.fixture
def qa_chain_wrapper(mock_vectorstore):
    """Create a QAChainWrapper instance."""
    prompt = ChatPromptTemplate.from_template("Test: {question}")
    return QAChainWrapper(mock_vectorstore, prompt)


def test_rerank_documents_no_reranker(qa_chain_wrapper, sample_documents):
    """Test rerank_documents when reranker is None (line 115)."""
    # Set reranker to None and ensure _get_reranker returns None
    qa_chain_wrapper._reranker = None
    with patch.object(qa_chain_wrapper, '_get_reranker', return_value=None):
        result = qa_chain_wrapper.rerank_documents("query", sample_documents[:3], top_k=2)
        assert len(result) == 2
        assert all(r[1] is None for r in result)  # All scores should be None


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_reranking_similarity(mock_format_history, mock_create_llm, qa_chain_wrapper, sample_documents):
    """Test streaming with reranking and similarity search (line 247)."""
    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    # Mock retriever for similarity search with reranking
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = sample_documents[:3]
    qa_chain_wrapper._vectorstore.as_retriever.return_value = mock_retriever

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
            "search_type": "similarity",
            "use_reranking": True,
        }

        results = list(qa_chain_wrapper.stream(inputs))
        assert len(results) > 0


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_similarity_doc_matching(mock_format_history, mock_create_llm, qa_chain_wrapper):
    """Test similarity search with document matching (lines 268-276)."""
    from langchain_core.documents import Document

    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    # Create documents with matching content (first 100 chars must match)
    doc_content = "A" * 50 + "B" * 50 + "C" * 50  # 150 chars total
    doc1 = Document(page_content=doc_content, metadata={"page": 1, "source": "test.pdf"})
    doc2 = Document(page_content=doc_content, metadata={"page": 1, "source": "test.pdf"})

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [doc1]
    qa_chain_wrapper._retriever = mock_retriever

    mock_vectorstore = qa_chain_wrapper._vectorstore
    mock_vectorstore.similarity_search_with_score.return_value = [
        (doc2, 0.5)  # Same content, should match (lines 269-274)
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
    # Verify that scores were matched
    assert results[0].get("docs_with_scores") is not None

    # Test case where doc doesn't match (else clause line 276)
    doc3 = Document(
        page_content="X" * 150,  # Different content
        metadata={"page": 2, "source": "other.pdf"}
    )
    mock_retriever.invoke.return_value = [doc3]
    mock_vectorstore.similarity_search_with_score.return_value = [
        (doc2, 0.5)  # Different content, shouldn't match
    ]

    results = list(qa_chain_wrapper.stream(inputs))
    assert len(results) > 0

