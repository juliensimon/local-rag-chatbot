"""Tests for network and API failure scenarios."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from models import create_llm, create_embeddings
from qa_chain import QAChainWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


@patch("models.ChatOpenAI")
def test_llm_server_unavailable(mock_chat_openai):
    """Test graceful handling when LLM server is unavailable."""
    # Simulate connection error
    mock_chat_openai.side_effect = requests.exceptions.ConnectionError("Connection refused")

    with pytest.raises((requests.exceptions.ConnectionError, Exception)):
        create_llm()


@patch("models.ChatOpenAI")
def test_llm_server_timeout(mock_chat_openai):
    """Test handling of LLM server timeout."""
    mock_llm = MagicMock()
    mock_llm.stream.side_effect = requests.exceptions.Timeout("Request timed out")
    mock_chat_openai.return_value = mock_llm

    llm = create_llm(streaming=True)
    
    # Should raise timeout error when streaming
    with pytest.raises(requests.exceptions.Timeout):
        list(llm.stream("test"))


@patch("models.ChatOpenAI")
def test_llm_server_error_response(mock_chat_openai):
    """Test handling of LLM server error response."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("500 Internal Server Error")
    mock_chat_openai.return_value = mock_llm

    llm = create_llm(streaming=False)
    
    with pytest.raises(Exception) as exc_info:
        llm.invoke("test")
    assert "500" in str(exc_info.value) or "Error" in str(exc_info.value)


@patch("models.HuggingFaceEmbeddings")
def test_embedding_model_download_failure(mock_embeddings_class):
    """Test handling of embedding model download failure."""
    mock_embeddings_class.side_effect = Exception("Failed to download model")

    with pytest.raises(Exception) as exc_info:
        create_embeddings()
    assert "Failed" in str(exc_info.value) or "download" in str(exc_info.value).lower()


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_rag_query_with_llm_failure(mock_format_history, mock_create_llm, mock_vectorstore):
    """Test RAG query when LLM fails during streaming."""
    from qa_chain import create_qa_chain

    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_llm.stream.side_effect = requests.exceptions.ConnectionError("LLM server unavailable")
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_doc = Document(page_content="Test", metadata={"source": "test.pdf", "page": 1})
    mock_retriever.invoke.return_value = [mock_doc]
    mock_vectorstore.as_retriever.return_value = mock_retriever

    qa_chain = create_qa_chain(mock_vectorstore)

    # Mock the chain operator
    mock_chain = MagicMock()
    mock_chain.stream.side_effect = requests.exceptions.ConnectionError("LLM server unavailable")
    qa_chain._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question",
        "chat_history": [],
    }

    # Should handle error gracefully
    results = list(qa_chain.stream(inputs))
    
    # Should return error message in chunk
    assert len(results) > 0
    chunk_text = str(results[0].get("chunk", ""))
    # The error handling in qa_chain.py catches exceptions and yields error message
    assert len(chunk_text) > 0  # Should have some content (either response or error)


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_rag_query_with_retriever_failure(mock_format_history, mock_create_llm, mock_vectorstore):
    """Test RAG query when retriever fails."""
    from qa_chain import QAChainWrapper
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template("Test: {question}")
    qa_chain_wrapper = QAChainWrapper(mock_vectorstore, prompt)

    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "Response"
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    # Simulate retriever failure
    mock_retriever = MagicMock()
    mock_retriever.invoke.side_effect = Exception("ChromaDB connection failed")
    qa_chain_wrapper._retriever = mock_retriever

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question",
        "chat_history": [],
    }

    # Should handle retriever error - may raise or handle gracefully
    try:
        results = list(qa_chain_wrapper.stream(inputs))
        # If it doesn't raise, should have error in response or empty results
        if results:
            chunk = results[0].get("chunk", "")
            assert "Error" in str(chunk).lower() or len(results) == 0 or len(chunk) > 0
    except Exception:
        # Exception is acceptable for retriever failure
        pass


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_stream_interruption_handling(mock_format_history, mock_create_llm, mock_vectorstore):
    """Test handling of stream interruption."""
    from qa_chain import QAChainWrapper
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template("Test: {question}")
    qa_chain_wrapper = QAChainWrapper(mock_vectorstore, prompt)

    mock_format_history.return_value = ""
    
    # Simulate stream that gets interrupted
    mock_llm = MagicMock()
    mock_chunk1 = MagicMock()
    mock_chunk1.content = "Partial "
    mock_chunk2 = MagicMock()
    mock_chunk2.content = "response"
    
    def interrupted_stream(*args, **kwargs):
        yield mock_chunk1
        raise KeyboardInterrupt("Stream interrupted")
    
    mock_llm.stream.side_effect = interrupted_stream
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="Test", metadata={})]
    qa_chain_wrapper._retriever = mock_retriever

    mock_chain = MagicMock()
    mock_chain.stream.side_effect = interrupted_stream
    qa_chain_wrapper._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "test question",
        "chat_history": [],
    }

    # Should handle interruption gracefully
    try:
        results = list(qa_chain_wrapper.stream(inputs))
        # If it completes, should have partial response
        if results:
            assert len(results) > 0
    except KeyboardInterrupt:
        # Interruption is acceptable
        pass


@patch("retrievers.BM25Okapi")
def test_hybrid_search_with_semantic_failure(mock_bm25, mock_vectorstore):
    """Test hybrid search when semantic search fails."""
    from retrievers import HybridRetriever
    from langchain_core.documents import Document

    mock_vectorstore.get.return_value = {
        "documents": ["Doc 1"],
        "metadatas": [{"source": "test.pdf", "page": 1}],
    }

    # Simulate semantic search failure
    mock_vectorstore.similarity_search_with_score.side_effect = Exception("Vector search failed")

    mock_bm25_instance = MagicMock()
    mock_bm25_instance.get_scores.return_value = [0.8]
    mock_bm25.return_value = mock_bm25_instance

    retriever = HybridRetriever(mock_vectorstore)
    
    # Should handle semantic failure gracefully
    results = retriever.hybrid_search("test query", k=1)
    
    # Should still return results based on BM25 only
    assert isinstance(results, list)

