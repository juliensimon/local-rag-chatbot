"""Tests for empty vectorstore and initial state scenarios."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from qa_chain import QAChainWrapper, create_qa_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from ui.handlers import create_stream_chat_response


@pytest.fixture
def empty_vectorstore():
    """Create an empty vectorstore mock."""
    mock_vs = MagicMock()
    mock_vs.get.return_value = {
        "documents": [],
        "metadatas": [],
        "ids": [],
    }
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []  # No documents
    mock_vs.as_retriever.return_value = mock_retriever
    return mock_vs


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_rag_query_with_empty_vectorstore(mock_format_history, mock_create_llm, empty_vectorstore):
    """Test RAG query when vectorstore is empty."""
    prompt = ChatPromptTemplate.from_template("Test: {question}")
    qa_chain = QAChainWrapper(empty_vectorstore, prompt)

    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "I don't have any documents to search."
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "What is RAG?",
        "chat_history": [],
    }

    results = list(qa_chain.stream(inputs))

    # Should handle empty vectorstore gracefully
    assert len(results) > 0
    # Should have empty source documents
    assert len(results[0]["source_documents"]) == 0
    # Context should be empty
    context = "\n\n".join(doc.page_content for doc in results[0]["source_documents"])
    assert context == ""


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_rag_query_with_no_retrieved_documents(mock_format_history, mock_create_llm, mock_vectorstore):
    """Test RAG query when retrieval returns no documents."""
    prompt = ChatPromptTemplate.from_template("Test: {question}")
    qa_chain = QAChainWrapper(mock_vectorstore, prompt)

    mock_format_history.return_value = ""
    mock_llm = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.content = "No relevant documents found."
    mock_llm.stream.return_value = [mock_chunk]
    mock_create_llm.return_value = mock_llm

    # Retriever returns empty list
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []  # No documents retrieved
    qa_chain._retriever = mock_retriever

    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk]
    qa_chain._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "What is RAG?",
        "chat_history": [],
    }

    results = list(qa_chain.stream(inputs))

    # Should handle no documents gracefully
    assert len(results) > 0
    assert len(results[0]["source_documents"]) == 0


def test_initialize_chain_with_empty_vectorstore():
    """Test chain initialization with empty vectorstore."""
    from ui.app import initialize_chain

    with patch("ui.app.create_embeddings") as mock_embeddings, \
         patch("ui.app.load_or_create_vectorstore") as mock_load_vs, \
         patch("ui.app.create_qa_chain") as mock_create_chain:

        mock_embeddings.return_value = MagicMock()
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.get.return_value = {
            "metadatas": [],  # Empty metadatas
        }
        mock_load_vs.return_value = mock_vectorstore
        
        mock_chain = MagicMock()
        mock_create_chain.return_value = mock_chain

        chain, sources = initialize_chain()

        assert chain == mock_chain
        assert sources == []  # Should return empty list


@patch("ui.handlers.create_llm")
def test_handlers_with_empty_vectorstore(mock_create_llm, empty_vectorstore):
    """Test handlers with empty vectorstore."""
    mock_qa_chain = MagicMock()
    mock_qa_chain.stream.return_value = [
        {
            "chunk": "No documents available.",
            "source_documents": [],
            "docs_with_scores": None,
            "rewritten_query": None,
            "hybrid_scores": None,
        }
    ]

    stream_fn = create_stream_chat_response(mock_qa_chain)
    
    results = list(
        stream_fn(
            "What is RAG?",
            [],
            "RAG",
            doc_filter=None,
            search_type="mmr",
        )
    )

    assert len(results) > 0
    # Should handle empty results gracefully
    assert len(results[0][1]) == 0  # source_documents is empty


@patch("retrievers.BM25Okapi")
def test_hybrid_search_with_empty_documents(mock_bm25, empty_vectorstore):
    """Test hybrid search with empty document collection."""
    from retrievers import HybridRetriever

    empty_vectorstore.get.return_value = {
        "documents": [],
        "metadatas": [],
    }

    empty_vectorstore.similarity_search_with_score.return_value = []

    retriever = HybridRetriever(empty_vectorstore)
    
    results = retriever.hybrid_search("test query", k=5)

    # Should return empty list
    assert results == []


@patch("vectorstore.Chroma")
@patch("vectorstore.get_pdf_files")
def test_create_new_vectorstore_no_pdfs(mock_get_pdfs, mock_chroma):
    """Test creating new vectorstore when no PDFs exist."""
    from vectorstore import create_new_vectorstore

    mock_get_pdfs.return_value = []  # No PDF files

    with pytest.raises(FileNotFoundError):
        create_new_vectorstore(MagicMock())


def test_rerank_with_empty_documents(mock_vectorstore):
    """Test re-ranking with empty document list."""
    from qa_chain import QAChainWrapper
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template("Test: {question}")
    qa_chain_wrapper = QAChainWrapper(mock_vectorstore, prompt)
    
    result = qa_chain_wrapper.rerank_documents("query", [], top_k=5)
    
    # Should return empty list
    assert result == []


def test_hybrid_retriever_with_empty_collection(empty_vectorstore):
    """Test hybrid retriever initialization with empty collection."""
    from retrievers import HybridRetriever

    retriever = HybridRetriever(empty_vectorstore)
    retriever._build_bm25_index()

    # BM25 should be None when no documents
    assert retriever._bm25 is None
    assert retriever._documents == []

