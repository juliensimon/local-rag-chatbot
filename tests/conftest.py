"""Pytest configuration and shared fixtures."""

import os
from unittest.mock import MagicMock, Mock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings model."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 384
    return mock


@pytest.fixture
def mock_vectorstore():
    """Create a mock vectorstore."""
    mock = MagicMock()
    mock.get.return_value = {
        "documents": ["Document 1", "Document 2"],
        "metadatas": [
            {"source": "pdf/test1.pdf", "page": 1},
            {"source": "pdf/test2.pdf", "page": 2},
        ],
    }
    return mock


@pytest.fixture
def sample_documents():
    """Create sample document objects."""
    return [
        Document(
            page_content="This is a test document about machine learning.",
            metadata={"source": "pdf/test1.pdf", "page": 1},
        ),
        Document(
            page_content="Another document about neural networks.",
            metadata={"source": "pdf/test2.pdf", "page": 2},
        ),
        Document(
            page_content="Document in references section.",
            metadata={"source": "pdf/test3.pdf", "page": 3, "section": "references"},
        ),
    ]


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Mocked response")
    mock.stream.return_value = [
        MagicMock(content="Chunk "),
        MagicMock(content="1 "),
        MagicMock(content="2"),
    ]
    return mock


@pytest.fixture
def mock_reranker():
    """Create a mock reranker."""
    mock = MagicMock()
    mock.predict.return_value = [0.9, 0.8, 0.7]
    return mock


@pytest.fixture
def sample_chat_history():
    """Create sample chat history."""
    return [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG is Retrieval-Augmented Generation."},
        {"role": "user", "content": "How does it work?"},
    ]


@pytest.fixture
def sample_chat_history_tuples():
    """Create sample chat history as tuples."""
    return [
        ("What is RAG?", "RAG is Retrieval-Augmented Generation."),
        ("How does it work?", "It retrieves documents and generates answers."),
    ]


@pytest.fixture
def temp_pdf_dir(tmp_path):
    """Create a temporary directory for PDF files."""
    pdf_dir = tmp_path / "pdf"
    pdf_dir.mkdir()
    return str(pdf_dir)


@pytest.fixture
def temp_vectorstore_dir(tmp_path):
    """Create a temporary directory for vectorstore."""
    vs_dir = tmp_path / "vectorstore"
    vs_dir.mkdir()
    return str(vs_dir)


@pytest.fixture
def mock_hybrid_results():
    """Create mock hybrid search results."""
    doc1 = Document(
        page_content="Test content 1",
        metadata={"source": "pdf/test1.pdf", "page": 1},
    )
    doc2 = Document(
        page_content="Test content 2",
        metadata={"source": "pdf/test2.pdf", "page": 2},
    )
    return [
        {
            "doc": doc1,
            "fused_score": 0.9,
            "semantic_score": 0.85,
            "keyword_score": 0.95,
        },
        {
            "doc": doc2,
            "fused_score": 0.8,
            "semantic_score": 0.75,
            "keyword_score": 0.85,
        },
    ]

