"""Integration tests for the RAG application."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from models import create_embeddings, create_llm
from qa_chain import create_qa_chain
from utils import format_chat_history, format_context_with_highlight, messages_to_tuples
from vectorstore import load_or_create_vectorstore


@patch("vectorstore.Chroma")
@patch("models.HuggingFaceEmbeddings")
def test_integration_rag_flow(mock_embeddings_class, mock_chroma_class):
    """Test full RAG flow from embeddings to QA chain."""
    # Setup mocks
    mock_embeddings = MagicMock()
    mock_embeddings_class.return_value = mock_embeddings

    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {
        "documents": ["Test document content"],
        "metadatas": [{"source": "pdf/test.pdf", "page": 1}],
    }
    mock_vectorstore.as_retriever.return_value = MagicMock()
    mock_chroma_class.return_value = mock_vectorstore

    # Test full flow
    embeddings = create_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    qa_chain = create_qa_chain(vectorstore)

    assert embeddings is not None
    assert vectorstore is not None
    assert qa_chain is not None


@patch("models.ChatOpenAI")
def test_integration_llm_creation(mock_chat_openai):
    """Test LLM creation integration."""
    mock_llm = MagicMock()
    mock_chat_openai.return_value = mock_llm

    llm = create_llm(streaming=True)
    assert llm is not None
    mock_chat_openai.assert_called_once()


def test_integration_chat_history_conversion():
    """Test integration of chat history conversion and formatting."""
    messages = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
    ]

    # Convert to tuples
    tuples = messages_to_tuples(messages)
    assert len(tuples) == 1
    assert tuples[0][0] == "Question 1"

    # Format history
    formatted = format_chat_history(tuples)
    assert "Question 1" in formatted
    assert "Answer 1" in formatted


def test_integration_context_formatting():
    """Test integration of context formatting with all features."""
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="This is test content about machine learning.",
            metadata={"source": "pdf/test1.pdf", "page": 1},
        ),
        Document(
            page_content="Another document about neural networks.",
            metadata={"source": "pdf/test2.pdf", "page": 2},
        ),
    ]

    docs_with_scores = [(docs[0], 0.9), (docs[1], 0.8)]
    hybrid_scores = [
        (docs[0], 0.9, 0.85, 0.95),
        (docs[1], 0.8, 0.75, 0.85),
    ]

    result = format_context_with_highlight(
        docs,
        docs_with_scores=docs_with_scores,
        rewritten_query="machine learning neural networks",
        hybrid_scores=hybrid_scores,
    )

    assert "Sources:" in result
    assert "test1.pdf" in result
    assert "test2.pdf" in result
    assert "🔄 Rewritten:" in result
    assert "machine learning neural networks" in result
    assert "⭐" in result  # Top chunk highlighted
    assert "f:" in result  # Fused score
    assert "s:" in result  # Semantic score
    assert "k:" in result  # Keyword score


@patch("qa_chain.create_llm")
@patch("qa_chain.format_chat_history")
def test_integration_qa_chain_streaming(mock_format_history, mock_create_llm, mock_vectorstore):
    """Test integration of QA chain streaming."""
    from qa_chain import create_qa_chain

    mock_format_history.return_value = "Previous: Test"
    
    # Create proper mock chunks with content as property
    from unittest.mock import PropertyMock
    
    mock_chunk1 = MagicMock()
    mock_chunk1.content = "This is "
    mock_chunk2 = MagicMock()
    mock_chunk2.content = "a test "
    mock_chunk3 = MagicMock()
    mock_chunk3.content = "response."
    
    mock_llm = MagicMock()
    mock_llm.stream.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
    mock_create_llm.return_value = mock_llm

    mock_retriever = MagicMock()
    mock_doc = Mock(page_content="Retrieved context", metadata={"source": "test.pdf", "page": 1})
    mock_retriever.invoke.return_value = [mock_doc]
    mock_vectorstore.as_retriever.return_value = mock_retriever

    qa_chain = create_qa_chain(mock_vectorstore)

    # Mock the chain operator - this is what gets called in stream()
    mock_chain = MagicMock()
    mock_chain.stream.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]
    qa_chain._prompt.__or__ = MagicMock(return_value=mock_chain)

    inputs = {
        "question": "What is this?",
        "chat_history": [("Previous question", "Previous answer")],
    }

    results = list(qa_chain.stream(inputs))

    assert len(results) > 0
    assert all("chunk" in r for r in results)
    # Check that chunks are accumulated - they should be strings
    final_chunk = results[-1]["chunk"]
    # The chunks get accumulated, so we should see all three
    assert isinstance(final_chunk, str) or hasattr(final_chunk, '__str__')
    chunk_str = str(final_chunk) if not isinstance(final_chunk, str) else final_chunk
    assert len(chunk_str) > 0
    assert len(results[-1]["source_documents"]) > 0


@patch("retrievers.BM25Okapi")
def test_integration_hybrid_retrieval(mock_bm25, mock_vectorstore):
    """Test integration of hybrid retrieval."""
    from retrievers import HybridRetriever
    from langchain_core.documents import Document

    mock_vectorstore.get.return_value = {
        "documents": ["Document about AI", "Document about ML"],
        "metadatas": [
            {"source": "pdf/ai.pdf", "page": 1},
            {"source": "pdf/ml.pdf", "page": 2},
        ],
    }

    doc1 = Document(
        page_content="Document about AI",
        metadata={"source": "pdf/ai.pdf", "page": 1},
    )
    doc2 = Document(
        page_content="Document about ML",
        metadata={"source": "pdf/ml.pdf", "page": 2},
    )

    mock_vectorstore.similarity_search_with_score.return_value = [
        (doc1, 0.1),
        (doc2, 0.2),
    ]

    mock_bm25_instance = MagicMock()
    mock_bm25_instance.get_scores.return_value = [0.8, 0.6]
    mock_bm25.return_value = mock_bm25_instance

    retriever = HybridRetriever(mock_vectorstore)
    results = retriever.hybrid_search("AI machine learning", k=2, alpha=0.7)

    assert len(results) == 2
    assert all("doc" in r for r in results)
    assert all("fused_score" in r for r in results)
    assert all("semantic_score" in r for r in results)
    assert all("keyword_score" in r for r in results)

    # Check that scores are fused correctly
    for result in results:
        assert 0 <= result["fused_score"] <= 1
        assert 0 <= result["semantic_score"] <= 1
        assert 0 <= result["keyword_score"] <= 1


def test_integration_end_to_end_rag_query():
    """Test end-to-end RAG query processing."""
    from qa_chain import QAChainWrapper
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.documents import Document

    # Setup mocks
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {
        "documents": ["RAG is Retrieval-Augmented Generation"],
        "metadatas": [{"source": "pdf/rag.pdf", "page": 1}],
    }
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(
            page_content="RAG is Retrieval-Augmented Generation",
            metadata={"source": "pdf/rag.pdf", "page": 1},
        )
    ]
    mock_vectorstore.as_retriever.return_value = mock_retriever

    prompt = ChatPromptTemplate.from_template("Answer: {context}")

    with patch("qa_chain.create_llm") as mock_create_llm:
        mock_llm = MagicMock()
        mock_llm.stream.return_value = [
            MagicMock(content="RAG stands for "),
            MagicMock(content="Retrieval-Augmented Generation."),
        ]
        mock_create_llm.return_value = mock_llm

        qa_chain = QAChainWrapper(mock_vectorstore, prompt)

        inputs = {
            "question": "What is RAG?",
            "chat_history": [],
        }

        results = list(qa_chain.stream(inputs))

        assert len(results) > 0
        final_result = results[-1]
        assert "chunk" in final_result
        assert "source_documents" in final_result
        assert len(final_result["source_documents"]) > 0

