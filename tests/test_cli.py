"""Tests for cli.py entry point."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from cli import main


@patch("cli.create_llm")
@patch("cli.create_qa_chain")
@patch("cli.load_or_create_vectorstore")
@patch("cli.create_embeddings")
def test_cli_main_vanilla_llm(
    mock_create_embeddings,
    mock_load_vectorstore,
    mock_create_qa_chain,
    mock_create_llm,
    capsys,
):
    """Test CLI main function with vanilla LLM."""
    # Setup mocks
    mock_embeddings = MagicMock()
    mock_create_embeddings.return_value = mock_embeddings

    mock_vectorstore = MagicMock()
    mock_load_vectorstore.return_value = mock_vectorstore

    mock_qa_chain = MagicMock()
    mock_create_qa_chain.return_value = mock_qa_chain

    mock_llm = MagicMock()
    mock_llm.stream.return_value = [
        MagicMock(content="Hello "),
        MagicMock(content="World"),
    ]
    mock_create_llm.return_value = mock_llm

    # Run main
    main()

    # Check output
    captured = capsys.readouterr()
    assert "Vanilla Response" in captured.out
    assert "Hello World" in captured.out


@patch("cli.create_llm")
@patch("cli.create_qa_chain")
@patch("cli.load_or_create_vectorstore")
@patch("cli.create_embeddings")
def test_cli_main_rag_response(
    mock_create_embeddings,
    mock_load_vectorstore,
    mock_create_qa_chain,
    mock_create_llm,
    capsys,
    sample_documents,
):
    """Test CLI main function with RAG response."""
    # Setup mocks
    mock_embeddings = MagicMock()
    mock_create_embeddings.return_value = mock_embeddings

    mock_vectorstore = MagicMock()
    mock_load_vectorstore.return_value = mock_vectorstore

    mock_qa_chain = MagicMock()
    mock_qa_chain.stream.return_value = [
        {
            "chunk": "RAG ",
            "source_documents": sample_documents[:2],
            "docs_with_scores": None,
            "rewritten_query": None,
            "hybrid_scores": None,
        },
        {
            "chunk": "response",
            "source_documents": sample_documents[:2],
            "docs_with_scores": None,
            "rewritten_query": None,
            "hybrid_scores": None,
        },
    ]
    mock_create_qa_chain.return_value = mock_qa_chain

    mock_llm = MagicMock()
    mock_create_llm.return_value = mock_llm

    # Run main
    main()

    # Check output
    captured = capsys.readouterr()
    assert "RAG Response" in captured.out
    assert "RAG response" in captured.out
    assert "Sources:" in captured.out


@patch("cli.create_llm")
@patch("cli.create_qa_chain")
@patch("cli.load_or_create_vectorstore")
@patch("cli.create_embeddings")
def test_cli_main_with_filter(
    mock_create_embeddings,
    mock_load_vectorstore,
    mock_create_qa_chain,
    mock_create_llm,
    capsys,
    sample_documents,
):
    """Test CLI main function with metadata filter."""
    # Setup mocks
    mock_embeddings = MagicMock()
    mock_create_embeddings.return_value = mock_embeddings

    mock_vectorstore = MagicMock()
    mock_load_vectorstore.return_value = mock_vectorstore

    mock_qa_chain = MagicMock()
    mock_qa_chain.stream.return_value = [
        {
            "chunk": "Filtered ",
            "source_documents": sample_documents[:1],
            "docs_with_scores": None,
            "rewritten_query": None,
            "hybrid_scores": None,
        },
        {
            "chunk": "response",
            "source_documents": sample_documents[:1],
            "docs_with_scores": None,
            "rewritten_query": None,
            "hybrid_scores": None,
        },
    ]
    mock_create_qa_chain.return_value = mock_qa_chain

    mock_llm = MagicMock()
    mock_create_llm.return_value = mock_llm

    # Run main
    main()

    # Check output
    captured = capsys.readouterr()
    assert "Metadata Filter" in captured.out
    assert "Sources (filtered):" in captured.out


@patch("cli.create_llm")
@patch("cli.create_qa_chain")
@patch("cli.load_or_create_vectorstore")
@patch("cli.create_embeddings")
def test_cli_main_vanilla_error(
    mock_create_embeddings,
    mock_load_vectorstore,
    mock_create_qa_chain,
    mock_create_llm,
    capsys,
):
    """Test CLI main function with vanilla LLM error."""
    # Setup mocks
    mock_embeddings = MagicMock()
    mock_create_embeddings.return_value = mock_embeddings

    mock_vectorstore = MagicMock()
    mock_load_vectorstore.return_value = mock_vectorstore

    mock_qa_chain = MagicMock()
    mock_create_qa_chain.return_value = mock_qa_chain

    mock_llm = MagicMock()
    mock_llm.stream.side_effect = Exception("LLM Error")
    mock_create_llm.return_value = mock_llm

    # Run main
    main()

    # Check error handling
    captured = capsys.readouterr()
    assert "Error" in captured.out


@patch("cli.create_llm")
@patch("cli.create_qa_chain")
@patch("cli.load_or_create_vectorstore")
@patch("cli.create_embeddings")
def test_cli_main_rag_error(
    mock_create_embeddings,
    mock_load_vectorstore,
    mock_create_qa_chain,
    mock_create_llm,
    capsys,
):
    """Test CLI main function with RAG error."""
    # Setup mocks
    mock_embeddings = MagicMock()
    mock_create_embeddings.return_value = mock_embeddings

    mock_vectorstore = MagicMock()
    mock_load_vectorstore.return_value = mock_vectorstore

    mock_qa_chain = MagicMock()
    mock_qa_chain.stream.side_effect = Exception("RAG Error")
    mock_create_qa_chain.return_value = mock_qa_chain

    mock_llm = MagicMock()
    mock_create_llm.return_value = mock_llm

    # Run main
    main()

    # Check error handling
    captured = capsys.readouterr()
    assert "Error" in captured.out



