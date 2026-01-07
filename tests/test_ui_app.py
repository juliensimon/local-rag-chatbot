"""Tests for ui/app.py module."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ui.app import create_app, initialize_chain


@patch("ui.app.create_qa_chain")
@patch("ui.app.load_or_create_vectorstore")
@patch("ui.app.create_embeddings")
def test_initialize_chain(mock_create_embeddings, mock_load_vectorstore, mock_create_qa_chain):
    """Test initialize_chain function."""
    # Setup mocks
    mock_embeddings = MagicMock()
    mock_create_embeddings.return_value = mock_embeddings

    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {
        "metadatas": [
            {"source": "pdf/test1.pdf"},
            {"source": "pdf/test2.pdf"},
        ]
    }
    mock_load_vectorstore.return_value = mock_vectorstore

    mock_qa_chain = MagicMock()
    mock_create_qa_chain.return_value = mock_qa_chain

    # Test
    chain, sources = initialize_chain()

    assert chain == mock_qa_chain
    assert len(sources) == 2
    assert "test1.pdf" in sources
    assert "test2.pdf" in sources


@patch("ui.app.create_qa_chain")
@patch("ui.app.load_or_create_vectorstore")
@patch("ui.app.create_embeddings")
def test_initialize_chain_empty_metadatas(
    mock_create_embeddings, mock_load_vectorstore, mock_create_qa_chain
):
    """Test initialize_chain with empty metadatas."""
    # Setup mocks
    mock_embeddings = MagicMock()
    mock_create_embeddings.return_value = mock_embeddings

    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {"metadatas": []}
    mock_load_vectorstore.return_value = mock_vectorstore

    mock_qa_chain = MagicMock()
    mock_create_qa_chain.return_value = mock_qa_chain

    # Test
    chain, sources = initialize_chain()

    assert chain == mock_qa_chain
    assert sources == []


@patch("ui.app.create_qa_chain")
@patch("ui.app.load_or_create_vectorstore")
@patch("ui.app.create_embeddings")
def test_initialize_chain_no_collection(
    mock_create_embeddings, mock_load_vectorstore, mock_create_qa_chain
):
    """Test initialize_chain with no collection."""
    # Setup mocks
    mock_embeddings = MagicMock()
    mock_create_embeddings.return_value = mock_embeddings

    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = None
    mock_load_vectorstore.return_value = mock_vectorstore

    mock_qa_chain = MagicMock()
    mock_create_qa_chain.return_value = mock_qa_chain

    # Test
    chain, sources = initialize_chain()

    assert chain == mock_qa_chain
    assert sources == []


@patch("ui.app.update_hybrid_alpha_visibility")
@patch("ui.app.update_rag_controls")
@patch("ui.app.create_respond_handler")
@patch("ui.app.create_stream_chat_response")
@patch("ui.app.create_ui_components")
@patch("ui.app.initialize_chain")
def test_create_app(
    mock_initialize,
    mock_create_components,
    mock_create_stream,
    mock_create_respond,
    mock_update_rag,
    mock_update_hybrid,
):
    """Test create_app function."""
    # Setup mocks
    mock_qa_chain = MagicMock()
    mock_initialize.return_value = (mock_qa_chain, ["test1.pdf", "test2.pdf"])

    mock_components = {
        "demo": MagicMock(),
        "msg": MagicMock(),
        "chatbot": MagicMock(),
        "rag_enabled": MagicMock(),
        "search_type": MagicMock(),
        "doc_filter": MagicMock(),
        "query_rewriting": MagicMock(),
        "reranking": MagicMock(),
        "hybrid_alpha": MagicMock(),
        "context_box": MagicMock(),
        "search_col": MagicMock(),
        "filter_col": MagicMock(),
        "context_section": MagicMock(),
        "advanced_options": MagicMock(),
        "submit": MagicMock(),
        "clear": MagicMock(),
    }
    mock_create_components.return_value = mock_components

    mock_stream_fn = MagicMock()
    mock_create_stream.return_value = mock_stream_fn

    mock_respond_fn = MagicMock()
    mock_create_respond.return_value = mock_respond_fn

    # Test
    app = create_app()

    assert app == mock_components["demo"]
    mock_initialize.assert_called_once()
    mock_create_components.assert_called_once()
    mock_create_stream.assert_called_once_with(mock_qa_chain, ["test1.pdf", "test2.pdf"])
    mock_create_respond.assert_called_once_with(mock_stream_fn)

    # Check event handlers were attached
    assert mock_components["msg"].submit.called
    assert mock_components["submit"].click.called
    assert mock_components["clear"].click.called
    assert mock_components["rag_enabled"].change.called
    assert mock_components["search_type"].change.called

