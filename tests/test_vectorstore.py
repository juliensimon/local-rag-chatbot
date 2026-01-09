"""Tests for vectorstore module."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from vectorstore import (
    create_new_vectorstore,
    filter_metadata,
    get_pdf_files,
    get_text_splitter,
    handle_existing_vectorstore,
    load_or_create_vectorstore,
    process_documents,
    update_vectorstore,
)


def test_get_text_splitter():
    """Test text splitter creation."""
    splitter = get_text_splitter()
    assert splitter._chunk_size == 512
    assert splitter._chunk_overlap == 128


def test_get_pdf_files_nonexistent_dir(tmp_path):
    """Test getting PDF files from non-existent directory."""
    with patch("vectorstore.PDF_PATH", str(tmp_path / "nonexistent")):
        files = get_pdf_files()
        assert files == []


def test_get_pdf_files_empty_dir(tmp_path):
    """Test getting PDF files from empty directory."""
    pdf_dir = tmp_path / "pdf"
    pdf_dir.mkdir()
    with patch("vectorstore.PDF_PATH", str(pdf_dir)):
        files = get_pdf_files()
        assert files == []


def test_filter_metadata_keep():
    """Test metadata filter keeps valid documents."""
    doc = Mock(metadata={"section": "introduction"})
    assert filter_metadata(doc) is True


def test_filter_metadata_skip_references():
    """Test metadata filter skips references section."""
    doc = Mock(metadata={"section": "references"})
    assert filter_metadata(doc) is False


def test_filter_metadata_skip_acknowledgments():
    """Test metadata filter skips acknowledgments section."""
    doc = Mock(metadata={"section": "acknowledgments"})
    assert filter_metadata(doc) is False


def test_filter_metadata_skip_appendix():
    """Test metadata filter skips appendix section."""
    doc = Mock(metadata={"section": "appendix"})
    assert filter_metadata(doc) is False


def test_filter_metadata_no_section():
    """Test metadata filter with no section."""
    doc = Mock(metadata={})
    assert filter_metadata(doc) is True


def test_process_documents(sample_documents):
    """Test document processing."""
    splitter = get_text_splitter()
    result = process_documents(sample_documents, splitter)
    # Should filter out references document
    assert len(result) < len(sample_documents)
    assert all(not ("references" in doc.metadata.get("section", "").lower()) for doc in result)


@patch("vectorstore.Chroma")
@patch("vectorstore.os.path.exists")
def test_load_or_create_vectorstore_existing(mock_exists, mock_chroma, mock_embeddings):
    """Test loading existing vectorstore."""
    mock_exists.return_value = True
    mock_vectorstore = MagicMock()
    mock_chroma.return_value = mock_vectorstore

    with patch("vectorstore.handle_existing_vectorstore") as mock_handle:
        mock_handle.return_value = mock_vectorstore
        result = load_or_create_vectorstore(mock_embeddings)
        assert result == mock_vectorstore
        mock_handle.assert_called_once_with(mock_embeddings)


@patch("vectorstore.Chroma")
@patch("vectorstore.os.path.exists")
def test_load_or_create_vectorstore_new(mock_exists, mock_chroma, mock_embeddings):
    """Test creating new vectorstore."""
    mock_exists.return_value = False
    mock_vectorstore = MagicMock()
    mock_chroma.from_documents.return_value = mock_vectorstore

    with patch("vectorstore.create_new_vectorstore") as mock_create:
        mock_create.return_value = mock_vectorstore
        result = load_or_create_vectorstore(mock_embeddings)
        assert result == mock_vectorstore
        mock_create.assert_called_once_with(mock_embeddings)


@patch("vectorstore.get_pdf_files")
@patch("vectorstore.Chroma")
def test_handle_existing_vectorstore(mock_chroma, mock_get_pdfs, mock_embeddings):
    """Test handling existing vectorstore."""
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {
        "metadatas": [
            {"source": "pdf/existing.pdf"},
            {"source": "pdf/another.pdf"},
        ]
    }
    mock_chroma.return_value = mock_vectorstore
    mock_get_pdfs.return_value = ["pdf/existing.pdf", "pdf/new.pdf"]

    with patch("vectorstore.update_vectorstore") as mock_update:
        result = handle_existing_vectorstore(mock_embeddings)
        assert result == mock_vectorstore
        mock_update.assert_called_once()


@patch("vectorstore.get_pdf_files")
@patch("vectorstore.Chroma")
def test_handle_existing_vectorstore_no_new_files(mock_chroma, mock_get_pdfs, mock_embeddings):
    """Test handling existing vectorstore with no new files."""
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {
        "metadatas": [{"source": "pdf/existing.pdf"}]
    }
    mock_chroma.return_value = mock_vectorstore
    mock_get_pdfs.return_value = ["pdf/existing.pdf"]

    with patch("vectorstore.update_vectorstore") as mock_update:
        result = handle_existing_vectorstore(mock_embeddings)
        assert result == mock_vectorstore
        mock_update.assert_not_called()


@patch("vectorstore.DirectoryLoader")
@patch("vectorstore.process_documents")
def test_update_vectorstore(mock_process, mock_loader, mock_vectorstore):
    """Test updating vectorstore with new documents."""
    mock_loader_instance = MagicMock()
    mock_loader.return_value = mock_loader_instance
    mock_loader_instance.load.return_value = [
        Mock(metadata={"source": "pdf/new.pdf"}),
    ]

    mock_docs = [Mock()]
    mock_process.return_value = mock_docs

    update_vectorstore(mock_vectorstore, ["pdf/new.pdf"], {"pdf/old.pdf"})
    mock_vectorstore.add_documents.assert_called_once_with(mock_docs)


@patch("vectorstore.get_pdf_files")
@patch("vectorstore.DirectoryLoader")
@patch("vectorstore.Chroma")
def test_create_new_vectorstore(mock_chroma, mock_loader, mock_get_pdfs, mock_embeddings):
    """Test creating new vectorstore."""
    mock_get_pdfs.return_value = ["pdf/test1.pdf", "pdf/test2.pdf"]
    mock_loader_instance = MagicMock()
    mock_loader.return_value = mock_loader_instance
    mock_loader_instance.load.return_value = [Mock()]

    mock_vectorstore = MagicMock()
    mock_chroma.from_documents.return_value = mock_vectorstore

    with patch("vectorstore.process_documents") as mock_process:
        mock_process.return_value = [Mock()]
        result = create_new_vectorstore(mock_embeddings)
        assert result == mock_vectorstore
        mock_chroma.from_documents.assert_called_once()


# Tests merged from test_vectorstore_edge_cases.py


@patch("vectorstore.get_pdf_files")
@patch("vectorstore.Chroma")
def test_handle_existing_vectorstore_no_pdfs(mock_chroma, mock_get_pdfs):
    """Test handle_existing_vectorstore with no PDF files."""
    mock_vectorstore = MagicMock()
    mock_chroma.return_value = mock_vectorstore
    mock_get_pdfs.return_value = []

    with pytest.raises(FileNotFoundError):
        handle_existing_vectorstore(MagicMock())


@patch("vectorstore.get_pdf_files")
@patch("vectorstore.Chroma")
def test_handle_existing_vectorstore_empty_collection(mock_chroma, mock_get_pdfs):
    """Test handle_existing_vectorstore with empty collection."""
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = None
    mock_chroma.return_value = mock_vectorstore
    mock_get_pdfs.return_value = ["pdf/test.pdf"]

    with patch("vectorstore.update_vectorstore") as mock_update:
        result = handle_existing_vectorstore(MagicMock())
        assert result == mock_vectorstore
        # Should still try to update with new PDFs
        mock_update.assert_called_once()


@patch("vectorstore.get_pdf_files")
@patch("vectorstore.Chroma")
def test_handle_existing_vectorstore_no_metadatas(mock_chroma, mock_get_pdfs):
    """Test handle_existing_vectorstore with no metadatas."""
    mock_vectorstore = MagicMock()
    mock_vectorstore.get.return_value = {"metadatas": None}
    mock_chroma.return_value = mock_vectorstore
    mock_get_pdfs.return_value = ["pdf/test.pdf"]

    with patch("vectorstore.update_vectorstore") as mock_update:
        result = handle_existing_vectorstore(MagicMock())
        assert result == mock_vectorstore
        mock_update.assert_called_once()


# Tests merged from test_vectorstore_remaining.py


@patch("vectorstore.get_pdf_files")
@patch("vectorstore.DirectoryLoader")
@patch("vectorstore.Chroma")
def test_create_new_vectorstore_no_pdfs(mock_chroma, mock_loader, mock_get_pdfs):
    """Test create_new_vectorstore with no PDF files (lines 157-159)."""
    mock_get_pdfs.return_value = []

    with pytest.raises(FileNotFoundError):
        create_new_vectorstore(MagicMock())



