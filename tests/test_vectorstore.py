"""Tests for vectorstore module."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from config import DEFAULT_COLLECTION_NAME, PDF_PATH
from vectorstore import (
    create_new_vectorstore,
    filter_metadata,
    get_pdf_files,
    get_text_splitter,
    get_vectorstore_sources,
    handle_existing_vectorstore,
    load_or_create_vectorstore,
    process_documents,
    update_vectorstore,
    user_paths,
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
        mock_handle.assert_called_once_with(mock_embeddings, PDF_PATH, DEFAULT_COLLECTION_NAME)


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
        mock_create.assert_called_once_with(mock_embeddings, PDF_PATH, DEFAULT_COLLECTION_NAME)


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


# Per-user collections


def test_user_paths_are_isolated():
    alice_dir, alice_coll = user_paths("alice")
    bob_dir, bob_coll = user_paths("bob")
    assert alice_dir != bob_dir
    assert alice_coll != bob_coll


def test_user_corpus_is_outside_shared_corpus():
    # The shared corpus loads PDF_PATH with a recursive glob, so any user
    # directory beneath it would leak into the shared index.
    user_dir = os.path.abspath(user_paths("alice")[0])
    shared_dir = os.path.abspath(PDF_PATH)
    assert os.path.commonpath([user_dir, shared_dir]) != shared_dir


def test_user_collection_never_collides_with_shared_collection():
    assert user_paths(DEFAULT_COLLECTION_NAME)[1] != DEFAULT_COLLECTION_NAME


@pytest.mark.parametrize("user_id", [None, "", "../x", "a/b", "_a", "a_", "a" * 65])
def test_user_paths_rejects_unsafe_ids(user_id):
    with pytest.raises(ValueError):
        user_paths(user_id)


@patch("vectorstore.update_vectorstore")
@patch("vectorstore.get_pdf_files")
@patch("vectorstore.Chroma")
def test_existing_store_opens_the_requested_collection(mock_chroma, mock_get_pdfs, mock_update):
    mock_chroma.return_value.get.return_value = {"metadatas": []}
    mock_get_pdfs.return_value = ["pdf_users/alice/x.pdf"]

    handle_existing_vectorstore("emb", "pdf_users/alice", "user_alice")

    assert mock_chroma.call_args.kwargs["collection_name"] == "user_alice"
    mock_get_pdfs.assert_called_once_with("pdf_users/alice")
    # A new collection in an existing store is filled from the user's directory
    mock_update.assert_called_once_with(
        mock_chroma.return_value, ["pdf_users/alice/x.pdf"], set(), "pdf_users/alice"
    )


@patch("vectorstore.process_documents", return_value=[Mock()])
@patch("vectorstore.get_pdf_files", return_value=["pdf_users/alice/x.pdf"])
@patch("vectorstore.DirectoryLoader")
@patch("vectorstore.Chroma")
def test_new_store_indexes_the_users_directory_into_their_collection(
    mock_chroma, mock_loader, _get_pdfs, _process
):
    create_new_vectorstore("emb", "pdf_users/alice", "user_alice")

    assert mock_loader.call_args.args[0] == "pdf_users/alice"
    assert mock_chroma.from_documents.call_args.kwargs["collection_name"] == "user_alice"


def test_get_vectorstore_sources():
    store = MagicMock()
    store.get.return_value = {
        "metadatas": [{"source": "d/b.pdf"}, {"source": "d/a.pdf"}, {"source": "d/a.pdf"}, {}, None]
    }
    assert get_vectorstore_sources(store) == ["a.pdf", "b.pdf"]


@pytest.mark.parametrize("result", [None, {"metadatas": []}, {"metadatas": None}])
def test_get_vectorstore_sources_empty(result):
    store = MagicMock()
    store.get.return_value = result
    assert get_vectorstore_sources(store) == []

