"""Additional edge case tests for vectorstore module."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from vectorstore import handle_existing_vectorstore


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



