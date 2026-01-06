"""Tests to cover remaining lines in vectorstore.py."""

from unittest.mock import MagicMock, patch

import pytest

from vectorstore import create_new_vectorstore


@patch("vectorstore.get_pdf_files")
@patch("vectorstore.DirectoryLoader")
@patch("vectorstore.Chroma")
def test_create_new_vectorstore_no_pdfs(mock_chroma, mock_loader, mock_get_pdfs):
    """Test create_new_vectorstore with no PDF files (lines 157-159)."""
    mock_get_pdfs.return_value = []

    with pytest.raises(SystemExit):
        create_new_vectorstore(MagicMock())

