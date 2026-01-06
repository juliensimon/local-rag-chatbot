"""Tests for models module."""

from unittest.mock import MagicMock, patch

import pytest

from models import create_embeddings, create_llm


@patch("models.ChatOpenAI")
def test_create_llm(mock_chat_openai):
    """Test LLM creation."""
    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance

    result = create_llm(streaming=False)
    mock_chat_openai.assert_called_once()
    assert result == mock_instance


@patch("models.ChatOpenAI")
def test_create_llm_streaming(mock_chat_openai):
    """Test LLM creation with streaming enabled."""
    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance

    result = create_llm(streaming=True)
    call_kwargs = mock_chat_openai.call_args[1]
    assert call_kwargs["streaming"] is True
    assert result == mock_instance


@patch("models.HuggingFaceEmbeddings")
def test_create_embeddings(mock_embeddings):
    """Test embeddings creation."""
    mock_instance = MagicMock()
    mock_embeddings.return_value = mock_instance

    result = create_embeddings()
    mock_embeddings.assert_called_once()
    call_kwargs = mock_embeddings.call_args[1]
    assert call_kwargs["model_kwargs"]["device"] == "cpu"
    assert call_kwargs["encode_kwargs"]["normalize_embeddings"] is True
    assert result == mock_instance

