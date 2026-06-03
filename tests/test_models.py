"""Tests for models module."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from models import create_embeddings, create_llm, warn_if_reasoning_model


def _mock_openai_returning(message):
    """Build a patched openai.OpenAI whose chat completion returns `message`."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=message)]
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return MagicMock(return_value=client)


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


def test_warn_reasoning_model_via_reasoning_content(caplog):
    """A reasoning model (reasoning_content populated) triggers the loud warning."""
    message = MagicMock(reasoning_content="step 1: think...", content="")
    with patch("openai.OpenAI", _mock_openai_returning(message)):
        with caplog.at_level(logging.ERROR):
            warn_if_reasoning_model()
    assert "--reasoning off" in caplog.text


def test_warn_reasoning_model_via_think_tags(caplog):
    """Thinking left inline as <think> tags also triggers the warning."""
    message = MagicMock(reasoning_content=None, content="<think>hmm</think> ok")
    with patch("openai.OpenAI", _mock_openai_returning(message)):
        with caplog.at_level(logging.ERROR):
            warn_if_reasoning_model()
    assert "--reasoning off" in caplog.text


def test_warn_reasoning_model_clean_model_silent(caplog):
    """A normal instruct model (clean content, no reasoning) stays silent."""
    message = MagicMock(reasoning_content=None, content="ok")
    with patch("openai.OpenAI", _mock_openai_returning(message)):
        with caplog.at_level(logging.ERROR):
            warn_if_reasoning_model()
    assert "--reasoning off" not in caplog.text


def test_warn_reasoning_model_probe_failure_swallowed(caplog):
    """A probe failure logs a warning and never raises."""
    factory = MagicMock()
    factory.return_value.chat.completions.create.side_effect = RuntimeError("no server")
    with patch("openai.OpenAI", factory):
        with caplog.at_level(logging.WARNING):
            warn_if_reasoning_model()  # must not raise
    assert "probe failed" in caplog.text.lower()



