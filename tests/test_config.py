"""Tests for config module."""

import os

import pytest

from config import (
    CHAT_HISTORY_LIMIT,
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    HYBRID_ALPHA_DEFAULT,
    MMR_LAMBDA,
    PDF_PATH,
    RAG_PROMPT_TEMPLATE,
    RETRIEVER_FETCH_K,
    RETRIEVER_K,
    RERANKER_MODEL,
)


def test_config_constants():
    """Test that configuration constants are set."""
    assert RETRIEVER_K == 5
    assert RETRIEVER_FETCH_K == 15
    assert MMR_LAMBDA == 0.7
    assert CHAT_HISTORY_LIMIT == 5
    assert HYBRID_ALPHA_DEFAULT == 0.7
    assert CHUNK_SIZE == 512
    assert CHUNK_OVERLAP == 128
    assert EMBEDDING_MODEL_NAME == "BAAI/bge-small-en-v1.5"
    assert EMBEDDING_DEVICE == "cpu"
    assert RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_config_paths():
    """Test that path constants are strings."""
    assert isinstance(CHROMA_PATH, str)
    assert isinstance(PDF_PATH, str)


def test_rag_prompt_template():
    """Test that RAG prompt template contains required placeholders."""
    assert "{context}" in RAG_PROMPT_TEMPLATE
    assert "{question}" in RAG_PROMPT_TEMPLATE
    assert "{chat_history}" in RAG_PROMPT_TEMPLATE


def test_environment_variables():
    """Test that environment variables can be read."""
    # These should have defaults or be set
    assert isinstance(os.getenv("CHROMA_PATH", "vectorstore"), str)
    assert isinstance(os.getenv("PDF_PATH", "pdf"), str)



