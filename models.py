"""Model creation utilities for LLM and embeddings."""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from config import (
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_URL,
)


def create_llm(streaming=False):
    """Initialize the OpenAI-compatible language model.

    Args:
        streaming (bool): Whether to enable response streaming

    Returns:
        ChatOpenAI: Configured language model instance
    """
    return ChatOpenAI(
        model=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_URL,
        streaming=streaming,
    )


def create_embeddings():
    """Initialize the embedding model.

    Returns:
        HuggingFaceEmbeddings: Configured embedding model instance
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )



