"""Model creation utilities for LLM and embeddings."""

import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from config import (
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_URL,
)

logger = logging.getLogger(__name__)


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


def warn_if_reasoning_model():
    """Probe the LLM at startup and warn loudly if it is a reasoning model with
    thinking still enabled.

    Reasoning models (Qwen3, DeepSeek-R1, ...) route their thoughts into
    ``reasoning_content`` (or a ``<think>`` block) and leave ``content`` empty
    until thinking finishes. This app -- and mem0's memory extraction -- read
    ``content``, so thinking-on means empty/garbled chat answers and silently
    failing memory. The fix is operator-side: launch llama-server with
    ``--reasoning off``. This probe turns that silent failure into one actionable
    log line. Best-effort: a probe failure is logged and ignored.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_URL)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=32,
            temperature=0,
        )
        message = resp.choices[0].message
        thinking = bool(getattr(message, "reasoning_content", None)) or "<think>" in (
            message.content or ""
        )
    except Exception as e:
        logger.warning("Reasoning-model startup probe failed (skipping check): %s", e)
        return

    if thinking:
        logger.error(
            "Reasoning model detected with thinking ENABLED: chat answers will come "
            "back empty/garbled and mem0 memory extraction will fail silently. Restart "
            "llama-server with `--reasoning off`."
        )



