"""Optional persistent-memory layer (mem0) — self-hosted and fully local.

Gated by the MEM0_ENABLED env flag so the app runs unchanged when it is off
(mem0 is only imported when enabled). When on, it reuses the local llama-server
as the LLM, the same bge embeddings, and a local Chroma store — nothing leaves
the machine. Per-request use is controlled by the `memory_enabled` request flag.

NOTE: mem0's fact-extraction prompt is large (~8k+ tokens). Run llama-server with
a wide context window (``-c 32768``); at the default 8192, extraction silently
fails with a 400 "exceeds context size" and nothing gets stored.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from config import (
    EMBEDDING_MODEL_NAME,
    MEM0_ENABLED,
    MEM0_PATH,
    MEM0_USER_ID,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_URL,
)

logger = logging.getLogger(__name__)

_memory = None
_init_failed = False

# Single background worker for writes: mem0.add() is a second LLM round-trip
# (fact extraction), so it runs off the request path. One worker serializes
# writes to avoid concurrent Chroma access.
_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mem0-remember")


def _get_memory():
    """Lazily build the mem0 Memory singleton, or None if disabled/unavailable."""
    global _memory, _init_failed
    if not MEM0_ENABLED or _init_failed:
        return None
    if _memory is not None:
        return _memory
    try:
        from mem0 import Memory

        _memory = Memory.from_config(
            {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": OPENAI_MODEL,
                        "openai_base_url": OPENAI_URL,
                        "api_key": OPENAI_API_KEY,
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {"model": EMBEDDING_MODEL_NAME},
                },
                "vector_store": {
                    "provider": "chroma",
                    "config": {"collection_name": "memories", "path": MEM0_PATH},
                },
            }
        )
        logger.info("mem0 memory layer initialized (path=%s)", MEM0_PATH)
        return _memory
    except Exception as e:
        logger.warning("mem0 unavailable — memory disabled: %s", e)
        _init_failed = True
        return None


def _facts(search_result):
    """Normalize a mem0 search result (list, or {'results': [...]}) to fact strings."""
    items = (
        search_result.get("results", [])
        if isinstance(search_result, dict)
        else search_result
    )
    return [it.get("memory", "") for it in (items or []) if it.get("memory")]


def recall(query, user_id=MEM0_USER_ID):
    """Return a prompt preamble of relevant remembered facts, or None.

    None means "no memory to inject" — either the layer is off, unavailable,
    or there is nothing relevant stored yet.
    """
    mem = _get_memory()
    if mem is None:
        return None
    try:
        facts = _facts(mem.search(query=query, filters={"user_id": user_id}))
    except Exception as e:
        logger.warning("mem0 recall failed: %s", e)
        return None
    if not facts:
        return None
    bullets = "\n".join(f"- {f}" for f in facts)
    return (
        "What you remember about this user from earlier conversations:\n"
        f"{bullets}\n"
        "Use these facts to personalize your answer when relevant."
    )


def _remember_sync(user_message, assistant_message, user_id):
    """The actual mem0 write (a second LLM call for fact extraction).

    Runs on the background worker, never on the request path. Best-effort: a
    failure is logged and swallowed — memory is auxiliary to answering.
    """
    mem = _get_memory()
    if mem is None:
        return
    try:
        mem.add(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ],
            user_id=user_id,
        )
    except Exception as e:
        logger.warning("mem0 remember failed: %s", e)


def remember(user_message, assistant_message, user_id=MEM0_USER_ID):
    """Queue the latest turn for extraction + storage, off the request path.

    Returns immediately: the mem0 add is a second LLM round-trip, so it runs on
    a background worker instead of blocking the chat response. Trade-off: a fact
    stated this turn may not be recalled until the write lands, so rapid
    same-session follow-ups can miss it. Returns the Future for callers/tests
    that want to await completion.
    """
    return _writer.submit(_remember_sync, user_message, assistant_message, user_id)
