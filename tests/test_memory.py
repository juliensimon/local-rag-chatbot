"""Tests for the optional mem0 persistent-memory layer (memory.py).

mem0 itself is fully mocked — these tests verify *our* wiring and the
intent behind it: the layer stays off unless enabled, recall/remember
degrade safely on failure (memory is auxiliary, never break the chat),
and the mem0 2.0.x API contract (search via `filters=`) is honored.
"""

from unittest.mock import MagicMock, patch

import pytest

import memory


@pytest.fixture(autouse=True)
def reset_memory_state():
    """Each test starts from a clean module state (singleton + latch reset)."""
    memory._memory = None
    memory._init_failed = False
    yield
    memory._memory = None
    memory._init_failed = False


# --------------------------------------------------------------- _facts

def test_facts_list_shape():
    """A plain list of memory dicts yields their 'memory' strings."""
    res = [{"memory": "likes EVs"}, {"memory": "skip oil"}]
    assert memory._facts(res) == ["likes EVs", "skip oil"]


def test_facts_results_dict_shape():
    """The v1.1 {'results': [...]} shape is unwrapped to the same list."""
    res = {"results": [{"memory": "a"}, {"memory": "b"}]}
    assert memory._facts(res) == ["a", "b"]


def test_facts_empty_inputs():
    """Empty list / empty results / None all yield no facts (no crash)."""
    assert memory._facts([]) == []
    assert memory._facts({"results": []}) == []
    assert memory._facts(None) == []


def test_facts_skips_items_without_memory():
    """Items missing a 'memory' field (or with an empty one) are dropped."""
    res = [{"memory": "keep"}, {"id": "x"}, {"memory": ""}]
    assert memory._facts(res) == ["keep"]


# ----------------------------------------------------------- _get_memory

def test_get_memory_disabled_returns_none():
    """Feature flag off: no Memory is built (mem0 is never even imported)."""
    with patch("memory.MEM0_ENABLED", False):
        assert memory._get_memory() is None


def test_get_memory_returns_cached_instance():
    """An already-built Memory is reused, not rebuilt."""
    sentinel = object()
    memory._memory = sentinel
    with patch("memory.MEM0_ENABLED", True):
        assert memory._get_memory() is sentinel


def test_get_memory_builds_and_caches():
    """When enabled, Memory.from_config is called once and the result cached."""
    fake_mem = MagicMock()
    with patch("memory.MEM0_ENABLED", True), patch(
        "mem0.Memory.from_config", return_value=fake_mem
    ) as mock_build:
        assert memory._get_memory() is fake_mem
        assert memory._get_memory() is fake_mem  # second call hits the cache
        mock_build.assert_called_once()


def test_get_memory_init_failure_latches():
    """A build failure is swallowed, returns None, and is NOT retried."""
    with patch("memory.MEM0_ENABLED", True), patch(
        "mem0.Memory.from_config", side_effect=RuntimeError("boom")
    ) as mock_build:
        assert memory._get_memory() is None
        assert memory._get_memory() is None  # latched
        assert memory._init_failed is True
        mock_build.assert_called_once()


# --------------------------------------------------------------- recall

def test_recall_none_when_memory_unavailable():
    """recall returns None (nothing to inject) when the layer is off/unavailable."""
    with patch("memory._get_memory", return_value=None):
        assert memory.recall("anything") is None


def test_recall_formats_found_facts():
    """recall returns a preamble with the facts, scoped by the user_id filter."""
    fake_mem = MagicMock()
    fake_mem.search.return_value = {
        "results": [{"memory": "likes EVs"}, {"memory": "skip oil"}]
    }
    with patch("memory._get_memory", return_value=fake_mem):
        out = memory.recall("what do I like?")

    assert out is not None
    assert "likes EVs" in out and "skip oil" in out
    # mem0 2.0.x: user scoping is via filters=, NOT user_id=
    _, kwargs = fake_mem.search.call_args
    assert kwargs["filters"] == {"user_id": memory.MEM0_USER_ID}


def test_recall_none_when_no_facts():
    """No stored facts → nothing to inject → None (not an empty preamble)."""
    fake_mem = MagicMock()
    fake_mem.search.return_value = []
    with patch("memory._get_memory", return_value=fake_mem):
        assert memory.recall("q") is None


def test_recall_swallows_search_error():
    """A search failure degrades to None rather than raising into the request."""
    fake_mem = MagicMock()
    fake_mem.search.side_effect = RuntimeError("db down")
    with patch("memory._get_memory", return_value=fake_mem):
        assert memory.recall("q") is None


# -------------------------------------------------------------- remember

def test_remember_sync_noop_when_unavailable():
    """The write worker is a safe no-op when the layer is off."""
    with patch("memory._get_memory", return_value=None):
        memory._remember_sync("hi", "hello", "julien")  # must not raise


def test_remember_sync_adds_turn_with_user_id():
    """The write worker stores the user+assistant turn under the given user_id."""
    fake_mem = MagicMock()
    with patch("memory._get_memory", return_value=fake_mem):
        memory._remember_sync("hi", "hello", "julien")

    args, kwargs = fake_mem.add.call_args
    assert args[0] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert kwargs["user_id"] == "julien"


def test_remember_sync_swallows_add_error():
    """An add failure must never propagate (it runs off the request path)."""
    fake_mem = MagicMock()
    fake_mem.add.side_effect = RuntimeError("boom")
    with patch("memory._get_memory", return_value=fake_mem):
        memory._remember_sync("hi", "hello", "julien")  # must not raise


def test_remember_runs_in_background_and_persists():
    """remember() returns immediately; the write completes on the worker thread."""
    fake_mem = MagicMock()
    with patch("memory._get_memory", return_value=fake_mem):
        future = memory.remember("hi", "hello")
        future.result(timeout=5)  # wait for the background write to land
    fake_mem.add.assert_called_once()
    assert fake_mem.add.call_args.kwargs["user_id"] == memory.MEM0_USER_ID


# ----------------------------------------------- prompt-injection wiring

def test_stream_vanilla_injects_memories_into_system_prompt():
    """Recalled memories are prepended to the vanilla system prompt."""
    from api import streaming

    chunk = MagicMock(content="hi")
    llm = MagicMock()
    llm.stream.return_value = [chunk]

    list(
        streaming.stream_vanilla_response(
            llm, "q", memories="REMEMBERED FACTS", remember_turn=False
        )
    )

    system_msg = llm.stream.call_args[0][0][0]  # first message of the call
    assert "REMEMBERED FACTS" in system_msg.content


def test_stream_vanilla_remembers_after_completion():
    """With remember_turn set, the full turn is persisted after streaming finishes."""
    from api import streaming

    chunk = MagicMock(content="abc")
    llm = MagicMock()
    llm.stream.return_value = [chunk]

    with patch("memory.remember") as mock_remember:
        list(
            streaming.stream_vanilla_response(
                llm, "user msg", memories=None, remember_turn=True
            )
        )

    mock_remember.assert_called_once_with("user msg", "abc")


def test_stream_rag_passes_memories_and_remembers():
    """RAG streaming forwards memories into the chain and persists the full turn."""
    from api import streaming

    qa = MagicMock()
    qa.stream.return_value = [
        {"chunk": "Hello ", "source_documents": [], "docs_with_scores": None,
         "rewritten_query": None, "hybrid_scores": None},
        {"chunk": "world", "source_documents": [], "docs_with_scores": None,
         "rewritten_query": None, "hybrid_scores": None},
    ]

    with patch("memory.remember") as mock_remember:
        events = list(
            streaming.stream_rag_response(
                qa, "q", [], "mmr", None, False, False, 0.7,
                memories="REMEMBERED", remember_turn=True,
            )
        )

    # memories are threaded into the chain inputs...
    stream_input = qa.stream.call_args[0][0]
    assert stream_input["memories"] == "REMEMBERED"
    # ...and the accumulated answer is persisted once the stream is done.
    mock_remember.assert_called_once_with("q", "Hello world")
    assert any("Hello " in e for e in events)
