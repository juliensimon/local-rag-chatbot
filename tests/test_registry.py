"""Tests for the per-user collection registry."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from api.registry import (
    Collection,
    CollectionRegistry,
    UnknownUserError,
    build_collection,
    check_roots_disjoint,
    create_registry,
)
from config import DEFAULT_COLLECTION_NAME, PDF_PATH


@pytest.fixture
def user_root(tmp_path):
    with patch("vectorstore.USER_PDF_ROOT", str(tmp_path)):
        yield tmp_path


@pytest.fixture
def registry():
    shared = Collection(qa_chain=MagicMock(), pdf_path=PDF_PATH, sources=["shared.pdf"])
    return CollectionRegistry(embeddings="emb", shared=shared)


def test_no_user_means_shared_corpus(registry):
    assert registry.get(None).sources == ["shared.pdf"]


@patch("api.registry.build_collection")
def test_user_collection_is_built_from_the_users_directory(mock_build, registry, user_root):
    (user_root / "alice").mkdir()
    (user_root / "alice" / "doc.pdf").write_bytes(b"%PDF")

    assert registry.get("alice") is mock_build.return_value
    mock_build.assert_called_once_with("emb", str(user_root / "alice"), "user_alice")


@patch("api.registry.build_collection")
def test_user_collection_is_cached(mock_build, registry, user_root):
    # Indexing embeds every chunk; repeating it per request would make chat unusable
    (user_root / "alice").mkdir()
    (user_root / "alice" / "doc.pdf").write_bytes(b"%PDF")

    first = registry.get("alice")
    assert registry.get("alice") is first
    mock_build.assert_called_once()


@patch("api.registry.build_collection")
def test_unknown_user_does_not_create_a_directory(mock_build, registry, user_root):
    with pytest.raises(UnknownUserError):
        registry.get("bob")
    assert not (user_root / "bob").exists()
    mock_build.assert_not_called()


@patch("api.registry.build_collection")
def test_user_with_empty_directory_is_unknown(mock_build, registry, user_root):
    (user_root / "carol").mkdir()
    with pytest.raises(UnknownUserError):
        registry.get("carol")
    mock_build.assert_not_called()


def test_unsafe_user_id_is_rejected(registry, user_root):
    with pytest.raises(ValueError):
        registry.get("../pdf")


@patch("api.registry.create_qa_chain")
@patch("api.registry.load_or_create_vectorstore")
def test_build_collection(mock_load, mock_chain):
    mock_load.return_value.get.return_value = {
        "metadatas": [{"source": "dir/b.pdf"}, {"source": "dir/a.pdf"}]
    }
    collection = build_collection("emb", "dir", "coll")

    mock_load.assert_called_once_with("emb", "dir", "coll")
    mock_chain.assert_called_once_with(mock_load.return_value)
    assert collection == Collection(mock_chain.return_value, "dir", ["a.pdf", "b.pdf"])


def test_users_index_concurrently(registry, user_root):
    # A slow first index for one user must not block another user's first request
    for user in ("alice", "bob"):
        (user_root / user).mkdir()
        (user_root / user / "doc.pdf").write_bytes(b"%PDF")
    alice_indexing = threading.Event()
    bob_done = threading.Event()

    def build(embeddings, pdf_path, collection_name):
        if collection_name == "user_alice":
            alice_indexing.set()
            # Only returns if bob's load completes while alice is still indexing
            assert bob_done.wait(timeout=5)
        return MagicMock()

    errors = []

    def load_alice():
        try:
            registry.get("alice")
        except Exception as exc:  # surfaced below; thread exceptions are otherwise lost
            errors.append(exc)

    with patch("api.registry.build_collection", side_effect=build):
        alice = threading.Thread(target=load_alice)
        alice.start()
        assert alice_indexing.wait(timeout=5)
        registry.get("bob")
        bob_done.set()
        alice.join(timeout=5)

    assert not alice.is_alive()
    assert errors == []


def test_same_user_is_indexed_once_under_concurrency(registry, user_root):
    (user_root / "alice").mkdir()
    (user_root / "alice" / "doc.pdf").write_bytes(b"%PDF")
    started = threading.Event()
    release = threading.Event()

    def build(*args):
        started.set()
        release.wait(timeout=5)
        return MagicMock()

    with patch("api.registry.build_collection", side_effect=build) as mock_build:
        first = threading.Thread(target=registry.get, args=("alice",))
        second = threading.Thread(target=registry.get, args=("alice",))
        first.start()
        started.wait(timeout=5)
        second.start()
        release.set()
        first.join(timeout=5)
        second.join(timeout=5)

    mock_build.assert_called_once()


@pytest.mark.parametrize("user_root", ["pdf", "pdf/users"])
def test_user_root_inside_shared_root_is_rejected(tmp_path, user_root):
    # The shared corpus globs recursively, so it would ingest every user's PDFs
    with pytest.raises(ValueError):
        check_roots_disjoint(str(tmp_path / "pdf"), str(tmp_path / user_root))


@pytest.mark.parametrize("user_root", ["pdf_users", "other/pdf"])
def test_disjoint_roots_are_accepted(tmp_path, user_root):
    # "pdf_users" shares a string prefix with "pdf" but is a sibling directory
    check_roots_disjoint(str(tmp_path / "pdf"), str(tmp_path / user_root))


@patch("api.registry.build_collection")
def test_create_registry_refuses_overlapping_roots_before_indexing(mock_build):
    with patch("api.registry.USER_PDF_ROOT", f"{PDF_PATH}/users"):
        with pytest.raises(ValueError):
            create_registry("emb")
    mock_build.assert_not_called()


@patch("api.registry.build_collection")
def test_create_registry_loads_shared_corpus(mock_build):
    registry = create_registry("emb")
    mock_build.assert_called_once_with("emb", PDF_PATH, DEFAULT_COLLECTION_NAME)
    assert registry.get(None) is mock_build.return_value
