"""Tests for the per-user collection registry."""

from unittest.mock import MagicMock, patch

import pytest

from api.registry import (
    Collection,
    CollectionRegistry,
    UnknownUserError,
    build_collection,
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


@patch("api.registry.build_collection")
def test_create_registry_loads_shared_corpus(mock_build):
    registry = create_registry("emb")
    mock_build.assert_called_once_with("emb", PDF_PATH, DEFAULT_COLLECTION_NAME)
    assert registry.get(None) is mock_build.return_value
