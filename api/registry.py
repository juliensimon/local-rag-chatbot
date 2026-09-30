"""Per-user document collections, loaded lazily and cached for the process lifetime."""

import os
import threading
from dataclasses import dataclass
from typing import Optional

from config import DEFAULT_COLLECTION_NAME, PDF_PATH
from qa_chain import create_qa_chain
from vectorstore import (
    get_pdf_files,
    get_vectorstore_sources,
    load_or_create_vectorstore,
    user_paths,
)


class UnknownUserError(LookupError):
    """Raised when a user has no PDF corpus on disk."""


@dataclass
class Collection:
    """A loaded corpus: its QA chain, where its PDFs live and which files it holds."""

    qa_chain: object
    pdf_path: str
    sources: list[str]


def build_collection(embeddings, pdf_path: str, collection_name: str) -> Collection:
    """Load (or index) a corpus and wrap it in a QA chain."""
    vectorstore = load_or_create_vectorstore(embeddings, pdf_path, collection_name)
    return Collection(
        qa_chain=create_qa_chain(vectorstore),
        pdf_path=pdf_path,
        sources=get_vectorstore_sources(vectorstore),
    )


class CollectionRegistry:
    """Maps a user id to that user's collection; None maps to the shared corpus."""

    def __init__(self, embeddings, shared: Collection):
        self._embeddings = embeddings
        self._shared = shared
        self._users: dict[str, Collection] = {}
        self._lock = threading.Lock()

    def get(self, user_id: Optional[str]) -> Collection:
        """Return the collection for user_id, indexing it on first use.

        Raises:
            ValueError: If user_id is not a safe identifier
            UnknownUserError: If the user has no PDFs
        """
        if user_id is None:
            return self._shared
        collection = self._users.get(user_id)
        if collection is not None:
            return collection
        # Serialize first loads so two concurrent requests don't index the same corpus twice
        with self._lock:
            if user_id not in self._users:
                self._users[user_id] = self._load_user(user_id)
            return self._users[user_id]

    def _load_user(self, user_id: str) -> Collection:
        pdf_path, collection_name = user_paths(user_id)
        # Check isdir first: get_pdf_files() creates missing directories, and
        # arbitrary request input must not create folders on disk.
        if not os.path.isdir(pdf_path) or not get_pdf_files(pdf_path):
            raise UnknownUserError(user_id)
        return build_collection(self._embeddings, pdf_path, collection_name)


def create_registry(embeddings) -> CollectionRegistry:
    """Build a registry with the shared corpus loaded eagerly."""
    shared = build_collection(embeddings, PDF_PATH, DEFAULT_COLLECTION_NAME)
    return CollectionRegistry(embeddings, shared)
