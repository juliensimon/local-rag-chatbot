"""Retrieval strategies for document search."""

import logging
import re

from langchain_core.documents import Document

logger = logging.getLogger(__name__)
from rank_bm25 import BM25Okapi

from config import (
    HYBRID_ALPHA_DEFAULT,
    HYBRID_INITIAL_K,
    RETRIEVER_K,
)


class HybridRetriever:
    """Combines semantic (vector) and keyword (BM25) search for hybrid retrieval."""

    def __init__(self, vectorstore):
        """Initialize hybrid retriever with BM25 index.

        Args:
            vectorstore: Chroma vectorstore instance
        """
        self._vectorstore = vectorstore
        self._bm25 = None
        self._documents = None

    def _tokenize(self, text):
        """Simple tokenization for BM25.

        Args:
            text: Text to tokenize

        Returns:
            List[str]: Tokenized words
        """
        return re.findall(r"\b\w+\b", text.lower())

    def _build_bm25_index(self):
        """Build BM25 index from all vectorstore documents (build once, reuse)."""
        if self._bm25 is not None:
            return  # Already built

        # Get all documents from vectorstore
        collection = self._vectorstore.get()
        if not collection or not collection.get("documents"):
            self._documents = []
            self._bm25 = None
            return

        all_docs = []
        documents = collection.get("documents", [])
        metadatas = collection.get("metadatas", [])

        for i, (text, metadata) in enumerate(zip(documents, metadatas)):
            doc = Document(page_content=text, metadata=metadata if metadata else {})
            all_docs.append(doc)

        self._documents = all_docs

        # Build BM25 index
        tokenized_docs = [self._tokenize(doc.page_content) for doc in all_docs]
        if tokenized_docs:
            self._bm25 = BM25Okapi(tokenized_docs)
        else:
            self._bm25 = None

    def _matches_filter(self, doc, metadata_filter):
        """Check if document matches metadata filter.

        Args:
            doc: Document to check
            metadata_filter: Filter dictionary

        Returns:
            bool: True if document matches filter
        """
        if not metadata_filter:
            return True

        for key, condition in metadata_filter.items():
            if key not in doc.metadata:
                return False
            # Simple filter matching
            if isinstance(condition, dict):
                op = list(condition.keys())[0]
                value = condition[op]
                if op == "$eq" and doc.metadata[key] != value:
                    return False
                elif op == "$in" and doc.metadata[key] not in value:
                    return False
        return True

    def _normalize_scores(self, scores):
        """Normalize scores to 0-1 range using min-max scaling.

        Args:
            scores: List or array of raw scores

        Returns:
            List of normalized scores (0-1 range)
        """
        if len(scores) == 0:
            return []
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [0.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def hybrid_search(
        self, query, k=RETRIEVER_K, alpha=HYBRID_ALPHA_DEFAULT, metadata_filter=None
    ):
        """Perform hybrid search combining semantic and keyword search.

        Args:
            query: User's question
            k: Number of documents to return
            alpha: Weight for semantic search (0.0 = pure keyword, 1.0 = pure semantic)
            metadata_filter: Optional metadata filter

        Returns:
            List of dictionaries with 'doc', 'fused_score', 'semantic_score', 'keyword_score'
        """
        # Build BM25 index once (lazy initialization)
        self._build_bm25_index()

        if not self._documents:
            return []

        # Semantic search (vector)
        semantic_k = min(HYBRID_INITIAL_K, len(self._documents))
        try:
            semantic_docs = self._vectorstore.similarity_search_with_score(
                query, k=semantic_k, filter=metadata_filter if metadata_filter else None
            )
        except Exception as e:
            logger.warning(f"Semantic search failed, falling back to keyword only: {e}")
            semantic_docs = []

        # Keyword search (BM25)
        query_tokens = self._tokenize(query)
        if self._bm25:
            bm25_scores = self._bm25.get_scores(query_tokens)
        else:
            bm25_scores = [0.0] * len(self._documents)

        # Normalize BM25 scores to 0-1 range
        bm25_scores = self._normalize_scores(bm25_scores)

        # Create document score map (match by content and metadata, not object ID)
        doc_scores = {}

        # Helper to create a unique key for a document
        def doc_key(doc):
            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page", "")
            content_start = doc.page_content[:50]  # First 50 chars for matching
            return f"{source}:{page}:{content_start}"

        # Add semantic scores (convert distance to similarity: lower distance = higher score)
        for doc, distance in semantic_docs:
            key = doc_key(doc)
            # Convert distance to similarity score (0-1 range)
            semantic_score = 1.0 / (1.0 + distance) if distance >= 0 else 1.0
            doc_scores[key] = {"doc": doc, "semantic": semantic_score, "keyword": 0.0}

        # Add keyword scores - match BM25 documents with semantic documents
        # Only include documents that match the filter
        for i, bm25_doc in enumerate(self._documents):
            # Apply metadata filter to BM25 results
            if not self._matches_filter(bm25_doc, metadata_filter):
                continue

            key = doc_key(bm25_doc)
            keyword_score = bm25_scores[i] if i < len(bm25_scores) else 0.0

            if key not in doc_scores:
                # New document from BM25 (not found in semantic results)
                doc_scores[key] = {"doc": bm25_doc, "semantic": 0.0, "keyword": 0.0}
            doc_scores[key]["keyword"] = keyword_score

        # Fuse scores
        fused_results = []
        for doc_id, scores in doc_scores.items():
            fused_score = alpha * scores["semantic"] + (1 - alpha) * scores["keyword"]
            fused_results.append(
                {
                    "doc": scores["doc"],
                    "fused_score": fused_score,
                    "semantic_score": scores["semantic"],
                    "keyword_score": scores["keyword"],
                }
            )

        # Sort by fused score and return top-k
        fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
        return fused_results[:k]

