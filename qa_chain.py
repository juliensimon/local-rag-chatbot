"""Question-answering chain with RAG capabilities."""

import logging
import os

from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Sentinel to indicate reranker load failed (prevent retry loop)
_RERANKER_LOAD_FAILED = object()

from config import (
    HYBRID_ALPHA_DEFAULT,
    MMR_LAMBDA,
    RAG_PROMPT_TEMPLATE,
    RERANK_INITIAL_K,
    RERANK_TOP_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_K,
)
from models import create_llm
from retrievers import HybridRetriever
from utils import format_chat_history


class QAChainWrapper:
    """Wrapper for RAG question-answering with streaming support and metadata filtering."""

    def __init__(self, vectorstore, prompt):
        """Initialize the QA chain wrapper.

        Args:
            vectorstore: Chroma vectorstore instance
            prompt: ChatPromptTemplate for generating responses
        """
        self._vectorstore = vectorstore
        self._prompt = prompt
        # Default retriever using MMR for diverse results
        self._retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVER_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": MMR_LAMBDA,
            },
        )
        # Initialize hybrid retriever (lazy-loaded)
        self._hybrid_retriever = HybridRetriever(vectorstore)
        # Cross-encoder for re-ranking (lazy-loaded)
        self._reranker = None

    @property
    def retriever(self):
        """Return the retriever for external access."""
        return self._retriever

    def _get_reranker(self):
        """Lazy-load cross-encoder reranker.

        Returns:
            CrossEncoder or None if loading fails
        """
        if self._reranker is None:
            try:
                from config import RERANKER_MODEL

                self._reranker = CrossEncoder(RERANKER_MODEL)
            except Exception as e:
                logger.warning(f"Could not load cross-encoder: {e}")
                self._reranker = _RERANKER_LOAD_FAILED

        if self._reranker is _RERANKER_LOAD_FAILED:
            return None
        return self._reranker

    def rewrite_query(self, question, chat_history=None):
        """Rewrite query to improve retrieval quality.

        Args:
            question: Original user question
            chat_history: Optional chat history for context

        Returns:
            Rewritten query string, or original if rewriting fails
        """
        try:
            rewrite_prompt = f"""Rewrite the following question to improve document retrieval. 
Focus on key technical terms, remove conversational filler, and expand important concepts.
Keep numbers, model names, and specific technical terms exactly as they appear.

Original question: {question}

Rewritten query (keywords and key phrases only, be concise):"""

            llm = create_llm(streaming=False)
            rewritten = llm.invoke(rewrite_prompt).content.strip()

            # Fall back to original if rewritten is too short or same
            if len(rewritten) < len(question) * 0.3 or rewritten.lower() == question.lower():
                return question

            return rewritten
        except Exception as e:
            print(f"Query rewriting failed: {e}")
            return question

    def rerank_documents(self, query, documents, top_k=RETRIEVER_K):
        """Re-rank documents using cross-encoder for better relevance.

        Args:
            query: User's question
            documents: List of Document objects from initial retrieval
            top_k: Number of top documents to return after re-ranking

        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if not documents:
            return []

        reranker = self._get_reranker()
        if reranker is None:
            # Fall back to original order if reranker unavailable
            return [(doc, None) for doc in documents[:top_k]]

        try:
            # Prepare pairs: (query, document_text)
            pairs = [(query, doc.page_content) for doc in documents]

            # Get relevance scores (higher = more relevant)
            scores = reranker.predict(pairs)

            # Sort by score (descending) and return top-k
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return [(doc, score) for doc, score in scored_docs[:top_k]]
        except Exception as e:
            print(f"Re-ranking failed: {e}")
            # Fall back to original order
            return [(doc, None) for doc in documents[:top_k]]

    def get_retriever_with_filter(self, metadata_filter=None, search_type="mmr"):
        """Get a retriever with optional metadata filtering and search type.

        Args:
            metadata_filter: Dict for Chroma where clause, e.g.:
                - {"source": {"$eq": "pdf/doc.pdf"}} - exact match
                - {"page": {"$gte": 5}} - page >= 5
                - {"source": {"$in": ["pdf/doc1.pdf", "pdf/doc2.pdf"]}} - source in list
            search_type: "mmr" for Maximal Marginal Relevance (diverse results) or
                        "similarity" for pure similarity search (most relevant)

        Returns:
            Retriever configured with specified search type and optional filter
        """
        if search_type == "mmr":
            search_kwargs = {
                "k": RETRIEVER_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": MMR_LAMBDA,
            }
        else:  # similarity search
            search_kwargs = {"k": RETRIEVER_K}

        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        return self._vectorstore.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    def stream(self, inputs):
        """Stream the chain response token by token.

        Args:
            inputs: Dict with 'question', optional 'chat_history', optional 'filter',
                    optional 'search_type', optional 'use_query_rewriting',
                    optional 'use_reranking', optional 'hybrid_alpha'
                - filter: Chroma metadata filter dict
                - search_type: "mmr", "similarity", or "hybrid" (default: "mmr")
                - use_query_rewriting: Whether to rewrite query before retrieval
                - use_reranking: Whether to re-rank results with cross-encoder
                - hybrid_alpha: Weight for semantic search in hybrid (0-1, default 0.7)

        Yields:
            dict: Contains 'chunk' (token text), 'source_documents', 'docs_with_scores',
                  'rewritten_query', and optional 'hybrid_scores'
        """
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])
        metadata_filter = inputs.get("filter")
        search_type = inputs.get("search_type", "mmr")
        use_query_rewriting = inputs.get("use_query_rewriting", False)
        use_reranking = inputs.get("use_reranking", False)
        hybrid_alpha = inputs.get("hybrid_alpha", HYBRID_ALPHA_DEFAULT)

        # Step 1: Query rewriting (if enabled)
        rewritten_query = None
        retrieval_query = question
        if use_query_rewriting:
            rewritten_query = self.rewrite_query(question, chat_history)
            retrieval_query = rewritten_query

        # Step 2: Retrieve documents based on search type
        docs = []
        docs_with_scores = None
        hybrid_scores = None

        if search_type == "hybrid":
            # Hybrid search: semantic + keyword
            initial_k = RERANK_INITIAL_K if use_reranking else RETRIEVER_K
            hybrid_results = self._hybrid_retriever.hybrid_search(
                retrieval_query,
                k=initial_k,
                alpha=hybrid_alpha,
                metadata_filter=metadata_filter,
            )
            docs = [r["doc"] for r in hybrid_results]
            # Store hybrid scores for display
            hybrid_scores = [
                (
                    r["doc"],
                    r["fused_score"],
                    r["semantic_score"],
                    r["keyword_score"],
                )
                for r in hybrid_results
            ]
            docs_with_scores = [(r["doc"], r["fused_score"]) for r in hybrid_results]
        else:
            # Traditional semantic search (MMR or similarity)
            initial_k = RERANK_INITIAL_K if use_reranking else RETRIEVER_K

            # Get retriever (with optional filter and search type)
            if metadata_filter or search_type != "mmr":
                retriever = self.get_retriever_with_filter(
                    metadata_filter, search_type=search_type
                )
            else:
                retriever = self._retriever

            # Retrieve more candidates if re-ranking enabled
            if use_reranking and search_type == "mmr":
                # For MMR, we need to adjust fetch_k
                retriever = self._vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": initial_k,
                        "fetch_k": initial_k * 2,
                        "lambda_mult": MMR_LAMBDA,
                        "filter": metadata_filter if metadata_filter else None,
                    },
                )
            elif use_reranking and search_type == "similarity":
                retriever = self._vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": initial_k,
                        "filter": metadata_filter if metadata_filter else None,
                    },
                )

            docs = retriever.invoke(retrieval_query)

            # Get similarity scores for highlighting (if using similarity search)
            if search_type == "similarity":
                try:
                    scored_docs = self._vectorstore.similarity_search_with_score(
                        retrieval_query,
                        k=initial_k,
                        filter=metadata_filter if metadata_filter else None,
                    )
                    # Create a mapping to match docs by content
                    docs_with_scores = []
                    for doc in docs:
                        for scored_doc, score in scored_docs:
                            if (
                                doc.page_content[:100] == scored_doc.page_content[:100]
                                and doc.metadata.get("page") == scored_doc.metadata.get("page")
                            ):
                                docs_with_scores.append((doc, score))
                                break
                        else:
                            docs_with_scores.append((doc, None))
                except Exception:
                    pass

        # Step 3: Re-ranking (if enabled)
        if use_reranking and docs:
            reranked = self.rerank_documents(retrieval_query, docs, top_k=RERANK_TOP_K)
            docs = [doc for doc, _ in reranked]
            # Update scores with re-ranking scores
            docs_with_scores = reranked
        elif not docs_with_scores:
            # Create placeholder scores if none exist
            docs_with_scores = [(doc, None) for doc in docs]

        # Limit to final k
        docs = docs[:RETRIEVER_K]
        docs_with_scores = docs_with_scores[:RETRIEVER_K]

        # Identify top chunk for emphasis
        top_chunk_idx = 0
        if docs_with_scores and len(docs_with_scores) > 0:
            best_score = float("-inf")
            for i, (doc, score) in enumerate(docs_with_scores):
                if score is not None:
                    # Handle both similarity scores (higher is better, <= 1.0) 
                    # and distance scores (lower is better, > 1.0)
                    if score <= 1.0:
                        if score > best_score:
                            best_score = score
                            top_chunk_idx = i
                    else:
                        # For distance scores, lower is better
                        if -score > best_score:
                            best_score = -score
                            top_chunk_idx = i

        # Format context with emphasis on top chunk and contextual headers
        context_parts = []
        for i, doc in enumerate(docs):
            # Add contextual header with source and page info
            source = doc.metadata.get("source", "Unknown")
            source_name = os.path.basename(source) if source != "Unknown" else "Unknown"
            page = doc.metadata.get("page", "unknown")
            header = f"[Document: {source_name}, Page: {page}]"
            
            # Emphasize top chunk explicitly
            if i == top_chunk_idx:
                content = f"{header}\n\n[MOST RELEVANT CONTEXT]\n{doc.page_content}\n[/MOST RELEVANT CONTEXT]"
            else:
                content = f"{header}\n\n{doc.page_content}"
            
            context_parts.append(content)
        
        context = "\n\n---\n\n".join(context_parts)

        # Format chat history
        history_str = format_chat_history(chat_history)

        # Create streaming LLM
        streaming_llm = create_llm(streaming=True)

        # Stream the response with error handling
        chain = self._prompt | streaming_llm
        try:
            for chunk in chain.stream(
                {
                    "question": question,  # Use original question in prompt
                    "context": context,
                    "chat_history": history_str,
                }
            ):
                yield {
                    "chunk": chunk.content,
                    "source_documents": docs,
                    "docs_with_scores": docs_with_scores,
                    "rewritten_query": rewritten_query,
                    "hybrid_scores": hybrid_scores,
                }
        except Exception as e:
            yield {
                "chunk": f"\n\n[Error: {e}]",
                "source_documents": docs,
                "docs_with_scores": docs_with_scores,
                "rewritten_query": rewritten_query,
                "hybrid_scores": hybrid_scores,
            }


def create_qa_chain(vectorstore):
    """Create the question-answering chain with MMR and metadata filtering support.

    Args:
        vectorstore: Chroma vectorstore instance

    Returns:
        QAChainWrapper: Configured QA chain with MMR retrieval and streaming support
    """
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    return QAChainWrapper(vectorstore, prompt)

