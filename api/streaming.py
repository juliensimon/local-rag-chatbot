"""Server-Sent Events (SSE) streaming utilities."""

import json
from typing import Any, Generator, Optional

from api.schemas import ContextResponse, SourceDocument
from utils import get_document_source, get_top_chunk_index


def format_sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Event message.

    Args:
        event: Event type name
        data: Data to serialize as JSON

    Returns:
        Formatted SSE string
    """
    json_data = json.dumps(data)
    return f"event: {event}\ndata: {json_data}\n\n"


def build_context_response(
    source_documents: list,
    docs_with_scores: Optional[list] = None,
    rewritten_query: Optional[str] = None,
    hybrid_scores: Optional[list] = None,
) -> ContextResponse:
    """Build a ContextResponse from source documents.

    Args:
        source_documents: List of LangChain Document objects
        docs_with_scores: Optional list of (doc, score) tuples
        rewritten_query: Optional rewritten query string
        hybrid_scores: Optional list of (doc, fused, semantic, keyword) tuples

    Returns:
        ContextResponse with formatted source documents
    """
    if not source_documents:
        return ContextResponse(sources=[], rewritten_query=rewritten_query)

    top_idx = get_top_chunk_index(docs_with_scores) if docs_with_scores else 0

    sources = []
    for i, doc in enumerate(source_documents):
        source_doc = SourceDocument(
            content=doc.page_content,
            source=get_document_source(doc),
            page=doc.metadata.get("page", 0),
            is_top=(i == top_idx),
        )

        # Add scores if available (convert to float for JSON serialization)
        if hybrid_scores and i < len(hybrid_scores):
            _, fused, semantic, keyword = hybrid_scores[i]
            source_doc.score = float(fused) if fused is not None else None
            source_doc.semantic_score = float(semantic) if semantic is not None else None
            source_doc.keyword_score = float(keyword) if keyword is not None else None
        elif docs_with_scores and i < len(docs_with_scores):
            _, score = docs_with_scores[i]
            source_doc.score = float(score) if score is not None else None

        sources.append(source_doc)

    return ContextResponse(sources=sources, rewritten_query=rewritten_query)


def stream_rag_response(
    qa_chain,
    message: str,
    chat_history: list,
    search_type: str,
    metadata_filter: Optional[dict],
    use_query_rewriting: bool,
    use_reranking: bool,
    hybrid_alpha: float,
) -> Generator[str, None, None]:
    """Stream RAG response as SSE events.

    Args:
        qa_chain: QAChainWrapper instance
        message: User's question
        chat_history: List of (user, assistant) tuples
        search_type: Search type (mmr, similarity, hybrid)
        metadata_filter: Optional metadata filter dict
        use_query_rewriting: Whether to use query rewriting
        use_reranking: Whether to use re-ranking
        hybrid_alpha: Hybrid search alpha value (0-1)

    Yields:
        SSE formatted strings
    """
    stream_input = {
        "question": message,
        "chat_history": chat_history,
        "search_type": search_type,
        "use_query_rewriting": use_query_rewriting,
        "use_reranking": use_reranking,
        "hybrid_alpha": hybrid_alpha,
    }
    if metadata_filter:
        stream_input["filter"] = metadata_filter

    source_docs = []
    docs_with_scores = None
    rewritten_query = None
    hybrid_scores = None

    for chunk_data in qa_chain.stream(stream_input):
        chunk = chunk_data.get("chunk", "")
        source_docs = chunk_data.get("source_documents", [])
        docs_with_scores = chunk_data.get("docs_with_scores")
        rewritten_query = chunk_data.get("rewritten_query")
        hybrid_scores = chunk_data.get("hybrid_scores")

        if chunk:
            yield format_sse_event("token", {"content": chunk})

    # Send context at the end
    context = build_context_response(
        source_docs, docs_with_scores, rewritten_query, hybrid_scores
    )
    yield format_sse_event("context", context.model_dump())
    yield format_sse_event("done", {})


def stream_vanilla_response(llm, message: str) -> Generator[str, None, None]:
    """Stream vanilla LLM response as SSE events.

    Args:
        llm: Streaming LLM instance
        message: User's question

    Yields:
        SSE formatted strings
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    vanilla_system = SystemMessage(
        content=(
            "Answer naturally and helpfully. "
            "Do not include citations, sources, references, or a 'Sources:' section. "
            "If you are unsure, say so."
        )
    )

    try:
        for chunk in llm.stream([vanilla_system, HumanMessage(content=message)]):
            if chunk.content:
                yield format_sse_event("token", {"content": chunk.content})
    except Exception as e:
        yield format_sse_event("error", {"message": str(e)})
        return

    yield format_sse_event("done", {})
