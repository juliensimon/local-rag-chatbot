"""API route handlers."""

import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SourcesResponse,
)
from api.streaming import (
    build_context_response,
    stream_rag_response,
    stream_vanilla_response,
)
from config import PDF_PATH
from utils import messages_to_tuples

router = APIRouter()

# These will be set by the main app during initialization
_qa_chain = None
_available_sources: set = set()


def init_routes(qa_chain, available_sources: list):
    """Initialize routes with QA chain and available sources.

    Args:
        qa_chain: QAChainWrapper instance
        available_sources: List of available document source filenames
    """
    global _qa_chain, _available_sources
    _qa_chain = qa_chain
    _available_sources = set(available_sources)


def validate_doc_filter(doc_filter: Optional[str]) -> Optional[dict]:
    """Validate document filter and return metadata filter dict.

    Args:
        doc_filter: Document filename to filter by

    Returns:
        Metadata filter dict or None
    """
    if not doc_filter or doc_filter == "All Documents":
        return None
    if doc_filter not in _available_sources:
        return None
    full_path = os.path.join(PDF_PATH, doc_filter)
    return {"source": {"$eq": full_path}}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    vectorstore_ready = _qa_chain is not None
    llm_ready = True  # LLM is lazily loaded, assume ready

    return HealthResponse(
        status="healthy" if vectorstore_ready else "degraded",
        vectorstore_ready=vectorstore_ready,
        llm_ready=llm_ready,
    )


@router.get("/sources", response_model=SourcesResponse)
async def get_sources():
    """Get list of available document sources."""
    return SourcesResponse(sources=sorted(_available_sources))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    if _qa_chain is None:
        raise HTTPException(status_code=503, detail="QA chain not initialized")

    chat_history = messages_to_tuples(
        [{"role": m.role, "content": m.content} for m in request.history]
    )

    if request.rag_enabled:
        metadata_filter = validate_doc_filter(request.doc_filter)
        hybrid_alpha = request.hybrid_alpha / 100.0

        # Collect full response from stream
        full_response = ""
        source_docs = []
        docs_with_scores = None
        rewritten_query = None
        hybrid_scores = None

        stream_input = {
            "question": request.message,
            "chat_history": chat_history,
            "search_type": request.search_type,
            "use_query_rewriting": request.use_query_rewriting,
            "use_reranking": request.use_reranking,
            "hybrid_alpha": hybrid_alpha,
        }
        if metadata_filter:
            stream_input["filter"] = metadata_filter

        for chunk_data in _qa_chain.stream(stream_input):
            full_response += chunk_data.get("chunk", "")
            source_docs = chunk_data.get("source_documents", [])
            docs_with_scores = chunk_data.get("docs_with_scores")
            rewritten_query = chunk_data.get("rewritten_query")
            hybrid_scores = chunk_data.get("hybrid_scores")

        context = build_context_response(
            source_docs, docs_with_scores, rewritten_query, hybrid_scores
        )
        return ChatResponse(response=full_response, context=context)
    else:
        # Vanilla LLM
        from models import create_llm

        llm = create_llm(streaming=False)
        from langchain_core.messages import HumanMessage, SystemMessage

        vanilla_system = SystemMessage(
            content=(
                "Answer naturally and helpfully. "
                "Do not include citations, sources, references, or a 'Sources:' section. "
                "If you are unsure, say so."
            )
        )
        response = llm.invoke([vanilla_system, HumanMessage(content=request.message)])
        return ChatResponse(response=response.content, context=None)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    if _qa_chain is None:
        raise HTTPException(status_code=503, detail="QA chain not initialized")

    chat_history = messages_to_tuples(
        [{"role": m.role, "content": m.content} for m in request.history]
    )

    if request.rag_enabled:
        metadata_filter = validate_doc_filter(request.doc_filter)
        hybrid_alpha = request.hybrid_alpha / 100.0

        return StreamingResponse(
            stream_rag_response(
                _qa_chain,
                request.message,
                chat_history,
                request.search_type,
                metadata_filter,
                request.use_query_rewriting,
                request.use_reranking,
                hybrid_alpha,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        from models import create_llm

        llm = create_llm(streaming=True)

        return StreamingResponse(
            stream_vanilla_response(llm, request.message),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
