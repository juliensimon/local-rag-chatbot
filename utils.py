"""Utility functions for formatting and chat history."""

import os
from langchain_core.messages import AIMessage, HumanMessage

from config import CHAT_HISTORY_LIMIT


def format_chat_history(chat_history, limit=CHAT_HISTORY_LIMIT):
    """Format chat history for inclusion in prompts.

    Args:
        chat_history: List of message tuples, dicts, or Message objects
        limit: Maximum number of recent messages to include

    Returns:
        str: Formatted chat history string
    """
    if not chat_history:
        return ""

    history_parts = []
    for msg in chat_history[-limit:]:
        if isinstance(msg, tuple):
            history_parts.append(f"Human: {msg[0]}\nAssistant: {msg[1]}")
        elif isinstance(msg, dict):
            # OpenAI-style message format
            role = "Human" if msg.get("role") == "user" else "Assistant"
            history_parts.append(f"{role}: {msg.get('content', '')}")
        elif isinstance(msg, HumanMessage):
            history_parts.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            history_parts.append(f"Assistant: {msg.content}")

    return "\n".join(history_parts)


def messages_to_tuples(messages):
    """Convert OpenAI-style messages to tuples for the QA chain.

    Args:
        messages: List of dicts with 'role' and 'content' keys

    Returns:
        List of (user, assistant) tuples
    """
    tuples = []
    user_msg = None
    for msg in messages:
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant" and user_msg is not None:
            tuples.append((user_msg, msg["content"]))
            user_msg = None
    return tuples


def format_context_with_highlight(
    source_documents,
    docs_with_scores=None,
    rewritten_query=None,
    hybrid_scores=None,
):
    """Format context with highlighting for the top matching chunk.

    Args:
        source_documents: List of document chunks
        docs_with_scores: Optional list of (doc, score) tuples
        rewritten_query: Optional rewritten query string
        hybrid_scores: Optional list of (doc, fused_score, semantic_score, keyword_score) tuples

    Returns:
        Formatted context markdown string with highlighting for top chunk and sources list
    """
    if not source_documents:
        return ""

    # Show rewritten query if available (compact)
    query_info = ""
    if rewritten_query:
        query_info = f"**🔄 Rewritten:** `{rewritten_query}`\n\n"

    # Get unique sources for summary (compact) with hyperlinks
    seen_sources = set()
    sources_list = []
    for doc in source_documents:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "unknown")
        source_key = f"{source}:{page}"
        if source_key not in seen_sources:
            sources_list.append(f"`{source}` (p.{page})")
            seen_sources.add(source_key)

    # Identify top chunk
    top_chunk_idx = 0
    if docs_with_scores and len(docs_with_scores) > 0:
        best_score = float("-inf")
        for i, (doc, score) in enumerate(docs_with_scores):
            if score is not None:
                if score <= 1.0:
                    if score > best_score:
                        best_score = score
                        top_chunk_idx = i
                else:
                    if -score > best_score:
                        best_score = -score
                        top_chunk_idx = i

    # Compact sources header
    sources_header = f"**Sources:** {', '.join(sources_list)}\n\n---\n\n"

    # Format chunks more compactly
    formatted_chunks = []
    for i, doc in enumerate(source_documents):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "unknown")
        content = doc.page_content

        # Compact header
        is_top = i == top_chunk_idx
        star = "⭐ " if is_top else ""
        header = f"**{star}{source}** (page {page})"

        # Add score info (compact)
        score_info = ""
        if hybrid_scores and i < len(hybrid_scores):
            _, fused, sem, kw = hybrid_scores[i]
            score_info = f" *[f:{fused:.2f} s:{sem:.2f} k:{kw:.2f}]*"
        elif (
            docs_with_scores
            and i < len(docs_with_scores)
            and docs_with_scores[i][1] is not None
        ):
            score = docs_with_scores[i][1]
            if score <= 1.0:
                score_info = f" *[rel:{score:.3f}]*"
            else:
                score_info = f" *[dist:{score:.3f}]*"

        # Compact content display
        chunk_text = f"{header}{score_info}\n\n{content}\n\n---"
        formatted_chunks.append(chunk_text)

    return query_info + sources_header + "\n\n".join(formatted_chunks)

