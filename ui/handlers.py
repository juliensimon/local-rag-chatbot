"""Event handlers for the Gradio interface."""

import os

import gradio as gr

from config import MAX_QUERY_LENGTH, PDF_PATH
from models import create_llm
from qa_chain import QAChainWrapper
from utils import format_context_with_highlight, messages_to_tuples
from langchain_core.messages import HumanMessage, SystemMessage


def validate_doc_filter(doc_filter, available_sources):
    """Validate document filter against allowed sources to prevent path traversal.

    Args:
        doc_filter: User-provided document filter value
        available_sources: Set/list of valid source filenames

    Returns:
        Full path if valid, None otherwise
    """
    if not doc_filter or doc_filter == "All Documents":
        return None
    # Only allow filenames that exist in our indexed sources
    if doc_filter not in available_sources:
        return None  # Invalid filter, ignore
    return os.path.join(PDF_PATH, doc_filter)


def create_stream_chat_response(qa_chain: QAChainWrapper, available_sources=None):
    """Create a stream_chat_response function bound to the QA chain.

    Args:
        qa_chain: QAChainWrapper instance
        available_sources: List of valid document source filenames

    Returns:
        Function that streams chat responses
    """
    sources_set = set(available_sources) if available_sources else set()

    def stream_chat_response(
        message,
        history,
        query_type,
        doc_filter=None,
        search_type="mmr",
        use_query_rewriting=False,
        use_reranking=False,
        hybrid_alpha=0.7,
    ):
        """Stream chat responses token by token.

        Args:
            message: User's question
            history: List of message dicts with 'role' and 'content' keys
            query_type: "RAG" or "Vanilla LLM"
            doc_filter: Optional document name to filter by
            search_type: "mmr", "similarity", or "hybrid"
            use_query_rewriting: Whether to rewrite query before retrieval
            use_reranking: Whether to re-rank results with cross-encoder
            hybrid_alpha: Weight for semantic search in hybrid (0-1)

        Yields:
            tuple: (accumulated_response, source_documents, docs_with_scores, rewritten_query, hybrid_scores)
        """
        # Convert messages format to tuples for QA chain
        chat_history = messages_to_tuples(history) if history else []
        accumulated_response = ""
        source_docs = []
        docs_with_scores = None
        rewritten_query = None
        hybrid_scores = None

        if query_type == "RAG":
            # Build filter if document is selected (with path traversal protection)
            metadata_filter = None
            validated_path = validate_doc_filter(doc_filter, sources_set)
            if validated_path:
                metadata_filter = {"source": {"$eq": validated_path}}

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

            for chunk_data in qa_chain.stream(stream_input):
                chunk = chunk_data.get("chunk", "")
                source_docs = chunk_data.get("source_documents", [])
                docs_with_scores = chunk_data.get("docs_with_scores")
                rewritten_query = chunk_data.get("rewritten_query")
                hybrid_scores = chunk_data.get("hybrid_scores")
                accumulated_response += chunk
                yield (
                    accumulated_response,
                    source_docs,
                    docs_with_scores,
                    rewritten_query,
                    hybrid_scores,
                )
        else:
            streaming_llm = create_llm(streaming=True)
            try:
                vanilla_system = SystemMessage(
                    content=(
                        "Answer naturally and helpfully. "
                        "Do not include citations, sources, references, or a 'Sources:' section. "
                        "If you are unsure, say so."
                    )
                )
                for chunk in streaming_llm.stream(
                    [vanilla_system, HumanMessage(content=message)]
                ):
                    accumulated_response += chunk.content
                    yield accumulated_response, [], None, None, None
            except Exception as e:
                yield f"{accumulated_response}\n\n[Error: {e}]", [], None, None, None

    return stream_chat_response


def create_respond_handler(stream_chat_response_fn):
    """Create a respond handler function.

    Args:
        stream_chat_response_fn: Function that streams chat responses

    Returns:
        Function that handles user responses
    """

    def respond(
        message,
        chat_history,
        is_rag_enabled,
        search_type_choice,
        selected_doc,
        use_query_rewriting,
        use_reranking,
        hybrid_alpha_value,
    ):
        """Stream response to user message.

        Args:
            message: User input
            chat_history: Previous conversation (list of message dicts)
            is_rag_enabled: Whether to use RAG
            search_type_choice: "mmr", "similarity", or "hybrid"
            selected_doc: Document filter selection
            use_query_rewriting: Whether to rewrite query
            use_reranking: Whether to re-rank results
            hybrid_alpha_value: Hybrid search balance (0-100)

        Yields:
            tuple: (cleared_msg, history, context, rag_state, doc_filter)
        """
        if not message or not message.strip():
            yield "", chat_history, "", is_rag_enabled, selected_doc
            return

        # Truncate extremely long inputs to prevent resource exhaustion
        if len(message) > MAX_QUERY_LENGTH:
            message = message[:MAX_QUERY_LENGTH]

        query_type = "RAG" if is_rag_enabled else "Vanilla LLM"
        new_history = list(chat_history)

        # Add user message
        new_history.append({"role": "user", "content": message})
        # Add empty assistant message for streaming
        new_history.append({"role": "assistant", "content": ""})

        source_docs = []
        context = ""
        docs_with_scores = None
        rewritten_query = None
        hybrid_scores = None

        # Pass document filter and search type to stream function
        doc_filter_value = selected_doc if is_rag_enabled else None
        search_type_value = search_type_choice if is_rag_enabled else "mmr"
        hybrid_alpha_normalized = hybrid_alpha_value / 100.0 if hybrid_alpha_value else 0.7

        for (
            partial_response,
            docs,
            scores,
            rewritten,
            hybrid,
        ) in stream_chat_response_fn(
            message,
            chat_history,
            query_type,
            doc_filter_value,
            search_type_value,
            use_query_rewriting if is_rag_enabled else False,
            use_reranking if is_rag_enabled else False,
            hybrid_alpha_normalized,
        ):
            source_docs = docs
            docs_with_scores = scores
            rewritten_query = rewritten
            hybrid_scores = hybrid
            new_history[-1] = {"role": "assistant", "content": partial_response}
            # Format context with highlighting
            if is_rag_enabled and source_docs:
                context = format_context_with_highlight(
                    source_docs, docs_with_scores, rewritten_query, hybrid_scores
                )
            yield "", new_history, context, is_rag_enabled, selected_doc

        # Add context after streaming completes (sources shown in context box only)
        if is_rag_enabled and source_docs:
            # Don't add sources to chatbot output, only show in context box
            context = format_context_with_highlight(
                source_docs, docs_with_scores, rewritten_query, hybrid_scores
            )

        yield "", new_history, context, is_rag_enabled, selected_doc

    return respond


def update_rag_controls(is_rag_enabled):
    """Show/hide RAG-related controls based on toggle.

    Args:
        is_rag_enabled: Whether RAG mode is enabled

    Returns:
        Tuple of gr.update() calls for visibility
    """
    if is_rag_enabled:
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def update_hybrid_alpha_visibility(search_type_choice):
    """Show/hide hybrid alpha slider based on search type.

    Args:
        search_type_choice: Selected search type

    Returns:
        gr.update() for hybrid alpha visibility
    """
    if search_type_choice == "hybrid":
        return gr.update(visible=True, interactive=True)
    return gr.update(visible=False, interactive=False)

