"""
Gradio web application for RAG-powered document chat using local LLM.

This module provides a user interface for querying documents using either
Retrieval-Augmented Generation (RAG) or vanilla LLM responses.
"""

import os
import gradio as gr
from langchain_core.messages import HumanMessage, SystemMessage
from demo import (
    create_embeddings,
    create_llm,
    create_qa_chain,
    load_or_create_vectorstore,
    PDF_PATH,
)


def initialize_chain():
    """Initialize the RAG chain components.

    Returns:
        tuple: (QAChainWrapper, list of available sources)
    """
    embeddings = create_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    chain = create_qa_chain(vectorstore)

    # Get available document sources for the filter dropdown
    collection = vectorstore.get()
    sources = sorted(set(
        os.path.basename(meta.get("source", ""))
        for meta in collection["metadatas"]
        if meta and meta.get("source")
    ))

    return chain, sources


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


def stream_chat_response(message, history, query_type, doc_filter=None, search_type="mmr"):
    """Stream chat responses token by token.

    Args:
        message: User's question
        history: List of message dicts with 'role' and 'content' keys
        query_type: "RAG" or "Vanilla LLM"
        doc_filter: Optional document name to filter by
        search_type: "mmr" for diverse results or "similarity" for most relevant

    Yields:
        tuple: (accumulated_response, source_documents)
    """
    # Convert messages format to tuples for QA chain
    chat_history = messages_to_tuples(history) if history else []
    accumulated_response = ""
    source_docs = []

    if query_type == "RAG":
        # Build filter if document is selected
        metadata_filter = None
        if doc_filter and doc_filter != "All Documents":
            # Construct full path for exact match (ChromaDB doesn't support $contains)
            full_source_path = os.path.join(PDF_PATH, doc_filter)
            metadata_filter = {"source": {"$eq": full_source_path}}

        stream_input = {
            "question": message,
            "chat_history": chat_history,
            "search_type": search_type,
        }
        if metadata_filter:
            stream_input["filter"] = metadata_filter

        docs_with_scores = None
        for chunk_data in qa_chain.stream(stream_input):
            chunk = chunk_data.get("chunk", "")
            source_docs = chunk_data.get("source_documents", [])
            docs_with_scores = chunk_data.get("docs_with_scores")
            accumulated_response += chunk
            yield accumulated_response, source_docs, docs_with_scores
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
            for chunk in streaming_llm.stream([vanilla_system, HumanMessage(content=message)]):
                accumulated_response += chunk.content
                yield accumulated_response, [], None
        except Exception as e:
            yield f"{accumulated_response}\n\n[Error: {e}]", [], None


def format_context_with_highlight(source_documents, docs_with_scores=None):
    """Format context with highlighting for the top matching chunk.
    
    Args:
        source_documents: List of document chunks
        docs_with_scores: Optional list of (doc, score) tuples from similarity search
    
    Returns:
        Formatted context markdown string with highlighting for top chunk and sources list
    """
    if not source_documents:
        return ""
    
    # Get unique sources for summary
    seen_sources = set()
    sources_list = []
    for doc in source_documents:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "unknown")
        source_key = f"{source}:{page}"
        if source_key not in seen_sources:
            sources_list.append(f"- {source}, page {page}")
            seen_sources.add(source_key)
    
    # Identify top chunk (first one, or one with best score)
    top_chunk_idx = 0
    if docs_with_scores and len(docs_with_scores) > 0:
        # Find chunk with lowest score (most similar)
        best_score = float('inf')
        for i, (doc, score) in enumerate(docs_with_scores):
            if score is not None and score < best_score:
                best_score = score
                top_chunk_idx = i
    # For MMR, first chunk is typically the most relevant
    
    # Add sources summary at the top
    sources_header = "### 📚 Sources\n" + "\n".join(sources_list) + "\n\n---\n\n"
    
    formatted_chunks = []
    for i, doc in enumerate(source_documents):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "unknown")
        content = doc.page_content
        
        # Header with source info
        header_parts = [f"### 📄 {source}, page {page}"]
        if docs_with_scores and i < len(docs_with_scores) and docs_with_scores[i][1] is not None:
            score = docs_with_scores[i][1]
            header_parts.append(f"*(similarity distance: {score:.4f}; lower is closer)*")
        if i == top_chunk_idx:
            header_parts.append("**⭐ TOP MATCH**")
        header = "\n".join(header_parts)
        
        # Render chunk text as a code block for readability; emphasize top match via the header.
        if i == top_chunk_idx:
            chunk_text = f"{header}\n\n```text\n{content}\n```"
        else:
            chunk_text = f"{header}\n\n```text\n{content}\n```"
        
        formatted_chunks.append(chunk_text)
    
    return sources_header + "\n\n".join(formatted_chunks)


def format_sources(source_documents, question=None):
    """Format source documents for display, filtering to only include relevant sources.
    
    Args:
        source_documents: List of document chunks
        question: Optional question to filter sources by relevance
    """
    if not source_documents:
        return ""

    sources = []
    seen_sources = set()
    
    # Extract key terms from question for relevance checking
    key_terms = set()
    critical_terms = set()  # Terms that must be present together
    
    if question:
        import re
        # Extract meaningful keywords (nouns, numbers, specific terms)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'why', 'when', 'where', 
                      'for', 'to', 'of', 'in', 'on', 'at', 'by', 'and', 'or', 'but', 'with',
                      'exact', 'number', 'mentioned', 'are', 'is', 'what'}
        words = re.findall(r'\b\w+\b', question.lower())
        key_terms = {w for w in words if w not in stop_words and len(w) > 3}
        # Also include any numbers or specific terms
        numbers = re.findall(r'\d+[,\d]*', question)
        key_terms.update(numbers)
        
        # For fact-based questions, identify critical terms that should appear together
        # e.g., "WMT" and "training steps" should both appear for WMT training steps question
        if 'wmt' in question.lower():
            critical_terms.add('wmt')
        if 'training steps' in question.lower() or 'training step' in question.lower():
            critical_terms.add('training')

    for doc in source_documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "unknown")
        source_key = f"{source}:{page}"

        if source_key not in seen_sources:
            content_lower = doc.page_content.lower()
            content = doc.page_content
            is_relevant = True
            
            # For fact-based questions with critical terms, require ALL critical terms
            if critical_terms:
                has_all_critical = all(term in content_lower for term in critical_terms)
                if not has_all_critical:
                    is_relevant = False
            
            # Also check for key terms: chunk should contain at least 2 key terms
            if key_terms and is_relevant:
                matches = sum(1 for term in key_terms if term.lower() in content_lower)
                # Require at least 2 matches, or all if there are fewer than 2 key terms
                threshold = min(2, len(key_terms))
                is_relevant = matches >= threshold
            
            # Always include if we can't determine relevance (no question provided)
            if is_relevant:
                sources.append(f"- {os.path.basename(source)}, page {page}")
                seen_sources.add(source_key)

    if sources:
        return "\n\nSources:\n" + "\n".join(sources)
    return ""


# Initialize QA chain and get available sources
qa_chain, available_sources = initialize_chain()

# Custom CSS to remove orange highlight during streaming
custom_css = """
.generating {
    border-color: transparent !important;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# RAG-Powered Document Chat with Local LLM")
    gr.Markdown(
        "Ask questions about your documents. Choose between **MMR** (diverse results) "
        "or **Similarity** (most relevant) search. Optionally filter by specific document."
    )

    with gr.Row():
        rag_enabled = gr.Checkbox(
            value=False,
            label="Enable RAG",
            info="Toggle between RAG-powered document search or vanilla LLM responses",
        )
        search_type = gr.Radio(
            choices=[("MMR (Diverse)", "mmr"), ("Similarity (Most Relevant)", "similarity")],
            value="mmr",
            label="Search Type",
            info="MMR: diverse results | Similarity: most relevant chunks",
            interactive=False,
            visible=False,
        )
        doc_filter = gr.Dropdown(
            choices=["All Documents"] + available_sources,
            value="All Documents",
            label="Filter by Document",
            info="Limit search to a specific PDF",
            interactive=False,
            visible=False,
        )

    chatbot = gr.Chatbot(type="messages")
    context_header = gr.Markdown("### Retrieved Context (⭐ = Top Match)", visible=False)
    context_box = gr.Markdown(visible=False)

    with gr.Row():
        msg = gr.Textbox(label="Query", scale=8)
        with gr.Column(scale=1):
            submit = gr.Button("Submit")
            clear = gr.Button("Clear")

    def respond(message, chat_history, is_rag_enabled, search_type_choice, selected_doc):
        """Stream response to user message.

        Args:
            message: User input
            chat_history: Previous conversation (list of message dicts)
            is_rag_enabled: Whether to use RAG
            search_type_choice: "mmr" or "similarity"
            selected_doc: Document filter selection

        Yields:
            tuple: (cleared_msg, history, context, rag_state, doc_filter)
        """
        if not message:
            yield "", chat_history, "", is_rag_enabled, selected_doc
            return

        query_type = "RAG" if is_rag_enabled else "Vanilla LLM"
        new_history = list(chat_history)

        # Add user message
        new_history.append({"role": "user", "content": message})
        # Add empty assistant message for streaming
        new_history.append({"role": "assistant", "content": ""})

        source_docs = []
        context = ""
        docs_with_scores = None

        # Pass document filter and search type to stream function
        doc_filter_value = selected_doc if is_rag_enabled else None
        search_type_value = search_type_choice if is_rag_enabled else "mmr"

        for partial_response, docs, scores in stream_chat_response(
            message, chat_history, query_type, doc_filter_value, search_type_value
        ):
            source_docs = docs
            docs_with_scores = scores
            new_history[-1] = {"role": "assistant", "content": partial_response}
            # Format context with highlighting
            if is_rag_enabled and source_docs:
                context = format_context_with_highlight(source_docs, docs_with_scores)
            yield "", new_history, context, is_rag_enabled, selected_doc

        # Add context after streaming completes (sources shown in context box only)
        if is_rag_enabled and source_docs:
            # Don't add sources to chatbot output, only show in context box
            context = format_context_with_highlight(source_docs, docs_with_scores)

        yield "", new_history, context, is_rag_enabled, selected_doc

    def update_rag_controls(is_rag_enabled):
        """Show/hide RAG-related controls based on toggle."""
        if is_rag_enabled:
            return (
                gr.update(visible=True),
                gr.update(visible=True, interactive=True),
                gr.update(visible=True, interactive=True),
                gr.update(visible=True),
            )
        return (
            gr.update(visible=False, value=""),
            gr.update(visible=False, interactive=False),
            gr.update(visible=False, interactive=False),
            gr.update(visible=False),
        )

    # Event handlers
    msg.submit(
        respond,
        [msg, chatbot, rag_enabled, search_type, doc_filter],
        [msg, chatbot, context_box, rag_enabled, doc_filter],
    )
    submit.click(
        respond,
        [msg, chatbot, rag_enabled, search_type, doc_filter],
        [msg, chatbot, context_box, rag_enabled, doc_filter],
    )
    clear.click(
        lambda: [[], "", False, "mmr", "All Documents"],
        None,
        [chatbot, context_box, rag_enabled, search_type, doc_filter],
        queue=False,
    )
    rag_enabled.change(
        update_rag_controls,
        rag_enabled,
        [context_box, search_type, doc_filter, context_header],
    )

    gr.Examples(
        examples=[
            "What are the exact ROUGE-2 scores for T5-Small, T5-Base, and T5-Large on the XSum dataset?",
            "What are the exact model sizes in parameters for T5-Small, T5-Base, T5-Large, and T5-XL?",
            "What are the exact values for beta1, beta2, and epsilon hyperparameters used in training?",
            "What percentage improvement does GKD achieve over Supervised KD according to the figures?",
            "What is the exact number of training steps mentioned for WMT en-de experiments?",
        ],
        inputs=msg,
    )

if __name__ == "__main__":
    demo.launch(share=False)
