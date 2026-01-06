"""
Gradio web application for RAG-powered document chat using local LLM.

This module provides a user interface for querying documents using either
Retrieval-Augmented Generation (RAG) or vanilla LLM responses.
"""

import os
import gradio as gr
from demo import (
    create_embeddings,
    create_llm,
    create_qa_chain,
    load_or_create_vectorstore,
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


def stream_chat_response(message, history, query_type, doc_filter=None):
    """Stream chat responses token by token.

    Args:
        message: User's question
        history: List of message dicts with 'role' and 'content' keys
        query_type: "RAG" or "Vanilla LLM"
        doc_filter: Optional document name to filter by

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
            metadata_filter = {"source": {"$contains": doc_filter}}

        stream_input = {
            "question": message,
            "chat_history": chat_history,
        }
        if metadata_filter:
            stream_input["filter"] = metadata_filter

        for chunk_data in qa_chain.stream(stream_input):
            chunk = chunk_data.get("chunk", "")
            source_docs = chunk_data.get("source_documents", [])
            accumulated_response += chunk
            yield accumulated_response, source_docs
    else:
        streaming_llm = create_llm(streaming=True)
        try:
            for chunk in streaming_llm.stream(message):
                accumulated_response += chunk.content
                yield accumulated_response, []
        except Exception as e:
            yield f"{accumulated_response}\n\n[Error: {e}]", []


def format_sources(source_documents):
    """Format source documents for display."""
    if not source_documents:
        return ""

    sources = []
    seen_sources = set()

    for doc in source_documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "unknown")
        source_key = f"{source}:{page}"

        if source_key not in seen_sources:
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
        "Ask questions about your documents. Uses **MMR** for diverse results. "
        "Optionally filter by specific document."
    )

    with gr.Row():
        rag_enabled = gr.Checkbox(
            value=True,
            label="Enable RAG",
            info="Toggle between RAG-powered document search or vanilla LLM responses",
        )
        doc_filter = gr.Dropdown(
            choices=["All Documents"] + available_sources,
            value="All Documents",
            label="Filter by Document",
            info="Limit search to a specific PDF",
            interactive=True,
        )

    chatbot = gr.Chatbot(type="messages")
    context_box = gr.Textbox(
        label="Retrieved Context",
        interactive=False,
        visible=True,
        lines=5,
    )

    with gr.Row():
        msg = gr.Textbox(label="Query", scale=8)
        with gr.Column(scale=1):
            submit = gr.Button("Submit")
            clear = gr.Button("Clear")

    def respond(message, chat_history, is_rag_enabled, selected_doc):
        """Stream response to user message.

        Args:
            message: User input
            chat_history: Previous conversation (list of message dicts)
            is_rag_enabled: Whether to use RAG
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

        # Pass document filter to stream function
        doc_filter_value = selected_doc if is_rag_enabled else None

        for partial_response, docs in stream_chat_response(
            message, chat_history, query_type, doc_filter_value
        ):
            source_docs = docs
            new_history[-1] = {"role": "assistant", "content": partial_response}
            yield "", new_history, context, is_rag_enabled, selected_doc

        # Add sources and context after streaming completes
        if is_rag_enabled and source_docs:
            final_response = new_history[-1]["content"] + format_sources(source_docs)
            new_history[-1] = {"role": "assistant", "content": final_response}
            context = "\n\n".join(doc.page_content for doc in source_docs)

        yield "", new_history, context, is_rag_enabled, selected_doc

    def update_rag_controls(is_rag_enabled):
        """Show/hide RAG-related controls based on toggle."""
        return (
            gr.update(visible=is_rag_enabled, value="" if not is_rag_enabled else None),
            gr.update(interactive=is_rag_enabled),
        )

    # Event handlers
    msg.submit(
        respond,
        [msg, chatbot, rag_enabled, doc_filter],
        [msg, chatbot, context_box, rag_enabled, doc_filter],
    )
    submit.click(
        respond,
        [msg, chatbot, rag_enabled, doc_filter],
        [msg, chatbot, context_box, rag_enabled, doc_filter],
    )
    clear.click(
        lambda: [[], "", True, "All Documents"],
        None,
        [chatbot, context_box, rag_enabled, doc_filter],
        queue=False,
    )
    rag_enabled.change(
        update_rag_controls,
        rag_enabled,
        [context_box, doc_filter],
    )

    gr.Examples(
        examples=[
            "Tell me about Arcee Fusion.",
            "How does deepseek-R1 differ from deepseek-v3?",
            "What is the main innovation in DELLA merging?",
        ],
        inputs=msg,
    )

if __name__ == "__main__":
    demo.launch(share=False)
