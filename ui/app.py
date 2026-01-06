"""Main Gradio application for RAG-powered document chat."""

import os

from models import create_embeddings
from qa_chain import create_qa_chain
from vectorstore import load_or_create_vectorstore

from .components import create_ui_components
from .handlers import (
    create_respond_handler,
    create_stream_chat_response,
    update_hybrid_alpha_visibility,
    update_rag_controls,
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
    if not collection or not collection.get("metadatas"):
        sources = []
    else:
        sources = sorted(
            set(
                os.path.basename(meta.get("source", ""))
                for meta in collection["metadatas"]
                if meta and meta.get("source")
            )
        )

    return chain, sources


def create_app():
    """Create and configure the Gradio application.

    Returns:
        gr.Blocks: Configured Gradio interface
    """
    # Initialize QA chain and get available sources
    qa_chain, available_sources = initialize_chain()

    # Create UI components
    components = create_ui_components(available_sources)
    demo = components["demo"]

    # Create handlers
    stream_chat_response_fn = create_stream_chat_response(qa_chain)
    respond_fn = create_respond_handler(stream_chat_response_fn)

    # Event handlers
    components["msg"].submit(
        respond_fn,
        [
            components["msg"],
            components["chatbot"],
            components["rag_enabled"],
            components["search_type"],
            components["doc_filter"],
            components["query_rewriting"],
            components["reranking"],
            components["hybrid_alpha"],
        ],
        [
            components["msg"],
            components["chatbot"],
            components["context_box"],
            components["rag_enabled"],
            components["doc_filter"],
        ],
    )
    components["submit"].click(
        respond_fn,
        [
            components["msg"],
            components["chatbot"],
            components["rag_enabled"],
            components["search_type"],
            components["doc_filter"],
            components["query_rewriting"],
            components["reranking"],
            components["hybrid_alpha"],
        ],
        [
            components["msg"],
            components["chatbot"],
            components["context_box"],
            components["rag_enabled"],
            components["doc_filter"],
        ],
    )
    components["clear"].click(
        lambda: [[], "", False, "mmr", "All Documents", False, False, 70],
        None,
        [
            components["chatbot"],
            components["context_box"],
            components["rag_enabled"],
            components["search_type"],
            components["doc_filter"],
            components["query_rewriting"],
            components["reranking"],
            components["hybrid_alpha"],
        ],
        queue=False,
    )
    components["rag_enabled"].change(
        update_rag_controls,
        components["rag_enabled"],
        [
            components["search_col"],
            components["filter_col"],
            components["context_section"],
            components["advanced_options"],
        ],
    )
    components["search_type"].change(
        update_hybrid_alpha_visibility,
        components["search_type"],
        [components["hybrid_alpha"]],
    )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, server_port=7860)

