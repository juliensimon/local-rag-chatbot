"""UI component definitions for the Gradio interface."""

import gradio as gr

from config import HYBRID_ALPHA_UI_DEFAULT

# Modern, clean CSS styling
CUSTOM_CSS = """
.generating {
    border-color: transparent !important;
}

/* Cleaner header */
h1 {
    font-size: 2rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Compact controls */
.control-row {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Better spacing for controls */
.form-group {
    margin-bottom: 0.75rem;
}

/* Cleaner chatbot */
.chatbot-container {
    border-radius: 12px;
    overflow: hidden;
}

/* Context panel styling */
.context-panel {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    max-height: 600px;
    overflow-y: auto;
}

/* Better buttons */
button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}

button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

/* Cleaner input */
input[type="text"], textarea {
    border-radius: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* Compact accordion */
.accordion {
    border-radius: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: rgba(255, 255, 255, 0.02) !important;
}

/* Better radio buttons */
.radio-group {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

/* Hide info text by default, show on hover */
.info-text {
    font-size: 0.85rem;
    color: rgba(255, 255, 255, 0.6);
    margin-top: 0.25rem;
}
"""


def create_ui_components(available_sources):
    """Create and return all UI components.

    Args:
        available_sources: List of available document sources

    Returns:
        dict: Dictionary containing all UI components
    """
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
        # Clean header
        gr.Markdown("# 📚 Document Q&A")
        gr.Markdown(
            "Ask questions about your research papers. Toggle RAG on to search documents, or off for general chat.",
            elem_classes=["info-text"],
        )

        # Main controls in a compact row
        with gr.Row(elem_classes=["control-row"]):
            with gr.Column(scale=1, min_width=120):
                rag_enabled = gr.Checkbox(
                    value=False,
                    label="🔍 RAG Mode",
                    info="",
                )
            with gr.Column(scale=2, min_width=200, visible=False) as search_col:
                search_type = gr.Radio(
                    choices=[
                        ("MMR", "mmr"),
                        ("Similarity", "similarity"),
                        ("Hybrid", "hybrid"),
                    ],
                    value="mmr",
                    label="Search",
                    info="",
                    interactive=True,
                )
            with gr.Column(scale=2, min_width=180, visible=False) as filter_col:
                doc_filter = gr.Dropdown(
                    choices=["All Documents"] + available_sources,
                    value="All Documents",
                    label="Document",
                    info="",
                    interactive=True,
                )

        # Advanced options - more compact
        with gr.Accordion("⚙️ Advanced", open=False, visible=False) as advanced_options:
            with gr.Row():
                with gr.Column(scale=1):
                    query_rewriting = gr.Checkbox(
                        value=False,
                        label="Query Rewriting",
                        info="",
                    )
                with gr.Column(scale=1):
                    reranking = gr.Checkbox(
                        value=False,
                        label="Re-ranking",
                        info="",
                    )
                hybrid_alpha = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=HYBRID_ALPHA_UI_DEFAULT,
                    step=5,
                    label="Hybrid Balance",
                    info="",
                    visible=False,
                    interactive=False,
                )

        # Chat interface - main focus
        chatbot = gr.Chatbot(
            type="messages",
            height=500,
            show_copy_button=True,
        )

        # Context panel - cleaner, shown below chatbot when RAG is enabled
        with gr.Column(visible=False) as context_section:
            context_header = gr.Markdown(
                "### 📄 Retrieved Context", elem_classes=["info-text"]
            )
            context_box = gr.Markdown(
                elem_classes=["context-panel"],
            )

        # Input area - cleaner
        with gr.Row():
            msg = gr.Textbox(
                label="",
                placeholder="Ask a question about your documents...",
                scale=9,
                container=False,
            )
            with gr.Column(scale=1, min_width=100):
                submit = gr.Button("Send", variant="primary", scale=1)
                clear = gr.Button("Clear", scale=1, variant="secondary")

        # Examples - cleaner presentation
        with gr.Accordion("💡 Example Questions", open=False):
            gr.Examples(
                examples=[
                    "What are the main challenges for achieving net zero emissions by 2050?",
                    "What is the projected global renewable energy capacity for 2028?",
                    "How is the global EV market expected to evolve in the coming years?",
                    "What percentage of global electricity generation came from renewables in 2023?",
                    "What are the key policy recommendations for accelerating clean energy transitions?",
                    "What is the projected investment in clean energy technologies for 2024?",
                    "How do critical minerals supply chains impact renewable energy deployment?",
                    "What are the exact CO2 emissions reduction targets in the Net Zero Roadmap?",
                    "What role does energy efficiency play in reducing global energy demand?",
                    "What are the specific challenges facing developing countries in clean energy transitions?",
                ],
                inputs=msg,
                label="",
            )

    return {
        "demo": demo,
        "rag_enabled": rag_enabled,
        "search_type": search_type,
        "doc_filter": doc_filter,
        "query_rewriting": query_rewriting,
        "reranking": reranking,
        "hybrid_alpha": hybrid_alpha,
        "chatbot": chatbot,
        "context_box": context_box,
        "msg": msg,
        "submit": submit,
        "clear": clear,
        "search_col": search_col,
        "filter_col": filter_col,
        "context_section": context_section,
        "advanced_options": advanced_options,
    }


