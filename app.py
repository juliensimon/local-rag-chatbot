import gradio as gr
from demo import create_llm, create_embeddings, load_or_create_vectorstore, create_qa_chain

def initialize_chain():
    """Initialize the RAG chain and return it."""
    llm = create_llm()
    embeddings = create_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    return create_qa_chain(llm, vectorstore)

def chat_response(message, history, query_type):
    """Process chat messages and return responses using either RAG or vanilla LLM.
    
    Args:
        message (str): The user's input message
        history (list): List of previous message-response pairs
        query_type (str): Either "RAG" or "Vanilla LLM"
    
    Returns:
        str: The formatted response text including sources for RAG queries
    """
    # Convert Gradio history to list of tuples
    chat_history = [(msg, resp) for msg, resp in history] if history else []
    
    if query_type == "RAG":
        result = qa_chain.invoke({
            "question": message,
            "chat_history": chat_history
        })
        response_text = result["answer"]
        
        # Add source citations if available
        if result.get("source_documents"):
            sources = []
            seen_sources = set()
            
            # Deduplicate sources while preserving order
            for doc in result["source_documents"]:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'unknown')
                source_key = f"{source}:{page}"
                
                if source_key not in seen_sources:
                    sources.append(f"- {source}, page {page}")
                    seen_sources.add(source_key)
            
            if sources:
                response_text += "\n\nSources:\n" + "\n".join(sources)
    else:
        # Use vanilla LLM for direct responses
        result = llm.invoke(message)
        response_text = result.content
    
    return response_text

# Initialize components globally
llm = create_llm()
qa_chain = initialize_chain()

# Create the Gradio interface
with gr.Blocks() as demo:
    # UI Setup
    gr.Markdown("# RAG-Powered Document Chat with Arcee Conductor")
    gr.Markdown("Ask questions about your documents. The system will provide answers based on the content of your PDFs.")
    
    rag_enabled = gr.Checkbox(
        value=True,
        label="Enable RAG",
        info="Toggle between RAG-powered document search or vanilla LLM responses"
    )
    
    chatbot = gr.Chatbot()
    context_box = gr.Textbox(
        label="Retrieved Context",
        interactive=False,
        visible=True,
        lines=5
    )
    
    # Input controls
    with gr.Row():
        msg = gr.Textbox(label="Query", scale=8)
        with gr.Column(scale=1):
            submit = gr.Button("Submit")
            clear = gr.Button("Clear")
    
    def respond(message, chat_history, is_rag_enabled):
        """Handle user messages and generate responses.
        
        Args:
            message (str): User input message
            chat_history (list): Previous conversation history
            is_rag_enabled (bool): Whether to use RAG or vanilla LLM
            
        Returns:
            tuple: (cleared message, updated history, context, RAG state)
        """
        if not message:
            return "", chat_history, "", is_rag_enabled
        
        query_type = "RAG" if is_rag_enabled else "Vanilla LLM"
        bot_message = chat_response(message, chat_history, query_type)
        
        # Create new history to avoid modifying the original
        new_history = list(chat_history)
        new_history.append((message, bot_message))
        
        # Get relevant context for RAG queries
        context = ""
        if is_rag_enabled and qa_chain.retriever:
            docs = qa_chain.retriever.get_relevant_documents(message)
            context = "\n\n".join(doc.page_content for doc in docs)
        
        return "", new_history, context, is_rag_enabled

    def update_context_visibility(is_rag_enabled):
        """Update context box visibility based on RAG toggle."""
        return gr.update(visible=is_rag_enabled, value="" if not is_rag_enabled else None)

    # Wire up event handlers
    msg.submit(respond, [msg, chatbot, rag_enabled], [msg, chatbot, context_box, rag_enabled])
    submit.click(respond, [msg, chatbot, rag_enabled], [msg, chatbot, context_box, rag_enabled])
    clear.click(lambda: [[], "", True], None, [chatbot, context_box, rag_enabled], queue=False)
    rag_enabled.change(update_context_visibility, rag_enabled, context_box)

    # Add example queries
    gr.Examples(
        examples=[
            "Tell me about Arcee Fusion.",
            "How does deepseek-R1 differ from deepseek-v3?"
        ],
        inputs=msg
    )

if __name__ == "__main__":
    demo.launch(share=False)
