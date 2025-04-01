# Import necessary libraries
import gradio as gr
from demo import create_embeddings  # Function to create text embeddings
from demo import create_llm  # Function to initialize the language model
from demo import create_qa_chain  # Function to create question-answering chain
from demo import (
    load_or_create_vectorstore,  # Function to load or create vector database
)


def initialize_chain():
    """Initialize the RAG (Retrieval-Augmented Generation) chain and return it.

    This function sets up all components needed for document-based Q&A:
    1. Language model for generating responses
    2. Embeddings model for converting text to vectors
    3. Vector store for document retrieval
    4. QA chain that combines retrieval with generation
    """
    llm = create_llm()  # Initialize the language model
    embeddings = create_embeddings()  # Initialize the embeddings model
    vectorstore = load_or_create_vectorstore(
        embeddings
    )  # Load or create the vector database
    return create_qa_chain(llm, vectorstore)  # Create and return the QA chain


def chat_response(message, history, query_type):
    """Process chat messages and return responses using either RAG or vanilla LLM.

    This function handles two modes of operation:
    1. RAG mode: Retrieves relevant document chunks and generates answers based on them
    2. Vanilla LLM mode: Directly queries the LLM without document context

    Args:
        message (str): The user's input message/question
        history (list): List of previous message-response pairs from the chat
        query_type (str): Either "RAG" or "Vanilla LLM" to determine response method

    Returns:
        str: The formatted response text including sources for RAG queries
    """
    # Convert Gradio history format to the format expected by our chain
    chat_history = [(msg, resp) for msg, resp in history] if history else []

    if query_type == "RAG":
        # Use RAG to get answer with context from documents
        result = qa_chain.invoke({"question": message, "chat_history": chat_history})
        response_text = result["answer"]

        # Add source citations if available
        if result.get("source_documents"):
            # Track seen sources to avoid duplicates
            sources = []
            seen_sources = set()

            # Deduplicate sources while preserving order
            # This ensures we don't list the same source multiple times
            for doc in result["source_documents"]:
                source = doc.metadata.get("source", "Unknown")  # Get document source
                page = doc.metadata.get("page", "unknown")  # Get page number
                source_key = f"{source}:{page}"  # Create unique key for deduplication

                # Only add source if we haven't seen it before
                if source_key not in seen_sources:
                    sources.append(f"- {source}, page {page}")
                    seen_sources.add(source_key)

            # Append sources to response if any were found
            if sources:
                response_text += "\n\nSources:\n" + "\n".join(sources)
    else:
        # Use vanilla LLM for direct responses without document context
        # This bypasses the retrieval step and directly queries the language model
        result = llm.invoke(message)
        response_text = result.content

    return response_text


# Initialize components once at startup for better performance
# This avoids recreating these expensive objects on each request
llm = create_llm()  # Global LLM instance
qa_chain = initialize_chain()  # Global QA chain instance

# Create the Gradio interface
with gr.Blocks() as demo:
    # Header and description section
    gr.Markdown("# RAG-Powered Document Chat with Arcee Conductor")
    gr.Markdown(
        "Ask questions about your documents. The system will provide answers based on the content of your PDFs."
    )

    # RAG toggle switch - allows switching between RAG and vanilla LLM modes
    rag_enabled = gr.Checkbox(
        value=True,  # Default to RAG mode
        label="Enable RAG",
        info="Toggle between RAG-powered document search or vanilla LLM responses",
    )

    # Chat interface components
    chatbot = gr.Chatbot()  # Main chat interface showing conversation history
    context_box = gr.Textbox(
        label="Retrieved Context",  # Shows the raw context retrieved from documents
        interactive=False,  # User can't edit this
        visible=True,  # Initially visible when RAG is enabled
        lines=5,  # Height of the text box
    )

    # Input area for user queries
    with gr.Row():
        msg = gr.Textbox(label="Query", scale=8)  # Text input for user questions
        with gr.Column(scale=1):
            submit = gr.Button("Submit")  # Submit button
            clear = gr.Button("Clear")  # Clear conversation button

    def respond(message, chat_history, is_rag_enabled):
        """Generate response to user message and update the UI.

        This function:
        1. Processes the user's message
        2. Gets a response using the appropriate method (RAG or vanilla)
        3. Updates the chat history
        4. Retrieves context for display (in RAG mode)

        Args:
            message (str): User input message
            chat_history (list): Previous conversation history
            is_rag_enabled (bool): Whether to use RAG or vanilla LLM

        Returns:
            tuple: (cleared message, updated history, context, RAG state)
                - cleared message: Empty string to clear the input box
                - updated history: New chat history with the latest exchange
                - context: Retrieved document context (for RAG mode)
                - RAG state: Current state of the RAG toggle
        """
        # Return early if message is empty
        if not message:
            return "", chat_history, "", is_rag_enabled

        # Get response using appropriate method based on RAG toggle
        query_type = "RAG" if is_rag_enabled else "Vanilla LLM"
        bot_message = chat_response(message, chat_history, query_type)

        # Update chat history with new exchange
        # Create a new list to avoid modifying the original
        new_history = list(chat_history)
        new_history.append((message, bot_message))

        # Get context for display (only for RAG mode)
        # This shows the raw document chunks that were retrieved
        context = ""
        if is_rag_enabled and qa_chain.retriever:
            docs = qa_chain.retriever.get_relevant_documents(message)
            context = "\n\n".join(doc.page_content for doc in docs)

        # Return values to update the UI
        return "", new_history, context, is_rag_enabled

    def update_context_visibility(is_rag_enabled):
        """Show/hide context box based on RAG toggle.

        Args:
            is_rag_enabled (bool): Whether RAG mode is enabled

        Returns:
            gr.update: Update object to modify the context box visibility
        """
        return gr.update(
            visible=is_rag_enabled,  # Only show context box in RAG mode
            value="" if not is_rag_enabled else None,  # Clear content when hiding
        )

    # Connect event handlers to UI components
    # When user presses Enter in the message box
    msg.submit(
        respond, [msg, chatbot, rag_enabled], [msg, chatbot, context_box, rag_enabled]
    )
    # When user clicks the Submit button
    submit.click(
        respond, [msg, chatbot, rag_enabled], [msg, chatbot, context_box, rag_enabled]
    )
    # When user clicks the Clear button - reset everything
    clear.click(
        lambda: [[], "", True], None, [chatbot, context_box, rag_enabled], queue=False
    )
    # When user toggles the RAG switch - update context box visibility
    rag_enabled.change(update_context_visibility, rag_enabled, context_box)

    # Example queries to help users get started
    gr.Examples(
        examples=[
            "Tell me about Arcee Fusion.",
            "How does deepseek-R1 differ from deepseek-v3?",
            "What is the main innovation in DELLA merging?",
        ],
        inputs=msg,
    )

# Run the app when this script is executed directly
if __name__ == "__main__":
    demo.launch(share=False)  # Launch the Gradio interface locally
