"""
RAG (Retrieval-Augmented Generation) utilities for document processing and querying.

This module provides functions for:
1. Loading and processing PDF documents
2. Creating and managing vector embeddings
3. Setting up language models and QA chains
4. Handling both RAG and vanilla LLM responses
"""

import glob
import os
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Disable tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration - use environment variables with sensible defaults
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "dummy-key-not-needed"
OPENAI_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "dummy")
CHROMA_PATH = os.getenv("CHROMA_PATH", "vectorstore")
PDF_PATH = os.getenv("PDF_PATH", "pdf")

# RAG configuration
RETRIEVER_K = 3  # Number of documents to retrieve
CHAT_HISTORY_LIMIT = 5  # Number of recent messages to include in context


def create_llm(streaming=False):
    """Initialize the OpenAI-compatible language model.

    Args:
        streaming (bool): Whether to enable response streaming

    Returns:
        ChatOpenAI: Configured language model instance
    """
    return ChatOpenAI(
        model=OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        base_url=OPENAI_URL,
        streaming=streaming,
    )


def create_embeddings():
    """Initialize the embedding model."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_text_splitter():
    """Create text splitter with optimal settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        length_function=len,
        add_start_index=True,
    )


def get_pdf_files():
    """Get list of PDF files from the specified directory."""
    if not os.path.exists(PDF_PATH):
        os.makedirs(PDF_PATH)
        return []
    return list(glob.glob(os.path.join(PDF_PATH, "*.pdf")))


def filter_metadata(doc):
    """Filter out unwanted sections from documents.

    Args:
        doc: Document object with metadata

    Returns:
        bool: True if document should be kept, False if filtered out
    """
    skip_sections = {"references", "acknowledgments", "appendix"}
    section = doc.metadata.get("section", "").lower()
    return not any(s in section for s in skip_sections)


def process_documents(documents, text_splitter):
    """Process and filter documents into chunks."""
    chunks = text_splitter.split_documents(documents)
    return [chunk for chunk in chunks if filter_metadata(chunk)]


def load_or_create_vectorstore(embeddings):
    """Load existing vectorstore or create a new one."""
    if os.path.exists(CHROMA_PATH):
        return handle_existing_vectorstore(embeddings)
    return create_new_vectorstore(embeddings)


def handle_existing_vectorstore(embeddings):
    """Handle loading and updating existing vectorstore.

    Args:
        embeddings: Embedding model instance

    Returns:
        Chroma: Loaded and potentially updated vectorstore

    Exits if no PDF files are found.
    """
    print("Loading existing Chroma database...")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    current_pdfs = get_pdf_files()
    if not current_pdfs:
        print("No PDF files found in directory.")
        sys.exit(1)

    collection = vectorstore.get()
    processed_files = {
        meta.get("source")
        for meta in collection["metadatas"]
        if meta and meta.get("source")
    }

    new_pdfs = [pdf for pdf in current_pdfs if pdf not in processed_files]

    if new_pdfs:
        update_vectorstore(vectorstore, new_pdfs, processed_files)
    else:
        print("No new PDF files to process.")

    return vectorstore


def update_vectorstore(vectorstore, new_pdfs, processed_files):
    """Update existing vectorstore with new documents."""
    print(f"Found {len(new_pdfs)} new PDF files to process...")
    loader = DirectoryLoader(PDF_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    new_documents = [
        doc for doc in documents if doc.metadata.get("source") not in processed_files
    ]

    filtered_chunks = process_documents(new_documents, get_text_splitter())
    if filtered_chunks:
        print("Adding new documents to existing database...")
        vectorstore.add_documents(filtered_chunks)
        print("Database updated successfully!")


def create_new_vectorstore(embeddings):
    """Create a new vectorstore from documents."""
    print("Creating new Chroma database...")
    pdf_files = get_pdf_files()
    if not pdf_files:
        print(f"No PDF files found in '{PDF_PATH}' directory!")
        print(f"Please add your PDF files to the '{PDF_PATH}' directory and run again.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF files to process...")
    print("(This may take a while as documents need to be processed and embedded)")

    os.makedirs(CHROMA_PATH, exist_ok=True)

    loader = DirectoryLoader(PDF_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    filtered_chunks = process_documents(documents, get_text_splitter())

    return Chroma.from_documents(
        documents=filtered_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )


def format_chat_history(chat_history, limit=CHAT_HISTORY_LIMIT):
    """Format chat history for inclusion in prompts.

    Args:
        chat_history: List of message tuples or Message objects
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
        elif isinstance(msg, HumanMessage):
            history_parts.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            history_parts.append(f"Assistant: {msg.content}")

    return "\n".join(history_parts)


# Prompt template for RAG responses
RAG_PROMPT_TEMPLATE = """Answer the question using your own knowledge and the provided context.

Context:
{context}

Question: {question}

Previous conversation:
{chat_history}

Answer:"""


class QAChainWrapper:
    """Wrapper for RAG question-answering with streaming support."""

    def __init__(self, retriever, prompt):
        """Initialize the QA chain wrapper.

        Args:
            retriever: Document retriever instance
            prompt: ChatPromptTemplate for generating responses
        """
        self._retriever = retriever
        self._prompt = prompt

    @property
    def retriever(self):
        """Return the retriever for external access."""
        return self._retriever

    def stream(self, inputs):
        """Stream the chain response token by token.

        Args:
            inputs: Dict with 'question' and optional 'chat_history'

        Yields:
            dict: Contains 'chunk' (token text) and 'source_documents'
        """
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])

        # Retrieve documents first
        docs = self._retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Format chat history
        history_str = format_chat_history(chat_history)

        # Create streaming LLM
        streaming_llm = create_llm(streaming=True)

        # Stream the response with error handling
        chain = self._prompt | streaming_llm
        try:
            for chunk in chain.stream({
                "question": question,
                "context": context,
                "chat_history": history_str
            }):
                yield {"chunk": chunk.content, "source_documents": docs}
        except Exception as e:
            yield {"chunk": f"\n\n[Error: {e}]", "source_documents": docs}


def create_qa_chain(vectorstore):
    """Create the question-answering chain.

    Args:
        vectorstore: Vector store instance for document retrieval

    Returns:
        QAChainWrapper: Configured QA chain with streaming support
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVER_K}
    )
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    return QAChainWrapper(retriever, prompt)


def main():
    """Main execution function for CLI testing."""
    import time

    embeddings = create_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    qa_chain = create_qa_chain(vectorstore)

    query = "Tell me about Arcee Fusion."
    chat_history = []

    # Test vanilla response
    print("\n=== Vanilla Response (No RAG) ===")
    streaming_llm = create_llm(streaming=True)
    print("Answer: ", end="", flush=True)
    try:
        for chunk in streaming_llm.stream(query):
            print(chunk.content, end="", flush=True)
        print()
    except Exception as e:
        print(f"\nError: {e}")

    # Test RAG response
    print("\n=== RAG Response ===")
    print("Answer: ", end="", flush=True)
    try:
        for chunk_data in qa_chain.stream({"question": query, "chat_history": chat_history}):
            print(chunk_data["chunk"], end="", flush=True)
        print()

        # Print sources
        if chunk_data.get("source_documents"):
            print("\nSources:")
            seen_sources = set()
            for doc in chunk_data["source_documents"]:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "unknown")
                source_key = f"{source}:{page}"
                if source_key not in seen_sources:
                    print(f"- {source}, page {page}")
                    seen_sources.add(source_key)
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
