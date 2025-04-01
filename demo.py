import glob
import os

# Set tokenizers parallelism and transformers cache before importing other libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "https://conductor.arcee.ai/v1"
OPENAI_MODEL = "auto"
CHROMA_PATH = "vectorstore"
# CHROMA_PATH = "/data/vectorstore"
PDF_PATH = "pdf"


def create_llm(streaming=False):
    """Initialize the OpenAI language model.

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
        exit(1)

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
        exit(1)

    print(f"Found {len(pdf_files)} PDF files to process...")
    print("(This may take a while as documents need to be processed and embedded)")

    os.makedirs(CHROMA_PATH, exist_ok=True)

    loader = DirectoryLoader(PDF_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    filtered_chunks = process_documents(documents, get_text_splitter())

    return Chroma.from_documents(
        documents=filtered_chunks, embedding=embeddings, persist_directory=CHROMA_PATH
    )


def create_qa_chain(llm, vectorstore):
    """Create the question-answering chain."""
    prompt_template = """Answer the question using your own knowledge and the provided context.

Context:
{context}

Question: {question}

Previous conversation:
{chat_history}

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        chain_type="stuff",
        verbose=True,
    )


def get_vanilla_response(llm, query):
    """Get response from vanilla chain without RAG with streaming."""
    streaming_llm = create_llm(streaming=True)

    vanilla_prompt = ChatPromptTemplate.from_template(
        """
Question: {question}

Instructions:
- If you don't know the answer, say so
- Be concise and clear
- Only state what you're confident about

Answer:"""
    )

    chain = vanilla_prompt | streaming_llm

    print("\n=== Vanilla Response (No RAG) ===")

    try:
        print("Answer: ", end="", flush=True)
        for chunk in chain.stream({"question": query}):
            print(chunk.content, end="", flush=True)
        print()  # New line after streaming completes

    except Exception as e:
        print(f"\nError getting vanilla response: {str(e)}")


def get_rag_response(qa_chain, query, chat_history):
    """Get response from RAG-powered chain with streaming.

    Args:
        qa_chain: The QA chain instance
        query (str): User's question
        chat_history (list): Previous conversation history
    """
    print("\n=== RAG Response ===")

    try:
        # First get the result to ensure we have the answer and sources
        result = qa_chain.invoke({"question": query, "chat_history": chat_history})

        # Print the answer with character-by-character "streaming" simulation
        print("\nAnswer: ", end="", flush=True)
        answer = result.get("answer", "No answer found.")
        for char in answer:
            print(char, end="", flush=True)
            time.sleep(0.005)  # Small delay to simulate streaming
        print()

        # Print sources
        if result.get("source_documents"):
            print("\nSources:")
            seen_sources = set()
            for doc in result["source_documents"]:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "unknown")
                source_key = f"{source}:{page}"

                if source_key not in seen_sources:
                    print(f"- {source}, page {page}")
                    seen_sources.add(source_key)

    except Exception as e:
        print(f"\nError: {str(e)}")


def main():
    """Main execution function."""
    # Create streaming LLM for RAG
    llm = create_llm(streaming=True)
    embeddings = create_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    qa_chain = create_qa_chain(llm, vectorstore)

    chat_history = []
    query = "Tell me about Arcee Fusion."

    # Get both vanilla and RAG responses
    get_vanilla_response(llm, query)
    get_rag_response(qa_chain, query, chat_history)


if __name__ == "__main__":
    main()
