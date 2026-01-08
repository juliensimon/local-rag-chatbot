"""Vectorstore management for document storage and retrieval."""

import glob
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_PATH, CHUNK_OVERLAP, CHUNK_SIZE, PDF_PATH


def get_text_splitter():
    """Create text splitter with optimal settings.

    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )


def get_pdf_files():
    """Get list of PDF files from the specified directory.

    Returns:
        List[str]: List of PDF file paths
    """
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
    """Process and filter documents into chunks.

    Args:
        documents: List of document objects
        text_splitter: Text splitter instance

    Returns:
        List[Document]: Filtered document chunks
    """
    chunks = text_splitter.split_documents(documents)
    return [chunk for chunk in chunks if filter_metadata(chunk)]


def load_or_create_vectorstore(embeddings):
    """Load existing vectorstore or create a new one.

    Args:
        embeddings: Embedding model instance

    Returns:
        Chroma: Loaded or newly created vectorstore
    """
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
        raise FileNotFoundError("No PDF files found in directory.")

    collection = vectorstore.get()
    if not collection or not collection.get("metadatas"):
        processed_files = set()
    else:
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


def add_documents_in_batches(vectorstore, documents, batch_size=5000):
    """Add documents to vectorstore in batches to avoid exceeding ChromaDB limits.

    Args:
        vectorstore: Chroma vectorstore instance
        documents: List of documents to add
        batch_size: Maximum documents per batch (ChromaDB limit is 5461)
    """
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        print(f"Adding batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} ({len(batch)} documents)...")
        vectorstore.add_documents(batch)


def update_vectorstore(vectorstore, new_pdfs, processed_files):
    """Update existing vectorstore with new documents.

    Args:
        vectorstore: Existing Chroma vectorstore
        new_pdfs: List of new PDF file paths
        processed_files: Set of already processed file paths
    """
    print(f"Found {len(new_pdfs)} new PDF files to process...")
    loader = DirectoryLoader(PDF_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    new_documents = [
        doc for doc in documents if doc.metadata.get("source") not in processed_files
    ]

    filtered_chunks = process_documents(new_documents, get_text_splitter())
    if filtered_chunks:
        print(f"Adding {len(filtered_chunks)} new document chunks to existing database...")
        add_documents_in_batches(vectorstore, filtered_chunks)
        print("Database updated successfully!")


def create_new_vectorstore(embeddings):
    """Create a new vectorstore from documents.

    Args:
        embeddings: Embedding model instance

    Returns:
        Chroma: Newly created vectorstore

    Exits if no PDF files are found.
    """
    print("Creating new Chroma database...")
    pdf_files = get_pdf_files()
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{PDF_PATH}' directory. "
            f"Please add PDF files and run again."
        )

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

