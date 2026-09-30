"""Vectorstore management for document storage and retrieval."""

import glob
import os
import re

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_COLLECTION_NAME,
    PDF_PATH,
    USER_COLLECTION_PREFIX,
    USER_ID_PATTERN,
    USER_PDF_ROOT,
)


def user_paths(user_id):
    """Resolve a user's PDF directory and Chroma collection name.

    Args:
        user_id: User identifier matching USER_ID_PATTERN

    Returns:
        tuple: (pdf_path, collection_name)

    Raises:
        ValueError: If user_id is not a safe identifier
    """
    if not re.fullmatch(USER_ID_PATTERN, user_id or ""):
        raise ValueError(f"Invalid user id: {user_id!r}")
    return os.path.join(USER_PDF_ROOT, user_id), f"{USER_COLLECTION_PREFIX}{user_id}"


def get_vectorstore_sources(vectorstore):
    """List the distinct source filenames stored in a vectorstore.

    Args:
        vectorstore: Chroma vectorstore instance

    Returns:
        List[str]: Sorted basenames of indexed PDF files
    """
    collection = vectorstore.get()
    if not collection or not collection.get("metadatas"):
        return []
    return sorted(
        {
            os.path.basename(meta["source"])
            for meta in collection["metadatas"]
            if meta and meta.get("source")
        }
    )


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


def get_pdf_files(pdf_path=None):
    """Get list of PDF files from the specified directory.

    Args:
        pdf_path: Directory to scan (defaults to PDF_PATH)

    Returns:
        List[str]: List of PDF file paths
    """
    pdf_path = pdf_path or PDF_PATH
    if not os.path.exists(pdf_path):
        os.makedirs(pdf_path)
        return []
    return list(glob.glob(os.path.join(pdf_path, "*.pdf")))


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


def load_or_create_vectorstore(embeddings, pdf_path=None, collection_name=None):
    """Load existing vectorstore or create a new one.

    Args:
        embeddings: Embedding model instance
        pdf_path: Directory holding the corpus PDFs (defaults to PDF_PATH)
        collection_name: Chroma collection (defaults to DEFAULT_COLLECTION_NAME)

    Returns:
        Chroma: Loaded or newly created vectorstore
    """
    pdf_path = pdf_path or PDF_PATH
    collection_name = collection_name or DEFAULT_COLLECTION_NAME
    # CHROMA_PATH is shared by all collections: a collection missing from an
    # existing store is created empty and filled by the incremental update path.
    if os.path.exists(CHROMA_PATH):
        return handle_existing_vectorstore(embeddings, pdf_path, collection_name)
    return create_new_vectorstore(embeddings, pdf_path, collection_name)


def handle_existing_vectorstore(embeddings, pdf_path=None, collection_name=None):
    """Handle loading and updating existing vectorstore.

    Args:
        embeddings: Embedding model instance
        pdf_path: Directory holding the corpus PDFs (defaults to PDF_PATH)
        collection_name: Chroma collection (defaults to DEFAULT_COLLECTION_NAME)

    Returns:
        Chroma: Loaded and potentially updated vectorstore

    Exits if no PDF files are found.
    """
    pdf_path = pdf_path or PDF_PATH
    collection_name = collection_name or DEFAULT_COLLECTION_NAME
    print(f"Loading existing Chroma database (collection '{collection_name}')...")
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    current_pdfs = get_pdf_files(pdf_path)
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
        update_vectorstore(vectorstore, new_pdfs, processed_files, pdf_path)
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


def update_vectorstore(vectorstore, new_pdfs, processed_files, pdf_path=None):
    """Update existing vectorstore with new documents.

    Args:
        vectorstore: Existing Chroma vectorstore
        new_pdfs: List of new PDF file paths
        processed_files: Set of already processed file paths
        pdf_path: Directory holding the corpus PDFs (defaults to PDF_PATH)
    """
    print(f"Found {len(new_pdfs)} new PDF files to process...")
    loader = DirectoryLoader(pdf_path or PDF_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    new_documents = [
        doc for doc in documents if doc.metadata.get("source") not in processed_files
    ]

    filtered_chunks = process_documents(new_documents, get_text_splitter())
    if filtered_chunks:
        print(f"Adding {len(filtered_chunks)} new document chunks to existing database...")
        add_documents_in_batches(vectorstore, filtered_chunks)
        print("Database updated successfully!")


def create_new_vectorstore(embeddings, pdf_path=None, collection_name=None):
    """Create a new vectorstore from documents.

    Args:
        embeddings: Embedding model instance
        pdf_path: Directory holding the corpus PDFs (defaults to PDF_PATH)
        collection_name: Chroma collection (defaults to DEFAULT_COLLECTION_NAME)

    Returns:
        Chroma: Newly created vectorstore

    Exits if no PDF files are found.
    """
    pdf_path = pdf_path or PDF_PATH
    collection_name = collection_name or DEFAULT_COLLECTION_NAME
    print(f"Creating new Chroma database (collection '{collection_name}')...")
    pdf_files = get_pdf_files(pdf_path)
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{pdf_path}' directory. "
            f"Please add PDF files and run again."
        )

    print(f"Found {len(pdf_files)} PDF files to process...")
    print("(This may take a while as documents need to be processed and embedded)")

    os.makedirs(CHROMA_PATH, exist_ok=True)

    loader = DirectoryLoader(pdf_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    filtered_chunks = process_documents(documents, get_text_splitter())

    return Chroma.from_documents(
        documents=filtered_chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_PATH,
    )
