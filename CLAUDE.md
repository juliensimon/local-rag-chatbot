# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Start local LLM server (required before running app)
llama-server -hf arcee-ai/Trinity-Mini-GGUF:Q8_0

# Run Gradio web app
python app.py

# Run CLI for testing RAG functionality
python cli.py

# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_utils.py

# Run specific test
pytest tests/test_utils.py::test_format_chat_history_empty
```

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) chatbot built with LangChain and Gradio.

### Core Data Flow
1. **Document Ingestion** (`vectorstore.py`): PDFs from `pdf/` → chunked with RecursiveCharacterTextSplitter → embedded via HuggingFace → stored in ChromaDB at `vectorstore/`
2. **Query Processing** (`qa_chain.py`): User question → optional query rewriting → retrieval (MMR/similarity/hybrid) → optional cross-encoder reranking → LLM response with streaming
3. **UI Layer** (`ui/`): Gradio interface handles user interaction and displays retrieved context with source highlighting

### Key Components

- **`config.py`**: All configurable parameters (chunk size, retrieval settings, prompt templates). Environment variables: `OPENAI_BASE_URL`, `OPENAI_MODEL`, `CHROMA_PATH`, `PDF_PATH`, `EMBEDDING_MODEL`, `RERANKER_MODEL`

- **`models.py`**: Factory functions for LLM (`ChatOpenAI` pointing to local llama-server) and embeddings (`BAAI/bge-small-en-v1.5`)

- **`qa_chain.py`**: `QAChainWrapper` orchestrates the RAG pipeline. Supports three search types:
  - `mmr`: Maximal Marginal Relevance for diverse results
  - `similarity`: Pure vector similarity
  - `hybrid`: Combines semantic + BM25 keyword search (via `HybridRetriever`)

- **`retrievers.py`**: `HybridRetriever` implements BM25 keyword search and fuses scores with semantic search using configurable alpha weighting

- **`vectorstore.py`**: Handles ChromaDB persistence, incremental PDF updates, and document batching to avoid ChromaDB size limits

- **`ui/handlers.py`**: Streaming response handlers, path traversal protection for document filters, message format conversion

### Search Type Implementation Details
- **MMR**: Uses `lambda_mult` (0.7 default) to balance relevance vs diversity
- **Hybrid**: BM25 scores normalized to 0-1, fused with semantic via `alpha` weight
- **Reranking**: Cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores top candidates

### Testing
Tests extensively mock LLMs and vectorstores. Coverage target: 80%. Key fixtures in `tests/conftest.py`: `mock_embeddings`, `mock_vectorstore`, `mock_llm`, `mock_reranker`.
