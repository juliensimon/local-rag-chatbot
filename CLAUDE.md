# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Start local LLM server (required before running app)
llama-server -hf arcee-ai/Trinity-Mini-GGUF:Q8_0

# Run FastAPI backend
uvicorn api.main:app --reload

# Run React frontend (in separate terminal)
cd frontend && npm run dev

# Run CLI for testing RAG functionality
python cli.py

# Run all tests with coverage (enforces 80% minimum)
pytest

# Run specific test file
pytest tests/test_utils.py

# Run specific test
pytest tests/test_utils.py::test_format_chat_history_empty

# Run tests without coverage (faster)
pytest --no-cov

# Frontend build/test
cd frontend && npm run build
cd frontend && npm run lint
```

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) chatbot built with LangChain, FastAPI, and React.

### Core Data Flow
1. **Document Ingestion** (`vectorstore.py`): PDFs from `pdf/` → chunked with RecursiveCharacterTextSplitter (512 chars, 128 overlap) → embedded via HuggingFace → stored in ChromaDB at `vectorstore/`
2. **Query Processing** (`qa_chain.py`): User question → optional query rewriting → retrieval (MMR/similarity/hybrid) → optional cross-encoder reranking → LLM response with streaming
3. **API Layer** (`api/`): FastAPI backend with REST endpoints and SSE streaming for the React frontend
4. **UI Layer** (`frontend/`): Modern React interface with real-time streaming and source highlighting

### Key Components

- **`config.py`**: All configurable parameters (chunk size, retrieval settings, prompt templates). Environment variables: `OPENAI_BASE_URL`, `OPENAI_MODEL`, `CHROMA_PATH`, `PDF_PATH`, `EMBEDDING_MODEL`, `RERANKER_MODEL`

- **`models.py`**: Factory functions for LLM (`ChatOpenAI` pointing to local llama-server at port 8080) and embeddings (`BAAI/bge-small-en-v1.5`)

- **`qa_chain.py`**: `QAChainWrapper` orchestrates the RAG pipeline. The `stream()` method is the main entry point - accepts inputs dict with `question`, `chat_history`, `filter`, `search_type`, `use_query_rewriting`, `use_reranking`, and `hybrid_alpha`. Supports three search types:
  - `mmr`: Maximal Marginal Relevance for diverse results
  - `similarity`: Pure vector similarity
  - `hybrid`: Combines semantic + BM25 keyword search (via `HybridRetriever`)

- **`retrievers.py`**: `HybridRetriever` implements BM25 keyword search and fuses scores with semantic search using configurable alpha weighting. Builds BM25 index lazily on first hybrid search.

- **`vectorstore.py`**: Handles ChromaDB persistence, incremental PDF updates, and document batching to avoid ChromaDB size limits

- **`api/routes.py`**: FastAPI endpoints for health, sources, and chat (streaming via SSE)

- **`api/streaming.py`**: SSE streaming implementation with events: `token`, `context`, `done`, `error`

### Search Type Implementation Details
- **MMR**: Uses `lambda_mult` (0.7 default) to balance relevance vs diversity. Fetches 10 candidates, returns 3.
- **Hybrid**: BM25 scores normalized to 0-1 via min-max scaling, fused with semantic via `alpha` weight (default 0.7 = 70% semantic)
- **Reranking**: Cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores top 20 candidates down to 3

### Testing
Tests extensively mock LLMs and vectorstores. Coverage enforced at 80% minimum via pytest config. Key fixtures in `tests/conftest.py`: `mock_embeddings`, `mock_vectorstore`, `mock_llm`, `mock_reranker`, `sample_documents`, `mock_hybrid_results`.

### React Frontend Architecture

The `frontend/` directory contains a modern React UI built with:
- **React 18 + TypeScript** with Vite for fast development
- **Tailwind CSS v3** with shadcn/ui components (accessible, customizable)
- **React Query** for server state management
- **SSE streaming** for real-time response display

Key directories:
- `src/api/` - API client and SSE streaming utilities
- `src/components/ui/` - shadcn/ui base components
- `src/components/chat/` - Chat interface (MessageList, ChatInput, ChatMessage)
- `src/components/controls/` - RAG settings (RagToggle, SearchTypeSelector, DocumentFilter)
- `src/components/context/` - Retrieved documents panel (ContextPanel, SourceCard)
- `src/context/` - React Context providers (ThemeContext, SettingsContext, ChatContext)
- `src/hooks/` - Custom hooks (useSources, useHealthCheck)
- `src/types/` - TypeScript types matching `api/schemas.py`

API endpoints consumed:
- `GET /api/health` - Health check
- `GET /api/sources` - Available documents
- `POST /api/chat/stream` - SSE streaming chat (events: `token`, `context`, `done`, `error`)

### Frontend Library Notes

**react-markdown v9+**: The `className` prop was removed. Wrap `<ReactMarkdown>` in a container div and apply styles there:
```tsx
// Correct pattern for v9+
<div className="prose prose-sm dark:prose-invert">
  <ReactMarkdown>{content}</ReactMarkdown>
</div>
```

**@tailwindcss/typography**: Provides `prose` utility class for markdown styling. Use `prose-invert` for dark mode. Custom spacing via modifiers like `prose-p:my-1`, `prose-ul:my-1`.

**Tailwind CSS v3**: This project uses Tailwind v3 (not v4). The v4 PostCSS plugin moved to a separate package and has breaking changes with shadcn/ui.

### Backend API Notes

**numpy float serialization**: When returning scores from hybrid search, convert numpy.float32 to native Python floats for JSON serialization:
```python
source_doc.score = float(score) if score is not None else None
```
