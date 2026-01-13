# Vector Store Explorer - Plan Mode Prompt

## Context
This is a RAG (Retrieval-Augmented Generation) system built with:
- **Backend**: FastAPI with LangChain, using ChromaDB vector store at `vectorstore/`
- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS (existing UI in `frontend/`)
- **Embeddings**: BAAI/bge-small-en-v1.5 (configurable via `EMBEDDING_MODEL_NAME`)
- **Chunking**: RecursiveCharacterTextSplitter with `CHUNK_SIZE=512`, `CHUNK_OVERLAP=128`
- **Retrieval**: Supports MMR, similarity, and hybrid (semantic + BM25) search
- **Metadata**: Documents include `source` (file path) and `page` (page number) metadata

The vector store is managed via `vectorstore.py` and uses LangChain's Chroma wrapper. The existing frontend uses Radix UI components, React Query, and follows a component-based architecture in `frontend/src/components/`.

## Objective
Design and implement a standalone web application in `explorer/` to explore, analyze, and optimize the Chroma vector store. The app should help users understand their vector store's health, identify optimization opportunities, and improve RAG system quality and accuracy.

## Requirements

### 1. Vector Store Statistics Dashboard
Display comprehensive statistics about the vector store:

**Collection-Level Stats:**
- Total number of documents/chunks
- Total number of unique source files
- Average chunk size (characters/tokens)
- Chunk size distribution (histogram)
- Embedding dimension (from embedding model)
- Database size on disk
- Last updated timestamp

**Source-Level Stats:**
- List of all source files with:
  - Number of chunks per source
  - Total pages per source
  - Average chunk size per source
  - Coverage (pages with chunks vs total pages)
- Source file size vs chunk count correlation

**Chunk Quality Metrics:**
- Chunk length distribution (identify too-short or too-long chunks)
- Overlap analysis (chunks with high similarity to neighbors)
- Metadata completeness (sources with missing metadata)
- Duplicate detection (near-duplicate chunks)

**Embedding Quality Indicators:**
- Vector similarity distribution (pairwise similarity histogram)
- Cluster analysis (identify document clusters)
- Embedding space coverage (dimensionality analysis)

### 2. RAG Optimization Recommendations
Provide actionable optimization tips based on analysis:

**Chunking Optimization:**
- Identify optimal chunk size based on current distribution
- Flag chunks that are too small (< 50 chars) or too large (> 2000 chars)
- Suggest overlap adjustments based on content boundaries
- Recommend chunking strategy improvements

**Retrieval Optimization:**
- Analyze query-document similarity patterns
- Identify low-quality retrievals (suggest re-ranking)
- Recommend optimal `k` values based on document distribution
- Suggest hybrid search alpha values based on content type

**Metadata Enhancement:**
- Flag missing or incomplete metadata
- Suggest metadata enrichment opportunities
- Identify sources with poor coverage

**Embedding Model Recommendations:**
- Compare current embedding model performance
- Suggest alternative models for specific use cases
- Flag embedding quality issues

**Content Quality:**
- Identify low-information chunks (very short, mostly whitespace, etc.)
- Flag potential noise (headers, footers, page numbers as standalone chunks)
- Suggest content filtering improvements

### 3. Interactive Exploration Features
- **Search Interface**: Test queries against the vector store with different search types (MMR, similarity, hybrid)
- **Document Browser**: Browse chunks by source file, page, or similarity
- **Similarity Visualization**: Visual representation of document relationships (t-SNE/UMAP if feasible)
- **Query Analysis**: Show how queries map to retrieved documents with scores
- **Filtering**: Filter by source file, page range, chunk size, etc.

### 4. Technical Requirements

**Backend (FastAPI):**
- New API endpoints in `explorer/api/` or extend existing `api/routes.py`
- Endpoints for:
  - `/api/explorer/stats` - Get comprehensive statistics
  - `/api/explorer/sources` - List all sources with details
  - `/api/explorer/chunks` - Get chunks with filtering/pagination
  - `/api/explorer/search` - Test search queries
  - `/api/explorer/recommendations` - Get optimization recommendations
  - `/api/explorer/analyze` - Run analysis (chunk quality, duplicates, etc.)
- Reuse existing vectorstore loading logic from `vectorstore.py`
- Reuse embedding model initialization from `models.py`
- Efficient batch processing for large vector stores

**Frontend (React + TypeScript):**
- New standalone app in `explorer/` directory
- Reuse UI components from `frontend/src/components/ui/` where possible
- Use same design system (Tailwind + Radix UI)
- Responsive layout with dashboard-style views
- Charts/visualizations (consider Chart.js, Recharts, or similar)
- Real-time updates for long-running analyses

**Architecture:**
- Can be integrated into existing app or standalone
- Share configuration from `config.py`
- Follow existing code patterns and structure

### 5. Success Criteria

**Functional:**
- ✅ All statistics load correctly for vector stores with 100+ documents
- ✅ Optimization recommendations are actionable and specific
- ✅ Search interface returns results matching main app behavior
- ✅ All visualizations render correctly and are interactive
- ✅ Performance: Statistics load in < 5 seconds for typical stores

**Quality:**
- ✅ Code follows existing patterns (type hints, docstrings, error handling)
- ✅ Frontend matches existing UI/UX patterns
- ✅ Comprehensive error handling for edge cases (empty store, corrupted data, etc.)
- ✅ Accessible UI (keyboard navigation, screen reader support)

**Testing:**
- ✅ Unit tests for statistics calculations
- ✅ Integration tests for API endpoints
- ✅ Frontend component tests for key features
- ✅ Test with various vector store sizes (small, medium, large)

### 6. Deliverables

**Documentation:**
- PRD (Product Requirements Document) with:
  - Detailed feature specifications
  - User stories and use cases
  - Technical architecture
  - API specifications
  - UI/UX mockups or wireframes
  - Testing strategy
  - Success metrics

**Implementation Tasks:**
- Backend API endpoints with proper error handling
- Frontend components and pages
- Statistics calculation logic
- Optimization recommendation engine
- Visualization components
- Integration with existing vectorstore infrastructure

**Testing:**
- Test suite covering core functionality
- Performance benchmarks
- Edge case handling

## Constraints
- Must work with existing ChromaDB structure at `vectorstore/`
- Should not modify existing RAG system code (read-only exploration)
- Must handle large vector stores efficiently (1000+ documents)
- Should be performant (avoid blocking operations)
- Must follow existing code style and patterns

## Optional Enhancements
- Export statistics as JSON/CSV
- Comparison mode (compare two vector stores)
- Historical tracking (track changes over time)
- Automated optimization suggestions with preview
- Integration with existing chat interface for testing
