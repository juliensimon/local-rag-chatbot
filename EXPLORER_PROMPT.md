# Improved Plan Mode Prompt

## Original Prompt
> In explorer/, design a simple web app to explore the Chroma vector store located at vectorstore/ I'm interested in vector store statistics, as well as optimization tips to improve the quality and accuracy of my RAG system. Expected output: a detailed PRD with tasks, tests, success criteria, etc.

## Improved Prompt

**Context:**
Build a vector store exploration web application in `explorer/` for the existing RAG system. The system uses ChromaDB (at `vectorstore/`) with LangChain, FastAPI backend, and React frontend. The vector store contains PDF documents chunked with RecursiveCharacterTextSplitter (512 chars, 128 overlap), embedded with BAAI/bge-small-en-v1.5, and supports MMR/similarity/hybrid retrieval. Existing frontend uses React 18, TypeScript, Tailwind CSS, and Radix UI components.

**Requirements:**

1. **Statistics Dashboard** - Display:
   - Collection stats: total chunks, unique sources, avg chunk size, size distribution, embedding dimensions, DB size
   - Source-level stats: chunks per source, page coverage, file-to-chunk correlation
   - Quality metrics: chunk length distribution, overlap analysis, metadata completeness, duplicate detection
   - Embedding quality: similarity distribution, cluster analysis, embedding space coverage

2. **Optimization Recommendations** - Provide actionable tips for:
   - Chunking: optimal size suggestions, flag outliers (<50 or >2000 chars), overlap adjustments
   - Retrieval: similarity pattern analysis, optimal k values, hybrid alpha suggestions
   - Metadata: completeness flags, enrichment opportunities
   - Embedding: model comparison, quality issues
   - Content: low-information chunks, noise detection (headers/footers)

3. **Interactive Features:**
   - Search interface: test queries with MMR/similarity/hybrid search
   - Document browser: browse by source, page, or similarity
   - Similarity visualization: document relationship graphs
   - Query analysis: query-to-document mapping with scores
   - Filtering: by source, page range, chunk size

4. **Technical Stack:**
   - Backend: FastAPI endpoints (`/api/explorer/*`) reusing `vectorstore.py` and `models.py`
   - Frontend: React + TypeScript in `explorer/`, reuse UI components from `frontend/src/components/ui/`
   - Design: Match existing Tailwind + Radix UI patterns
   - Performance: Handle 1000+ documents efficiently, <5s load time

5. **Success Criteria:**
   - Functional: All stats load correctly, recommendations are actionable, search matches main app behavior
   - Quality: Follows existing code patterns, comprehensive error handling, accessible UI
   - Testing: Unit tests for calculations, integration tests for APIs, component tests, edge cases

**Deliverables:**
- Detailed PRD with feature specs, user stories, technical architecture, API specs, UI mockups, testing strategy
- Implementation: Backend APIs, frontend components, statistics engine, recommendation system, visualizations
- Test suite with performance benchmarks

**Constraints:**
- Read-only access to existing vectorstore (no modifications to RAG system)
- Must work with existing ChromaDB structure
- Handle large stores efficiently
- Follow existing code style and patterns
