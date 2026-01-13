# Vector Store Explorer

A standalone web application for exploring, analyzing, and optimizing the Chroma vector store used by the RAG system.

## Features

- **Statistics Dashboard**: Comprehensive statistics about the vector store including collection-level, source-level, and quality metrics
- **Optimization Recommendations**: Actionable recommendations for improving chunking, retrieval, metadata, and content quality
- **Search Interface**: Test different search strategies (MMR, similarity, hybrid) to see how they perform
- **Document Browser**: Browse and filter document chunks with pagination

## Architecture

- **Backend**: FastAPI service running on port 8001
- **Frontend**: React + TypeScript app running on port 5174
- **Integration**: Read-only access to the existing ChromaDB vectorstore

## Setup

### Backend

The backend is a Python FastAPI application. It reuses the existing vectorstore and models from the main RAG system.

```bash
# Install dependencies (if not already installed)
pip install -r ../requirements.txt

# Run the backend
cd explorer/backend
python main.py
```

The backend will start on `http://localhost:8001`

### Frontend

The frontend is a React application built with Vite.

```bash
# Install dependencies
cd explorer/frontend
npm install

# Run the development server
npm run dev
```

The frontend will start on `http://localhost:5174`

## API Endpoints

All endpoints are prefixed with `/api/explorer`:

- `GET /health` - Health check
- `GET /stats` - Get all statistics
- `GET /stats/collection` - Collection-level stats
- `GET /stats/sources` - Source-level stats
- `GET /stats/quality` - Quality metrics
- `GET /recommendations` - Optimization recommendations
- `POST /search` - Perform search (MMR/similarity/hybrid)
- `GET /documents` - Browse documents with filtering
- `GET /documents/{doc_id}` - Get specific document
- `GET /similarity/{doc_id}` - Get similar documents

## Development

### Running Tests

```bash
# Backend tests
cd explorer/backend
pytest tests/

# Frontend tests (if configured)
cd explorer/frontend
npm test
```

## Notes

- The explorer has read-only access to the vectorstore
- Statistics are cached for 5 minutes to improve performance
- The frontend reuses UI components from the main frontend for consistency
