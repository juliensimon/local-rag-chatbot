"""FastAPI application entry point for explorer backend."""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Get absolute path to project root from this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VECTORSTORE_PATH = str(PROJECT_ROOT / "vectorstore")

# Add project root to Python path
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from explorer.backend.analyzer import VectorStoreAnalyzer
from explorer.backend.routes import init_routes, router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    print(f"Loading vectorstore from: {VECTORSTORE_PATH}")
    
    from models import create_embeddings
    from langchain_chroma import Chroma
    
    embeddings = create_embeddings()
    vectorstore = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
    
    collection = vectorstore.get()
    doc_count = len(collection.get("documents", []))
    print(f"Loaded {doc_count} documents")
    
    analyzer = VectorStoreAnalyzer(vectorstore)
    init_routes(analyzer)
    print("Explorer backend ready")
    
    yield


def create_explorer_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Vector Store Explorer API",
        description="API for exploring and analyzing the Chroma vector store",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/explorer")
    return app


app = create_explorer_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
