"""FastAPI application entry point."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import init_routes, router
from models import create_embeddings
from qa_chain import create_qa_chain
from vectorstore import load_or_create_vectorstore


def create_api_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    app = FastAPI(
        title="RAG Chatbot API",
        description="API for RAG-powered document question-answering",
        version="1.0.0",
    )

    # CORS middleware for React frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router, prefix="/api")

    return app


def initialize_qa_chain():
    """Initialize the QA chain and get available sources.

    Returns:
        tuple: (QAChainWrapper, list of available sources)
    """
    embeddings = create_embeddings()
    vectorstore = load_or_create_vectorstore(embeddings)
    chain = create_qa_chain(vectorstore)

    # Get available document sources
    collection = vectorstore.get()
    if not collection or not collection.get("metadatas"):
        sources = []
    else:
        sources = sorted(
            set(
                os.path.basename(meta.get("source", ""))
                for meta in collection["metadatas"]
                if meta and meta.get("source")
            )
        )

    return chain, sources


# Create the app instance
app = create_api_app()


@app.on_event("startup")
async def startup_event():
    """Initialize QA chain on startup."""
    qa_chain, sources = initialize_qa_chain()
    init_routes(qa_chain, sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
