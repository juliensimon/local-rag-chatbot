"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.registry import create_registry
from api.routes import init_routes, router
from models import create_embeddings


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


def initialize_registry():
    """Load the shared corpus; per-user collections load on first request.

    Returns:
        CollectionRegistry: Registry serving all collections
    """
    return create_registry(create_embeddings())


# Create the app instance
app = create_api_app()


@app.on_event("startup")
async def startup_event():
    """Initialize the collection registry on startup."""
    init_routes(initialize_registry())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
