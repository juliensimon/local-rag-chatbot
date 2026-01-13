"""API route handlers for explorer endpoints."""

import os
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

import sys
from pathlib import Path

# Add parent directory to path for imports
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from explorer.backend.analyzer import VectorStoreAnalyzer
from explorer.backend.schemas import (
    CollectionStats,
    DocumentResponse,
    DocumentsResponse,
    HealthResponse,
    QualityMetrics,
    RecommendationsResponse,
    SearchRequest,
    SearchResponse,
    SimilarityResponse,
    SourceStat,
    SourceStatsResponse,
    StatsResponse,
)
from retrievers import HybridRetriever

router = APIRouter()

# Global analyzer instance (initialized on startup)
_analyzer: Optional[VectorStoreAnalyzer] = None
_hybrid_retriever: Optional[HybridRetriever] = None


def init_routes(analyzer: VectorStoreAnalyzer):
    """Initialize routes with analyzer instance.

    Args:
        analyzer: VectorStoreAnalyzer instance
    """
    global _analyzer, _hybrid_retriever
    _analyzer = analyzer
    _hybrid_retriever = HybridRetriever(analyzer._vectorstore)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check explorer API health status."""
    if _analyzer is None:
        return HealthResponse(status="unhealthy", vectorstore_ready=False)

    try:
        # Quick check: try to get collection
        collection = _analyzer._vectorstore.get()
        ready = collection is not None and len(collection.get("documents", [])) > 0
        return HealthResponse(status="healthy" if ready else "degraded", vectorstore_ready=ready)
    except Exception:
        return HealthResponse(status="unhealthy", vectorstore_ready=False)


@router.get("/stats", response_model=StatsResponse)
async def get_all_stats():
    """Get all statistics (collection, source, quality, embedding)."""
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    try:
        collection_stats = _analyzer.get_collection_stats()
        source_stats = _analyzer.get_source_stats()
        quality_metrics = _analyzer.get_quality_metrics()
        embedding_quality = _analyzer.get_embedding_quality()

        return StatsResponse(
            collection=CollectionStats(**collection_stats),
            sources=SourceStatsResponse(
                sources=[SourceStat(**stat) for stat in source_stats]
            ),
            quality=QualityMetrics(**quality_metrics),
            embedding=embedding_quality,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing statistics: {str(e)}")


@router.get("/stats/collection", response_model=CollectionStats)
async def get_collection_stats():
    """Get collection-level statistics only."""
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    try:
        stats = _analyzer.get_collection_stats()
        return CollectionStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing collection stats: {str(e)}")


@router.get("/stats/sources", response_model=SourceStatsResponse)
async def get_source_stats():
    """Get source-level statistics."""
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    try:
        stats = _analyzer.get_source_stats()
        return SourceStatsResponse(sources=[SourceStat(**stat) for stat in stats])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing source stats: {str(e)}")


@router.get("/stats/quality", response_model=QualityMetrics)
async def get_quality_metrics():
    """Get quality metrics."""
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    try:
        metrics = _analyzer.get_quality_metrics()
        return QualityMetrics(**metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing quality metrics: {str(e)}")


@router.get("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations():
    """Get optimization recommendations."""
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    try:
        recommendations = _analyzer.get_recommendations()
        from explorer.backend.schemas import Recommendation

        return RecommendationsResponse(
            recommendations=[Recommendation(**rec) for rec in recommendations]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating recommendations: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform search with MMR, similarity, or hybrid search."""
    if _analyzer is None or _hybrid_retriever is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    try:
        # Build metadata filter if source_filter provided
        metadata_filter = None
        if request.source_filter:
            from config import PDF_PATH

            full_path = os.path.join(PDF_PATH, request.source_filter)
            metadata_filter = {"source": {"$eq": full_path}}

        # Perform search based on type
        results = []
        if request.search_type == "hybrid":
            hybrid_results = _hybrid_retriever.hybrid_search(
                request.query,
                k=request.k,
                alpha=request.hybrid_alpha,
                metadata_filter=metadata_filter,
            )
            for r in hybrid_results:
                doc = r["doc"]
                collection = _analyzer._vectorstore.get()
                ids = collection.get("ids", [])
                documents = collection.get("documents", [])
                # Find doc_id
                doc_id = None
                for i, d in enumerate(documents):
                    if d == doc.page_content:
                        doc_id = ids[i] if i < len(ids) else f"doc_{i}"
                        break

                results.append(
                    {
                        "content": doc.page_content,
                        "source": os.path.basename(doc.metadata.get("source", "")),
                        "page": doc.metadata.get("page", 0),
                        "score": r["fused_score"],
                        "semantic_score": r["semantic_score"],
                        "keyword_score": r["keyword_score"],
                        "doc_id": doc_id or "unknown",
                    }
                )
        elif request.search_type == "similarity":
            scored_docs = _analyzer._vectorstore.similarity_search_with_score(
                request.query, k=request.k, filter=metadata_filter
            )
            collection = _analyzer._vectorstore.get()
            ids = collection.get("ids", [])
            documents = collection.get("documents", [])

            for doc, score in scored_docs:
                # Find doc_id
                doc_id = None
                for i, d in enumerate(documents):
                    if d == doc.page_content:
                        doc_id = ids[i] if i < len(ids) else f"doc_{i}"
                        break

                # Convert distance to similarity (lower distance = higher similarity)
                similarity = 1.0 / (1.0 + score) if score >= 0 else 1.0

                results.append(
                    {
                        "content": doc.page_content,
                        "source": os.path.basename(doc.metadata.get("source", "")),
                        "page": doc.metadata.get("page", 0),
                        "score": similarity,
                        "semantic_score": similarity,
                        "keyword_score": None,
                        "doc_id": doc_id or "unknown",
                    }
                )
        else:  # MMR
            retriever = _analyzer._vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": request.k,
                    "fetch_k": request.k * 3,
                    "lambda_mult": 0.7,
                    "filter": metadata_filter,
                },
            )
            docs = retriever.invoke(request.query)
            collection = _analyzer._vectorstore.get()
            ids = collection.get("ids", [])
            documents = collection.get("documents", [])

            for doc in docs:
                # Find doc_id
                doc_id = None
                for i, d in enumerate(documents):
                    if d == doc.page_content:
                        doc_id = ids[i] if i < len(ids) else f"doc_{i}"
                        break

                results.append(
                    {
                        "content": doc.page_content,
                        "source": os.path.basename(doc.metadata.get("source", "")),
                        "page": doc.metadata.get("page", 0),
                        "score": None,  # MMR doesn't provide scores
                        "semantic_score": None,
                        "keyword_score": None,
                        "doc_id": doc_id or "unknown",
                    }
                )

        from explorer.backend.schemas import DocumentScore

        return SearchResponse(
            results=[DocumentScore(**r) for r in results],
            query=request.query,
            search_type=request.search_type,
            total_results=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.get("/documents", response_model=DocumentsResponse)
async def get_documents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
    source: Optional[str] = None,
    min_page: Optional[int] = Query(default=None, ge=0),
    max_page: Optional[int] = Query(default=None, ge=0),
    min_chunk_size: Optional[int] = Query(default=None, ge=0),
    max_chunk_size: Optional[int] = Query(default=None, ge=0),
):
    """Browse documents with filtering and pagination."""
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    try:
        collection = _analyzer._vectorstore.get()
        documents = collection.get("documents", [])
        metadatas = collection.get("metadatas", [])
        ids = collection.get("ids", [])

        # Apply filters
        filtered_docs = []
        for i, (doc, meta, doc_id) in enumerate(zip(documents, metadatas, ids)):
            if not meta:
                continue

            # Source filter
            if source:
                meta_source = os.path.basename(meta.get("source", ""))
                if meta_source != source:
                    continue

            # Page filter
            page_num = meta.get("page")
            if min_page is not None and (page_num is None or page_num < min_page):
                continue
            if max_page is not None and (page_num is None or page_num > max_page):
                continue

            # Chunk size filter
            chunk_size = len(doc)
            if min_chunk_size is not None and chunk_size < min_chunk_size:
                continue
            if max_chunk_size is not None and chunk_size > max_chunk_size:
                continue

            filtered_docs.append((i, doc, meta, doc_id))

        # Pagination
        total = len(filtered_docs)
        total_pages = (total + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_docs = filtered_docs[start_idx:end_idx]

        doc_responses = []
        for i, doc, meta, doc_id in paginated_docs:
            doc_responses.append(
                DocumentResponse(
                    doc_id=doc_id,
                    content=doc,
                    source=os.path.basename(meta.get("source", "")),
                    page=meta.get("page", 0),
                    chunk_size=len(doc),
                    metadata=meta or {},
                )
            )

        return DocumentsResponse(
            documents=doc_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching documents: {str(e)}")


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """Get a specific document chunk by ID."""
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    try:
        collection = _analyzer._vectorstore.get()
        documents = collection.get("documents", [])
        metadatas = collection.get("metadatas", [])
        ids = collection.get("ids", [])

        # Find document by ID
        if doc_id not in ids:
            raise HTTPException(status_code=404, detail="Document not found")

        idx = ids.index(doc_id)
        doc = documents[idx]
        meta = metadatas[idx] if idx < len(metadatas) else {}

        return DocumentResponse(
            doc_id=doc_id,
            content=doc,
            source=os.path.basename(meta.get("source", "")),
            page=meta.get("page", 0),
            chunk_size=len(doc),
            metadata=meta or {},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching document: {str(e)}")


@router.get("/similarity/{doc_id}", response_model=SimilarityResponse)
async def get_similar_documents(doc_id: str, k: int = Query(default=5, ge=1, le=20)):
    """Get similar documents to a given document chunk."""
    if _analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")

    try:
        collection = _analyzer._vectorstore.get()
        documents = collection.get("documents", [])
        metadatas = collection.get("metadatas", [])
        ids = collection.get("ids", [])

        # Find document by ID
        if doc_id not in ids:
            raise HTTPException(status_code=404, detail="Document not found")

        idx = ids.index(doc_id)
        doc = documents[idx]

        # Use the document content as query for similarity search
        similar_docs = _analyzer._vectorstore.similarity_search_with_score(
            doc, k=k + 1
        )  # +1 to exclude self

        # Filter out self and format results
        results = []
        for similar_doc, score in similar_docs:
            # Skip if it's the same document
            if similar_doc.page_content == doc:
                continue

            # Find doc_id
            similar_id = None
            for i, d in enumerate(documents):
                if d == similar_doc.page_content:
                    similar_id = ids[i] if i < len(ids) else f"doc_{i}"
                    break

            # Convert distance to similarity
            similarity = 1.0 / (1.0 + score) if score >= 0 else 1.0

            from explorer.backend.schemas import DocumentScore

            results.append(
                DocumentScore(
                    content=similar_doc.page_content,
                    source=os.path.basename(similar_doc.metadata.get("source", "")),
                    page=similar_doc.metadata.get("page", 0),
                    score=similarity,
                    semantic_score=similarity,
                    keyword_score=None,
                    doc_id=similar_id or "unknown",
                )
            )

            if len(results) >= k:
                break

        return SimilarityResponse(doc_id=doc_id, similar_docs=results)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar documents: {str(e)}")
