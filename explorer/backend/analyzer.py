"""Vector store analyzer for statistics and recommendations."""

import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma

from config import CHROMA_PATH, CHUNK_OVERLAP, CHUNK_SIZE
from models import create_embeddings
from vectorstore import load_or_create_vectorstore


class VectorStoreAnalyzer:
    """Analyzes vector store and generates statistics and recommendations."""

    def __init__(self, vectorstore: Optional[Chroma] = None):
        """Initialize analyzer with vectorstore.

        Args:
            vectorstore: Optional Chroma vectorstore instance. If None, loads from CHROMA_PATH.
        """
        if vectorstore is None:
            embeddings = create_embeddings()
            # Resolve CHROMA_PATH - make it absolute if relative
            chroma_path = CHROMA_PATH
            if not os.path.isabs(chroma_path):
                # Get project root (assuming we're in explorer/backend/)
                # Go up two levels from this file to get project root
                project_root = Path(__file__).parent.parent.parent
                chroma_path = str(project_root / chroma_path)
            
            # Ensure absolute path
            chroma_path = os.path.abspath(chroma_path)
            
            # Try to load existing vectorstore, don't create new one
            if os.path.exists(chroma_path):
                print(f"Loading vectorstore from: {chroma_path}")
                self._vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
                # Verify it has data
                try:
                    collection = self._vectorstore.get()
                    doc_count = len(collection.get("documents", []))
                    print(f"Vectorstore loaded: {doc_count} documents found")
                except Exception as e:
                    print(f"Warning: Could not read vectorstore collection: {e}")
            else:
                # If vectorstore doesn't exist, try load_or_create (might fail if no PDFs)
                try:
                    self._vectorstore = load_or_create_vectorstore(embeddings)
                except Exception as e:
                    # Create a dummy vectorstore that will return empty results
                    # This allows the app to start even if vectorstore isn't ready
                    print(f"Warning: Could not load vectorstore: {e}")
                    # Create an empty Chroma instance
                    self._vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        else:
            self._vectorstore = vectorstore

        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = 300  # 5 minutes

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/missing
        """
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any):
        """Set cached value.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = (value, time.time())

    def _get_collection(self) -> Dict[str, Any]:
        """Get collection data from vectorstore.

        Returns:
            Collection dictionary with documents, metadatas, ids
        """
        return self._vectorstore.get()

    def get_collection_stats(self) -> Dict[str, Any]:
        """Compute collection-level statistics.

        Returns:
            Dictionary with collection statistics
        """
        cache_key = "collection_stats"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        collection = self._get_collection()
        documents = collection.get("documents", [])
        metadatas = collection.get("metadatas", [])
        ids = collection.get("ids", [])

        if not documents:
            stats = {
                "total_chunks": 0,
                "unique_sources": 0,
                "avg_chunk_size": 0.0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "embedding_dimensions": 0,
                "db_size_bytes": None,
                "chunk_size_distribution": {},
            }
            self._set_cache(cache_key, stats)
            return stats

        # Compute chunk sizes
        chunk_sizes = [len(doc) for doc in documents]
        total_chunks = len(documents)
        avg_chunk_size = sum(chunk_sizes) / total_chunks if total_chunks > 0 else 0.0

        # Get unique sources
        unique_sources = set()
        for meta in metadatas:
            if meta and meta.get("source"):
                unique_sources.add(meta["source"])

        # Chunk size distribution (bins: 0-100, 100-200, ..., 1900-2000, 2000+)
        distribution = defaultdict(int)
        for size in chunk_sizes:
            if size < 100:
                distribution["0-100"] += 1
            elif size >= 2000:
                distribution["2000+"] += 1
            else:
                bin_start = (size // 100) * 100
                bin_end = bin_start + 100
                distribution[f"{bin_start}-{bin_end}"] += 1

        # Get embedding dimensions (sample first embedding)
        embedding_dimensions = 0
        if ids:
            try:
                result = self._vectorstore._collection.get(ids=[ids[0]], include=["embeddings"])
                if result and result.get("embeddings") and result["embeddings"][0]:
                    embedding_dimensions = len(result["embeddings"][0])
            except Exception:
                # Fallback: use model default
                embedding_dimensions = 384  # bge-small-en-v1.5 default

        # Get DB size
        db_size_bytes = None
        if os.path.exists(CHROMA_PATH):
            try:
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(CHROMA_PATH):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            total_size += os.path.getsize(filepath)
                db_size_bytes = total_size
            except Exception:
                pass

        stats = {
            "total_chunks": total_chunks,
            "unique_sources": len(unique_sources),
            "avg_chunk_size": round(avg_chunk_size, 2),
            "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            "embedding_dimensions": embedding_dimensions,
            "db_size_bytes": db_size_bytes,
            "chunk_size_distribution": dict(distribution),
        }

        self._set_cache(cache_key, stats)
        return stats

    def get_source_stats(self) -> List[Dict[str, Any]]:
        """Compute source-level statistics.

        Returns:
            List of source statistics dictionaries
        """
        cache_key = "source_stats"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        collection = self._get_collection()
        documents = collection.get("documents", [])
        metadatas = collection.get("metadatas", [])

        if not documents:
            self._set_cache(cache_key, [])
            return []

        # Group by source
        source_data = defaultdict(lambda: {"chunks": [], "pages": set(), "sizes": []})

        for doc, meta in zip(documents, metadatas):
            if not meta or not meta.get("source"):
                continue

            source = meta["source"]
            source_data[source]["chunks"].append(doc)
            source_data[source]["sizes"].append(len(doc))

            page = meta.get("page")
            if page is not None:
                source_data[source]["pages"].add(page)

        # Compute stats per source
        source_stats = []
        for source, data in source_data.items():
            filename = os.path.basename(source)
            chunk_count = len(data["chunks"])
            pages = sorted(data["pages"])
            total_pages = max(pages) + 1 if pages else 0  # Assume 0-indexed
            pages_with_chunks = len(pages)
            page_coverage = (
                (pages_with_chunks / total_pages * 100) if total_pages > 0 else 0.0
            )

            sizes = data["sizes"]
            avg_size = sum(sizes) / len(sizes) if sizes else 0.0

            source_stats.append(
                {
                    "source": source,
                    "filename": filename,
                    "chunk_count": chunk_count,
                    "total_pages": total_pages,
                    "pages_with_chunks": pages_with_chunks,
                    "page_coverage": round(page_coverage, 2),
                    "avg_chunk_size": round(avg_size, 2),
                    "min_chunk_size": min(sizes) if sizes else 0,
                    "max_chunk_size": max(sizes) if sizes else 0,
                }
            )

        # Sort by filename
        source_stats.sort(key=lambda x: x["filename"])

        self._set_cache(cache_key, source_stats)
        return source_stats

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Compute quality metrics.

        Returns:
            Dictionary with quality metrics
        """
        cache_key = "quality_metrics"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        collection = self._get_collection()
        documents = collection.get("documents", [])
        metadatas = collection.get("metadatas", [])

        if not documents:
            metrics = {
                "chunk_length_distribution": {},
                "outliers_short": 0,
                "outliers_long": 0,
                "overlap_analysis": {},
                "metadata_completeness": {},
                "duplicate_chunks": 0,
                "low_information_chunks": 0,
            }
            self._set_cache(cache_key, metrics)
            return metrics

        # Chunk length distribution
        chunk_sizes = [len(doc) for doc in documents]
        distribution = Counter(chunk_sizes)
        # Convert integer keys to strings for JSON serialization
        distribution_str = {str(k): v for k, v in distribution.items()}

        # Outliers
        outliers_short = sum(1 for size in chunk_sizes if size < 50)
        outliers_long = sum(1 for size in chunk_sizes if size > 2000)

        # Metadata completeness
        required_fields = ["source", "page"]
        completeness = {}
        for field in required_fields:
            present = sum(1 for meta in metadatas if meta and meta.get(field))
            completeness[field] = round((present / len(metadatas) * 100), 2) if metadatas else 0.0

        # Overlap analysis (simplified: check adjacent chunks from same source)
        overlap_info = {
            "sources_with_overlap": 0,
            "avg_overlap_ratio": 0.0,
        }

        # Group chunks by source and page
        source_chunks = defaultdict(list)
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            if meta and meta.get("source"):
                source = meta["source"]
                page = meta.get("page", 0)
                source_chunks[source].append((i, page, doc))

        overlap_ratios = []
        for source, chunks in source_chunks.items():
            # Sort by page
            chunks.sort(key=lambda x: x[1])
            has_overlap = False

            for j in range(len(chunks) - 1):
                idx1, page1, doc1 = chunks[j]
                idx2, page2, doc2 = chunks[j + 1]

                # Check if chunks are adjacent (same or consecutive pages)
                if page2 - page1 <= 1:
                    # Simple overlap detection: check if end of doc1 overlaps with start of doc2
                    overlap_len = min(len(doc1), len(doc2), CHUNK_OVERLAP * 2)
                    if overlap_len > 0:
                        end1 = doc1[-overlap_len:]
                        start2 = doc2[:overlap_len]
                        # Simple character overlap
                        common = sum(1 for c1, c2 in zip(end1, start2) if c1 == c2)
                        if overlap_len > 0:
                            ratio = common / overlap_len
                            if ratio > 0.3:  # 30% overlap threshold
                                overlap_ratios.append(ratio)
                                has_overlap = True

            if has_overlap:
                overlap_info["sources_with_overlap"] += 1

        if overlap_ratios:
            overlap_info["avg_overlap_ratio"] = round(sum(overlap_ratios) / len(overlap_ratios), 3)

        # Duplicate detection (simple: exact content matches)
        content_set = set()
        duplicates = 0
        for doc in documents:
            if doc in content_set:
                duplicates += 1
            else:
                content_set.add(doc)

        # Low information chunks (very short or repetitive)
        low_info = 0
        for doc in documents:
            if len(doc) < 30:
                low_info += 1
            elif len(set(doc.split())) < 5:  # Very few unique words
                low_info += 1

        metrics = {
            "chunk_length_distribution": dict(list(distribution_str.items())[:20]),  # Top 20
            "outliers_short": outliers_short,
            "outliers_long": outliers_long,
            "overlap_analysis": overlap_info,
            "metadata_completeness": completeness,
            "duplicate_chunks": duplicates,
            "low_information_chunks": low_info,
        }

        self._set_cache(cache_key, metrics)
        return metrics

    def get_embedding_quality(self) -> Optional[Dict[str, Any]]:
        """Compute embedding quality metrics (expensive operation).

        Returns:
            Dictionary with embedding quality metrics or None if computation fails
        """
        cache_key = "embedding_quality"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        collection = self._get_collection()
        documents = collection.get("documents", [])
        ids = collection.get("ids", [])

        if not documents or len(documents) < 2:
            return None

        # Sample pairwise similarities (limit to avoid expensive computation)
        sample_size = min(100, len(documents))
        sample_indices = list(range(0, len(documents), len(documents) // sample_size))[:sample_size]

        similarities = []
        try:
            for i in range(len(sample_indices) - 1):
                idx1 = sample_indices[i]
                idx2 = sample_indices[i + 1]

                # Get embeddings
                try:
                    emb1 = self._vectorstore._collection.get(ids=[ids[idx1]])["embeddings"][0]
                    emb2 = self._vectorstore._collection.get(ids=[ids[idx2]])["embeddings"][0]

                    # Cosine similarity
                    dot_product = sum(a * b for a, b in zip(emb1, emb2))
                    norm1 = sum(a * a for a in emb1) ** 0.5
                    norm2 = sum(b * b for b in emb2) ** 0.5

                    if norm1 > 0 and norm2 > 0:
                        similarity = dot_product / (norm1 * norm2)
                        similarities.append(similarity)
                except Exception:
                    continue

            if not similarities:
                return None

            # Distribution bins
            distribution = defaultdict(int)
            for sim in similarities:
                bin_val = round(sim, 1)  # Round to 0.1
                distribution[str(bin_val)] += 1

            quality = {
                "similarity_distribution": dict(distribution),
                "avg_similarity": round(sum(similarities) / len(similarities), 3),
                "min_similarity": round(min(similarities), 3),
                "max_similarity": round(max(similarities), 3),
            }

            self._set_cache(cache_key, quality)
            return quality
        except Exception:
            return None

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations.

        Returns:
            List of recommendation dictionaries
        """
        cache_key = "recommendations"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        recommendations = []

        # Get stats for recommendations
        collection_stats = self.get_collection_stats()
        quality_metrics = self.get_quality_metrics()
        source_stats = self.get_source_stats()

        # Chunking recommendations
        outliers_short = quality_metrics.get("outliers_short", 0)
        outliers_long = quality_metrics.get("outliers_long", 0)
        total_chunks = collection_stats.get("total_chunks", 0)

        if outliers_short > 0:
            pct = (outliers_short / total_chunks * 100) if total_chunks > 0 else 0
            severity = "high" if pct > 10 else "medium" if pct > 5 else "low"
            recommendations.append(
                {
                    "category": "chunking",
                    "severity": severity,
                    "title": f"Found {outliers_short} chunks shorter than 50 characters",
                    "description": f"{pct:.1f}% of chunks are too short, which may reduce retrieval quality.",
                    "action_items": [
                        "Review chunking strategy",
                        "Consider merging very short chunks with adjacent chunks",
                        "Check if text splitter settings are appropriate",
                    ],
                    "metrics": {"outliers_short": outliers_short, "percentage": round(pct, 2)},
                }
            )

        if outliers_long > 0:
            pct = (outliers_long / total_chunks * 100) if total_chunks > 0 else 0
            severity = "high" if pct > 10 else "medium" if pct > 5 else "low"
            recommendations.append(
                {
                    "category": "chunking",
                    "severity": severity,
                    "title": f"Found {outliers_long} chunks longer than 2000 characters",
                    "description": f"{pct:.1f}% of chunks exceed recommended size, which may reduce embedding quality.",
                    "action_items": [
                        "Review chunk size settings",
                        f"Current chunk size: {CHUNK_SIZE}, consider reducing if many chunks exceed 2000 chars",
                        "Check if documents need different chunking strategies",
                    ],
                    "metrics": {"outliers_long": outliers_long, "percentage": round(pct, 2)},
                }
            )

        # Optimal chunk size recommendation
        avg_size = collection_stats.get("avg_chunk_size", 0)
        if avg_size < 200 or avg_size > 800:
            recommendations.append(
                {
                    "category": "chunking",
                    "severity": "medium",
                    "title": f"Average chunk size ({avg_size:.0f} chars) may not be optimal",
                    "description": f"Current average is {avg_size:.0f} characters. Recommended range is 200-800 characters for most use cases.",
                    "action_items": [
                        f"Consider adjusting CHUNK_SIZE from {CHUNK_SIZE}",
                        "Test different chunk sizes and measure retrieval quality",
                    ],
                    "metrics": {"avg_chunk_size": avg_size, "current_setting": CHUNK_SIZE},
                }
            )

        # Metadata completeness
        completeness = quality_metrics.get("metadata_completeness", {})
        for field, pct in completeness.items():
            if pct < 100:
                severity = "high" if pct < 80 else "medium"
                recommendations.append(
                    {
                        "category": "metadata",
                        "severity": severity,
                        "title": f"Missing {field} metadata in {100 - pct:.1f}% of chunks",
                        "description": f"Only {pct:.1f}% of chunks have {field} metadata, which may limit filtering capabilities.",
                        "action_items": [
                            "Review document loading process",
                            "Ensure metadata is preserved during chunking",
                            "Check if source documents have required metadata",
                        ],
                        "metrics": {"field": field, "completeness": pct},
                    }
                )

        # Duplicate chunks
        duplicates = quality_metrics.get("duplicate_chunks", 0)
        if duplicates > 0:
            pct = (duplicates / total_chunks * 100) if total_chunks > 0 else 0
            severity = "high" if pct > 5 else "medium" if pct > 1 else "low"
            recommendations.append(
                {
                    "category": "content",
                    "severity": severity,
                    "title": f"Found {duplicates} duplicate chunks",
                    "description": f"{pct:.1f}% of chunks are exact duplicates, which wastes storage and may bias retrieval.",
                    "action_items": [
                        "Review document processing pipeline",
                        "Check for duplicate source documents",
                        "Consider deduplication before chunking",
                    ],
                    "metrics": {"duplicates": duplicates, "percentage": round(pct, 2)},
                }
            )

        # Low information chunks
        low_info = quality_metrics.get("low_information_chunks", 0)
        if low_info > 0:
            pct = (low_info / total_chunks * 100) if total_chunks > 0 else 0
            severity = "medium" if pct > 5 else "low"
            recommendations.append(
                {
                    "category": "content",
                    "severity": severity,
                    "title": f"Found {low_info} low-information chunks",
                    "description": f"{pct:.1f}% of chunks contain very little information (headers, footers, or repetitive content).",
                    "action_items": [
                        "Review document filtering logic",
                        "Consider filtering out headers and footers",
                        "Check if noise detection is working correctly",
                    ],
                    "metrics": {"low_info_chunks": low_info, "percentage": round(pct, 2)},
                }
            )

        # Overlap analysis
        overlap_info = quality_metrics.get("overlap_analysis", {})
        sources_with_overlap = overlap_info.get("sources_with_overlap", 0)
        if sources_with_overlap > 0:
            total_sources = collection_stats.get("unique_sources", 0)
            if total_sources > 0:
                pct = (sources_with_overlap / total_sources * 100)
                severity = "low" if pct < 50 else "medium"
                recommendations.append(
                    {
                        "category": "chunking",
                        "severity": severity,
                        "title": f"Overlap detected in {sources_with_overlap} sources",
                        "description": f"{pct:.1f}% of sources show chunk overlap. Current overlap setting: {CHUNK_OVERLAP} characters.",
                        "action_items": [
                            f"Review CHUNK_OVERLAP setting (currently {CHUNK_OVERLAP})",
                            "Consider adjusting overlap if retrieval quality is poor",
                            "Test different overlap values",
                        ],
                        "metrics": {
                            "sources_with_overlap": sources_with_overlap,
                            "current_overlap": CHUNK_OVERLAP,
                        },
                    }
                )

        # Retrieval recommendations (based on collection size)
        total_chunks = collection_stats.get("total_chunks", 0)
        if total_chunks > 0:
            if total_chunks < 100:
                recommendations.append(
                    {
                        "category": "retrieval",
                        "severity": "low",
                        "title": "Small collection - consider lower k values",
                        "description": f"With only {total_chunks} chunks, using k > 5 may return less relevant results.",
                        "action_items": [
                            "Use k=3-5 for retrieval",
                            "Consider similarity search over MMR for small collections",
                        ],
                        "metrics": {"total_chunks": total_chunks},
                    }
                )
            elif total_chunks > 10000:
                recommendations.append(
                    {
                        "category": "retrieval",
                        "severity": "low",
                        "title": "Large collection - consider hybrid search",
                        "description": f"With {total_chunks} chunks, hybrid search (semantic + keyword) may improve retrieval quality.",
                        "action_items": [
                            "Enable hybrid search for better results",
                            "Consider increasing k for initial retrieval",
                            "Test different hybrid_alpha values (0.5-0.8)",
                        ],
                        "metrics": {"total_chunks": total_chunks},
                    }
                )

        self._set_cache(cache_key, recommendations)
        return recommendations
