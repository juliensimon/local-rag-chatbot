"""Integration tests for explorer API routes."""

import pytest
from fastapi.testclient import TestClient
from explorer.backend.main import app
from explorer.backend.analyzer import VectorStoreAnalyzer
from unittest.mock import Mock, patch


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_analyzer():
    """Create a mock analyzer."""
    analyzer = Mock(spec=VectorStoreAnalyzer)
    analyzer.get_collection_stats.return_value = {
        "total_chunks": 100,
        "unique_sources": 5,
        "avg_chunk_size": 512.0,
        "min_chunk_size": 50,
        "max_chunk_size": 2000,
        "embedding_dimensions": 384,
        "db_size_bytes": 1024000,
        "chunk_size_distribution": {"0-100": 10, "100-200": 20},
    }
    analyzer.get_source_stats.return_value = [
        {
            "source": "/path/doc1.pdf",
            "filename": "doc1.pdf",
            "chunk_count": 20,
            "total_pages": 10,
            "pages_with_chunks": 10,
            "page_coverage": 100.0,
            "avg_chunk_size": 512.0,
            "min_chunk_size": 50,
            "max_chunk_size": 1000,
        }
    ]
    analyzer.get_quality_metrics.return_value = {
        "chunk_length_distribution": {},
        "outliers_short": 5,
        "outliers_long": 2,
        "overlap_analysis": {},
        "metadata_completeness": {"source": 100.0, "page": 100.0},
        "duplicate_chunks": 0,
        "low_information_chunks": 1,
    }
    analyzer.get_recommendations.return_value = [
        {
            "category": "chunking",
            "severity": "low",
            "title": "Test recommendation",
            "description": "Test description",
            "action_items": ["Action 1"],
            "metrics": {},
        }
    ]
    analyzer.get_embedding_quality.return_value = None
    analyzer._vectorstore = Mock()
    return analyzer


@patch("explorer.backend.routes._analyzer")
def test_health_check(mock_analyzer_global, mock_analyzer):
    """Test health check endpoint."""
    mock_analyzer_global = mock_analyzer
    mock_analyzer._vectorstore.get.return_value = {"documents": ["test"]}

    client = TestClient(app)
    response = client.get("/api/explorer/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "vectorstore_ready" in data


@patch("explorer.backend.routes._analyzer")
def test_get_stats(mock_analyzer_global, mock_analyzer):
    """Test stats endpoint."""
    mock_analyzer_global = mock_analyzer

    client = TestClient(app)
    response = client.get("/api/explorer/stats")

    assert response.status_code == 200
    data = response.json()
    assert "collection" in data
    assert "sources" in data
    assert "quality" in data


@patch("explorer.backend.routes._analyzer")
def test_get_recommendations(mock_analyzer_global, mock_analyzer):
    """Test recommendations endpoint."""
    mock_analyzer_global = mock_analyzer

    client = TestClient(app)
    response = client.get("/api/explorer/recommendations")

    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
