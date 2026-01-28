"""Tests for unified search API."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.knowledge_base.api.unified_search_api import (
    SearchMode,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
    router,
)


class TestSearchMode:
    """Tests for SearchMode enum."""

    def test_enum_values(self):
        """Test enum values are correct."""
        assert SearchMode.VECTOR.value == "vector"
        assert SearchMode.BM25.value == "bm25"
        assert SearchMode.HYBRID.value == "hybrid"
        assert SearchMode.RERANKED.value == "reranked"


class TestUnifiedSearchRequest:
    """Tests for UnifiedSearchRequest model."""

    def test_default_values(self):
        """Test default request values."""
        request = UnifiedSearchRequest(query="test query")
        assert request.query == "test query"
        assert request.mode == SearchMode.HYBRID
        assert request.top_k == 10
        assert request.filters is None
        assert request.domain is None
        assert request.vector_weight == 0.5
        assert request.bm25_weight == 0.5
        assert request.use_reranking is True
        assert request.initial_top_k == 50

    def test_custom_values(self):
        """Test custom request values."""
        request = UnifiedSearchRequest(
            query="machine learning",
            mode=SearchMode.VECTOR,
            top_k=20,
            filters={"domain": "tech"},
            vector_weight=0.7,
            bm25_weight=0.3,
        )
        assert request.query == "machine learning"
        assert request.mode == SearchMode.VECTOR
        assert request.top_k == 20
        assert request.filters == {"domain": "tech"}
        assert request.vector_weight == 0.7
        assert request.bm25_weight == 0.3

    def test_validation_top_k_range(self):
        """Test top_k validation."""
        with pytest.raises(ValueError):
            UnifiedSearchRequest(query="test", top_k=0)

        with pytest.raises(ValueError):
            UnifiedSearchRequest(query="test", top_k=101)

    def test_validation_weight_sum(self):
        """Test weight validation."""
        with pytest.raises(ValueError):
            UnifiedSearchRequest(query="test", vector_weight=0.7, bm25_weight=0.7)


class TestUnifiedSearchResponse:
    """Tests for UnifiedSearchResponse model."""

    def test_response_structure(self):
        """Test response model structure."""
        response = UnifiedSearchResponse(
            query="test",
            mode=SearchMode.HYBRID,
            total_results=5,
            results=[{"id": "1", "score": 0.9}],
            processing_time_ms=100.0,
            metadata={"filters": None},
        )
        assert response.query == "test"
        assert response.mode == SearchMode.HYBRID
        assert response.total_results == 5
        assert len(response.results) == 1
        assert response.processing_time_ms > 0


class TestUnifiedSearchAPI:
    """Tests for unified search API endpoints."""

    @pytest.fixture
    def mock_services(self):
        """Create mock search services."""
        mock_vector_store = MagicMock()
        mock_vector_store.search_similar_chunks = AsyncMock(return_value=[])

        mock_bm25_index = MagicMock()
        mock_bm25_index.search = AsyncMock(return_value=[])
        mock_bm25_index.get_stats = AsyncMock(
            return_value={"document_count": 0, "indexed": True}
        )

        mock_hybrid_engine = MagicMock()
        mock_hybrid_engine.search = AsyncMock(return_value=[])

        mock_reranking_pipeline = MagicMock()
        mock_reranking_pipeline.search = AsyncMock(return_value=[])

        return {
            "vector_store": mock_vector_store,
            "bm25_index": mock_bm25_index,
            "hybrid_engine": mock_hybrid_engine,
            "reranking_pipeline": mock_reranking_pipeline,
        }

    @pytest.fixture
    def client(self, mock_services):
        """Create test client with mocked services."""
        with patch(
            "src.knowledge_base.api.unified_search_api.get_search_services",
            return_value=mock_services,
        ):
            from fastapi import FastAPI

            test_app = FastAPI()
            test_app.include_router(router)
            return TestClient(test_app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        with patch(
            "src.knowledge_base.api.unified_search_api.get_search_services",
            return_value={
                "vector_store": MagicMock(),
                "bm25_index": MagicMock(),
            },
        ):
            with patch(
                "src.knowledge_base.api.unified_search_api.BM25Index.get_stats",
                new_callable=AsyncMock,
                return_value={"document_count": 10, "indexed": True},
            ):
                response = client.get("/unified-search/health")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert "services" in data

    def test_modes_endpoint(self, client):
        """Test search modes endpoint."""
        response = client.get("/unified-search/modes")

        assert response.status_code == 200
        modes = response.json()

        assert len(modes) == 4
        mode_names = [m["mode"] for m in modes]
        assert "vector" in mode_names
        assert "bm25" in mode_names
        assert "hybrid" in mode_names
        assert "reranked" in mode_names

    def test_search_request_validation(self, client):
        """Test search request validation."""
        response = client.post(
            "/unified-search/",
            json={"query": "test"},
        )
        assert response.status_code == 200

        response = client.post(
            "/unified-search/",
            json={"query": "", "top_k": 0},
        )
        assert response.status_code == 422

    def test_search_hybrid_mode(self, client, mock_services):
        """Test hybrid search mode."""
        mock_services["hybrid_engine"].search = AsyncMock(
            return_value=[
                MagicMock(
                    id="1",
                    text="test text",
                    vector_score=0.8,
                    bm25_score=0.6,
                    final_score=0.7,
                    metadata=None,
                    source="hybrid",
                )
            ]
        )

        response = client.post(
            "/unified-search/",
            json={"query": "test query", "mode": "hybrid", "top_k": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["mode"] == "hybrid"

    def test_search_vector_mode(self, client, mock_services):
        """Test vector search mode."""
        mock_services["vector_store"].search_similar_chunks = AsyncMock(
            return_value=[
                {
                    "id": "1",
                    "text": "test",
                    "similarity": 0.9,
                    "metadata": {"source": "doc1"},
                }
            ]
        )

        with patch(
            "src.knowledge_base.api.unified_search_api.EmbeddingClient"
        ) as MockClient:
            mock_client = MagicMock()
            mock_client.embed_texts = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
            MockClient.return_value = mock_client

            response = client.post(
                "/unified-search/",
                json={"query": "test", "mode": "vector", "top_k": 10},
            )

            assert response.status_code == 200

    def test_search_bm25_mode(self, client, mock_services):
        """Test BM25 search mode."""
        from src.knowledge_base.storage.bm25_index import SearchResult

        mock_services["bm25_index"].search = AsyncMock(
            return_value=[
                SearchResult(
                    id="1",
                    text="test text",
                    score=1.5,
                    metadata={"source": "doc1"},
                )
            ]
        )

        response = client.post(
            "/unified-search/",
            json={"query": "test", "mode": "bm25", "top_k": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "bm25"

    def test_search_reranked_mode(self, client, mock_services):
        """Test reranked search mode."""
        mock_services["reranking_pipeline"].search = AsyncMock(
            return_value=[
                MagicMock(
                    id="1",
                    text="test text",
                    reranked_score=0.95,
                    cross_encoder_score=0.9,
                    metadata=None,
                )
            ]
        )

        response = client.post(
            "/unified-search/",
            json={
                "query": "test",
                "mode": "reranked",
                "top_k": 10,
                "use_reranking": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "reranked"

    def test_search_with_filters(self, client, mock_services):
        """Test search with metadata filters."""
        mock_services["hybrid_engine"].search = AsyncMock(return_value=[])

        response = client.post(
            "/unified-search/",
            json={
                "query": "test",
                "mode": "hybrid",
                "top_k": 10,
                "filters": {"domain": "technology", "year": 2024},
            },
        )

        assert response.status_code == 200
        mock_services["hybrid_engine"].search.assert_called_once()

    def test_search_with_weights(self, client, mock_services):
        """Test search with custom weights."""
        mock_services["hybrid_engine"].search = AsyncMock(return_value=[])

        response = client.post(
            "/unified-search/",
            json={
                "query": "test",
                "mode": "hybrid",
                "top_k": 10,
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
            },
        )

        assert response.status_code == 200

    def test_search_response_timing(self, client, mock_services):
        """Test search response includes timing information."""
        mock_services["hybrid_engine"].search = AsyncMock(return_value=[])

        response = client.post(
            "/unified-search/",
            json={"query": "test", "mode": "hybrid"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], (int, float))
        assert data["processing_time_ms"] >= 0
