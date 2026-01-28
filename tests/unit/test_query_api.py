"""Unit tests for query API."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from knowledge_base.query_api import router
from knowledge_base.intelligence import (
    FederatedQueryRouter,
    HybridEntityRetriever,
    QueryDomain,
    ExecutionStrategy,
    FederatedQueryResult,
)
from knowledge_base.intelligence.v1.hybrid_retriever import (
    RetrievedEntity,
    HybridRetrievalResult,
)
from uuid import UUID


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = MagicMock()
    store.search_similar_entities = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_graph_store():
    """Create a mock graph store."""
    store = MagicMock()
    store.get_entity_neighborhood = AsyncMock(return_value=(None, []))
    return store


@pytest.fixture
def mock_retriever(mock_vector_store, mock_graph_store):
    """Create a mock hybrid entity retriever."""
    retriever = HybridEntityRetriever(
        vector_store=mock_vector_store, graph_store=mock_graph_store
    )
    return retriever


@pytest.fixture
def mock_router(mock_retriever):
    """Create a mock federated query router."""
    router = FederatedQueryRouter(retriever=mock_retriever)
    return router


class TestExecutionStrategyEnum:
    """Tests for ExecutionStrategyEnum."""

    def test_enum_values(self):
        """Test enum values are correct."""
        from knowledge_base.query_api import ExecutionStrategyEnum

        assert ExecutionStrategyEnum.SEQUENTIAL.value == "sequential"
        assert ExecutionStrategyEnum.PARALLEL.value == "parallel"
        assert ExecutionStrategyEnum.PRIORITY.value == "priority"


class TestFederatedQueryRequest:
    """Tests for FederatedQueryRequest model."""

    def test_default_values(self):
        """Test default values are applied correctly."""
        from knowledge_base.query_api import FederatedQueryRequest

        request = FederatedQueryRequest(query="test query")
        assert request.max_domains == 3
        assert request.min_confidence == 0.5
        assert request.strategy.value == "parallel"

    def test_custom_values(self):
        """Test custom values are accepted."""
        from knowledge_base.query_api import (
            FederatedQueryRequest,
            ExecutionStrategyEnum,
        )

        request = FederatedQueryRequest(
            query="test query",
            max_domains=5,
            strategy=ExecutionStrategyEnum.SEQUENTIAL,
            min_confidence=0.7,
        )
        assert request.max_domains == 5
        assert request.strategy == ExecutionStrategyEnum.SEQUENTIAL
        assert request.min_confidence == 0.7

    def test_validation_max_domains(self):
        """Test max_domains validation."""
        from knowledge_base.query_api import FederatedQueryRequest

        with pytest.raises(ValueError):
            FederatedQueryRequest(query="test", max_domains=0)

        with pytest.raises(ValueError):
            FederatedQueryRequest(query="test", max_domains=7)

    def test_validation_min_confidence(self):
        """Test min_confidence validation."""
        from knowledge_base.query_api import FederatedQueryRequest

        with pytest.raises(ValueError):
            FederatedQueryRequest(query="test", min_confidence=-0.1)

        with pytest.raises(ValueError):
            FederatedQueryRequest(query="test", min_confidence=1.1)


class TestFederatedQueryResponse:
    """Tests for FederatedQueryResponse model."""

    def test_response_creation(self):
        """Test response model can be created."""
        from knowledge_base.query_api import FederatedQueryResponse

        response = FederatedQueryResponse(
            query="test query",
            detected_domains=[
                {"domain": "technical", "confidence": 0.8, "keywords": ["api"]}
            ],
            strategy="parallel",
            total_results=5,
            domain_results=[{"name": "test", "type": "entity"}],
        )
        assert response.query == "test query"
        assert response.total_results == 5
        assert len(response.detected_domains) == 1
        assert len(response.domain_results) == 1


class TestGetQueryServices:
    """Tests for get_query_services dependency."""

    @pytest.mark.asyncio
    async def test_get_query_services_returns_tuple(self):
        """Test dependency returns tuple of services."""
        from knowledge_base.query_api import get_query_services

        with (
            patch("knowledge_base.query_api.VectorStore") as mock_vs,
            patch("knowledge_base.query_api.GraphStore") as mock_gs,
            patch(
                "knowledge_base.query_api.HybridEntityRetriever"
            ) as mock_retriever_class,
            patch("knowledge_base.query_api.FederatedQueryRouter") as mock_router_class,
        ):
            mock_vs.return_value = MagicMock()
            mock_gs.return_value = MagicMock()
            mock_retriever_instance = MagicMock()
            mock_retriever_class.return_value = mock_retriever_instance
            mock_router_instance = MagicMock()
            mock_router_class.return_value = mock_router_instance

            result = await get_query_services()

            assert isinstance(result, tuple)
            assert len(result) == 2
            mock_retriever_class.assert_called_once()
            mock_router_class.assert_called_once()


class TestFederatedQueryEndpoint:
    """Tests for /federated endpoint."""

    @pytest.mark.asyncio
    async def test_execute_federated_query_basic(self):
        """Test basic federated query execution."""
        from knowledge_base.query_api import (
            execute_federated_query,
            FederatedQueryRequest,
            ExecutionStrategyEnum,
        )

        mock_router = MagicMock(spec=FederatedQueryRouter)
        mock_router._domain_detector = MagicMock()
        mock_router._domain_detector.detect.return_value = [
            MagicMock(domain=QueryDomain.TECHNICAL, confidence=0.8, keywords=["api"])
        ]

        mock_result = MagicMock(spec=FederatedQueryResult)
        mock_result.results = {
            QueryDomain.TECHNICAL: [{"id": "1", "name": "API", "type": "service"}]
        }
        mock_result.total_results = 1
        mock_router.route_and_execute = AsyncMock(return_value=mock_result)

        mock_retriever = MagicMock()

        services = (mock_router, mock_retriever)

        request = FederatedQueryRequest(
            query="How do I use the API?",
            max_domains=3,
            strategy=ExecutionStrategyEnum.PARALLEL,
            min_confidence=0.5,
        )

        response = await execute_federated_query(request, services)

        assert response.query == "How do I use the API?"
        assert response.total_results == 1
        assert len(response.detected_domains) == 1
        assert response.detected_domains[0]["domain"] == "technical"
        assert len(response.domain_results) == 1

    @pytest.mark.asyncio
    async def test_execute_federated_query_handles_error(self):
        """Test federated query handles errors gracefully."""
        from knowledge_base.query_api import (
            execute_federated_query,
            FederatedQueryRequest,
            ExecutionStrategyEnum,
        )
        from fastapi import HTTPException

        mock_router = MagicMock(spec=FederatedQueryRouter)
        mock_router._domain_detector = MagicMock()
        mock_router._domain_detector.detect.side_effect = Exception("Test error")

        mock_retriever = MagicMock()
        services = (mock_router, mock_retriever)

        request = FederatedQueryRequest(query="test")

        with pytest.raises(HTTPException) as exc_info:
            await execute_federated_query(request, services)

        assert exc_info.value.status_code == 500
        assert "Failed to execute federated query" in exc_info.value.detail
