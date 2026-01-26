"""Unit tests for hybrid entity retriever."""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from knowledge_base.intelligence.v1.hybrid_retriever import (
    GraphEntity,
    HybridEntityRetriever,
    HybridRetrievalResult,
    RetrievedEntity,
)


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = AsyncMock()
    store.search_similar_entities = AsyncMock()
    return store


@pytest.fixture
def mock_graph_store():
    """Create a mock graph store."""
    store = AsyncMock()
    store.get_entity_neighborhood = AsyncMock()
    return store


@pytest.fixture
def sample_vector_results():
    """Sample vector search results."""
    return [
        {
            "id": uuid4(),
            "name": "Entity1",
            "entity_type": "Person",
            "description": "A person entity",
            "properties": {"domain": "test"},
            "confidence": 0.9,
            "similarity": 0.85,
        },
        {
            "id": uuid4(),
            "name": "Entity2",
            "entity_type": "Organization",
            "description": "An organization entity",
            "properties": {"domain": "test"},
            "confidence": 0.8,
            "similarity": 0.75,
        },
    ]


@pytest.fixture
def sample_graph_entities():
    """Sample entities from graph expansion."""
    entity1 = MagicMock()
    entity1.id = uuid4()
    entity1.name = "GraphEntity1"
    entity1.entity_type = "Location"
    entity1.description = "A location entity"
    entity1.properties = {}
    entity1.confidence = 0.7

    entity2 = MagicMock()
    entity2.id = uuid4()
    entity2.name = "GraphEntity2"
    entity2.entity_type = "Event"
    entity2.description = "An event entity"
    entity2.properties = {}
    entity2.confidence = 0.65

    edge1 = MagicMock()
    edge1.confidence = 0.7

    edge2 = MagicMock()
    edge2.confidence = 0.65

    return [
        (entity1, edge1),
        (entity2, edge2),
    ]


class TestHybridEntityRetriever:
    """Tests for HybridEntityRetriever class."""

    @pytest.mark.asyncio
    async def test_init_with_valid_weights(self, mock_vector_store, mock_graph_store):
        """Test initialization with valid weights."""
        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            vector_weight=0.7,
            graph_weight=0.3,
        )
        assert retriever._vector_weight == 0.7
        assert retriever._graph_weight == 0.3

    @pytest.mark.asyncio
    async def test_init_with_invalid_weights(self, mock_vector_store, mock_graph_store):
        """Test initialization with invalid weights raises error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            HybridEntityRetriever(
                vector_store=mock_vector_store,
                graph_store=mock_graph_store,
                vector_weight=0.5,
                graph_weight=0.7,
            )

    @pytest.mark.asyncio
    async def test_retrieve_vector_only(
        self,
        mock_vector_store,
        mock_graph_store,
        sample_vector_results,
    ):
        """Test retrieval with only vector results."""
        mock_vector_store.search_similar_entities.return_value = sample_vector_results
        mock_graph_store.get_entity_neighborhood.side_effect = ValueError("Not found")

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
        )

        result = await retriever.retrieve(
            query="test query",
            query_embedding=[0.1] * 100,
            graph_depth=1,
        )

        assert isinstance(result, HybridRetrievalResult)
        assert result.query == "test query"
        assert len(result.entities) == 2
        assert result.vector_results_count == 2
        assert result.graph_results_count == 0

    @pytest.mark.asyncio
    async def test_retrieve_with_graph_expansion(
        self,
        mock_vector_store,
        mock_graph_store,
        sample_vector_results,
        sample_graph_entities,
    ):
        """Test retrieval with vector and graph results."""
        mock_vector_store.search_similar_entities.return_value = sample_vector_results

        center_entity = MagicMock()
        mock_graph_store.get_entity_neighborhood.return_value = (
            center_entity,
            sample_graph_entities,
        )

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
        )

        result = await retriever.retrieve(
            query="test query",
            query_embedding=[0.1] * 100,
            graph_depth=1,
        )

        assert len(result.entities) >= 2
        assert result.vector_results_count == 2
        assert result.graph_results_count == 2

    @pytest.mark.asyncio
    async def test_retrieve_with_domain_filter(
        self,
        mock_vector_store,
        mock_graph_store,
    ):
        """Test retrieval with domain filtering."""
        mixed_domain_results = [
            {
                "id": uuid4(),
                "name": "Entity1",
                "entity_type": "Person",
                "description": "A person entity",
                "properties": {"domain": "test"},
                "confidence": 0.9,
                "similarity": 0.85,
            },
            {
                "id": uuid4(),
                "name": "Entity2",
                "entity_type": "Organization",
                "description": "An organization entity",
                "properties": {"domain": "other"},
                "confidence": 0.8,
                "similarity": 0.75,
            },
        ]
        mock_vector_store.search_similar_entities.return_value = mixed_domain_results
        mock_graph_store.get_entity_neighborhood.side_effect = ValueError("Not found")

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
        )

        result = await retriever.retrieve(
            query="test query",
            query_embedding=[0.1] * 100,
            domain="test",
        )

        mock_vector_store.search_similar_entities.assert_called_once()
        assert len(result.entities) == 1
        assert result.entities[0].name == "Entity1"

    @pytest.mark.asyncio
    async def test_retrieve_scores_combined_correctly(
        self,
        mock_vector_store,
        mock_graph_store,
        sample_vector_results,
    ):
        """Test that scores are combined with proper weights."""
        mock_vector_store.search_similar_entities.return_value = sample_vector_results
        mock_graph_store.get_entity_neighborhood.side_effect = ValueError("Not found")

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            vector_weight=0.6,
            graph_weight=0.4,
        )

        result = await retriever.retrieve(
            query="test query",
            query_embedding=[0.1] * 100,
        )

        for entity in result.entities:
            expected_score = entity.vector_score * 0.6
            assert abs(entity.final_score - expected_score) < 0.01

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(
        self,
        mock_vector_store,
        mock_graph_store,
    ):
        """Test retrieval with no results."""
        mock_vector_store.search_similar_entities.return_value = []

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
        )

        result = await retriever.retrieve(
            query="test query",
            query_embedding=[0.1] * 100,
        )

        assert len(result.entities) == 0
        assert result.vector_results_count == 0
        assert result.graph_results_count == 0

    @pytest.mark.asyncio
    async def test_retrieve_respects_limits(
        self,
        mock_vector_store,
        mock_graph_store,
        sample_vector_results,
        sample_graph_entities,
    ):
        """Test that retrieval respects configured limits."""
        mock_vector_store.search_similar_entities.return_value = sample_vector_results

        center_entity = MagicMock()
        mock_graph_store.get_entity_neighborhood.return_value = (
            center_entity,
            sample_graph_entities,
        )

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
        )

        result = await retriever.retrieve(
            query="test query",
            query_embedding=[0.1] * 100,
            vector_limit=5,
            graph_limit=1,
        )

        mock_vector_store.search_similar_entities.assert_called_once()
        call_kwargs = mock_vector_store.search_similar_entities.call_args.kwargs
        assert call_kwargs["limit"] == 5


class TestRetrievedEntity:
    """Tests for RetrievedEntity dataclass."""

    def test_retrieved_entity_defaults(self):
        """Test default values for RetrievedEntity."""
        entity_id = uuid4()
        entity = RetrievedEntity(
            id=entity_id,
            name="Test",
            entity_type="Type",
            description=None,
            properties=None,
            confidence=0.9,
            vector_score=0.8,
        )

        assert entity.graph_score is None
        assert entity.graph_hops is None
        assert entity.final_score == 0.0
        assert entity.source == "vector"

    def test_retrieved_entity_graph_source(self):
        """Test RetrievedEntity from graph source."""
        entity_id = uuid4()
        entity = RetrievedEntity(
            id=entity_id,
            name="Test",
            entity_type="Type",
            description=None,
            properties=None,
            confidence=0.9,
            vector_score=0.0,
            graph_score=0.7,
            graph_hops=2,
            source="graph",
        )

        assert entity.graph_score == 0.7
        assert entity.graph_hops == 2
        assert entity.source == "graph"


class TestHybridRetrievalResult:
    """Tests for HybridRetrievalResult dataclass."""

    def test_result_defaults(self):
        """Test default values for HybridRetrievalResult."""
        result = HybridRetrievalResult(query="test")

        assert result.entities == []
        assert result.vector_results_count == 0
        assert result.graph_results_count == 0
        assert result.total_hops_traversed == 0

    def test_result_with_entities(self):
        """Test result with entities."""
        entities = [
            RetrievedEntity(
                id=uuid4(),
                name="E1",
                entity_type="Type",
                description=None,
                properties=None,
                confidence=0.9,
                vector_score=0.8,
            ),
            RetrievedEntity(
                id=uuid4(),
                name="E2",
                entity_type="Type",
                description=None,
                properties=None,
                confidence=0.8,
                vector_score=0.7,
            ),
        ]
        result = HybridRetrievalResult(
            query="test",
            entities=entities,
            vector_results_count=2,
            graph_results_count=0,
            total_hops_traversed=0,
        )

        assert len(result.entities) == 2


class TestGraphEntity:
    """Tests for GraphEntity class."""

    def test_graph_entity_creation(self):
        """Test GraphEntity instantiation."""
        entity_id = uuid4()
        graph_entity = GraphEntity(
            id=entity_id,
            name="TestGraphEntity",
            entity_type="Location",
            description="A test location",
            properties={"country": "USA"},
            confidence=0.85,
            hops=2,
        )

        assert graph_entity.id == entity_id
        assert graph_entity.name == "TestGraphEntity"
        assert graph_entity.entity_type == "Location"
        assert graph_entity.confidence == 0.85
        assert graph_entity.hops == 2

    def test_graph_entity_hops_calculation(self):
        """Test hop count assignment."""
        entity1 = GraphEntity(
            id=uuid4(),
            name="E1",
            entity_type="Type",
            description=None,
            properties=None,
            confidence=0.9,
            hops=1,
        )

        entity2 = GraphEntity(
            id=uuid4(),
            name="E2",
            entity_type="Type",
            description=None,
            properties=None,
            confidence=0.8,
            hops=2,
        )

        assert entity1.hops == 1
        assert entity2.hops == 2


class TestHybridEntityRetrieverEdgeCases:
    """Edge case tests for HybridEntityRetriever."""

    @pytest.mark.asyncio
    async def test_retrieve_handles_graph_store_error(
        self,
        mock_vector_store,
        mock_graph_store,
    ):
        """Test retrieval handles graph store errors gracefully."""
        sample_results = [
            {
                "id": uuid4(),
                "name": "Entity1",
                "entity_type": "Person",
                "description": "A person entity",
                "properties": {},
                "confidence": 0.9,
                "similarity": 0.85,
            },
            {
                "id": uuid4(),
                "name": "Entity2",
                "entity_type": "Organization",
                "description": "An organization entity",
                "properties": {},
                "confidence": 0.8,
                "similarity": 0.75,
            },
        ]
        mock_vector_store.search_similar_entities.return_value = sample_results
        mock_graph_store.get_entity_neighborhood.side_effect = Exception("Graph error")

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
        )

        result = await retriever.retrieve(
            query="test query",
            query_embedding=[0.1] * 100,
        )

        assert len(result.entities) == 2
        assert result.vector_results_count == 2
        assert result.graph_results_count == 0

    @pytest.mark.asyncio
    async def test_retrieve_handles_missing_properties(
        self,
        mock_vector_store,
        mock_graph_store,
    ):
        """Test retrieval handles entities without properties."""
        results_without_properties = [
            {
                "id": uuid4(),
                "name": "Entity1",
                "entity_type": "Person",
                "description": "A person entity",
                "confidence": 0.9,
                "similarity": 0.85,
            },
        ]
        mock_vector_store.search_similar_entities.return_value = (
            results_without_properties
        )
        mock_graph_store.get_entity_neighborhood.side_effect = ValueError("Not found")

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
        )

        result = await retriever.retrieve(
            query="test query",
            query_embedding=[0.1] * 100,
        )

        assert len(result.entities) == 1
        assert result.entities[0].properties is None

    @pytest.mark.asyncio
    async def test_retrieve_entity_type_filtering(
        self,
        mock_vector_store,
        mock_graph_store,
    ):
        """Test entity type filtering in retrieval."""
        sample_results = [
            {
                "id": uuid4(),
                "name": "Entity1",
                "entity_type": "Person",
                "description": "A person entity",
                "properties": {},
                "confidence": 0.9,
                "similarity": 0.85,
            },
        ]
        mock_vector_store.search_similar_entities.return_value = sample_results
        mock_graph_store.get_entity_neighborhood.side_effect = ValueError("Not found")

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
        )

        result = await retriever.retrieve(
            query="test query",
            query_embedding=[0.1] * 100,
            entity_types=["Person", "Organization"],
        )

        mock_graph_store.get_entity_neighborhood.assert_called_once()
        call_kwargs = mock_graph_store.get_entity_neighborhood.call_args.kwargs
        assert call_kwargs["node_types"] == ["Person", "Organization"]

    @pytest.mark.asyncio
    async def test_retrieve_calculates_total_hops(
        self,
        mock_vector_store,
        mock_graph_store,
        sample_vector_results,
        sample_graph_entities,
    ):
        """Test that total hops are calculated correctly."""
        mock_vector_store.search_similar_entities.return_value = sample_vector_results

        center_entity = MagicMock()
        mock_graph_store.get_entity_neighborhood.return_value = (
            center_entity,
            sample_graph_entities,
        )

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
        )

        result = await retriever.retrieve(
            query="test query",
            query_embedding=[0.1] * 100,
            graph_depth=2,
        )

        assert result.total_hops_traversed > 0

    @pytest.mark.asyncio
    async def test_entities_ranked_by_final_score(
        self,
        mock_vector_store,
        mock_graph_store,
    ):
        """Test that entities are ranked by final score."""
        entity1_id = uuid4()
        entity2_id = uuid4()
        entity3_id = uuid4()

        vector_results = [
            {
                "id": entity1_id,
                "name": "Entity1",
                "entity_type": "Type",
                "description": None,
                "properties": {},
                "confidence": 0.9,
                "similarity": 0.9,
            },
            {
                "id": entity2_id,
                "name": "Entity2",
                "entity_type": "Type",
                "description": None,
                "properties": {},
                "confidence": 0.8,
                "similarity": 0.6,
            },
        ]
        mock_vector_store.search_similar_entities.return_value = vector_results

        graph_entity = MagicMock()
        graph_entity.id = entity3_id
        graph_entity.name = "GraphEntity"
        graph_entity.entity_type = "Type"
        graph_entity.description = None
        graph_entity.properties = {}
        graph_entity.confidence = 0.95

        graph_edge = MagicMock()
        graph_edge.confidence = 0.85

        mock_graph_store.get_entity_neighborhood.return_value = (
            MagicMock(),
            [(graph_entity, graph_edge)],
        )

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            vector_weight=0.6,
            graph_weight=0.4,
        )

        result = await retriever.retrieve(
            query="test query",
            query_embedding=[0.1] * 100,
        )

        scores = [e.final_score for e in result.entities]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_retrieve_uses_min_confidence_threshold(
        self,
        mock_vector_store,
        mock_graph_store,
        sample_vector_results,
    ):
        """Test that min confidence threshold is applied."""
        mock_vector_store.search_similar_entities.return_value = sample_vector_results
        mock_graph_store.get_entity_neighborhood.side_effect = ValueError("Not found")

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
        )

        result = await retriever.retrieve(
            query="test query",
            query_embedding=[0.1] * 100,
            min_confidence=0.75,
        )

        mock_vector_store.search_similar_entities.assert_called_once()
        call_kwargs = mock_vector_store.search_similar_entities.call_args.kwargs
        assert call_kwargs["similarity_threshold"] == 0.75

    @pytest.mark.asyncio
    async def test_default_weights(self, mock_vector_store, mock_graph_store):
        """Test that default weights are applied correctly."""
        mock_vector_store.search_similar_entities.return_value = []
        mock_graph_store.get_entity_neighborhood.side_effect = ValueError("Not found")

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
        )

        assert retriever._vector_weight == HybridEntityRetriever.DEFAULT_VECTOR_WEIGHT
        assert retriever._graph_weight == HybridEntityRetriever.DEFAULT_GRAPH_WEIGHT
