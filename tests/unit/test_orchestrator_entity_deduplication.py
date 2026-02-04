"""Tests for entity pipeline service."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from knowledge_base.orchestration.entity_pipeline_service import EntityPipelineService
from knowledge_base.persistence.v1.schema import Document, Entity, Chunk, Edge
from knowledge_base.ingestion.v1.gleaning_service import (
    ExtractedEntity,
    ExtractionResult,
)


@pytest.fixture
def entity_service():
    """Create a test entity pipeline service instance."""
    service = EntityPipelineService()
    return service


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id=uuid4(),
        name="test_document.pdf",
        source_uri="/path/to/test.pdf",
        mime_type="application/pdf",
        status="pending",
    )


@pytest.fixture
def sample_entity():
    """Create a sample entity for testing."""
    return Entity(
        id=uuid4(),
        name="Test Entity",
        entity_type="CONCEPT",
        description="A test entity",
        properties={"key": "value"},
        confidence=0.9,
    )


class TestEntityPipelineService:
    """Tests for EntityPipelineService."""

    def test_initialization(self, entity_service):
        """Test service initialization."""
        assert entity_service._multi_agent_extractor is None
        assert entity_service._entity_resolver is None
        assert entity_service._entity_typer is None
        assert entity_service._clustering_service is None
        assert entity_service._gleaning_service is None
        assert entity_service._hallucination_detector is None

    def test_set_extractors(self, entity_service):
        """Test setting extraction components."""
        mock_extractor = Mock()
        mock_resolver = Mock()
        mock_typer = Mock()
        mock_clustering = Mock()
        mock_gleaning = Mock()
        mock_detector = Mock()

        entity_service.set_extractors(
            multi_agent_extractor=mock_extractor,
            entity_resolver=mock_resolver,
            entity_typer=mock_typer,
            clustering_service=mock_clustering,
            gleaning_service=mock_gleaning,
            hallucination_detector=mock_detector,
        )

        assert entity_service._multi_agent_extractor == mock_extractor
        assert entity_service._entity_resolver == mock_resolver
        assert entity_service._entity_typer == mock_typer
        assert entity_service._clustering_service == mock_clustering
        assert entity_service._gleaning_service == mock_gleaning
        assert entity_service._hallucination_detector == mock_detector


class TestEntityConversion:
    """Tests for entity conversion methods."""

    def test_create_entities_from_extraction(self, entity_service):
        """Test creating entities from extraction result."""
        chunk_id = uuid4()
        extraction = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Test Entity",
                    entity_type="CONCEPT",
                    description="A test entity",
                    properties={"key": "value"},
                    confidence=0.9,
                )
            ],
            edges=[],
        )

        entities = entity_service._create_entities_from_extraction(extraction, chunk_id)

        assert len(entities) == 1
        assert entities[0].name == "Test Entity"
        assert entities[0].entity_type == "CONCEPT"
        assert entities[0].uri.startswith("entity:")

    def test_create_edges_from_extraction(self, entity_service):
        """Test creating edges from extraction result."""
        entity1_id = uuid4()
        entity2_id = uuid4()

        entities = [
            Entity(id=entity1_id, name="Entity 1", entity_type="PERSON"),
            Entity(id=entity2_id, name="Entity 2", entity_type="ORGANIZATION"),
        ]

        extraction = ExtractionResult(
            entities=[],
            edges=[
                {
                    "source": "Entity 1",
                    "target": "Entity 2",
                    "edge_type": "related_to",
                    "properties": {},
                    "confidence": 0.9,
                }
            ],
        )

        edges = entity_service._create_edges_from_extraction(extraction, entities)

        assert len(edges) == 1
        assert edges[0].source_id == entity1_id
        assert edges[0].target_id == entity2_id


class TestEntityResolution:
    """Tests for entity resolution methods."""

    def test_convert_extraction_to_entities(self, entity_service):
        """Test converting multi-agent extraction results."""
        from knowledge_base.intelligence.v1.multi_agent_extractor import (
            ExtractedEntity as MultiAgentExtractedEntity,
        )

        chunk_id = uuid4()
        extracted = [
            MultiAgentExtractedEntity(
                id=uuid4(),
                name="Test Entity",
                entity_type="CONCEPT",
                description="A test entity",
                properties={"key": "value"},
                source_text="Source text",
                confidence=0.9,
                chunk_id=chunk_id,
                linked_entities=[],
            )
        ]

        chunks = [Chunk(id=chunk_id, text="chunk text", chunk_index=0)]

        entities, edges = entity_service._convert_extraction_to_entities(
            extracted, chunks
        )

        assert len(entities) == 1
        assert entities[0].name == "Test Entity"
        assert entities[0].entity_type == "CONCEPT"
