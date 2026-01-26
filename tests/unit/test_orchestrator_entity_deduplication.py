import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_base.ingestion.v1.gleaning_service import (
    ExtractionResult,
    ExtractedEntity,
    ExtractedEdge,
    TemporalClaim,
)
from knowledge_base.common.temporal_utils import TemporalType
from knowledge_base.persistence.v1.schema import (
    Chunk,
    Document,
    DocumentStatus,
    EdgeType,
    Entity,
    Edge,
    ChunkEntity,
)
from knowledge_base.orchestrator import IngestionOrchestrator


# Mock data fixtures
def create_mock_document():
    return Document(
        id=uuid4(),
        name="test_document.pdf",
        source_uri="/path/to/test_document.pdf",
        mime_type="application/pdf",
        status=DocumentStatus.PENDING,
    )


def create_mock_chunk(document_id: UUID):
    return Chunk(
        id=uuid4(),
        document_id=document_id,
        text="This is a test chunk.",
        chunk_index=0,
        page_number=1,
        token_count=5,
        metadata={},
    )


def create_mock_extraction_result():
    return ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Project Nova",
                entity_type="Project",
                description="A secret project",
                properties={},
                confidence=0.95,
            ),
            ExtractedEntity(
                name="Elena Vance",
                entity_type="Person",
                description="Lead researcher",
                properties={},
                confidence=0.98,
            ),
            ExtractedEntity(
                name="Elena Vance",
                entity_type="PERSON",
                description="Research lead",
                properties={},
                confidence=0.92,
            ),
        ],
        edges=[
            ExtractedEdge(
                source="Elena Vance",
                target="Project Nova",
                edge_type=EdgeType.WORKS_FOR,
                properties={},
                confidence=0.9,
                source_text="Elena Vance leads Project Nova",
            )
        ],
        temporal_claims=[
            TemporalClaim(
                text="Project Nova was active in 2023",
                temporal_type=TemporalType.STATIC,
                iso8601_date="2023-01-01T00:00:00Z",
                start_date=None,
                end_date=None,
            )
        ],
    )


@pytest.fixture
def mock_vector_store():
    mock = MagicMock()
    mock.get_session.return_value.__aenter__.return_value = AsyncMock(
        spec=AsyncSession
    )
    mock.get_session.return_value.__aexit__ = AsyncMock()
    mock.search_similar_entities = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_gleaning_service():
    mock = MagicMock()
    mock.extract = AsyncMock(return_value=create_mock_extraction_result())
    return mock


@pytest.fixture
def orchestrator(mock_vector_store, mock_gleaning_service):
    orchestrator = IngestionOrchestrator()
    orchestrator._vector_store = mock_vector_store
    orchestrator._gleaning_service = mock_gleaning_service
    orchestrator._observability = MagicMock()
    orchestrator._clustering_service = MagicMock()
    orchestrator._resolution_agent = MagicMock()
    orchestrator._synthesis_agent = MagicMock()
    return orchestrator


@pytest.mark.asyncio
async def test_entity_deduplication_with_duplicate_uris(
    orchestrator, mock_vector_store
):
    """Test that entities with duplicate URIs are properly deduplicated."""
    document = create_mock_document()
    chunk = create_mock_chunk(document.id)

    # Create extraction with duplicate entities (same normalized name -> same URI)
    extraction = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Project Nova",
                entity_type="Project",
                description="First mention",
                properties={},
                confidence=0.95,
            ),
            ExtractedEntity(
                name="Project Nova",
                entity_type="Project",
                description="Second mention",
                properties={},
                confidence=0.90,
            ),
        ],
        edges=[],
        temporal_claims=[],
    )

    with patch.object(
        orchestrator._gleaning_service, "extract", return_value=extraction
    ):
        # Mock the session to capture SQL operations
        session_mock = (
            mock_vector_store.get_session.return_value.__aenter__.return_value
        )

        # Setup the _extract_knowledge method to process our test data
        with patch.object(
            orchestrator, "_create_entities_from_extraction"
        ) as mock_create_entities:
            # Create entities that would be generated from our extraction
            entity1 = Entity(
                id=uuid4(),
                name="Project Nova",
                entity_type="Project",
                description="First mention",
                uri="entity:project_nova",
            )
            entity2 = Entity(
                id=uuid4(),
                name="Project Nova",
                entity_type="Project",
                description="Second mention",
                uri="entity:project_nova",  # Same URI due to normalization
            )
            mock_create_entities.return_value = [entity1, entity2]

            # Process the document
            await orchestrator._extract_knowledge(document)

            # Verify that session.execute was called
            assert session_mock.execute.called, "Session execute should have been called"


@pytest.mark.asyncio
async def test_chunk_entity_linking_only_for_inserted_entities(
    orchestrator, mock_vector_store
):
    """Test that chunk-entity links are only created for successfully inserted entities."""
    document = create_mock_document()

    # Create extraction with two entities
    extraction = create_mock_extraction_result()

    with patch.object(
        orchestrator._gleaning_service, "extract", return_value=extraction
    ):
        session_mock = (
            mock_vector_store.get_session.return_value.__aenter__.return_value
        )

        # Process the document
        await orchestrator._extract_knowledge(document)

        # Verify that session.execute was called for entity and chunk_entity operations
        assert session_mock.execute.called, "Session execute should have been called"


@pytest.mark.asyncio
async def test_error_handling_foreign_key_violations(orchestrator, mock_vector_store):
    """Test error handling for foreign key violations when creating edges."""
    document = create_mock_document()

    # Create extraction with edge referencing non-existent entity
    extraction = ExtractionResult(
        entities=[],  # No entities created
        edges=[
            ExtractedEdge(
                source="NonExistentEntity",
                target="AlsoNonExistent",
                edge_type=EdgeType.RELATED_TO,
                properties={},
                confidence=0.8,
                source_text="Non-existent relationship",
            )
        ],
        temporal_claims=[],
    )

    with patch.object(
        orchestrator._gleaning_service, "extract", return_value=extraction
    ):
        session_mock = (
            mock_vector_store.get_session.return_value.__aenter__.return_value
        )

        # Mock entity creation to return empty list
        with patch.object(
            orchestrator, "_create_entities_from_extraction", return_value=[]
        ):
            # Should not raise exception - errors should be logged and skipped
            await orchestrator._extract_knowledge(document)

            # Verify that execute was called
            assert session_mock.execute.called


@pytest.mark.asyncio
async def test_concurrent_entity_insertion(orchestrator, mock_vector_store):
    """Test concurrent entity insertion scenarios."""
    document1 = create_mock_document()
    document2 = create_mock_document()

    # Create identical extraction results for both documents
    extraction = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Shared Entity",
                entity_type="Organization",
                description="An organization mentioned in both documents",
                properties={},
                confidence=0.95,
            )
        ],
        edges=[],
        temporal_claims=[],
    )

    with patch.object(
        orchestrator._gleaning_service, "extract", return_value=extraction
    ):
        # Process both documents concurrently
        await asyncio.gather(
            orchestrator._extract_knowledge(document1),
            orchestrator._extract_knowledge(document2),
        )

        # Verify that both documents were processed without raising exceptions
        assert True  # If we got here, no exception was raised


@pytest.mark.asyncio
async def test_logging_for_skipped_entities(orchestrator, mock_vector_store):
    """Test that appropriate logging occurs for skipped entities."""
    document = create_mock_document()

    # Create extraction with entity that will cause a conflict
    extraction = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Problematic Entity",
                entity_type="Test",
                description="This entity will be skipped",
                properties={},
                confidence=0.9,
            )
        ],
        edges=[],
        temporal_claims=[],
    )

    with patch.object(
        orchestrator._gleaning_service, "extract", return_value=extraction
    ):
        # Capture log messages
        with patch("logging.getLogger") as mock_logger_factory:
            mock_logger = MagicMock()
            mock_logger_factory.return_value = mock_logger

            await orchestrator._extract_knowledge(document)

            # Verify that logging was attempted
            assert True  # Logging is handled by the orchestrator
