"""Tests for new multi-agent extraction services in orchestrator."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from knowledge_base.orchestrator import IngestionOrchestrator
from knowledge_base.intelligence.v1.multi_agent_extractor import (
    EntityExtractionManager,
    ExtractedEntity,
    ExtractionQualityScore,
)
from knowledge_base.intelligence.v1.hallucination_detector import (
    HallucinationDetector,
    EntityVerification,
    RiskLevel,
    VerificationStatus,
)
from knowledge_base.intelligence.v1.cross_domain_detector import CrossDomainDetector
from knowledge_base.persistence.v1.schema import (
    Document,
    Chunk,
    Entity,
    Edge,
    ReviewQueue,
    ReviewStatus,
)


@pytest.fixture
def orchestrator():
    """Create a test orchestrator instance."""
    return IngestionOrchestrator()


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
def sample_chunks():
    """Create sample chunks for testing."""
    doc_id = uuid4()
    return [
        Chunk(
            id=uuid4(),
            document_id=doc_id,
            text="Test chunk 1 with some entity information.",
            chunk_index=0,
        ),
        Chunk(
            id=uuid4(),
            document_id=doc_id,
            text="Test chunk 2 with more entity details.",
            chunk_index=1,
        ),
    ]


@pytest.fixture
def sample_extracted_entities(sample_chunks):
    """Create sample extracted entities for testing."""
    return [
        ExtractedEntity(
            id=uuid4(),
            name="Test Entity",
            entity_type="CONCEPT",
            description="A test entity",
            properties={"key": "value"},
            source_text="Test chunk 1 with some entity information.",
            chunk_id=sample_chunks[0].id,
            confidence=0.9,
        ),
    ]


@pytest.fixture
def sample_quality_score():
    """Create a sample quality score."""
    return ExtractionQualityScore(
        overall_score=0.85,
        entity_quality=0.9,
        relationship_quality=0.8,
        coherence_score=0.85,
        quality_level="high",
        feedback="Good extraction",
    )


@pytest.fixture
def sample_entity_verification():
    """Create a sample entity verification."""
    return EntityVerification(
        entity_name="Test Entity",
        entity_type="CONCEPT",
        risk_level=RiskLevel.LOW,
        overall_confidence=0.9,
        total_attributes=1,
        supported_count=1,
        unsupported_count=0,
        inconclusive_count=0,
        is_hallucinated=False,
    )


class TestNewServiceAttributes:
    """Tests for new service attributes in orchestrator."""

    def test_extraction_manager_attribute_exists(self, orchestrator):
        """Test that _extraction_manager attribute exists."""
        assert hasattr(orchestrator, "_extraction_manager")
        assert orchestrator._extraction_manager is None

    def test_hallucination_detector_attribute_exists(self, orchestrator):
        """Test that _hallucination_detector attribute exists."""
        assert hasattr(orchestrator, "_hallucination_detector")
        assert orchestrator._hallucination_detector is None

    def test_cross_domain_detector_attribute_exists(self, orchestrator):
        """Test that _cross_domain_detector attribute exists."""
        assert hasattr(orchestrator, "_cross_domain_detector")
        assert orchestrator._cross_domain_detector is None


class TestConvertExtractionToEntities:
    """Tests for _convert_extraction_to_entities method."""

    def test_convert_extraction_to_entities_empty(self, orchestrator):
        """Test converting empty extraction results."""
        entities, edges = orchestrator._convert_extraction_to_entities([], [])
        assert entities == []
        assert edges == []

    def test_convert_extraction_to_entities_with_data(
        self, orchestrator, sample_extracted_entities, sample_chunks
    ):
        """Test converting extraction results with data."""
        entities, edges = orchestrator._convert_extraction_to_entities(
            sample_extracted_entities, sample_chunks
        )

        assert len(entities) == 1
        entity = entities[0]
        assert entity.name == "Test Entity"
        assert entity.entity_type == "CONCEPT"
        assert entity.confidence == 0.9
        assert entity.uri is not None

    def test_convert_extraction_creates_uri(
        self, orchestrator, sample_extracted_entities
    ):
        """Test that URIs are created properly."""
        entities, _ = orchestrator._convert_extraction_to_entities(
            sample_extracted_entities, []
        )

        assert len(entities) == 1
        assert entities[0].uri.startswith("entity:")

    def test_convert_extraction_with_links(self, orchestrator, sample_chunks):
        """Test converting extraction with entity links."""
        entity1_id = uuid4()
        entity2_id = uuid4()

        entities = [
            ExtractedEntity(
                id=entity1_id,
                name="Entity 1",
                entity_type="PERSON",
                description="First entity",
                properties={},
                source_text="Source text",
                chunk_id=sample_chunks[0].id,
                confidence=0.9,
                linked_entities=[entity2_id],
            ),
            ExtractedEntity(
                id=entity2_id,
                name="Entity 2",
                entity_type="ORGANIZATION",
                description="Second entity",
                properties={},
                source_text="Source text",
                chunk_id=sample_chunks[0].id,
                confidence=0.85,
            ),
        ]

        result_entities, result_edges = orchestrator._convert_extraction_to_entities(
            entities, sample_chunks
        )

        assert len(result_entities) == 2
        assert len(result_edges) == 1
        assert result_edges[0].edge_type.value == "related_to"


class TestRouteToReview:
    """Tests for _route_to_review method."""

    @pytest.mark.asyncio
    async def test_route_to_review_empty(self, orchestrator, sample_document):
        """Test routing with empty verifications."""
        await orchestrator._route_to_review([], sample_document, "source text")

    @pytest.mark.asyncio
    async def test_route_to_review_low_risk(
        self, orchestrator, sample_document, sample_entity_verification
    ):
        """Test routing low-risk entity (should not add to queue)."""
        sample_entity_verification.risk_level = RiskLevel.LOW
        sample_entity_verification.is_hallucinated = False

        mock_session = Mock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()

        mock_vector_store = Mock()
        mock_vector_store.get_session = Mock(return_value=mock_session)

        orchestrator._vector_store = mock_vector_store

        await orchestrator._route_to_review(
            [sample_entity_verification], sample_document, "source text"
        )

        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_to_review_high_risk(
        self, orchestrator, sample_document, sample_entity_verification
    ):
        """Test routing high-risk entity (should add to queue)."""
        sample_entity_verification.risk_level = RiskLevel.HIGH
        sample_entity_verification.is_hallucinated = True
        sample_entity_verification.hallucination_reasons = ["Unsupported attribute"]

        mock_session = Mock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()

        mock_vector_store = Mock()
        mock_vector_store.get_session = Mock(return_value=mock_session)

        mock_observability = Mock()
        mock_observability.log_metric = Mock()

        orchestrator._vector_store = mock_vector_store
        orchestrator._observability = mock_observability

        await orchestrator._route_to_review(
            [sample_entity_verification], sample_document, "source text"
        )

        mock_session.add.assert_called_once()

        call_args = mock_session.add.call_args[0][0]
        assert isinstance(call_args, ReviewQueue)
        assert call_args.item_type == "entity_verification"
        assert call_args.status == ReviewStatus.PENDING
        assert call_args.priority == 7

    @pytest.mark.asyncio
    async def test_route_to_review_critical_risk(
        self, orchestrator, sample_document, sample_entity_verification
    ):
        """Test routing critical-risk entity (should have high priority)."""
        sample_entity_verification.risk_level = RiskLevel.CRITICAL
        sample_entity_verification.is_hallucinated = True

        mock_session = Mock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()

        mock_vector_store = Mock()
        mock_vector_store.get_session = Mock(return_value=mock_session)

        mock_observability = Mock()
        mock_observability.log_metric = Mock()

        orchestrator._vector_store = mock_vector_store
        orchestrator._observability = mock_observability

        await orchestrator._route_to_review(
            [sample_entity_verification], sample_document, "source text"
        )

        mock_session.add.assert_called_once()

        call_args = mock_session.add.call_args[0][0]
        assert call_args.priority == 9


class TestMultiAgentExtractionMethod:
    """Tests for _extract_entities_multi_agent method structure."""

    def test_method_exists(self, orchestrator):
        """Test that _extract_entities_multi_agent method exists."""
        assert hasattr(orchestrator, "_extract_entities_multi_agent")
        assert callable(orchestrator._extract_entities_multi_agent)


class TestEnsureExtractionManager:
    """Tests for _ensure_extraction_manager method."""

    def test_method_exists(self, orchestrator):
        """Test that _ensure_extraction_manager method exists."""
        assert hasattr(orchestrator, "_ensure_extraction_manager")
        assert callable(orchestrator._ensure_extraction_manager)
