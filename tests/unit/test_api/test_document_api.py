"""
Unit tests for Document API endpoints.

Tests cover all endpoints defined in knowledge_base/document_api.py:
- GET /api/v1/documents/{document_id} - Document metadata
- GET /api/v1/documents/{document_id}/content - Document content
- GET /api/v1/documents/{document_id}/spans - Text spans with entity references
- GET /api/v1/documents/{document_id}/entities - Extracted entities
- POST /api/v1/documents:search - Hybrid search
- POST /api/v1/documents/{document_id}/annotations - W3C Annotations
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_base import document_api
from knowledge_base.common.api_models import APIResponse
from knowledge_base.persistence.v1.schema import Document, Chunk, Entity, ChunkEntity, DocumentStatus


@pytest.fixture
def test_document_id():
    """Test document UUID."""
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def sample_document(test_document_id):
    """Sample document for testing."""
    doc = Document(
        id=test_document_id,
        name="Test Document.pdf",
        source_uri="s3://test-bucket/test.pdf",
        status=DocumentStatus.COMPLETED,
        domain="finance",
        doc_metadata={"mime_type": "application/pdf", "processing_time_ms": 5000},
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        updated_at=datetime(2024, 1, 1, 12, 5, 0),
    )
    # Add metadata attribute to match document_api.py usage
    doc.metadata = doc.doc_metadata
    return doc


@pytest.fixture
def sample_chunks(test_document_id):
    """Sample document chunks for testing."""
    return [
        Chunk(
            id=uuid.uuid4(),
            document_id=test_document_id,
            chunk_index=0,
            text="This is the first chunk of text. ",
        ),
        Chunk(
            id=uuid.uuid4(),
            document_id=test_document_id,
            chunk_index=1,
            text="This is the second chunk of text. ",
        ),
    ]


@pytest.fixture
def sample_entities(test_document_id):
    """Sample entities for testing."""
    entity1 = Entity(
        id=uuid.uuid4(),
        name="Test Company",
        entity_type="Organization",
        description="A test organization",
        confidence=0.95,
        properties={"industry": "technology"},
        created_at=datetime(2024, 1, 1, 12, 1, 0),
    )
    entity2 = Entity(
        id=uuid.uuid4(),
        name="John Smith",
        entity_type="Person",
        description="CEO of Test Company",
        confidence=0.88,
        properties={},
        created_at=datetime(2024, 1, 1, 12, 2, 0),
    )
    return [entity1, entity2]


@pytest.fixture
def sample_chunk_entities(sample_chunks, sample_entities):
    """Sample chunk-entity relationships."""
    return [
        ChunkEntity(
            id=uuid.uuid4(),
            chunk_id=sample_chunks[0].id,
            entity_id=sample_entities[0].id,
            confidence=0.95,
            grounding_quote="Test Company",
        ),
        ChunkEntity(
            id=uuid.uuid4(),
            chunk_id=sample_chunks[1].id,
            entity_id=sample_entities[1].id,
            confidence=0.88,
            grounding_quote="John Smith",
        ),
    ]


@pytest.fixture
def mock_session_execute():
    """Helper to create properly mocked execute results."""
    def _create_mock_scalar_result(value):
        """Create a mock scalar result."""
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=value)
        if isinstance(value, list):
            result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=value)))
        else:
            result.scalar = MagicMock(return_value=value)
        return result

    return _create_mock_scalar_result


class TestGetDocument:
    """Tests for GET /api/v1/documents/{document_id}"""

    @pytest.mark.asyncio
    async def test_get_document_success(
        self,
        mock_session_execute,
        test_document_id,
        sample_document,
        sample_chunks,
    ):
        """Test successful document metadata retrieval."""
        # Create mock session
        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()

        # Setup execute to return different mock results
        session.execute.side_effect = [
            mock_session_execute(sample_document),  # Document query
            mock_session_execute(len(sample_chunks)),  # Chunk count
            mock_session_execute(2),  # Entity count
        ]

        response = await document_api.get_document(test_document_id, session)

        assert response.success is True
        assert response.error is None
        assert response.data.id == str(test_document_id)
        assert response.data.name == "Test Document.pdf"
        assert response.data.status == "completed"
        assert response.data.chunk_count == len(sample_chunks)
        assert response.data.entity_count == 2
        assert response.metadata["document_id"] == str(test_document_id)

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, test_document_id):
        """Test document not found returns 404."""
        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        session.execute.return_value = mock_result

        with pytest.raises(Exception) as exc_info:
            await document_api.get_document(test_document_id, session)

        assert "404" in str(exc_info.value)
        assert "not found" in str(exc_info.value)


class TestGetDocumentContent:
    """Tests for GET /api/v1/documents/{document_id}/content"""

    @pytest.mark.asyncio
    async def test_get_document_content_success(
        self,
        mock_session_execute,
        test_document_id,
        sample_document,
        sample_chunks,
    ):
        """Test successful document content retrieval."""
        # Create mock session
        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()

        # Setup execute to return different mock results
        doc_result = MagicMock()
        doc_result.scalar_one_or_none = MagicMock(return_value=sample_document)

        chunks_result = MagicMock()
        chunks_scalars = MagicMock()
        chunks_scalars.all = MagicMock(return_value=sample_chunks)
        chunks_result.scalars = MagicMock(return_value=chunks_scalars)

        session.execute.side_effect = [doc_result, chunks_result]

        response = await document_api.get_document_content(
            test_document_id, "text", session
        )

        assert response.success is True
        assert response.error is None
        assert response.data.document_id == str(test_document_id)
        assert "first chunk" in response.data.content
        assert "second chunk" in response.data.content
        assert response.data.mime_type == "application/pdf"
        assert response.data.chunk_count == len(sample_chunks)
        assert response.data.total_length == len(sample_chunks[0].text) + len(
            sample_chunks[1].text
        )

    @pytest.mark.asyncio
    async def test_get_document_content_not_found(self, test_document_id):
        """Test document not found for content retrieval."""
        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        session.execute.return_value = mock_result

        with pytest.raises(Exception) as exc_info:
            await document_api.get_document_content(test_document_id, "text", session)

        assert "404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_document_content_no_chunks(
        self, mock_session_execute, test_document_id, sample_document
    ):
        """Test document with no chunks returns empty content."""
        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()

        # Setup execute to return document but no chunks
        doc_result = MagicMock()
        doc_result.scalar_one_or_none = MagicMock(return_value=sample_document)

        chunks_result = MagicMock()
        chunks_scalars = MagicMock()
        chunks_scalars.all = MagicMock(return_value=[])
        chunks_result.scalars = MagicMock(return_value=chunks_scalars)

        session.execute.side_effect = [doc_result, chunks_result]

        with pytest.raises(Exception) as exc_info:
            await document_api.get_document_content(test_document_id, "text", session)

        assert "404" in str(exc_info.value)
        assert "No content found" in str(exc_info.value)


class TestGetDocumentSpans:
    """Tests for GET /api/v1/documents/{document_id}/spans"""

    @pytest.mark.asyncio
    async def test_get_document_spans_success(
        self,
        test_document_id,
        sample_document,
        sample_chunks,
        sample_entities,
        sample_chunk_entities,
    ):
        """Test successful retrieval of document text spans."""
        from unittest.mock import AsyncMock
        from knowledge_base.common.offset_service import TextSpan

        # Create mock session
        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()

        # Setup execute to return different mock results
        doc_result = MagicMock()
        doc_result.scalar_one_or_none = MagicMock(return_value=sample_document)

        chunks_result = MagicMock()
        chunks_scalars = MagicMock()
        chunks_scalars.all = MagicMock(return_value=sample_chunks)
        chunks_result.scalars = MagicMock(return_value=chunks_scalars)

        chunk_entities_result = MagicMock()
        chunk_entities_result.all = MagicMock(return_value=[
            (sample_chunk_entities[0], sample_entities[0]),
            (sample_chunk_entities[1], sample_entities[1]),
        ])

        session.execute.side_effect = [doc_result, chunks_result, chunk_entities_result]

        with patch(
            "knowledge_base.document_api.OffsetCalculationService.calculate_absolute_offsets"
        ) as mock_calculate:
            mock_calculate.return_value = [
                TextSpan(
                    start_offset=10,
                    end_offset=22,
                    text="Test Company",
                    entity_id=sample_entities[0].id,
                    entity_name=sample_entities[0].name,
                    entity_type=sample_entities[0].entity_type,
                    confidence=0.95,
                    grounding_quote="Test Company",
                ),
                TextSpan(
                    start_offset=35,
                    end_offset=45,
                    text="John Smith",
                    entity_id=sample_entities[1].id,
                    entity_name=sample_entities[1].name,
                    entity_type=sample_entities[1].entity_type,
                    confidence=0.88,
                    grounding_quote="John Smith",
                ),
            ]

            response = await document_api.get_document_spans(
                test_document_id, 0.0, None, False, session
            )

            assert response.success is True
            assert response.error is None
            assert response.data.document_id == str(test_document_id)
            assert len(response.data.spans) == 2
            assert response.data.total_spans == 2
            assert response.data.entities_found == 2
            assert response.data.coverage_percentage > 0.0

    @pytest.mark.asyncio
    async def test_get_document_spans_empty(
        self, mock_session_execute, test_document_id, sample_document
    ):
        """Test document with no spans returns empty response."""
        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()

        # Setup execute to return document but no chunks
        doc_result = MagicMock()
        doc_result.scalar_one_or_none = MagicMock(return_value=sample_document)

        chunks_result = MagicMock()
        chunks_scalars = MagicMock()
        chunks_scalars.all = MagicMock(return_value=[])
        chunks_result.scalars = MagicMock(return_value=chunks_scalars)

        session.execute.side_effect = [doc_result, chunks_result]

        response = await document_api.get_document_spans(
            test_document_id, 0.0, None, False, session
        )

        assert response.success is True
        assert response.error is None
        assert response.data.spans == []
        assert response.data.total_spans == 0
        assert response.data.coverage_percentage == 0.0


class TestSearchDocuments:
    """Tests for POST /api/v1/documents:search"""

    @pytest.mark.asyncio
    async def test_search_documents_success(self):
        """Test successful document search."""
        from knowledge_base.document_api import DocumentSearchRequest

        search_request = DocumentSearchRequest(
            query="test company",
            search_type="hybrid",
            domains=["finance"],
            min_confidence=0.5,
            limit=10,
            offset=0,
        )

        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()

        # Execute search
        response = await document_api.search_documents(search_request, session)

        assert response.success is True
        assert response.error is None
        assert response.data.query == "test company"
        assert response.data.search_type == "hybrid"
        assert response.data.total == 0  # Placeholder implementation
        assert response.data.results == []  # Placeholder implementation
        assert response.data.took_ms >= 0  # Should have taken some time

    @pytest.mark.asyncio
    async def test_search_documents_vector_only(self):
        """Test vector-only search."""
        from knowledge_base.document_api import DocumentSearchRequest

        search_request = DocumentSearchRequest(
            query="machine learning",
            search_type="vector",
            limit=5,
        )

        session = AsyncMock(spec=AsyncSession)

        response = await document_api.search_documents(search_request, session)

        assert response.success is True
        assert response.data.search_type == "vector"
        assert response.data.query == "machine learning"

    @pytest.mark.asyncio
    async def test_search_documents_keyword_only(self):
        """Test keyword-only search."""
        from knowledge_base.document_api import DocumentSearchRequest

        search_request = DocumentSearchRequest(
            query="artificial intelligence",
            search_type="keyword",
            limit=5,
        )

        session = AsyncMock(spec=AsyncSession)

        response = await document_api.search_documents(search_request, session)

        assert response.success is True
        assert response.data.search_type == "keyword"
        assert response.data.query == "artificial intelligence"


class TestCreateAnnotation:
    """Tests for POST /api/v1/documents/{document_id}/annotations"""

    @pytest.mark.asyncio
    async def test_create_annotation_success(
        self,
        test_document_id,
        sample_document,
    ):
        """Test successful W3C annotation creation."""
        from knowledge_base.document_api import W3CAnnotation

        # Create mock session
        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()

        # Mock document query
        doc_result = MagicMock()
        doc_result.scalar_one_or_none = MagicMock(return_value=sample_document)
        session.execute.return_value = doc_result

        annotation = W3CAnnotation(
            id="annotation-123",
            type="Annotation",
            target={
                "source": str(test_document_id),
                "selector": {
                    "type": "TextPositionSelector",
                    "start": 10,
                    "end": 25,
                },
            },
            body={
                "type": "TextualBody",
                "value": "Important quote",
            },
            creator="test-user@example.com",
            motivation="highlighting",
        )

        response = await document_api.create_annotation(
            test_document_id, annotation, session
        )

        assert response.success is True
        assert response.error is None
        assert response.data.id == "annotation-123"
        assert response.data.type == "Annotation"
        assert response.data.creator == "test-user@example.com"
        assert "created" in response.metadata

    @pytest.mark.asyncio
    async def test_create_annotation_invalid_structure(
        self, test_document_id, sample_document
    ):
        """Test annotation creation with invalid structure."""
        from knowledge_base.document_api import W3CAnnotation

        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()

        doc_result = MagicMock()
        doc_result.scalar_one_or_none = MagicMock(return_value=sample_document)
        session.execute.return_value = doc_result

        # Invalid annotation without selector in target
        annotation = W3CAnnotation(
            id="annotation-123",
            type="Annotation",
            target={"source": str(test_document_id)},  # Missing selector
            body={"type": "TextualBody", "value": "Important quote"},
        )

        with pytest.raises(Exception) as exc_info:
            await document_api.create_annotation(
                test_document_id, annotation, session
            )

        assert "400" in str(exc_info.value)
        assert "selector" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_annotation_document_not_found(self, test_document_id):
        """Test annotation creation when document doesn't exist."""
        from knowledge_base.document_api import W3CAnnotation

        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()

        doc_result = MagicMock()
        doc_result.scalar_one_or_none = MagicMock(return_value=None)
        session.execute.return_value = doc_result

        annotation = W3CAnnotation(
            id="annotation-123",
            type="Annotation",
            target={
                "source": str(test_document_id),
                "selector": {"type": "TextPositionSelector", "start": 10, "end": 25},
            },
            body={"type": "TextualBody", "value": "Important quote"},
        )

        with pytest.raises(Exception) as exc_info:
            await document_api.create_annotation(
                test_document_id, annotation, session
            )

        assert "404" in str(exc_info.value)


class TestDocumentAPIIntegration:
    """Integration tests using FastAPI TestClient."""

    @pytest.fixture
    def test_app(self):
        """Create test FastAPI app."""
        from fastapi import FastAPI
        from knowledge_base import document_api

        app = FastAPI()
        app.include_router(document_api.router, prefix="/api/v1/documents")
        return app

    @pytest.fixture
    def client(self, test_app):
        """Create test client."""
        return TestClient(test_app)

    def test_api_routes_exist(self):
        """Verify all expected routes are registered."""
        from knowledge_base import document_api

        routes = document_api.router.routes
        route_paths = [
            route.path for route in routes
        ]

        assert "/api/v1/documents/{document_id}" in route_paths
        assert "/api/v1/documents/{document_id}/content" in route_paths
        assert "/api/v1/documents/{document_id}/spans" in route_paths
        assert "/api/v1/documents/{document_id}/entities" in route_paths
        assert "/api/v1/documents/:search" in route_paths
        assert "/api/v1/documents/{document_id}/annotations" in route_paths

        # Verify HTTP methods
        search_routes = [r for r in routes if r.path == "/api/v1/documents/:search"]
        assert any("POST" in r.methods for r in search_routes)

        # Verify at least 6 routes exist
        assert len(routes) >= 6
