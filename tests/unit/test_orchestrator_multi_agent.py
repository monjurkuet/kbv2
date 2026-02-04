"""Tests for document pipeline service."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from knowledge_base.orchestration.document_pipeline_service import (
    DocumentPipelineService,
)
from knowledge_base.persistence.v1.schema import Document, Chunk


@pytest.fixture
def document_service():
    """Create a test document pipeline service instance."""
    service = DocumentPipelineService()
    return service


class TestDocumentPipelineService:
    """Tests for DocumentPipelineService."""

    def test_initialization(self, document_service):
        """Test service initialization."""
        assert document_service._chunker is None
        assert document_service._embedding_client is None

    def test_get_mime_type_pdf(self, document_service):
        """Test MIME type detection for PDF."""
        from pathlib import Path

        assert document_service._get_mime_type(Path("test.pdf")) == "application/pdf"

    def test_get_mime_type_txt(self, document_service):
        """Test MIME type detection for TXT."""
        from pathlib import Path

        assert document_service._get_mime_type(Path("test.txt")) == "text/plain"

    def test_get_mime_type_markdown(self, document_service):
        """Test MIME type detection for Markdown."""
        from pathlib import Path

        mime = document_service._get_mime_type(Path("test.md"))
        assert mime in ["text/markdown", "text/x-markdown"]
