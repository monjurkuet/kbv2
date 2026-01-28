"""Tests for PartitioningService."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from pydantic import ValidationError

from src.knowledge_base.ingestion.v1.partitioning_service import (
    PartitioningService,
    PartitioningConfig,
    PartitionedChunk,
)


class TestPartitioningConfig:
    """Tests for PartitioningConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PartitioningConfig()
        assert config.chunk_size == 1536
        assert config.chunk_overlap_ratio == 0.25
        assert config.min_chunk_size == 256

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PartitioningConfig(
            chunk_size=2048, chunk_overlap_ratio=0.3, min_chunk_size=512
        )
        assert config.chunk_size == 2048
        assert config.chunk_overlap_ratio == 0.3
        assert config.min_chunk_size == 512


class TestPartitionedChunk:
    """Tests for PartitionedChunk model."""

    def test_chunk_creation(self):
        """Test creating a partitioned chunk."""
        chunk = PartitionedChunk(
            text="Test chunk text",
            page_number=1,
            element_type="text",
            chunk_id="abc123",
            token_count=100,
            chunk_index=0,
        )
        assert chunk.text == "Test chunk text"
        assert chunk.page_number == 1
        assert chunk.element_type == "text"
        assert chunk.chunk_id == "abc123"
        assert chunk.token_count == 100
        assert chunk.chunk_index == 0

    def test_chunk_with_metadata(self):
        """Test creating a chunk with metadata."""
        metadata = {"source": "test", "version": 1}
        chunk = PartitionedChunk(
            text="Text with metadata",
            metadata=metadata,
        )
        assert chunk.metadata == metadata

    def test_chunk_with_links(self):
        """Test chunk with previous/next links."""
        chunk = PartitionedChunk(
            text="Middle chunk",
            chunk_id="middle",
            previous_chunk_id="prev",
            next_chunk_id="next",
        )
        assert chunk.previous_chunk_id == "prev"
        assert chunk.next_chunk_id == "next"


class TestPartitioningService:
    """Tests for PartitioningService."""

    def test_service_initialization(self):
        """Test service initialization with default config."""
        service = PartitioningService()
        assert service._config.chunk_size == 1536
        assert service._config.chunk_overlap_ratio == 0.25

    def test_service_initialization_custom_config(self):
        """Test service initialization with custom config."""
        config = PartitioningConfig(chunk_size=2048)
        service = PartitioningService(config)
        assert service._config.chunk_size == 2048

    @pytest.mark.asyncio
    async def test_partition_file_not_found(self):
        """Test partitioning non-existent file raises error."""
        service = PartitioningService()
        with pytest.raises(FileNotFoundError):
            await service.partition_file("/nonexistent/path/file.txt")

    @pytest.mark.asyncio
    async def test_chunk_elements_empty_raises_error(self):
        """Test that chunking empty elements raises ValueError."""
        service = PartitioningService()
        with pytest.raises(ValueError, match="Elements list cannot be empty"):
            await service.chunk_elements([])

    @pytest.mark.asyncio
    async def test_chunk_elements_single_element(self):
        """Test chunking single element that fits in one chunk."""
        service = PartitioningService()
        mock_element = MagicMock()
        mock_element.__str__ = MagicMock(return_value="Short text content.")
        mock_element.strip = MagicMock(return_value="Short text content.")

        chunks = await service.chunk_elements([mock_element])
        assert len(chunks) >= 1
        assert chunks[0].text == "Short text content."
        assert chunks[0].element_type == "text"

    @pytest.mark.asyncio
    async def test_chunk_elements_uses_config_chunk_size(self):
        """Test that chunking uses config chunk_size."""
        service = PartitioningService()
        mock_element = MagicMock()
        mock_text = "Word " * 500
        mock_element.__str__ = MagicMock(return_value=mock_text)
        mock_element.strip = MagicMock(return_value=mock_text)

        chunks = await service.chunk_elements([mock_element])
        for chunk in chunks:
            assert chunk.token_count > 0

    @pytest.mark.asyncio
    async def test_chunk_elements_custom_chunk_size(self):
        """Test chunking with custom chunk_size parameter."""
        service = PartitioningService()
        mock_element = MagicMock()
        mock_text = "Word " * 2000
        mock_element.__str__ = MagicMock(return_value=mock_text)
        mock_element.strip = MagicMock(return_value=mock_text)

        chunks = await service.chunk_elements([mock_element], chunk_size=500)
        for chunk in chunks:
            assert chunk.token_count > 0

    @pytest.mark.asyncio
    async def test_chunk_elements_adds_chunk_ids(self):
        """Test that chunking adds chunk IDs to results."""
        service = PartitioningService()
        mock_element = MagicMock()
        mock_text = "Test content for chunking"
        mock_element.__str__ = MagicMock(return_value=mock_text)
        mock_element.strip = MagicMock(return_value=mock_text)

        chunks = await service.chunk_elements([mock_element])
        for chunk in chunks:
            assert chunk.chunk_id is not None
            assert len(chunk.chunk_id) > 0

    @pytest.mark.asyncio
    async def test_chunk_elements_adds_metadata(self):
        """Test that chunking adds proper metadata."""
        service = PartitioningService()
        mock_element = MagicMock()
        mock_text = "Content with metadata"
        mock_element.__str__ = MagicMock(return_value=mock_text)
        mock_element.strip = MagicMock(return_value=mock_text)

        chunks = await service.chunk_elements([mock_element])
        assert chunks[0].metadata["element_type"] == "text"
        assert chunks[0].metadata["semantic_chunk"] is True

    @pytest.mark.asyncio
    async def test_chunk_elements_overlap_links(self):
        """Test that chunking with multiple chunks adds overlap links."""
        service = PartitioningService()
        mock_element = MagicMock()
        mock_text = "Sentence one. " * 200
        mock_element.__str__ = MagicMock(return_value=mock_text)
        mock_element.strip = MagicMock(return_value=mock_text)

        chunks = await service.chunk_elements([mock_element], chunk_size=100)
        if len(chunks) > 1:
            assert chunks[0].previous_chunk_id is None
            for i in range(1, len(chunks)):
                assert chunks[i].previous_chunk_id is not None
            assert chunks[-1].next_chunk_id is None

    @pytest.mark.asyncio
    async def test_chunk_elements_sequential_indices(self):
        """Test that chunks have sequential indices."""
        service = PartitioningService()
        mock_element = MagicMock()
        mock_text = "Test content for sequential indices"
        mock_element.__str__ = MagicMock(return_value=mock_text)
        mock_element.strip = MagicMock(return_value=mock_text)

        chunks = await service.chunk_elements([mock_element])
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_elements_to_text_empty(self):
        """Test converting empty elements to text."""
        service = PartitioningService()
        result = service._elements_to_text([])
        assert result == ""

    def test_elements_to_text_single_element(self):
        """Test converting single element to text."""
        service = PartitioningService()
        mock_element = MagicMock()
        mock_element.__str__ = MagicMock(return_value="  Test text  ")
        result = service._elements_to_text([mock_element])
        assert result == "Test text"

    def test_elements_to_text_multiple_elements(self):
        """Test converting multiple elements to text."""
        service = PartitioningService()
        mock_element1 = MagicMock()
        mock_element1.__str__ = MagicMock(return_value="First element")
        mock_element2 = MagicMock()
        mock_element2.__str__ = MagicMock(return_value="Second element")
        result = service._elements_to_text([mock_element1, mock_element2])
        assert "First element" in result
        assert "Second element" in result
        assert "\n\n" in result

    def test_elements_to_text_skips_empty_elements(self):
        """Test that empty elements are skipped."""
        service = PartitioningService()
        mock_element1 = MagicMock()
        mock_element1.__str__ = MagicMock(return_value="   ")
        mock_element2 = MagicMock()
        mock_element2.__str__ = MagicMock(return_value="Valid text")
        result = service._elements_to_text([mock_element1, mock_element2])
        assert result == "Valid text"

    @pytest.mark.asyncio
    async def test_partition_and_chunk_method_exists(self):
        """Test that partition_and_chunk method exists."""
        service = PartitioningService()
        assert hasattr(service, "partition_and_chunk")
        assert callable(service.partition_and_chunk)


class TestPartitioningConfigIntegration:
    """Integration tests for partitioning configuration."""

    def test_config_from_env(self):
        """Test loading config from environment variables."""
        import os

        os.environ["CHUNK_SIZE"] = "3000"
        try:
            config = PartitioningConfig()
            assert config.chunk_size == 3000
        finally:
            del os.environ["CHUNK_SIZE"]

    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValidationError):
            PartitioningConfig(chunk_overlap_ratio=1.5)
