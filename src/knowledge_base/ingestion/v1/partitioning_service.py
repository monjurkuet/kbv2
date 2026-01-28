"""Document partitioning service with enhanced semantic chunking."""

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition

from knowledge_base.partitioning.semantic_chunker import (
    SemanticChunker,
)


class PartitioningConfig(BaseSettings):
    """Partitioning configuration."""

    chunk_size: int = Field(default=1536, description="Default chunk size in tokens")
    chunk_overlap_ratio: float = Field(
        default=0.25, description="Overlap ratio between chunks (0.0 to 1.0)"
    )
    min_chunk_size: int = Field(default=256, description="Minimum chunk size in tokens")


class PartitionedChunk(BaseModel):
    """Partitioned chunk result with metadata."""

    text: str = Field(..., description="Chunk text")
    page_number: int | None = Field(None, description="Page number")
    element_type: str | None = Field(None, description="Element type")
    chunk_id: str = Field(default_factory=str, description="Unique chunk identifier")
    previous_chunk_id: str | None = Field(None, description="ID of previous chunk")
    next_chunk_id: str | None = Field(None, description="ID of next chunk")
    token_count: int = Field(default=0, description="Token count of the chunk")
    chunk_index: int = Field(default=0, description="Index of chunk in sequence")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class PartitioningService:
    """Service for partitioning documents into chunks with semantic awareness."""

    def __init__(self, config: PartitioningConfig | None = None) -> None:
        """Initialize partitioning service.

        Args:
            config: Partitioning configuration.
        """
        self._config = config or PartitioningConfig()
        self._semantic_chunker = SemanticChunker(
            chunk_size=self._config.chunk_size,
            overlap_ratio=self._config.chunk_overlap_ratio,
            min_chunk_size=self._config.min_chunk_size,
        )

    async def partition_file(
        self,
        file_path: str | Path,
    ) -> list[Element]:
        """Partition file into elements.

        Args:
            file_path: Path to file.

        Returns:
            List of document elements.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        elements = await asyncio.to_thread(partition, filename=str(path))

        return elements

    async def chunk_elements(
        self,
        elements: list[Element],
        chunk_size: int | None = None,
        chunk_overlap_ratio: float | None = None,
    ) -> list[PartitionedChunk]:
        """Chunk elements into semantic chunks.

        Args:
            elements: List of document elements.
            chunk_size: Maximum chunk size in tokens.
            chunk_overlap_ratio: Overlap ratio between chunks (0.0 to 1.0).

        Returns:
            List of partitioned chunks with metadata.

        Raises:
            ValueError: If elements list is empty.
        """
        if not elements:
            raise ValueError("Elements list cannot be empty")

        chunk_size = chunk_size or self._config.chunk_size
        overlap_ratio = chunk_overlap_ratio or self._config.chunk_overlap_ratio

        semantic_chunker = SemanticChunker(
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio,
            min_chunk_size=self._config.min_chunk_size,
        )

        combined_text = self._elements_to_text(elements)
        chunks_with_links = semantic_chunker.chunk_with_overlap(combined_text)

        result: list[PartitionedChunk] = []
        for i, chunk_link in enumerate(chunks_with_links):
            chunk = chunk_link.chunk
            result.append(
                PartitionedChunk(
                    text=chunk.text,
                    page_number=None,
                    element_type="text",
                    chunk_id=chunk.id,
                    previous_chunk_id=chunk_link.previous_chunk_id,
                    next_chunk_id=chunk_link.next_chunk_id,
                    token_count=chunk.token_count,
                    chunk_index=chunk.chunk_index,
                    metadata={
                        "element_id": str(i),
                        "element_type": "text",
                        "semantic_chunk": True,
                    },
                )
            )

        return result

    def _elements_to_text(self, elements: list[Element]) -> str:
        """Convert elements to combined text.

        Args:
            elements: List of document elements.

        Returns:
            Combined text from all elements.
        """
        texts = []
        for element in elements:
            element_text = str(element).strip()
            if element_text:
                texts.append(element_text)
        return "\n\n".join(texts)

    async def partition_and_chunk(
        self,
        file_path: str | Path,
        chunk_size: int | None = None,
        chunk_overlap_ratio: float | None = None,
    ) -> list[PartitionedChunk]:
        """Partition file and chunk in one operation.

        Args:
            file_path: Path to file.
            chunk_size: Maximum chunk size in tokens.
            chunk_overlap_ratio: Overlap ratio between chunks (0.0 to 1.0).

        Returns:
            List of partitioned chunks.
        """
        elements = await self.partition_file(file_path)
        return await self.chunk_elements(elements, chunk_size, chunk_overlap_ratio)
