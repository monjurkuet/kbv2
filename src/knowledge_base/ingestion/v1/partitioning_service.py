"""Document partitioning service."""

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition


class PartitioningConfig(BaseSettings):
    """Partitioning configuration."""

    chunk_size: int = 512
    chunk_overlap: int = 50


class PartitionedChunk(BaseModel):
    """Partitioned chunk result."""

    text: str = Field(..., description="Chunk text")
    page_number: int | None = Field(None, description="Page number")
    element_type: str | None = Field(None, description="Element type")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class PartitioningService:
    """Service for partitioning documents into chunks."""

    def __init__(self, config: PartitioningConfig | None = None) -> None:
        """Initialize partitioning service.

        Args:
            config: Partitioning configuration.
        """
        self._config = config or PartitioningConfig()

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
        chunk_overlap: int | None = None,
    ) -> list[PartitionedChunk]:
        """Chunk elements into semantic chunks.

        Args:
            elements: List of document elements.
            chunk_size: Maximum chunk size in tokens.
            chunk_overlap: Overlap between chunks.

        Returns:
            List of partitioned chunks.
        """
        chunk_size = chunk_size or self._config.chunk_size
        chunk_overlap = chunk_overlap or self._config.chunk_overlap

        chunks = await asyncio.to_thread(
            chunk_by_title,
            elements,
            max_characters=chunk_size * 4,
            new_after_n_chars=chunk_size * 4,
            overlap=chunk_overlap * 4,
        )

        result: list[PartitionedChunk] = []

        for chunk in chunks:
            page_number = getattr(chunk, "page_number", None)
            element_type = chunk.category if hasattr(chunk, "category") else None

            metadata: dict[str, Any] = {
                "element_id": chunk.id if hasattr(chunk, "id") else None,
                "element_type": element_type,
            }

            result.append(
                PartitionedChunk(
                    text=str(chunk),
                    page_number=page_number,
                    element_type=element_type,
                    metadata=metadata,
                )
            )

        return result

    async def partition_and_chunk(
        self,
        file_path: str | Path,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[PartitionedChunk]:
        """Partition file and chunk in one operation.

        Args:
            file_path: Path to file.
            chunk_size: Maximum chunk size in tokens.
            chunk_overlap: Overlap between chunks.

        Returns:
            List of partitioned chunks.
        """
        elements = await self.partition_file(file_path)
        return await self.chunk_elements(elements, chunk_size, chunk_overlap)
