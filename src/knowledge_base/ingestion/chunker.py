"""Semantic chunker for document splitting.

This module provides intelligent document chunking that:
1. Respects document structure (headers, paragraphs)
2. Maintains semantic coherence
3. Handles overlap for context preservation
4. Tracks chunk metadata and relationships
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""

    document_id: str
    chunk_index: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    token_count: int = 0
    char_count: int = 0
    start_char: int = 0
    end_char: int = 0
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A chunk of document content."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    chunk_index: int
    metadata: ChunkMetadata
    created_at: datetime = Field(default_factory=datetime.utcnow)


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""

    # Size settings
    chunk_size: int = 1000  # Target characters per chunk
    chunk_overlap: int = 150  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size

    # Token estimation
    chars_per_token: int = 4  # Approximate chars per token

    # Structure settings
    respect_headers: bool = True  # Split on headers
    respect_code_blocks: bool = True  # Don't split code blocks
    respect_tables: bool = True  # Don't split tables
    respect_lists: bool = True  # Try to keep lists together

    # Maximum limits
    max_chunk_size: int = 2000  # Maximum chunk size


class SemanticChunker:
    """Semantic document chunker.

    This chunker respects document structure and creates semantically
    coherent chunks. It handles:
    - Markdown headers
    - Code blocks
    - Tables
    - Lists
    - Paragraphs

    Example:
        >>> chunker = SemanticChunker()
        >>> chunks = chunker.chunk(document_content, document_id="doc_123")
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.chunk_index}: {len(chunk.text)} chars")
    """

    # Regex patterns for markdown structure
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    TABLE_PATTERN = re.compile(r'\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)*', re.MULTILINE)
    LIST_PATTERN = re.compile(r'^(\s*[-*+•]|\s*\d+\.)\s+.+$', re.MULTILINE)
    PARAGRAPH_PATTERN = re.compile(r'\n\s*\n')

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        """Initialize semantic chunker.

        Args:
            config: Chunking configuration.
        """
        self._config = config or ChunkingConfig()

    def chunk(
        self,
        content: str,
        document_id: str,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> list[Chunk]:
        """Chunk document content.

        Args:
            content: Document content to chunk.
            document_id: Document ID.
            extra_metadata: Additional metadata to include.

        Returns:
            List of chunks.
        """
        extra_metadata = extra_metadata or {}

        # Split into structural elements
        elements = self._split_into_elements(content)

        # Group elements into chunks
        chunks = self._group_into_chunks(elements, document_id, extra_metadata)

        # Link chunks together
        self._link_chunks(chunks)

        return chunks

    def _split_into_elements(self, content: str) -> list[dict[str, Any]]:
        """Split content into structural elements.

        Args:
            content: Document content.

        Returns:
            List of structural elements with metadata.
        """
        elements = []

        # Find all structural markers
        markers = []

        # Find headers
        for match in self.HEADER_PATTERN.finditer(content):
            markers.append({
                "type": "header",
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
                "start": match.start(),
                "end": match.end(),
            })

        # Find code blocks
        for match in self.CODE_BLOCK_PATTERN.finditer(content):
            markers.append({
                "type": "code_block",
                "start": match.start(),
                "end": match.end(),
            })

        # Find tables
        for match in self.TABLE_PATTERN.finditer(content):
            markers.append({
                "type": "table",
                "start": match.start(),
                "end": match.end(),
            })

        # Sort markers by position
        markers.sort(key=lambda x: x["start"])

        # If no markers found, split by paragraphs
        if not markers:
            return self._split_by_paragraphs(content)

        # Create elements from markers and inter-marker content
        current_pos = 0
        current_section = None

        for marker in markers:
            # Add content before this marker
            if marker["start"] > current_pos:
                pre_content = content[current_pos:marker["start"]].strip()
                if pre_content:
                    elements.append({
                        "type": "paragraph",
                        "content": pre_content,
                        "section": current_section,
                    })

            # Add the marker element
            element_content = content[marker["start"]:marker["end"]]

            if marker["type"] == "header":
                current_section = marker["title"]
                # Headers are included in the next element
            else:
                elements.append({
                    "type": marker["type"],
                    "content": element_content,
                    "section": current_section,
                })

            current_pos = marker["end"]

        # Add remaining content
        if current_pos < len(content):
            remaining = content[current_pos:].strip()
            if remaining:
                elements.append({
                    "type": "paragraph",
                    "content": remaining,
                    "section": current_section,
                })

        return elements

    def _split_by_paragraphs(self, content: str) -> list[dict[str, Any]]:
        """Split content by paragraphs.

        Args:
            content: Document content.

        Returns:
            List of paragraph elements.
        """
        elements = []
        paragraphs = self.PARAGRAPH_PATTERN.split(content)

        for para in paragraphs:
            para = para.strip()
            if para:
                # Check if it's a header
                header_match = self.HEADER_PATTERN.match(para)
                if header_match:
                    elements.append({
                        "type": "header",
                        "content": para,
                        "level": len(header_match.group(1)),
                        "title": header_match.group(2).strip(),
                        "section": header_match.group(2).strip(),
                    })
                else:
                    elements.append({
                        "type": "paragraph",
                        "content": para,
                        "section": None,
                    })

        return elements

    def _group_into_chunks(
        self,
        elements: list[dict[str, Any]],
        document_id: str,
        extra_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Group elements into chunks.

        Args:
            elements: Structural elements.
            document_id: Document ID.
            extra_metadata: Additional metadata.

        Returns:
            List of chunks.
        """
        chunks = []
        current_chunk_elements: list[dict[str, Any]] = []
        current_chunk_size = 0
        current_section = None
        chunk_index = 0

        for element in elements:
            element_size = len(element["content"])
            element_section = element.get("section")

            # Update current section
            if element_section:
                current_section = element_section

            # Decide whether to start a new chunk
            should_start_new = self._should_start_new_chunk(
                element, current_chunk_elements, current_chunk_size
            )

            if should_start_new and current_chunk_elements:
                # Create chunk from current elements
                chunk = self._create_chunk(
                    current_chunk_elements,
                    document_id,
                    chunk_index,
                    current_section,
                    extra_metadata,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk with overlap
                overlap_elements = self._get_overlap_elements(current_chunk_elements)
                current_chunk_elements = overlap_elements
                current_chunk_size = sum(len(e["content"]) for e in overlap_elements)

            current_chunk_elements.append(element)
            current_chunk_size += element_size

        # Create final chunk
        if current_chunk_elements:
            chunk = self._create_chunk(
                current_chunk_elements,
                document_id,
                chunk_index,
                current_section,
                extra_metadata,
            )
            chunks.append(chunk)

        return chunks

    def _should_start_new_chunk(
        self,
        element: dict[str, Any],
        current_elements: list[dict[str, Any]],
        current_size: int,
    ) -> bool:
        """Decide whether to start a new chunk.

        Args:
            element: Current element being considered.
            current_elements: Elements in current chunk.
            current_size: Current chunk size in characters.

        Returns:
            True if should start new chunk.
        """
        # Always respect code blocks and tables
        if self._config.respect_code_blocks and element["type"] == "code_block":
            return bool(current_elements)

        if self._config.respect_tables and element["type"] == "table":
            return bool(current_elements)

        # Check if current chunk is big enough
        if current_size >= self._config.chunk_size:
            # Start new chunk on header
            if self._config.respect_headers and element["type"] == "header":
                return True

            # Start new chunk if we're at max size
            if current_size >= self._config.max_chunk_size:
                return True

            # Start new chunk if adding would exceed max
            if current_size + len(element["content"]) > self._config.max_chunk_size:
                return True

        return False

    def _get_overlap_elements(
        self,
        elements: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Get elements for overlap from end of chunk.

        Args:
            elements: Current chunk elements.

        Returns:
            Elements to include in overlap.
        """
        if not elements or self._config.chunk_overlap == 0:
            return []

        overlap_elements = []
        overlap_size = 0

        # Take elements from the end until we have enough overlap
        for element in reversed(elements):
            element_size = len(element["content"])

            if overlap_size + element_size <= self._config.chunk_overlap:
                overlap_elements.insert(0, element)
                overlap_size += element_size
            else:
                break

        return overlap_elements

    def _create_chunk(
        self,
        elements: list[dict[str, Any]],
        document_id: str,
        chunk_index: int,
        section: Optional[str],
        extra_metadata: dict[str, Any],
    ) -> Chunk:
        """Create a chunk from elements.

        Args:
            elements: Elements to include.
            document_id: Document ID.
            chunk_index: Chunk index.
            section: Current section title.
            extra_metadata: Additional metadata.

        Returns:
            Created chunk.
        """
        # Combine element content
        content_parts = []
        for element in elements:
            content_parts.append(element["content"])

        text = "\n\n".join(content_parts)

        # Calculate metadata
        char_count = len(text)
        token_count = char_count // self._config.chars_per_token

        metadata = ChunkMetadata(
            document_id=document_id,
            chunk_index=chunk_index,
            section_title=section,
            token_count=token_count,
            char_count=char_count,
            extra=extra_metadata,
        )

        return Chunk(
            document_id=document_id,
            text=text,
            chunk_index=chunk_index,
            metadata=metadata,
        )

    def _link_chunks(self, chunks: list[Chunk]) -> None:
        """Link chunks together with previous/next IDs.

        Args:
            chunks: List of chunks to link.
        """
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.metadata.previous_chunk_id = chunks[i - 1].id
            if i < len(chunks) - 1:
                chunk.metadata.next_chunk_id = chunks[i + 1].id

    # ==================== Convenience Methods ====================

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count.
        """
        return len(text) // self._config.chars_per_token

    def get_chunk_stats(self, chunks: list[Chunk]) -> dict[str, Any]:
        """Get statistics for a list of chunks.

        Args:
            chunks: List of chunks.

        Returns:
            Statistics dictionary.
        """
        if not chunks:
            return {
                "count": 0,
                "total_chars": 0,
                "total_tokens": 0,
                "avg_chunk_size": 0,
            }

        total_chars = sum(len(c.text) for c in chunks)
        total_tokens = sum(c.metadata.token_count for c in chunks)

        return {
            "count": len(chunks),
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "avg_chunk_size": total_chars // len(chunks),
            "min_chunk_size": min(len(c.text) for c in chunks),
            "max_chunk_size": max(len(c.text) for c in chunks),
        }


def chunk_document(
    content: str,
    document_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Convenience function to chunk a document.

    Args:
        content: Document content.
        document_id: Document ID.
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of chunks.
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunker = SemanticChunker(config)
    return chunker.chunk(content, document_id)
