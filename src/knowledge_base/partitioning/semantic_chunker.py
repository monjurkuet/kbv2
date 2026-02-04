"""Semantic chunking implementation for document processing."""

import hashlib
import re
from typing import Any
from uuid import uuid4

import nltk
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel, Field

from knowledge_base.config.constants import (
    SEMANTIC_CHUNK_SIZE,
    OVERLAP_RATIO,
)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class Chunk(BaseModel):
    """Represents a chunk of text."""

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique chunk identifier"
    )
    text: str = Field(..., description="Chunk text content")
    page_number: int | None = Field(None, description="Page number if applicable")
    token_count: int = Field(..., description="Token count of the chunk")
    chunk_index: int = Field(..., description="Index of chunk in sequence")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ChunkWithLinks(BaseModel):
    """Represents a chunk with previous/next chunk links."""

    chunk: Chunk = Field(..., description="The chunk data")
    previous_chunk_id: str | None = Field(None, description="ID of previous chunk")
    next_chunk_id: str | None = Field(None, description="ID of next chunk")


class SemanticChunkerConfig(BaseModel):
    """Configuration for semantic chunker."""

    chunk_size: int = Field(
        default=SEMANTIC_CHUNK_SIZE, description="Target chunk size in tokens"
    )
    overlap_ratio: float = Field(
        default=OVERLAP_RATIO, description="Overlap ratio between chunks (0.0 to 1.0)"
    )
    min_chunk_size: int = Field(default=256, description="Minimum chunk size in tokens")
    max_chunk_size: int = Field(
        default=2048, description="Maximum chunk size in tokens"
    )


class SemanticChunker:
    """Semantic-aware text chunker with overlap support."""

    def __init__(
        self,
        chunk_size: int = SEMANTIC_CHUNK_SIZE,
        overlap_ratio: float = OVERLAP_RATIO,
        min_chunk_size: int = 256,
        max_chunk_size: int = 2048,
    ) -> None:
        """Initialize the semantic chunker.

        Args:
            chunk_size: Target chunk size in tokens (default 1536).
            overlap_ratio: Overlap ratio between consecutive chunks (default 0.25 = 25%).
            min_chunk_size: Minimum chunk size in tokens (default 256).
            max_chunk_size: Maximum chunk size in tokens (default 2048).

        Raises:
            ValueError: If chunk_size <= 0, overlap_ratio not in [0, 1), or min_chunk_size <= 0.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in range [0.0, 1.0)")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if min_chunk_size > chunk_size:
            raise ValueError("min_chunk_size cannot be larger than chunk_size")
        if max_chunk_size < chunk_size:
            raise ValueError("max_chunk_size cannot be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._overlap_ratio = overlap_ratio
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        self._overlap_tokens = int(chunk_size * overlap_ratio)

    @property
    def chunk_size(self) -> int:
        """Get the chunk size."""
        return self._chunk_size

    @property
    def overlap_ratio(self) -> float:
        """Get the overlap ratio."""
        return self._overlap_ratio

    @property
    def min_chunk_size(self) -> int:
        """Get the minimum chunk size."""
        return self._min_chunk_size

    @property
    def max_chunk_size(self) -> int:
        """Get the maximum chunk size."""
        return self._max_chunk_size

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using character-based approximation.

        Args:
            text: Input text.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        return len(text) // 4 + 1

    def _find_semantic_boundaries(self, text: str) -> list[int]:
        """Find natural semantic break points in text.

        Finds boundaries at paragraph breaks, sentence endings, and heading markers.

        Args:
            text: Input text to analyze.

        Returns:
            List of character indices where semantic boundaries occur.
        """
        if not text:
            return []

        boundaries: list[int] = [0]

        paragraph_breaks = [m.end() for m in re.finditer(r"\n\s*\n", text)]
        boundaries.extend(paragraph_breaks)

        sentences = list(sent_tokenize(text))
        sentence_starts: list[int] = []
        current_pos = 0
        for sentence in sentences:
            pos = text.find(sentence, current_pos)
            if pos != -1 and pos > current_pos:
                sentence_starts.append(pos)
            current_pos = pos + len(sentence)
        boundaries.extend(sentence_starts)

        heading_patterns = [
            r"(?m)^#{1,6}\s+.+$",
            r"(?m)^[A-Z][A-Z\s]+$",
            r"(?m)^\d+\.\s+.+$",
            r"(?m)^[IVX]+\.\s+.+$",
        ]
        for pattern in heading_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                if match.start() > 0:
                    boundaries.append(match.start())

        unique_boundaries = sorted(set(boundaries))
        unique_boundaries = [b for b in unique_boundaries if b < len(text)]

        return unique_boundaries

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences while preserving delimiters.

        Args:
            text: Input text.

        Returns:
            List of sentences with their trailing punctuation.
        """
        if not text:
            return []

        sentences: list[str] = []
        sent_tokenize_list = sent_tokenize(text)

        current_pos = 0
        for sentence in sent_tokenize_list:
            pos = text.find(sentence, current_pos)
            if pos == -1:
                continue
            if pos > current_pos:
                sentences.append(text[current_pos:pos])
            sentences.append(sentence)
            current_pos = pos + len(sentence)

        if current_pos < len(text):
            sentences.append(text[current_pos:])

        return sentences

    def _merge_small_chunks(self, chunks: list[Chunk], min_size: int) -> list[Chunk]:
        """Merge chunks smaller than minimum size with adjacent chunks.

        Args:
            chunks: List of chunks to process.
            min_size: Minimum chunk size in tokens.

        Returns:
            List of chunks with small chunks merged.
        """
        if len(chunks) <= 1:
            return chunks

        merged: list[Chunk] = []
        i = 0

        while i < len(chunks):
            current = chunks[i]
            token_count = current.token_count

            if token_count >= min_size:
                merged.append(current)
                i += 1
                continue

            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                combined_text = current.text + " " + next_chunk.text
                combined_token_count = self._count_tokens(combined_text)

                if combined_token_count <= self._chunk_size * 1.5:
                    merged.append(
                        Chunk(
                            id=current.id,
                            text=combined_text,
                            page_number=current.page_number,
                            token_count=combined_token_count,
                            chunk_index=len(merged),
                            metadata={**current.metadata, "merged": True},
                        )
                    )
                    i += 2
                    continue

            if i > 0 and token_count < min_size // 2:
                prev_chunk = merged[-1]
                combined_text = prev_chunk.text + " " + current.text
                combined_token_count = self._count_tokens(combined_text)

                if combined_token_count <= self._chunk_size * 1.5:
                    merged[-1] = Chunk(
                        id=prev_chunk.id,
                        text=combined_text,
                        page_number=prev_chunk.page_number,
                        token_count=combined_token_count,
                        chunk_index=len(merged) - 1,
                        metadata={**prev_chunk.metadata, "merged": True},
                    )
                    i += 1
                    continue

            if token_count > 0:
                merged.append(current)
            i += 1

        return merged

    def chunk(self, document_text: str) -> list[Chunk]:
        """Chunk document text into semantic chunks.

        Args:
            document_text: The document text to chunk.

        Returns:
            List of Chunk objects.

        Raises:
            ValueError: If document_text is empty.
        """
        if not document_text:
            raise ValueError("document_text cannot be empty")

        sentences = self._split_into_sentences(document_text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_chunk_text = ""
        current_token_count = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if (
                current_token_count + sentence_tokens > self._chunk_size
                and current_chunk_text
            ):
                chunks.append(
                    Chunk(
                        id=self._generate_chunk_id(chunks, chunk_index),
                        text=current_chunk_text.strip(),
                        token_count=current_token_count,
                        chunk_index=chunk_index,
                        page_number=None,
                    )
                )
                chunk_index += 1
                current_chunk_text = ""
                current_token_count = 0

            current_chunk_text += sentence + " "
            current_token_count += sentence_tokens

        if current_chunk_text:
            chunks.append(
                Chunk(
                    id=self._generate_chunk_id(chunks, chunk_index),
                    text=current_chunk_text.strip(),
                    token_count=current_token_count,
                    chunk_index=chunk_index,
                    page_number=None,
                )
            )

        return chunks

    def chunk_with_overlap(self, document_text: str) -> list[ChunkWithLinks]:
        """Chunk document text with overlap between consecutive chunks.

        Args:
            document_text: The document text to chunk.

        Returns:
            List of ChunkWithLinks objects with previous/next chunk references.

        Raises:
            ValueError: If document_text is empty.
        """
        if not document_text:
            raise ValueError("document_text cannot be empty")

        sentences = self._split_into_sentences(document_text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        overlap_sentences: list[str] = []
        current_chunk_text = ""
        current_token_count = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if (
                current_token_count + sentence_tokens > self._chunk_size
                and current_chunk_text
            ):
                overlap_size = self._overlap_tokens * 4

                overlap_for_next = []
                overlap_size_calc = overlap_size

                for overlap_candidate in reversed(overlap_sentences):
                    if overlap_size_calc <= 0:
                        break
                    overlap_for_next.insert(0, overlap_candidate)
                    overlap_size_calc -= len(overlap_candidate)

                overlap_sentences = overlap_for_next.copy()

                chunks.append(
                    Chunk(
                        id=self._generate_chunk_id(chunks, chunk_index),
                        text=current_chunk_text.strip(),
                        token_count=current_token_count,
                        chunk_index=chunk_index,
                        page_number=None,
                    )
                )
                chunk_index += 1

                current_chunk_text = ""
                current_token_count = 0
                for overlap_sent in overlap_sentences:
                    current_chunk_text += overlap_sent + " "
                    current_token_count += self._count_tokens(overlap_sent)

            if current_token_count + sentence_tokens <= self._max_chunk_size:
                current_chunk_text += sentence + " "
                current_token_count += sentence_tokens
                overlap_sentences.append(sentence)

        if current_chunk_text:
            chunks.append(
                Chunk(
                    id=self._generate_chunk_id(chunks, chunk_index),
                    text=current_chunk_text.strip(),
                    token_count=current_token_count,
                    chunk_index=chunk_index,
                    page_number=None,
                )
            )

        if len(chunks) > 1:
            chunks = self._merge_small_chunks(chunks, self._min_chunk_size)
            for i, chunk in enumerate(chunks):
                chunk.chunk_index = i

        result: list[ChunkWithLinks] = []
        for i, chunk in enumerate(chunks):
            link = ChunkWithLinks(
                chunk=chunk,
                previous_chunk_id=chunks[i - 1].id if i > 0 else None,
                next_chunk_id=chunks[i + 1].id if i < len(chunks) - 1 else None,
            )
            result.append(link)

        return result

    def _generate_chunk_id(self, chunks: list[Chunk], index: int) -> str:
        """Generate a unique chunk ID.

        Args:
            chunks: List of existing chunks.
            index: Chunk index.

        Returns:
            Unique chunk identifier.
        """
        unique_str = f"{index}-{len(chunks)}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]
